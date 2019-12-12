#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from __future__ import division

import time
import ctypes
import tempfile
from functools import reduce
import numpy

from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.cc import ccsd

import ctf
from ctf_helper import comm, rank, size, Logger, omp, static_partition
#comm = ctf.comm()
#rank = comm.rank()
#size = comm.np()
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

einsum = ctf.einsum
#einsum = lib.einsum

def update_amps(mycc, t1, t2, eris):
    time0 = time.clock(), time.time()
    log = Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc*nvir

    t1new = ctf.tensor(t1.shape)
    tau = t2 + einsum('ia,jb->ijab', t1, t1)
    t2new = einsum('acbd,ijab->ijcd', eris.vvvv, tau)
    t2new *= .5

    t1new += eris.fov
    foo = eris.foo + .5 * einsum('ia,ja->ij', eris.fov, t1)
    fvv = eris.fvv - .5 * einsum('ia,ib->ab', t1, eris.fov)

    foo += einsum('kc,jikc->ij',  2*t1, eris.ooov)
    foo += einsum('kc,jkic->ij', -1*t1, eris.ooov)
    woooo = einsum('ijka,la->ijkl', eris.ooov, t1)
    woooo = woooo + woooo.transpose(2,3,0,1)
    woooo += eris.oooo

    #for p0, p1 in lib.prange(0, nvir, vblk):
    fvv += einsum('kc,kcba->ab',  2*t1, eris.ovvv)
    fvv += einsum('kc,kbca->ab', -1*t1, eris.ovvv)

    woVoV = einsum('ijka,kb->ijab', eris.ooov, t1)
    woVoV -= einsum('jc,icab->ijab', t1, eris.ovvv)

    #:eris_ovvv = einsum('ial,bcl->iabc', ovL, vvL)
    #:tmp = einsum('ijcd,kcdb->kijb', tau, eris_ovvv)
    #:t2new += einsum('ka,kijb->jiba', -t1, tmp)
    tmp = einsum('ijcd,kcdb->kijb', tau, eris.ovvv)
    t2new -= einsum('ka,kijb->jiba', t1, tmp)
    tau = tmp = None

    wOVov  = einsum('ikjb,ka->ijba', eris.ooov, -1*t1)
    wOVov += einsum('jc,iabc->jiab', t1, eris.ovvv)
    t2new += wOVov.transpose(0,1,3,2)

    theta = t2.transpose(0,1,3,2) * 2 - t2
    t1new += einsum('ijcb,jcba->ia', theta, eris.ovvv)

    t2new += eris.ovov.transpose(0,2,1,3) * .5

    fov = eris.fov.copy()
    fov += einsum('kc,iakc->ia', t1, eris.ovov) * 2
    fov -= einsum('kc,icka->ia', t1, eris.ovov)

    t1new += einsum('jb,jiab->ia', fov, theta)
    t1new -= einsum('kijb,kjba->ia', eris.ooov, theta)
    theta = None

    wOVov += eris.ovov.transpose(0,2,3,1)
    wOVov -= .5 * einsum('icka,jkbc->jiab', eris.ovov, t2)
    tau = t2.transpose(0,2,1,3) * 2 - t2.transpose(0,3,1,2)
    tau -= einsum('ia,jb->ibja', t1*2, t1)
    wOVov += .5 * einsum('iakc,jbkc->jiab', eris.ovov, tau)

    theta = t2 * 2 - t2.transpose(0,1,3,2)
    t2new += einsum('ikac,jkcb->jiba', theta, wOVov)
    tau = theta = wOVov = None

    tau = einsum('ia,jb->ijab', t1*.5, t1) + t2
    theta = tau.transpose(0,1,3,2)*2 - tau
    fvv -= einsum('ijca,ibjc->ab', theta, eris.ovov)
    foo += einsum('iakb,jkba->ij', eris.ovov, theta)
    tau = theta = None

    tmp = einsum('ic,jkbc->jibk', t1, eris.oovv)
    t2new -= einsum('ka,jibk->jiab', t1, tmp)
    tmp = einsum('ic,jbkc->jibk', t1, eris.ovov)
    t2new -= einsum('ka,jibk->jiba', t1, tmp)
    tmp = None

    t1new += einsum('jb,iajb->ia',  2*t1, eris.ovov)
    t1new += einsum('jb,ijba->ia', -1*t1, eris.oovv)

    woVoV -= eris.oovv

    tau = t2 + einsum('ia,jb->ijab', t1, t1)
    woooo += einsum('iajb,klab->ikjl', eris.ovov, tau)
    t2new += .5 * einsum('kilj,klab->ijab', woooo, tau)
    tau -= t2 * .5
    woVoV += einsum('jkca,ickb->ijba', tau, eris.ovov)
    t2new += einsum('kicb,kjac->ijab', woVoV, t2)
    t2new += einsum('kica,kjcb->ijab', woVoV, t2)
    woooo = tau = woVoV = None

    ft_ij = foo + einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - einsum('ia,ib->ab', .5*t1, fov)
    t2new += einsum('ijac,bc->ijab', t2, ft_ab)
    t2new -= einsum('ki,kjab->ijab', ft_ij, t2)

    mo_e = eris.mo_energy
    eia = mo_e[:nocc].reshape(nocc,1) - mo_e[nocc:].reshape(1,nvir)
    t1new += einsum('ib,ab->ia', t1, fvv)
    t1new -= einsum('ja,ji->ia', t1, foo)
    t1new /= eia

    t2new = t2new + t2new.transpose(1,0,3,2)
    dijab = eia.reshape(nocc,1,nvir,1) + eia.reshape(1,nocc,1,nvir)
    t2new /= dijab

    if rank == 0:
        time0 = log.timer_debug1('update t1 t2', *time0)
    return t1new, t2new


class CCSD(ccsd.CCSD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None)
        self.diis = None

    update_amps = update_amps

    def ao2mo(self, mo_coeff=None):
        return _make_eris(self, mo_coeff)

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        mo_e = eris.mo_energy
        nocc = self.nocc
        nvir = mo_e.size - nocc

        eia = mo_e[:nocc].reshape(nocc,1) - mo_e[nocc:].reshape(1,nvir)
        t1 = eris.fov.conj() / eia

        dijab = eia.reshape(nocc,1,nvir,1) + eia.reshape(1,nocc,1,nvir)
        t2 = eris.ovov.conj().transpose(0,2,1,3) / dijab
        emp2 = einsum('ijab,iajb->', t2, eris.ovov) * 2
        emp2-= einsum('ijab,ibja->', t2, eris.ovov)
        self.emp2 = emp2.to_nparray()

        if rank == 0:
            logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
            logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def energy(self, t1, t2, eris):
        nocc, nvir = t1.shape
        e  = einsum('ia,ia->', eris.fov, t1) * 2
        t1 = ctf.astensor(t1)
        t2 = ctf.astensor(t2)

        e += einsum('ijab,iajb->', t2, eris.ovov) * 2
        e -= einsum('ijab,ibja->', t2, eris.ovov)
        e += einsum('iajb,ia,jb->', eris.ovov, t1, t1) * 2
        e -= einsum('jaib,ia,jb->', eris.ovov, t1, t1)
        return e.to_nparray()

    def kernel(self):
        eris = self.ao2mo(self.mo_coeff)
        log = logger.new_logger(self, self.verbose)
        cput1 = cput0 = (time.clock(), time.time())

        max_cycle = self.max_cycle
        tol = self.conv_tol
        tolnormt = self.conv_tol_normt

        t1, t2 = self.init_amps(eris)[1:]
        nocc, nvir = t1.shape
        eold = 0
        eccsd = 0

        conv = False
        for istep in range(self.max_cycle):
            t1new, t2new = self.update_amps(t1, t2, eris)
            normt = (t1new-t1).norm2() + (t2new-t2).norm2()
            t1, t2 = t1new, t2new
            t1new = t2new = None
            eold, eccsd = eccsd, self.energy(t1, t2, eris)
            if rank == 0:
                log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                         istep, eccsd, eccsd - eold, normt)
                cput1 = log.timer('CCSD iter', *cput1)
            if abs(eccsd-eold) < tol and normt < tolnormt:
                conv = True
                break
        if rank == 0:
            log.timer('CCSD', *cput0)
        self.converged = conv
        self.e_corr = eccsd
        self.t1 = t1
        self.t2 = t2
        return conv, eccsd, t1, t2

CC = CCSD


def _make_ao_ints(mol, mo_coeff, nocc):
    NS = ctf.SYM.NS
    SY = ctf.SYM.SY
    ao_loc = mol.ao_loc_nr()
    mo = numpy.asarray(mo_coeff, order='F')
    nao, nmo = mo.shape
    nvir = nmo - nocc

    ppoo = ctf.tensor((nao,nao,nocc,nocc), sym=[SY,NS,NS,NS], dtype='d')
    ppov = ctf.tensor((nao,nao,nocc,nvir), sym=[SY,NS,NS,NS], dtype='d')
    ppvv = ctf.tensor((nao,nao,nvir,nvir), sym=[SY,NS,SY,NS], dtype='d')
    print 'ao ints init', rank, lib.current_memory(), 'nao', nao
    intor = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    blksize = int(max(4, min(nao/3, nao/size**.5, 2000e6/8/nao**3)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, blksize)
    tasks = []
    for k, (ish0, ish1, di) in enumerate(sh_ranges):
        for jsh0, jsh1, dj in sh_ranges[:k+1]:
            tasks.append((ish0,ish1,jsh0,jsh1))

    sqidx = numpy.arange(nao**2).reshape(nao,nao)
    trilidx = sqidx[numpy.tril_indices(nao)]
    vsqidx = numpy.arange(nvir**2).reshape(nvir,nvir)
    vtrilidx = vsqidx[numpy.tril_indices(nvir)]

    subtasks = list(static_partition(tasks))
    ntasks = max(comm.allgather(len(subtasks)))
    print 'ao ints', rank, lib.current_memory(), 'nao', nao, 'ntasks', len(subtasks)
    for itask in range(ntasks):
        if itask >= len(subtasks):
            ppoo.write([], [])
            ppov.write([], [])
            ppvv.write([], [])
            continue

        shls_slice = subtasks[itask]
        ish0, ish1, jsh0, jsh1 = shls_slice
        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
        di = i1 - i0
        dj = j1 - j0
        if i0 != j0:
            eri = gto.moleintor.getints4c(intor, mol._atm, mol._bas, mol._env,
                                          shls_slice=shls_slice, aosym='s2kl',
                                          ao_loc=ao_loc, cintopt=ao2mopt._cintopt)
            idx = sqidx[i0:i1,j0:j1].ravel()

            eri = _ao2mo.nr_e2(eri.reshape(di*dj,-1), mo, (0,nmo,0,nmo), 's2kl', 's1')
        else:
            eri = gto.moleintor.getints4c(intor, mol._atm, mol._bas, mol._env,
                                          shls_slice=shls_slice, aosym='s4',
                                          ao_loc=ao_loc, cintopt=ao2mopt._cintopt)
            eri = _ao2mo.nr_e2(eri, mo, (0,nmo,0,nmo), 's4', 's1')
            idx = sqidx[i0:i1,j0:j1][numpy.tril_indices(i1-i0)]

        print 'ao ints memory before ctf write', rank, itask, lib.current_memory()
        ooidx = idx[:,None] * nocc**2 + numpy.arange(nocc**2)
        ovidx = idx[:,None] * (nocc*nvir) + numpy.arange(nocc*nvir)
        vvidx = idx[:,None] * nvir**2 + vtrilidx
        eri = eri.reshape(-1,nmo,nmo)
        ppoo.write(ooidx.ravel(), eri[:,:nocc,:nocc].ravel())
        ppov.write(ovidx.ravel(), eri[:,:nocc,nocc:].ravel())
        ppvv.write(vvidx.ravel(), lib.pack_tril(eri[:,nocc:,nocc:]).ravel())
        idx = eri = None
    print 'ao_ints Done'
    return ppoo, ppov, ppvv

def _make_eris(mycc, mo_coeff=None):
    mol = mycc.mol
    NS = ctf.SYM.NS
    SY = ctf.SYM.SY

    eris = ccsd._ChemistsERIs()
    if mo_coeff is None:
        mo_coeff = mycc.mo_coeff
    eris.mo_coeff = ccsd._mo_without_core(mycc, mo_coeff)
    nao, nmo = eris.mo_coeff.shape
    nocc = mycc.nocc
    nvir = nmo - nocc
    nvir_pair = nvir*(nvir+1)//2
    nao_pair = nao * (nao+1) // 2

    mo = ctf.astensor(eris.mo_coeff)
    ppoo, ppov, ppvv = _make_ao_ints(mol, eris.mo_coeff, nocc)
    eris.nocc = mycc.nocc
    eris.mol = mycc.mol

    eris.fock = ctf.tensor((nmo,nmo))
    with omp(16):
        if rank == 0:
            dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
            fockao = mycc._scf.get_hcore() + mycc._scf.get_veff(mycc.mol, dm)
            fock = reduce(numpy.dot, (eris.mo_coeff.T, fockao, eris.mo_coeff))
            eris.fock.write(numpy.arange(nmo**2), fock.ravel())
        else:
            eris.fock.write([], [])

    orbo = mo[:,:nocc]
    orbv = mo[:,nocc:]

    print 'before contraction', rank, lib.current_memory()
    tmp = ctf.tensor([nao,nocc,nvir,nvir], sym=[NS,NS,SY,NS])
    otmp = ctf.einsum('pqrs,qj->pjrs', ppvv, orbo, out=tmp)
    eris.oovv = ctf.tensor([nocc,nocc,nvir,nvir], sym=[NS,NS,SY,NS])
    eris.ovvv = ctf.einsum('pjrs,pi->ijrs', otmp, orbo, out=eris.oovv)
    otmp = tmp = None

    print '___________  vvvv', rank, lib.current_memory()
    tmp = ctf.tensor([nao,nvir,nvir,nvir], sym=[NS,NS,SY,NS])
    print '___________  vvvv sub1', rank, lib.current_memory()
    vtmp = ctf.einsum('pqrs,qj->pjrs', ppvv, orbv, out=tmp)

    eris.ovvv = ctf.tensor([nocc,nvir,nvir,nvir], sym=[NS,NS,SY,NS])
    eris.ovvv = ctf.einsum('pjrs,pi->ijrs', vtmp, orbo, out=eris.ovvv)
    ppvv = None

    tmp = ctf.tensor([nvir,nvir,nvir,nvir], sym=[NS,NS,SY,NS])
    print '___________  vvvv sub2', rank, lib.current_memory()
    vtmp = ctf.einsum('pjrs,pi->ijrs', vtmp, orbv, out=tmp)
    eris.vvvv = ctf.tensor(sym=[SY,NS,SY,NS], copy=vtmp)
    vtmp = tmp = None

    vtmp = ctf.einsum('pqrs,qj->pjrs', ppov, orbv)
    eris.ovov = ctf.einsum('pjrs,pi->ijrs', vtmp, orbo)
    vtmp = None

    otmp = ctf.einsum('pqrs,qj->pjrs', ppov, orbo)
    eris.ooov = ctf.einsum('pjrs,pi->ijrs', otmp, orbo)
    ppov = otmp = None

    otmp = ctf.einsum('pqrs,qj->pjrs', ppoo, orbo)
    eris.oooo = ctf.einsum('pjrs,pi->ijrs', otmp, orbo)
    ppoo = otmp = None


    print '___________  fock', rank, lib.current_memory()
    eris.mo_energy = eris.fock.diagonal()
    eris.foo = eris.fock[:nocc,:nocc].copy()
    eris.foo.i('ii') << (eris.mo_energy[:nocc]*-1).i('i')
    eris.fvv = eris.fock[nocc:,nocc:].copy()
    eris.fvv.i('ii') << (eris.mo_energy[nocc:]*-1).i('i')
    eris.fov = ctf.astensor(eris.fock[:nocc,nocc:])
    return eris


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'aug-ccpvqz'
    mol.build()
    mol.verbose = 4
    mol.max_memory = 1
    mf = scf.RHF(mol)
    #mf.chkfile = 'h2o-aqz.chk'
    #mf.run() #.density_fit().run()
    mf.__dict__.update(scf.chkfile.load('h2o-aqz.chk', 'scf'))
    CCSD(mf).run()
    exit()

    nocc, nvir = 5, mol.nao_nr() - 5
    nmo = nocc + nvir
    nmo_pair = nmo*(nmo+1)//2
    numpy.random.seed(12)
    mf._eri = numpy.random.random(nmo_pair*(nmo_pair+1)//2)
    mf.mo_coeff = numpy.random.random((nmo,nmo))
    mf.mo_energy = numpy.arange(0., nmo)
    mf.mo_occ = numpy.zeros(nmo)
    mf.mo_occ[:nocc] = 2
    vhf = mf.get_veff(mol, mf.make_rdm1())
    cinv = numpy.linalg.inv(mf.mo_coeff)
    mf.get_hcore = lambda *args: (reduce(numpy.dot, (cinv.T*mf.mo_energy, cinv)) - vhf)
    mycc = CCSD(mf)
    if 0:
        print(abs(ao2mo.restore(1, mol.intor('int2e_sph'), mol.nao_nr()) -
                  _make_ao_ints(mol).to_nparray()).max())
    eris = mycc.ao2mo()

    t1 = numpy.random.random((nocc,nvir)) * .1
    t2 = numpy.random.random((nocc,nocc,nvir,nvir)) * .1
    t2 = t2 + t2.transpose(1,0,3,2)

    emp2, t1a, t2a = mycc.init_amps(eris)
    t1a = t1a.to_nparray()
    t2a = t2a.to_nparray()
    if rank == 0:
        print(emp2 - -5817752.3455295097)
        print(lib.finger(t1a) - -0.1436792653009763)
        print(lib.finger(t2a) - -47.895887814531207)

    t1a, t2a = update_amps(mycc, ctf.astensor(t1), ctf.astensor(t2), eris)
    e = mycc.energy(t1a, t2a, eris)
    t1a = t1a.to_nparray()
    t2a = t2a.to_nparray()
    if rank == 0:
        print(lib.finger(t1a) - -14862.627009452939)
        print(lib.finger(t2a) - 6355.2161046378124 )
        print(e*1e-8 - 11157.900866549678)
    exit()

    eris.naux = 200
    vvL = numpy.random.random((nvir*(nvir+1)//2,eris.naux))
    eris.vvL = ctf.astensor(lib.unpack_tril(vvL,axis=0))
    eris.ovL = ctf.astensor(numpy.random.random((nocc,nvir,eris.naux)))

    t1a, t2a = update_amps(mycc, ctf.astensor(t1), ctf.astensor(t2), eris)
    t1a = t1a.to_nparray()
    t2a = t2a.to_nparray()
    if rank == 0:
        print(lib.finger(t1a) - -1909.8213601597431)
        print(lib.finger(t2a) - 7246.8212135621161 )

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'mycc-pvqz'
    mol.build()
    mf = scf.RHF(mol).density_fit().run()

    mycc = CCSD(mf)
    mycc.max_cycle = 2
    mycc.verbose = 4
    mycc.run()
    print(mycc.e_corr - -0.28122723906352487)

#    mycc = dfccsd.RCCSD(mf).run()
#    print(mycc.e_corr - -0.28122723906352487)

#    natm, bas, nproc = 50, 'ccpvdz', 14
#    mol = gto.M(atom=[['H', 0, 0, i*1.8] for i in range(int(natm))],
#                unit='bohr', basis=bas, verbose=3)
#    mf = scf.RHF(mol).density_fit()
#    with omp(int(nproc)):
#        if rank == 0:
#            mf.verbose = 4
#            mf.run()
#    mf.mo_occ = comm.bcast(mf.mo_occ)
#    mf.mo_coeff = comm.bcast(mf.mo_coeff)
#    mf.mo_energy = comm.bcast(mf.mo_energy)
#
#
#    mycc = CCSD(mf)
#    mycc.verbose = 6
#    mycc.conv_tol = 1e-4
#    mycc.conv_tol_normt = 1e-2
#    mycc.max_cycle = 2
#    mycc.run()
