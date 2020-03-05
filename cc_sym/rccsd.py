#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
'''
Restricted CCSD with ctf
'''

from functools import reduce
import time
import numpy as np
from pyscf.cc import ccsd
from pyscf import lib as pyscflib
from pyscf import ao2mo, gto
from pyscf.ao2mo import _ao2mo
from symtensor import sym_ctf as lib
import ctf
from cc_sym.linalg_helper.diis import DIIS
import cc_sym.rintermediates as imd

from cc_sym import settings
comm = settings.comm
rank = settings.rank
size = settings.size
Logger = settings.Logger
static_partition = settings.static_partition

tensor = lib.tensor
zeros = lib.zeros

def kernel(mycc, eris, t1, t2):
    if eris is None: eris = mycc.ao2mo(self.mo_coeff)
    if t1 is None: t1, t2 = mycc.init_amps(eris)[1:]
    log = Logger(mycc.stdout, mycc.verbose)
    max_cycle = mycc.max_cycle
    tol = mycc.conv_tol
    tolnormt = mycc.conv_tol_normt
    if isinstance(mycc.diis, DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = DIIS(mycc)
        adiis.space = mycc.diis_space
    else:
        adiis = None
    cput1 = cput0 = (time.clock(), time.time())
    nocc, nvir = t1.shape
    eold = 0
    eccsd = 0
    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        normt = (t1new-t1).norm() + (t2new-t2).norm()
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            t1new = (1-alpha) * t1 + alpha * t1new
            t2new *= alpha
            t2new += (1-alpha) * t2
        t1, t2 = t1new, t2new
        t1new = t2new = None
        tmpt0 = (time.clock(), time.time())
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
        log.timer("running diis", *tmpt0)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info('cycle = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)

    return conv, eccsd, t1, t2

def update_amps(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    cput1 = cput0 = (time.clock(), time.time())
    log = Logger(cc.stdout, cc.verbose)
    fov = eris.fov.copy()
    foo = eris.foo.copy()
    fvv = eris.fvv.copy()
    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # Move energy terms to the other side
    Foo -= eris._foo
    Fvv -= eris._fvv

    # T1 equation
    t1new = fov.conj().copy()
    tmp    =   lib.einsum('kc,ka->ac', fov, t1)
    t1new +=-2*lib.einsum('ac,ic->ia', tmp, t1)
    t1new +=   lib.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -lib.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*lib.einsum('kc,kica->ia', Fov, t2)
    t1new +=  -lib.einsum('kc,ikca->ia', Fov, t2)
    tmp    =   lib.einsum('kc,ic->ki', Fov, t1)
    t1new +=   lib.einsum('ki,ka->ia', tmp, t1)
    t1new += 2*lib.einsum('kcai,kc->ia', eris.ovvo, t1)
    t1new +=  -lib.einsum('kiac,kc->ia', eris.oovv, t1)
    t1new += 2*lib.einsum('kdac,ikcd->ia', eris.ovvv, t2)
    t1new +=  -lib.einsum('kcad,ikcd->ia', eris.ovvv, t2)
    tmp    =   lib.einsum('kdac,kd->ac', eris.ovvv, t1)
    t1new += 2*lib.einsum('ac,ic->ia', tmp, t1)
    tmp    =   lib.einsum('kcad,kd->ca', eris.ovvv, t1)
    t1new +=  -lib.einsum('ca,ic->ia', tmp, t1)
    t1new +=-2*lib.einsum('kilc,klac->ia', eris.ooov, t2)
    t1new +=   lib.einsum('likc,klac->ia', eris.ooov, t2)
    tmp    =   lib.einsum('kilc,lc->ki', eris.ooov, t1)
    t1new +=-2*lib.einsum('ki,ka->ia', tmp, t1)
    tmp    =   lib.einsum('likc,lc->ik', eris.ooov, t1)
    t1new +=   lib.einsum('ik,ka->ia', tmp, t1)

    cput1 = log.timer("updating t1", *cput1)
    # T2 equation
    t2new = eris.ovov.conj().transpose(0,2,1,3).copy()
    if cc.cc2:
        Woooo2  = eris.oooo.transpose(0,2,1,3).copy()
        Woooo2 += lib.einsum('kilc,jc->klij', eris.ooov, t1)
        Woooo2 += lib.einsum('ljkc,ic->klij', eris.ooov, t1)
        tmp     = lib.einsum('kcld,ic->kild', eris.ovov, t1)
        Woooo2 += lib.einsum('kild,jd->klij', tmp, t1)
        tmp     = lib.einsum('klij,ka->alij', Woooo2, t1)
        t2new  += lib.einsum('alij,lb->ijab', tmp, t1)
        Wvvvv   = lib.einsum('kcbd,ka->abcd', eris.ovvv, -t1)
        Wvvvv   = Wvvvv + Wvvvv.transpose(1,0,3,2)
        Wvvvv  += eris.vvvv.transpose(0,2,1,3)
        tmp     = lib.einsum('abcd,ic->abid', Wvvvv, t1)
        t2new  += lib.einsum('abid,jd->ijab', tmp, t1)
        Lvv2    = fvv - lib.einsum('kc,ka->ac', fov, t1)
        Lvv2   -= fvv.diagonal(preserve_shape=True)
        tmp     = lib.einsum('ac,ijcb->ijab', Lvv2, t2)
        t2new  += (tmp + tmp.transpose(1,0,3,2))
        Loo2    = foo + lib.einsum('kc,ic->ki', fov, t1)
        Loo2   -= foo.diagonal(preserve_shape=True)
        tmp     = lib.einsum('ki,kjab->ijab', Loo2, t2)
        t2new  -= (tmp + tmp.transpose(1,0,3,2))
    else:
        Loo = imd.Loo(t1, t2, eris)
        Lvv = imd.Lvv(t1, t2, eris)
        Loo -= eris._foo
        Lvv -= eris._fvv
        Woooo = imd.cc_Woooo(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)
        Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
        tau = t2 + lib.einsum('ia,jb->ijab', t1, t1)
        t2new += lib.einsum('klij,klab->ijab', Woooo, tau)
        t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)
        tmp = lib.einsum('ac,ijcb->ijab', Lvv, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('ki,kjab->ijab', Loo, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp  = 2*lib.einsum('akic,kjcb->ijab', Wvoov, t2)
        tmp -=   lib.einsum('akci,kjcb->ijab', Wvovo, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('akic,kjbc->ijab', Wvoov, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))

    tmp2  = lib.einsum('kibc,ka->abic', eris.oovv, -t1)
    tmp2 += eris.ovvv.conj().transpose(1,3,0,2)
    tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp2  = lib.einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += eris.ooov.transpose(3,1,2,0).conj()
    tmp = lib.einsum('akij,kb->ijab', tmp2, t1)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    t1new /= eris.eia
    t2new /= eris.eijab
    cput1 = log.timer("updating t2", *cput1)
    return t1new, t2new

def energy(cc, t1, t2, eris):
    e = 2*lib.einsum('ia,ia', eris.fov, t1)
    tau = lib.einsum('ia,jb->ijab',t1,t1)
    tau += t2
    e += 2*lib.einsum('ijab,iajb', tau, eris.ovov)
    e +=  -lib.einsum('ijab,ibja', tau, eris.ovov)
    return e.real

class RCCSD(ccsd.CCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, SYMVERBOSE=0):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self._backend = 'ctf'
        self.symlib = None
        self.lib  = lib
        self.SYMVERBOSE = SYMVERBOSE

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    def dump_flags(self, verbose=None):
        if verbose is None: verbose=self.verbose
        log = Logger(self.stdout, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('CC2 = %g', self.cc2)
        log.info('CCSD nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not 0:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('direct = %d', self.direct)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_normt = %s', self.conv_tol_normt)
        log.info('diis_space = %d', self.diis_space)
        log.info('diis_start_cycle = %d', self.diis_start_cycle)
        log.info('diis_start_energy_diff = %g', self.diis_start_energy_diff)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, pyscflib.current_memory()[0])
        return self

    def init_amps(self, eris=None):
        if eris is None: eris = self.ao2mo()
        log = Logger(self.stdout, self.verbose)
        t1 = eris.fov.conj() / eris.eia
        t2 = eris.ovov.transpose(0,2,1,3).conj() / eris.eijab
        self.emp2  = 2*lib.einsum('ijab,iajb', t2, eris.ovov)
        self.emp2 -=   lib.einsum('ijab,ibja', t2, eris.ovov)
        log.info('Init t2, MP2 energy = %.15g', self.emp2.real)
        return self.emp2, t1, t2

    def amplitudes_to_vector(self, t1, t2, out=None):
        vector = ctf.hstack((t1.array.ravel(), t2.array.ravel()))
        return vector

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        nov = nocc * nvir
        t1 = vec[:nov].reshape(nocc,nvir)
        t2 = vec[nov:].reshape(nocc,nocc,nvir,nvir)
        t1  = tensor(t1, verbose=self.SYMVERBOSE)
        t2  = tensor(t2, verbose=self.SYMVERBOSE)
        return t1, t2

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False, cc2=False):
        return self.ccsd(t1, t2, eris, mbpt2, cc2)

    def ipccsd(self, nroots=1, koopmans=True, guess=None, left=False,
               eris=None, imds=None, partition=None, kptlist=None,
               dtype=None, **kwargs):
        from cc_sym.eom_rccsd import EOMIP
        myeom = EOMIP(self)
        return myeom.kernel(nroots, koopmans, guess, left,
                   eris, imds, partition, kptlist,
                   dtype, **kwargs)

    def eaccsd(self, nroots=1, koopmans=True, guess=None, left=False,
               eris=None, imds=None, partition=None, kptlist=None,
               dtype=None, **kwargs):
        from cc_sym.eom_rccsd import EOMEA
        myeom = EOMEA(self)
        return myeom.kernel(nroots, koopmans, guess, left,
                   eris, imds, partition, kptlist,
                   dtype, **kwargs)

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False, cc2=False):
        if mbpt2 and cc2:
            raise RuntimeError('MBPT2 and CC2 are mutually exclusive approximations to the CCSD ground state.')
        log = Logger(self.stdout, self.verbose)

        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.eris = eris
        self.dump_flags()
        if mbpt2:
            cctyp = 'MBPT2'
            self.e_corr, self.t1, self.t2 = self.init_amps(eris)
        else:
            if cc2:
                cctyp = 'CC2'
                self.cc2 = True
            else:
                cctyp = 'CCSD'
                self.cc2 = False
            self.converged, self.e_corr, self.t1, self.t2 = \
                    kernel(self, eris, t1, t2)
            if self.converged:
                log.info('%s converged', cctyp)
            else:
                log.info('%s not converged', cctyp)
        if self._scf.e_tot == 0:
            log.note('E_corr = %.16g', self.e_corr)
        else:
            log.note('E(%s) = %.16g  E_corr = %.16g',
                        cctyp, self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    energy = energy
    update_amps = update_amps


class _ChemistsERIs:
    def __init__(self, cc, mo_coeff=None):
        self.lib = cc.lib
        log = Logger(cc.stdout, cc.verbose)
        nocc, nmo = cc.nocc, cc.nmo
        nvir = nmo - nocc
        cput1 = cput0 = (time.clock(), time.time())

        if mo_coeff is None:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, cc.mo_coeff)
        else:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, mo_coeff)

        fock = None
        if rank==0:
            dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
            fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
            fock = reduce(np.dot, (mo_coeff.T, fockao, mo_coeff))

        fock = comm.bcast(fock, root=0)
        self.dtype = dtype = np.result_type(fock)

        self.foo = zeros([nocc,nocc], dtype, verbose=cc.SYMVERBOSE)
        self.fov = zeros([nocc,nvir], dtype, verbose=cc.SYMVERBOSE)
        self.fvv = zeros([nvir,nvir], dtype, verbose=cc.SYMVERBOSE)
        self.eia = zeros([nocc,nvir], verbose=cc.SYMVERBOSE)

        if rank==0:
            mo_e = fock.diagonal().real
            eia  = mo_e[:nocc,None]- mo_e[None,nocc:]
            self.foo.write(range(nocc*nocc), fock[:nocc,:nocc].ravel())
            self.fov.write(range(nocc*nvir), fock[:nocc,nocc:].ravel())
            self.fvv.write(range(nvir*nvir), fock[nocc:,nocc:].ravel())
            self.eia.write(range(nocc*nvir), eia.ravel())
        else:
            self.foo.write([], [])
            self.fov.write([], [])
            self.fvv.write([], [])
            self.eia.write([], [])
        cput1 = log.timer("Writing Fock", *cput1)
        self._foo = self.foo.diagonal(preserve_shape=True)
        self._fvv = self.fvv.diagonal(preserve_shape=True)
        eijab = self.eia.array.reshape(nocc,1,nvir,1) + self.eia.array.reshape(1,nocc,1,nvir)
        self.eijab = tensor(eijab, verbose=cc.SYMVERBOSE)

        ppoo, ppov, ppvv = _make_ao_ints(cc.mol, mo_coeff, nocc, dtype)
        cput1 = log.timer('making ao integrals', *cput1)
        mo = ctf.astensor(mo_coeff)
        orbo, orbv = mo[:,:nocc], mo[:,nocc:]

        tmp = ctf.einsum('uvmn,ui->ivmn', ppoo, orbo.conj())
        oooo = ctf.einsum('ivmn,vj->ijmn', tmp, orbo)
        ooov = ctf.einsum('ivmn,va->mnia', tmp, orbv)

        tmp = ctf.einsum('uvma,vb->ubma', ppov, orbv)
        ovov = ctf.einsum('ubma,ui->ibma', tmp, orbo.conj())
        tmp = ctf.einsum('uvma,ub->mabv', ppov, orbv.conj())
        ovvo = ctf.einsum('mabv,vi->mabi', tmp, orbo)

        tmp = ctf.einsum('uvab,ui->ivab', ppvv, orbo.conj())
        oovv = ctf.einsum('ivab,vj->ijab', tmp, orbo)

        tmp = ctf.einsum('uvab,vc->ucab', ppvv, orbv)
        ovvv = ctf.einsum('ucab,ui->icab', tmp, orbo.conj())
        vvvv = ctf.einsum('ucab,ud->dcab', tmp, orbv.conj())

        self.oooo = tensor(oooo, verbose=cc.SYMVERBOSE)
        self.ooov = tensor(ooov, verbose=cc.SYMVERBOSE)
        self.ovov = tensor(ovov, verbose=cc.SYMVERBOSE)
        self.oovv = tensor(oovv, verbose=cc.SYMVERBOSE)
        self.ovvo = tensor(ovvo, verbose=cc.SYMVERBOSE)
        self.ovvv = tensor(ovvv, verbose=cc.SYMVERBOSE)
        self.vvvv = tensor(vvvv, verbose=cc.SYMVERBOSE)
        log.timer('ao2mo transformation', *cput0)

def _make_ao_ints(mol, mo_coeff, nocc, dtype):
    NS = ctf.SYM.NS
    SY = ctf.SYM.SY

    ao_loc = mol.ao_loc_nr()
    mo = np.asarray(mo_coeff, order='F')
    nao, nmo = mo.shape
    nvir = nmo - nocc

    ppoo = ctf.tensor((nao,nao,nocc,nocc), sym=[SY,NS,NS,NS], dtype=dtype)
    ppov = ctf.tensor((nao,nao,nocc,nvir), sym=[SY,NS,NS,NS], dtype=dtype)
    ppvv = ctf.tensor((nao,nao,nvir,nvir), sym=[SY,NS,SY,NS], dtype=dtype)
    intor = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    blksize = int(max(4, min(nao/3, nao/size**.5, 2000e6/8/nao**3)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, blksize)
    tasks = []
    for k, (ish0, ish1, di) in enumerate(sh_ranges):
        for jsh0, jsh1, dj in sh_ranges[:k+1]:
            tasks.append((ish0,ish1,jsh0,jsh1))

    sqidx = np.arange(nao**2).reshape(nao,nao)
    trilidx = sqidx[np.tril_indices(nao)]
    vsqidx = np.arange(nvir**2).reshape(nvir,nvir)
    vtrilidx = vsqidx[np.tril_indices(nvir)]

    subtasks = list(static_partition(tasks))
    ntasks = max(comm.allgather(len(subtasks)))
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
            idx = sqidx[i0:i1,j0:j1][np.tril_indices(i1-i0)]

        ooidx = idx[:,None] * nocc**2 + np.arange(nocc**2)
        ovidx = idx[:,None] * (nocc*nvir) + np.arange(nocc*nvir)
        vvidx = idx[:,None] * nvir**2 + vtrilidx
        eri = eri.reshape(-1,nmo,nmo)
        ppoo.write(ooidx.ravel(), eri[:,:nocc,:nocc].ravel())
        ppov.write(ovidx.ravel(), eri[:,:nocc,nocc:].ravel())
        ppvv.write(vvidx.ravel(), pyscflib.pack_tril(eri[:,nocc:,nocc:]).ravel())
        idx = eri = None
    return ppoo, ppov, ppvv


if __name__ == '__main__':
    from pyscf import scf
    import os

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 4
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol)
    if rank==0:
        mf.kernel()

    comm.barrier()
    mf.mo_coeff  = comm.bcast(mf.mo_coeff, root=0)
    mf.mo_occ = comm.bcast(mf.mo_occ, root=0)

    mycc = RCCSD(mf)
    mycc.kernel()
