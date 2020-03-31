#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
from functools import reduce
import time
import numpy as np
from pyscf import lib as pyscflib
from symtensor import sym_ctf as lib
from symtensor.symlib import SYMLIB
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
import ctf
from cc_sym import rccsd
from pyscf.pbc import tools, df
from functools import reduce
from cc_sym import settings
comm = settings.comm
rank = settings.rank
size = settings.size
Logger = settings.Logger
static_partition = settings.static_partition

tensor = lib.tensor
zeros = lib.zeros


def energy(cc, t1, t2, eris):
    log = Logger(cc.stdout, cc.verbose)
    nkpts = cc.nkpts
    e = 2*lib.einsum('ia,ia', eris.fov, t1)
    tau = lib.einsum('ia,jb->ijab',t1,t1)
    tau += t2
    e += 2*lib.einsum('ijab,iajb', tau, eris.ovov)
    e +=  -lib.einsum('ijab,ibja', tau, eris.ovov)
    if abs(e.imag)>1e-4:
        log.warn('Non-zero imaginary part found in KRCCSD energy %s', e)
    return e.real / nkpts

class KRCCSD(rccsd.RCCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, SYMVERBOSE=0):
        rccsd.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ, SYMVERBOSE)
        self.kpts = mf.kpts
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])
        self.symlib = SYMLIB('ctf')
        self.make_symlib()

    energy = energy
    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    @property
    def nkpts(self):
        return len(self.kpts)

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        log = Logger(self.stdout, self.verbose)
        nocc = self.nocc
        nvir = self.nmo - nocc
        kpts = self.kpts
        nkpts = self.nkpts
        gvec = self._scf.cell.reciprocal_vectors()
        sym1 = ['+-', [kpts,]*2, None, gvec]
        t1 = lib.zeros([nocc,nvir], eris.dtype, sym1, symlib=self.symlib, verbose=self.SYMVERBOSE)
        t2 = eris.ovov.transpose(0,2,1,3).conj() / eris.eijab
        self.emp2  = 2*lib.einsum('ijab,iajb', t2, eris.ovov)
        self.emp2 -=   lib.einsum('ijab,ibja', t2, eris.ovov)
        self.emp2 = self.emp2.real/nkpts
        log.info('Init t2, MP2 energy (with fock eigenvalue shift) = %.15g', self.emp2)
        log.timer('init mp2', *time0)
        return self.emp2, t1, t2

    def make_symlib(self):
        kpts = self.kpts
        gvec = self._scf.cell.reciprocal_vectors()
        sym1 = ['+-',[kpts,]*2, None, gvec]
        sym2 = ['++--',[kpts,]*4, None, gvec]
        sym3 = ['++-',[kpts,]*3, None, gvec]
        self.symlib.update(sym1, sym2, sym3)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        nkpts = self.nkpts
        nov = nkpts * nocc * nvir
        t1 = vec[:nov].reshape(nkpts,nocc,nvir)
        t2 = vec[nov:].reshape(nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir)

        kpts = self.kpts
        gvec = self._scf.cell.reciprocal_vectors()
        sym1 = ['+-',[kpts,]*2, None, gvec]
        sym2 = ['++--',[kpts,]*4, None, gvec]
        t1  = tensor(t1, sym1, symlib=self.symlib, verbose=self.SYMVERBOSE)
        t2  = tensor(t2, sym2, symlib=self.symlib, verbose=self.SYMVERBOSE)
        return t1, t2

    def ipccsd(self, nroots=1, koopmans=False, guess=None, left=False,
               eris=None, partition=None, kptlist=None):
        from cc_sym import eom_kccsd_rhf
        return eom_kccsd_rhf.EOMIP(self).kernel(nroots=nroots, koopmans=koopmans,
                                                guess=guess, left=left,
                                                eris=eris, partition=partition,
                                                kptlist=kptlist)

    def eaccsd(self, nroots=1, koopmans=False, guess=None, left=False,
               eris=None, partition=None, kptlist=None):
        from cc_sym import eom_kccsd_rhf
        return eom_kccsd_rhf.EOMEA(self).kernel(nroots=nroots, koopmans=koopmans,
                                                guess=guess, left=left,
                                                eris=eris, partition=partition,
                                                kptlist=kptlist)

def _make_fftdf_eris(mycc, eris):
    mydf = mycc._scf.with_df
    mo_coeff = eris.mo_coeff
    kpts = mycc.kpts
    logger = Logger(mycc.stdout, mycc.verbose)
    cell = mydf.cell
    gvec = cell.reciprocal_vectors()
    nao = cell.nao_nr()
    coords = cell.gen_uniform_grids(mydf.mesh)
    ngrids = len(coords)
    nkpts = len(kpts)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    cput1 = cput0 = (time.clock(), time.time())
    ijG = ctf.zeros([nkpts,nkpts,nocc,nocc,ngrids], dtype=np.complex128)
    iaG = ctf.zeros([nkpts,nkpts,nocc,nvir,ngrids], dtype=np.complex128)
    abG = ctf.zeros([nkpts,nkpts,nvir,nvir,ngrids], dtype=np.complex128)

    ijR = ctf.zeros([nkpts,nkpts,nocc,nocc,ngrids], dtype=np.complex128)
    iaR = ctf.zeros([nkpts,nkpts,nocc,nvir,ngrids], dtype=np.complex128)
    aiR = ctf.zeros([nkpts,nkpts,nvir,nocc,ngrids], dtype=np.complex128)
    abR = ctf.zeros([nkpts,nkpts,nvir,nvir,ngrids], dtype=np.complex128)

    jobs = []
    for ki in range(nkpts):
        for kj in range(ki,nkpts):
            jobs.append([ki,kj])

    tasks = list(static_partition(jobs))
    ntasks = max(comm.allgather(len(tasks)))
    idx_ooG = np.arange(nocc*nocc*ngrids)
    idx_ovG = np.arange(nocc*nvir*ngrids)
    idx_vvG = np.arange(nvir*nvir*ngrids)

    for itask in range(ntasks):
        if itask >= len(tasks):
            ijR.write([], [])
            iaR.write([], [])
            aiR.write([], [])
            abR.write([], [])
            ijR.write([], [])
            iaR.write([], [])
            aiR.write([], [])
            abR.write([], [])

            ijG.write([], [])
            iaG.write([], [])
            abG.write([], [])
            ijG.write([], [])
            iaG.write([], [])
            abG.write([], [])
            continue
        ki, kj = tasks[itask]
        kpti, kptj = kpts[ki], kpts[kj]
        ao_kpti = mydf._numint.eval_ao(cell, coords, kpti)[0]
        ao_kptj = mydf._numint.eval_ao(cell, coords, kptj)[0]
        q = kptj - kpti
        coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
        wcoulG = coulG * (cell.vol/ngrids)
        fac = np.exp(-1j * np.dot(coords, q))
        mo_kpti = np.dot(ao_kpti, mo_coeff[ki]).T
        mo_kptj = np.dot(ao_kptj, mo_coeff[kj]).T
        mo_pairs = np.einsum('ig,jg->ijg', mo_kpti.conj(), mo_kptj)
        mo_pairs_G = tools.fft(mo_pairs.reshape(-1,ngrids)*fac, mydf.mesh)

        off = ki * nkpts + kj
        ijR.write(off*idx_ooG.size+idx_ooG, mo_pairs[:nocc,:nocc].ravel())
        iaR.write(off*idx_ovG.size+idx_ovG, mo_pairs[:nocc,nocc:].ravel())
        aiR.write(off*idx_ovG.size+idx_ovG, mo_pairs[nocc:,:nocc].ravel())
        abR.write(off*idx_vvG.size+idx_vvG, mo_pairs[nocc:,nocc:].ravel())

        off = kj * nkpts + ki
        mo_pairs = mo_pairs.transpose(1,0,2).conj()
        ijR.write(off*idx_ooG.size+idx_ooG, mo_pairs[:nocc,:nocc].ravel())
        iaR.write(off*idx_ovG.size+idx_ovG, mo_pairs[:nocc,nocc:].ravel())
        aiR.write(off*idx_ovG.size+idx_ovG, mo_pairs[nocc:,:nocc].ravel())
        abR.write(off*idx_vvG.size+idx_vvG, mo_pairs[nocc:,nocc:].ravel())

        mo_pairs = None
        mo_pairs_G*= wcoulG
        v = tools.ifft(mo_pairs_G, mydf.mesh)
        v *= fac.conj()
        v = v.reshape(nmo,nmo,ngrids)

        off = ki * nkpts + kj
        ijG.write(off*idx_ooG.size+idx_ooG, v[:nocc,:nocc].ravel())
        iaG.write(off*idx_ovG.size+idx_ovG, v[:nocc,nocc:].ravel())
        abG.write(off*idx_vvG.size+idx_vvG, v[nocc:,nocc:].ravel())

        off = kj * nkpts + ki
        v = v.transpose(1,0,2).conj()
        ijG.write(off*idx_ooG.size+idx_ooG, v[:nocc,:nocc].ravel())
        iaG.write(off*idx_ovG.size+idx_ovG, v[:nocc,nocc:].ravel())
        abG.write(off*idx_vvG.size+idx_vvG, v[nocc:,nocc:].ravel())

    cput1 = logger.timer("Generating ijG", *cput1)
    sym1 = ["+-+", [kpts,]*3, None, gvec]
    sym2 = ["+--", [kpts,]*3, None, gvec]

    ooG = tensor(ijG, sym1, verbose=mycc.SYMVERBOSE)
    ovG = tensor(iaG, sym1, verbose=mycc.SYMVERBOSE)
    vvG = tensor(abG, sym1, verbose=mycc.SYMVERBOSE)

    ooR = tensor(ijR, sym2, verbose=mycc.SYMVERBOSE)
    ovR = tensor(iaR, sym2, verbose=mycc.SYMVERBOSE)
    voR = tensor(aiR, sym2, verbose=mycc.SYMVERBOSE)
    vvR = tensor(abR, sym2, verbose=mycc.SYMVERBOSE)

    eris.oooo = lib.einsum('ijg,klg->ijkl', ooG, ooR)/ nkpts
    eris.ooov = lib.einsum('ijg,kag->ijka', ooG, ovR)/ nkpts
    eris.oovv = lib.einsum('ijg,abg->ijab', ooG, vvR)/ nkpts
    ooG = ooR = ijG = ijR = None
    eris.ovvo = lib.einsum('iag,bjg->iabj', ovG, voR)/ nkpts
    eris.ovov = lib.einsum('iag,jbg->iajb', ovG, ovR)/ nkpts
    ovR = iaR = voR = aiR = None
    eris.ovvv = lib.einsum('iag,bcg->iabc', ovG, vvR)/ nkpts
    ovG = iaG = None
    eris.vvvv = lib.einsum('abg,cdg->abcd', vvG, vvR)/ nkpts
    cput1 = logger.timer("ijG to eri", *cput1)

def _make_df_eris(mycc, eris):
    from cc_sym import mpigdf
    mydf = mycc._scf.with_df
    mo_coeff = eris.mo_coeff
    if mydf.j3c is None: mydf.build()
    gvec = mydf.cell.reciprocal_vectors()
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    kpts = mydf.kpts
    nkpts = len(kpts)
    nao, naux = mydf.j3c.shape[2:]
    ijL = ctf.zeros([nkpts,nkpts,nocc,nocc,naux], dtype=mydf.j3c.dtype)
    iaL = ctf.zeros([nkpts,nkpts,nocc,nvir,naux], dtype=mydf.j3c.dtype)
    aiL = ctf.zeros([nkpts,nkpts,nvir,nocc,naux], dtype=mydf.j3c.dtype)
    abL = ctf.zeros([nkpts,nkpts,nvir,nvir,naux], dtype=mydf.j3c.dtype)
    jobs = []
    for ki in range(nkpts):
        for kj in range(ki,nkpts):
            jobs.append([ki,kj])
    tasks = static_partition(jobs)
    ntasks = max(comm.allgather((len(tasks))))
    idx_j3c = np.arange(nao**2*naux)
    idx_ooL = np.arange(nocc**2*naux)
    idx_ovL = np.arange(nocc*nvir*naux)
    idx_vvL = np.arange(nvir**2*naux)
    log = Logger(mydf.stdout, mydf.verbose)
    cput1 = cput0 = (time.clock(), time.time())
    for itask in range(ntasks):
        if itask >= len(tasks):
            mydf.j3c.read([])
            ijL.write([], [])
            iaL.write([], [])
            aiL.write([], [])
            abL.write([], [])

            ijL.write([], [])
            iaL.write([], [])
            aiL.write([], [])
            abL.write([], [])
            continue
        ki, kj = tasks[itask]
        ijid, ijdagger = mpigdf.get_member(kpts[ki], kpts[kj], mydf.kptij_lst)
        uvL = mydf.j3c.read(ijid*idx_j3c.size+idx_j3c).reshape(nao,nao,naux)
        if ijdagger: uvL = uvL.transpose(1,0,2).conj()
        pvL = np.einsum("up,uvL->pvL", mo_coeff[ki].conj(), uvL, optimize=True)
        uvL = None
        pqL = np.einsum('vq,pvL->pqL', mo_coeff[kj], pvL, optimize=True)

        off = ki * nkpts + kj
        ijL.write(off*idx_ooL.size+idx_ooL, pqL[:nocc,:nocc].ravel())
        iaL.write(off*idx_ovL.size+idx_ovL, pqL[:nocc,nocc:].ravel())
        aiL.write(off*idx_ovL.size+idx_ovL, pqL[nocc:,:nocc].ravel())
        abL.write(off*idx_vvL.size+idx_vvL, pqL[nocc:,nocc:].ravel())

        off = kj * nkpts + ki
        pqL = pqL.transpose(1,0,2).conj()
        ijL.write(off*idx_ooL.size+idx_ooL, pqL[:nocc,:nocc].ravel())
        iaL.write(off*idx_ovL.size+idx_ovL, pqL[:nocc,nocc:].ravel())
        aiL.write(off*idx_ovL.size+idx_ovL, pqL[nocc:,:nocc].ravel())
        abL.write(off*idx_vvL.size+idx_vvL, pqL[nocc:,nocc:].ravel())

    cput1 = log.timer("j3c transformation", *cput1)
    sym1 = ["+-+", [kpts,]*3, None, gvec]
    sym2 = ["+--", [kpts,]*3, None, gvec]

    ooL = tensor(ijL, sym1, verbose=mycc.SYMVERBOSE)
    ovL = tensor(iaL, sym1, verbose=mycc.SYMVERBOSE)
    voL = tensor(aiL, sym1, verbose=mycc.SYMVERBOSE)
    vvL = tensor(abL, sym1, verbose=mycc.SYMVERBOSE)

    ooL2 = tensor(ijL, sym2, verbose=mycc.SYMVERBOSE)
    ovL2 = tensor(iaL, sym2, verbose=mycc.SYMVERBOSE)
    voL2 = tensor(aiL, sym2, verbose=mycc.SYMVERBOSE)
    vvL2 = tensor(abL, sym2, verbose=mycc.SYMVERBOSE)

    eris.oooo = lib.einsum('ijg,klg->ijkl', ooL, ooL2) / nkpts
    eris.ooov = lib.einsum('ijg,kag->ijka', ooL, ovL2) / nkpts
    eris.oovv = lib.einsum('ijg,abg->ijab', ooL, vvL2) / nkpts
    eris.ovvo = lib.einsum('iag,bjg->iabj', ovL, voL2) / nkpts
    eris.ovov = lib.einsum('iag,jbg->iajb', ovL, ovL2) / nkpts
    eris.ovvv = lib.einsum('iag,bcg->iabc', ovL, vvL2) / nkpts
    eris.vvvv = lib.einsum('abg,cdg->abcd', vvL, vvL2) / nkpts

    cput1 = log.timer("integral transformation", *cput1)

class _ChemistsERIs:
    def __init__(self, cc, mo_coeff=None):
        from pyscf.pbc.cc.ccsd import _adjust_occ
        import pyscf.pbc.tools.pbc as tools
        if mo_coeff is None:
            mo_coeff = cc.mo_coeff
        cput0 = (time.clock(), time.time())
        symlib = cc.symlib
        log = Logger(cc.stdout, cc.verbose)
        self.lib = lib
        nocc, nmo, nkpts = cc.nocc, cc.nmo, cc.nkpts
        nvir = nmo - nocc
        cell, kpts = cc._scf.cell, cc.kpts
        gvec = cell.reciprocal_vectors()
        sym1 = ['+-', [kpts,]*2, None, gvec]
        sym2 = ['+-+-', [kpts,]*4, None, gvec]
        mo_coeff = self.mo_coeff = padded_mo_coeff(cc, mo_coeff)
        nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")
        madelung = tools.madelung(cell, kpts)
        fock = None
        if rank==0:
            dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
            with pyscflib.temporary_env(cc._scf, exxdiv=None):
                fockao = cc._scf.get_hcore() + cc._scf.get_veff(cell, dm)
            fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                                for k, mo in enumerate(mo_coeff)])
        fock = comm.bcast(fock, root=0)
        self.dtype = dtype = np.result_type(*(mo_coeff, fock)).char
        self.foo = zeros([nocc,nocc], dtype, sym1, symlib=symlib, verbose=cc.SYMVERBOSE)
        self.fov = zeros([nocc,nvir], dtype, sym1, symlib=symlib, verbose=cc.SYMVERBOSE)
        self.fvv = zeros([nvir,nvir], dtype, sym1, symlib=symlib, verbose=cc.SYMVERBOSE)
        self.eia = zeros([nocc,nvir], np.float64, sym1, symlib=symlib, verbose=cc.SYMVERBOSE)
        self._foo = zeros([nocc,nocc], dtype, sym1, symlib=symlib, verbose=cc.SYMVERBOSE)
        self._fvv = zeros([nvir,nvir], dtype, sym1, symlib=symlib, verbose=cc.SYMVERBOSE)

        foo = fock[:,:nocc,:nocc]
        fov = fock[:,:nocc,nocc:]
        fvv = fock[:,nocc:,nocc:]

        mo_energy = [fock[k].diagonal().real for k in range(nkpts)]
        mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                          for k, mo_e in enumerate(mo_energy)]

        mo_e_o = [e[:nocc] for e in mo_energy]
        mo_e_v = [e[nocc:] + cc.level_shift for e in mo_energy]
        foo_ = np.asarray([np.diag(e) for e in mo_e_o])
        fvv_ = np.asarray([np.diag(e) for e in mo_e_v])
        eia = np.zeros([nkpts,nocc,nvir])
        for ki in range(nkpts):
            eia[ki] = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                        [0,nvir,ki,mo_e_v,nonzero_vpadding],
                        fac=[1.0,-1.0])
        if rank ==0:
            self.foo.write(range(foo.size), foo.ravel())
            self.fov.write(range(fov.size), fov.ravel())
            self.fvv.write(range(fvv.size), fvv.ravel())
            self.eia.write(range(eia.size), eia.ravel())
            self._foo.write(range(foo_.size), foo_.ravel())
            self._fvv.write(range(fvv_.size), fvv_.ravel())
        else:
            self.foo.write([],[])
            self.fov.write([],[])
            self.fvv.write([],[])
            self.eia.write([],[])
            self._foo.write([],[])
            self._fvv.write([],[])

        self.eijab = zeros([nocc,nocc,nvir,nvir], np.float64, sym2, symlib=symlib, verbose=cc.SYMVERBOSE)

        kconserv = cc.khelper.kconserv
        khelper = cc.khelper

        idx_oovv = np.arange(nocc*nocc*nvir*nvir)
        jobs = list(khelper.symm_map.keys())
        tasks = static_partition(jobs)
        ntasks = max(comm.allgather(len(tasks)))
        nwrite = 0
        for itask in tasks:
            ikp, ikq, ikr = itask
            pqr = np.asarray(khelper.symm_map[(ikp,ikq,ikr)])
            nwrite += len(np.unique(pqr, axis=0))

        nwrite_max = max(comm.allgather(nwrite))
        write_count = 0
        for itask in range(ntasks):
            if itask >= len(tasks):
                continue
            ikp, ikq, ikr = tasks[itask]
            iks = kconserv[ikp,ikq,ikr]
            done = np.zeros([nkpts,nkpts,nkpts])
            for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                if done[kp,kq,kr]: continue
                ks = kconserv[kp,kq,kr]
                eia = _get_epq([0,nocc,kp,mo_e_o,nonzero_opadding],
                               [0,nvir,kq,mo_e_v,nonzero_vpadding],
                               fac=[1.0,-1.0])
                ejb = _get_epq([0,nocc,kr,mo_e_o,nonzero_opadding],
                               [0,nvir,ks,mo_e_v,nonzero_vpadding],
                               fac=[1.0,-1.0])
                eijab = eia[:,None,:,None] + ejb[None,:,None,:]
                off = kp * nkpts**2 + kr * nkpts + kq
                self.eijab.write(off*idx_oovv.size+idx_oovv, eijab.ravel())
                done[kp,kq,kr] = 1
                write_count += 1

        for i in range(nwrite_max-write_count):
            self.eijab.write([], [])

        if type(cc._scf.with_df) is df.FFTDF:
            _make_fftdf_eris(cc, self)
        else:
            from cc_sym import mpigdf
            if type(cc._scf.with_df) is mpigdf.GDF:
                _make_df_eris(cc, self)
            elif type(cc._scf.with_df) is df.GDF:
                log.warn("GDF converted to an MPIGDF object")
                cc._scf.with_df = mpigdf.from_serial(cc._scf.with_df)
                _make_df_eris(cc, self)
            else:
                raise NotImplementedError("DF object not recognized")
        log.timer("ao2mo transformation", *cput0)

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    import os
    cell = gto.Cell()
    cell.atom='''
    H 0.000000000000   0.000000000000   0.000000000000
    H 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.mesh = [15,15,15]
    cell.verbose = 4
    cell.build()

    kpts = cell.make_kpts([1,1,3])
    mf = scf.KRHF(cell,kpts, exxdiv=None)

    if rank==0:
        mf.kernel()

    comm.barrier()
    mf.mo_coeff = comm.bcast(mf.mo_coeff, root=0)
    mf.mo_occ = comm.bcast(mf.mo_occ, root=0)
    mycc = KRCCSD(mf)
    mycc.kernel()
