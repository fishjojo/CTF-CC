#!/usr/bin/env python

import time
import numpy as np
from pyscf import lib as pyscflib
from symtensor import sym_ctf as lib
from symtensor.symlib import SYMLIB
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM
import ctf
from cc_sym import rccsd

rank = rccsd.rank
comm = rccsd.comm
size = rccsd.size
tensor = lib.tensor
zeros = lib.zeros
Logger = rccsd.Logger

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

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        rccsd.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
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
        t1 = lib.zeros([nocc,nvir], eris.dtype, sym1, symlib=self.symlib)
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
        self.symlib.update(sym1, sym2)

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
        t1  = tensor(t1, sym1, symlib=self.symlib)
        t2  = tensor(t2, sym2, symlib=self.symlib)
        return t1, t2


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
        self.dtype = dtype = np.result_type(*(cc.mo_coeff, fock)).char
        self.foo = zeros([nocc,nocc], dtype, sym1, symlib=symlib)
        self.fov = zeros([nocc,nvir], dtype, sym1, symlib=symlib)
        self.fvv = zeros([nvir,nvir], dtype, sym1, symlib=symlib)
        self.eia = zeros([nocc,nvir], np.float64, sym1, symlib=symlib)
        self._foo = zeros([nocc,nocc], dtype, sym1, symlib=symlib)
        self._fvv = zeros([nvir,nvir], dtype, sym1, symlib=symlib)

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
        self.oooo = zeros([nocc,nocc,nocc,nocc], dtype, sym2, symlib=symlib)
        self.ooov = zeros([nocc,nocc,nocc,nvir], dtype, sym2, symlib=symlib)
        self.ovov = zeros([nocc,nvir,nocc,nvir], dtype, sym2, symlib=symlib)
        self.oovv = zeros([nocc,nocc,nvir,nvir], dtype, sym2, symlib=symlib)
        self.ovvo = zeros([nocc,nvir,nvir,nocc], dtype, sym2, symlib=symlib)
        self.ovvv = zeros([nocc,nvir,nvir,nvir], dtype, sym2, symlib=symlib)
        self.vvvv = zeros([nvir,nvir,nvir,nvir], dtype, sym2, symlib=symlib)
        self.eijab = zeros([nocc,nocc,nvir,nvir], np.float64, sym2, symlib=symlib)

        with_df = cc._scf.with_df
        fao2mo = cc._scf.with_df.ao2mo
        kconserv = cc.khelper.kconserv
        khelper = cc.khelper

        idx_oooo = np.arange(nocc*nocc*nocc*nocc)
        idx_ooov = np.arange(nocc*nocc*nocc*nvir)
        idx_ovov = np.arange(nocc*nvir*nocc*nvir)
        idx_oovv = np.arange(nocc*nocc*nvir*nvir)
        idx_ovvo = np.arange(nocc*nvir*nvir*nocc)
        idx_ovvv = np.arange(nocc*nvir*nvir*nvir)
        idx_vvvv = np.arange(nvir*nvir*nvir*nvir)

        jobs = list(khelper.symm_map.keys())
        tasks = rccsd.static_partition(jobs)
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
                if hasattr(with_df,'j3c'):
                    j3c = with_df.j3c.read([])
                    j3c = with_df.j3c.read([])
                continue
            ikp, ikq, ikr = tasks[itask]
            iks = kconserv[ikp,ikq,ikr]
            eri_kpt = fao2mo((mo_coeff[ikp],mo_coeff[ikq],mo_coeff[ikr],mo_coeff[iks]),
                             (kpts[ikp],kpts[ikq],kpts[ikr],kpts[iks]), compact=False)
            if dtype == np.float: eri_kpt = eri_kpt.real
            eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo) / nkpts
            done = np.zeros([nkpts,nkpts,nkpts])
            for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                if done[kp,kq,kr]: continue
                eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr)
                oooo = eri_kpt_symm[:nocc,:nocc,:nocc,:nocc].ravel()
                ooov = eri_kpt_symm[:nocc,:nocc,:nocc,nocc:].ravel()
                ovov = eri_kpt_symm[:nocc,nocc:,:nocc,nocc:].ravel()
                oovv = eri_kpt_symm[:nocc,:nocc,nocc:,nocc:].ravel()
                ovvo = eri_kpt_symm[:nocc,nocc:,nocc:,:nocc].ravel()
                ovvv = eri_kpt_symm[:nocc,nocc:,nocc:,nocc:].ravel()
                vvvv = eri_kpt_symm[nocc:,nocc:,nocc:,nocc:].ravel()
                ks = kconserv[kp,kq,kr]
                eia = _get_epq([0,nocc,kp,mo_e_o,nonzero_opadding],
                               [0,nvir,kq,mo_e_v,nonzero_vpadding],
                               fac=[1.0,-1.0])
                ejb = _get_epq([0,nocc,kr,mo_e_o,nonzero_opadding],
                               [0,nvir,ks,mo_e_v,nonzero_vpadding],
                               fac=[1.0,-1.0])
                eijab = eia[:,None,:,None] + ejb[None,:,None,:]

                off = kp * nkpts**2 + kq * nkpts + kr
                self.oooo.write(off*idx_oooo.size+idx_oooo, oooo)
                self.ooov.write(off*idx_ooov.size+idx_ooov, ooov)
                self.ovov.write(off*idx_ovov.size+idx_ovov, ovov)
                self.oovv.write(off*idx_oovv.size+idx_oovv, oovv)
                self.ovvo.write(off*idx_ovvo.size+idx_ovvo, ovvo)
                self.ovvv.write(off*idx_ovvv.size+idx_ovvv, ovvv)
                self.vvvv.write(off*idx_vvvv.size+idx_vvvv, vvvv)
                off = kp * nkpts**2 + kr * nkpts + kq
                self.eijab.write(off*idx_oovv.size+idx_oovv, eijab.ravel())
                done[kp,kq,kr] = 1
                write_count += 1
        for i in range(nwrite_max-write_count):
            self.oooo.write([], [])
            self.ooov.write([], [])
            self.ovov.write([], [])
            self.oovv.write([], [])
            self.ovvo.write([], [])
            self.ovvv.write([], [])
            self.vvvv.write([], [])
            self.eijab.write([], [])
        log.timer("ao2mo transformation", *cput0)


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    import os
    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 5
    cell.mesh = [5,5,5]
    cell.build()

    kpts = cell.make_kpts([1,1,3])
    mf = scf.KRHF(cell,kpts, exxdiv=None)
    chkfile = 'dmd_113.chk'
    if os.path.isfile(chkfile):
        mf.__dict__.update(scf.chkfile.load(chkfile, 'scf'))
    else:
        mf.chkfile = chkfile
        mf.kernel()
    mycc = KRCCSD(mf)
    mycc.max_cycle=100
    eris = mycc.ao2mo()
    _, t1, t2 = mycc.init_amps(eris)
    t10, t20 = mycc.update_amps(t1, t2, eris)
    '''
    refcc = cc.KRCCSD(mf)

    erisref = refcc.ao2mo()
    _, t1r, t2r= refcc.init_amps(erisref)
    t1r0, t2r0= refcc.update_amps(t1r, t2r, erisref)

    print(np.linalg.norm(eris.oooo.transpose(0,2,1,3).array.to_nparray()-erisref.oooo))
    print(np.linalg.norm(eris.ooov.transpose(0,2,1,3).array.to_nparray()-erisref.ooov))
    print(np.linalg.norm(eris.ovov.transpose(0,2,1,3).array.to_nparray()-erisref.oovv))
    print(np.linalg.norm(eris.oovv.transpose(0,2,1,3).array.to_nparray()-erisref.ovov))
    print(np.linalg.norm(eris.ovvo.transpose(2,0,3,1).array.to_nparray()-erisref.voov))
    print(np.linalg.norm(eris.ovvv.transpose(2,0,3,1).array.to_nparray()-erisref.vovv))
    print(np.linalg.norm(eris.vvvv.transpose(0,2,1,3).array.to_nparray()-erisref.vvvv))

    print("t1", np.linalg.norm(t10.array.to_nparray()-t1r0))
    print("t2", np.linalg.norm(t20.array.to_nparray()-t2r0))
    '''
    mycc.kernel()
    #refcc.kernel()
