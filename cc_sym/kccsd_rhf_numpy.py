#!/usr/bin/env python


'''
Restricted Kpoint CCSD with Numpy
'''

from functools import reduce
import time
import numpy as np
from pyscf import lib as pyscflib
from cc_sym import rccsd_numpy
from symtensor import sym
from symtensor.symlib import SYMLIB
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.lib import kpts_helper
from pyscf.lib import logger

tensor = sym.tensor
zeros = sym.zeros

def energy(mycc, t1, t2, eris):
    nkpts = len(mycc.kpts)
    lib = mycc.lib
    e = 2*lib.einsum('ia,ia', eris.fov, t1)
    tau = lib.einsum('ia,jb->ijab',t1,t1)
    tau += t2
    e += 2*lib.einsum('ijab,iajb', tau, eris.ovov)
    e +=  -lib.einsum('ijab,ibja', tau, eris.ovov)
    if abs(e.imag)>1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in KRCCSD energy %s', e)
    return e.real / nkpts

def init_amps(mycc, eris):
    time0 = time.clock(), time.time()
    lib = mycc.lib
    nocc = mycc.nocc
    nvir = mycc.nmo - nocc
    nkpts = mycc.nkpts

    t1 = lib.zeros([nocc,nvir], eris.dtype, mycc._sym1)
    t2 = eris.ovov.transpose(0,2,1,3).conj() / eris.eijab
    mycc.emp2  = 2*lib.einsum('ijab,iajb', t2, eris.ovov)
    mycc.emp2 -=   lib.einsum('ijab,ibja', t2, eris.ovov)
    mycc.emp2  =   mycc.emp2.real/nkpts
    logger.info(mycc, 'Init t2, MP2 energy (with fock eigenvalue shift) = %.15g', mycc.emp2)
    logger.timer(mycc, 'init mp2', *time0)
    return mycc.emp2, t1, t2

class KRCCSD(rccsd_numpy.RCCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, SYMVERBOSE=0):
        rccsd_numpy.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ, SYMVERBOSE)
        self.kpts = mf.kpts
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])
        self.lib = sym
        self.symlib = SYMLIB(self._backend)
        self.make_symlib()

    @property
    def nkpts(self):
        return len(self.kpts)

    @property
    def _sym1(self):
        kpts = self.kpts
        gvec = self._scf.cell.reciprocal_vectors()
        return ['+-',[kpts,]*2, None, gvec]

    @property
    def _sym2(self):
        kpts = self.kpts
        gvec = self._scf.cell.reciprocal_vectors()
        return ['++--',[kpts,]*4, None, gvec]

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
        t1  = self.lib.tensor(t1, self._sym1, symlib=self.symlib, verbose=self.SYMVERBOSE)
        t2  = self.lib.tensor(t2, self._sym2, symlib=self.symlib, verbose=self.SYMVERBOSE)
        return t1, t2

    def make_symlib(self):
        kpts = self.kpts
        gvec = self._scf.cell.reciprocal_vectors()
        sym3 = ['++-',[kpts,]*3, None, gvec]
        self.symlib.update(self._sym1, self._sym2, sym3)

    energy = energy
    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    init_amps = init_amps

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    def ipccsd(self, nroots=1, koopmans=False, guess=None, left=False,
               eris=None, partition=None, kptlist=None):
        from cc_sym import eom_kccsd_rhf_numpy
        return eom_kccsd_rhf_numpy.EOMIP(self).kernel(nroots=nroots, koopmans=koopmans,
                                                guess=guess, left=left,
                                                eris=eris, partition=partition,
                                                kptlist=kptlist)

    def eaccsd(self, nroots=1, koopmans=False, guess=None, left=False,
               eris=None, partition=None, kptlist=None):
        from cc_sym import eom_kccsd_rhf_numpy
        return eom_kccsd_rhf_numpy.EOMEA(self).kernel(nroots=nroots, koopmans=koopmans,
                                                guess=guess, left=left,
                                                eris=eris, partition=partition,
                                                kptlist=kptlist)
def _eris_common_init(eris, mycc, mo_coeff):
    eris.lib = mycc.lib
    if mo_coeff is None:
        mo_coeff = mycc.mo_coeff
    cell = mycc._scf.cell
    eris.mo_coeff = mo_coeff = padded_mo_coeff(mycc, mo_coeff)
    dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
    with pyscflib.temporary_env(mycc._scf, exxdiv=None):
        fockao = mycc._scf.get_hcore() + mycc._scf.get_veff(cell, dm)
    eris.fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                        for k, mo in enumerate(mo_coeff)])

class _ChemistsERIs:
    def __init__(self, mycc, mo_coeff=None):
        from pyscf.pbc.cc.ccsd import _adjust_occ
        import pyscf.pbc.tools.pbc as tools

        cput0 = (time.clock(), time.time())

        nocc, nmo, nkpts = mycc.nocc, mycc.nmo, mycc.nkpts
        nvir = nmo - nocc
        cell, kpts = mycc._scf.cell, mycc.kpts
        gvec = cell.reciprocal_vectors()
        sym2 = ['+-+-', [kpts,]*4, None, gvec]

        nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")
        madelung = tools.madelung(cell, kpts)

        _eris_common_init(self, mycc, mo_coeff)

        fock = self.fock
        mo_coeff = self.mo_coeff

        self.dtype = dtype = np.result_type(*fock).char
        self.foo = tensor(fock[:,:nocc,:nocc], mycc._sym1)
        self.fov = tensor(fock[:,:nocc,nocc:], mycc._sym1)
        self.fvv = tensor(fock[:,nocc:,nocc:], mycc._sym1)

        mo_energy = [fock[k].diagonal().real for k in range(nkpts)]
        mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                          for k, mo_e in enumerate(mo_energy)]

        mo_e_o = [e[:nocc] for e in mo_energy]
        mo_e_v = [e[nocc:] + mycc.level_shift for e in mo_energy]
        foo_ = np.asarray([np.diag(e) for e in mo_e_o])
        fvv_ = np.asarray([np.diag(e) for e in mo_e_v])
        self._foo = tensor(foo_, mycc._sym1)
        self._fvv = tensor(fvv_, mycc._sym1)

        eia = np.zeros([nkpts,nocc,nvir])
        for ki in range(nkpts):
            eia[ki] = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                        [0,nvir,ki,mo_e_v,nonzero_vpadding],
                        fac=[1.0,-1.0])

        self.eia = tensor(eia, mycc._sym1)
        self.oooo = zeros([nocc,nocc,nocc,nocc], dtype, sym2)
        self.ooov = zeros([nocc,nocc,nocc,nvir], dtype, sym2)
        self.ovov = zeros([nocc,nvir,nocc,nvir], dtype, sym2)
        self.oovv = zeros([nocc,nocc,nvir,nvir], dtype, sym2)
        self.ovvo = zeros([nocc,nvir,nvir,nocc], dtype, sym2)
        self.ovvv = zeros([nocc,nvir,nvir,nvir], dtype, sym2)
        self.vvvv = zeros([nvir,nvir,nvir,nvir], dtype, sym2)
        self.eijab = zeros([nocc,nocc,nvir,nvir], np.float64, mycc._sym2)

        with_df = mycc._scf.with_df
        fao2mo = mycc._scf.with_df.ao2mo
        kconserv = mycc.khelper.kconserv
        khelper = mycc.khelper

        jobs = list(khelper.symm_map.keys())
        for itask in jobs:
            ikp, ikq, ikr = itask
            iks = kconserv[ikp,ikq,ikr]
            eri_kpt = fao2mo((mo_coeff[ikp],mo_coeff[ikq],mo_coeff[ikr],mo_coeff[iks]),
                             (kpts[ikp],kpts[ikq],kpts[ikr],kpts[iks]), compact=False)
            if dtype == np.float: eri_kpt = eri_kpt.real
            eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo) / nkpts
            done = np.zeros([nkpts,nkpts,nkpts])
            for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                if done[kp,kq,kr]: continue
                eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr)
                oooo = eri_kpt_symm[:nocc,:nocc,:nocc,:nocc]
                ooov = eri_kpt_symm[:nocc,:nocc,:nocc,nocc:]
                ovov = eri_kpt_symm[:nocc,nocc:,:nocc,nocc:]
                oovv = eri_kpt_symm[:nocc,:nocc,nocc:,nocc:]
                ovvo = eri_kpt_symm[:nocc,nocc:,nocc:,:nocc]
                ovvv = eri_kpt_symm[:nocc,nocc:,nocc:,nocc:]
                vvvv = eri_kpt_symm[nocc:,nocc:,nocc:,nocc:]
                ks = kconserv[kp,kq,kr]
                eia = _get_epq([0,nocc,kp,mo_e_o,nonzero_opadding],
                               [0,nvir,kq,mo_e_v,nonzero_vpadding],
                               fac=[1.0,-1.0])
                ejb = _get_epq([0,nocc,kr,mo_e_o,nonzero_opadding],
                               [0,nvir,ks,mo_e_v,nonzero_vpadding],
                               fac=[1.0,-1.0])
                eijab = eia[:,None,:,None] + ejb[None,:,None,:]

                self.oooo.array[kp,kq,kr] = oooo
                self.ooov.array[kp,kq,kr] = ooov
                self.ovov.array[kp,kq,kr] = ovov
                self.oovv.array[kp,kq,kr] = oovv
                self.ovvo.array[kp,kq,kr] = ovvo
                self.ovvv.array[kp,kq,kr] = ovvv
                self.vvvv.array[kp,kq,kr] = vvvv
                self.eijab.array[kp,kr,kq] = eijab
                done[kp,kq,kr] = 1

        logger.timer(mycc, "ao2mo transformation", *cput0)
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

    mf.kernel()

    mycc = KRCCSD(mf)
    mycc.kernel()

    refcc = cc.KRCCSD(mf)
    refcc.kernel()
