#!/usr/bin/env python


'''
Mole GCCSD with Numpy
'''

import numpy as np
import time
from functools import reduce

from pyscf.lib import logger
from pyscf.mp import ump2
from pyscf import ao2mo
from pyscf.lib.parameters import LARGE_DENOM
from symtensor.symlib import SYMLIB
from cc_sym import gccsd_numpy as gccsd

from symtensor import sym
tensor = sym.tensor

class UCCSD(gccsd.GCCSD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, SYMVERBOSE=0):
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ, SYMVERBOSE)
        self.symlib = SYMLIB(self._backend)
        self.make_symlib()

    get_nocc = ump2.get_nocc
    get_nmo = ump2.get_nmo
    get_frozen_mask = ump2.get_frozen_mask

    def make_symlib(self):
        self.symlib.update(self._sym1, self._sym2)

    @property
    def _sym1(self):
        return ['+-',[[-1,1]]*2, None, None]

    @property
    def _sym2(self):
        return ['++--',[[-1,1]]*4, None, None]

    @property
    def _no(self):
        return max(self.nocc)

    @property
    def _nv(self):
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        return max(nmoa-nocca, nmob-noccb)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        nocc = self._no
        nvir = self._nv
        nov = 2 * nocc * nvir
        t1 = vec[:nov].reshape(2,nocc,nvir)
        t2 = vec[nov:].reshape(2,2,2,nocc,nocc,nvir,nvir)
        t1  = self.lib.tensor(t1, self._sym1, symlib=self.symlib, verbose=self.SYMVERBOSE)
        t2  = self.lib.tensor(t2, self._sym2, symlib=self.symlib, verbose=self.SYMVERBOSE)
        return t1, t2

    def ao2mo(self, mo_coeff=None):
        if getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            return _PhysicistsERIs(self, mo_coeff)

def _eris_common_init(eris, mycc, mo_coeff):
    if mo_coeff is None:
        mo_coeff = mycc.mo_coeff
    mo_idx = mycc.get_frozen_mask()
    eris.mo_coeff = mo_coeff = \
            (mo_coeff[0][:,mo_idx[0]], mo_coeff[1][:,mo_idx[1]])

    dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
    vhf = mycc._scf.get_veff(mycc.mol, dm)
    fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
    eris.focka = reduce(np.dot, (mo_coeff[0].conj().T, fockao[0], mo_coeff[0]))
    eris.fockb = reduce(np.dot, (mo_coeff[1].conj().T, fockao[1], mo_coeff[1]))

class _PhysicistsERIs:
    '''<pq||rs> = <pq|rs> - <pq|sr>'''
    def __init__(self, mycc, mo_coeff=None):
        _eris_common_init(self, mycc, mo_coeff)
        lib = self.lib = mycc.lib
        nocca, noccb = mycc.nocc
        nmoa, nmob = mycc.nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb

        nocc, nvir = mycc._no, mycc._nv
        mo_coeff = self.mo_coeff

        def fill_1e(aa, bb):
            a_shape = aa.shape
            b_shape = bb.shape
            dtype = np.result_type(*(aa, bb))
            shape = [2,]
            for ki, i in enumerate(a_shape):
                shape.append(max(i, b_shape[ki]))
            array = np.zeros(shape, dtype=dtype)
            idx_aa = np.ix_(*[[0]]+[np.arange(i) for i in aa.shape])
            idx_bb = np.ix_(*[[1]]+[np.arange(i) for i in bb.shape])
            array[idx_aa] = aa
            array[idx_bb] = bb
            return lib.tensor(array, mycc._sym1, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)

        #fock = fill_1e(self.focka, self.fockb)
        self.foo = fill_1e(self.focka[:nocca,:nocca], self.fockb[:noccb,:noccb])
        self.fov = fill_1e(self.focka[:nocca,nocca:], self.fockb[:noccb,noccb:])
        self.fvv = fill_1e(self.focka[nocca:,nocca:], self.fockb[noccb:,noccb:])
        self._foo = self.foo.diagonal(preserve_shape=True)
        self._fvv = self.fvv.diagonal(preserve_shape=True)

        mo_ea = self.focka.diagonal().real
        mo_eb = self.fockb.diagonal().real
        self.mo_energy = (mo_ea, mo_eb)

        eia_a  = mo_ea[:nocca,None]- mo_ea[None,nocca:]
        eia_b  = mo_eb[:noccb,None]- mo_eb[None,noccb:]

        eia = np.ones([2,nocc,nvir]) * LARGE_DENOM
        eia[0,:nocca,:nvira] = eia_a
        eia[1,:noccb,:nvirb] = eia_b
        self.eia = lib.tensor(eia, sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)

        eijab = np.ones([2,2,2,nocc,nocc,nvir, nvir]) * LARGE_DENOM

        eijab[0,0,0,:nocca,:nocca,:nvira,:nvira] = eia_a[:,None,:,None] + eia_a[None,:,None,:]
        eijab[1,1,1,:noccb,:noccb,:nvirb,:nvirb] = eia_b[:,None,:,None] + eia_b[None,:,None,:]
        eijab[0,1,0,:nocca,:noccb,:nvira,:nvirb] = eia_a[:,None,:,None] + eia_b[None,:,None,:]
        eijab[0,1,1,:nocca,:noccb,:nvirb,:nvira] = eia_a[:,None,None,:] + eia_b[None,:,:,None]
        eijab[1,0,0,:noccb,:nocca,:nvira,:nvirb] = eia_a[None,:,:,None] + eia_b[:,None,None,:]
        eijab[1,0,1,:noccb,:nocca,:nvirb,:nvira] = eia_a[None,:,None,:] + eia_b[:,None,:,None]

        self.eijab = lib.tensor(eijab, sym=mycc._sym2, verbose=mycc.SYMVERBOSE)
        gap_a = abs(mo_ea[:nocca,None] - mo_ea[None,nocca:])
        gap_b = abs(mo_eb[:noccb,None] - mo_eb[None,noccb:])
        if gap_a.size > 0:
            gap_a = gap_a.min()
        else:
            gap_a = 1e9
        if gap_b.size > 0:
            gap_b = gap_b.min()
        else:
            gap_b = 1e9
        if gap_a < 1e-5 or gap_b < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap (%s,%s) too small for UCCSD',
                        gap_a, gap_b)

        moa = mo_coeff[0]
        mob = mo_coeff[1]

        eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, moa), nmoa).reshape(nmoa,nmoa,nmoa,nmoa)
        eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, mob), nmob).reshape(nmob,nmob,nmob,nmob)
        eri_ab = ao2mo.general(mycc._scf._eri, (moa,moa,mob,mob), compact=False).reshape(nmoa,nmoa,nmob,nmob)



        eri_aaaa = eri_aa.transpose(0,2,1,3) - eri_aa.transpose(0,2,3,1)
        eri_bbbb = eri_bb.transpose(0,2,1,3) - eri_bb.transpose(0,2,3,1)
        eri_abab = eri_ab.reshape(nmoa,nmoa,nmob,nmob).transpose(0,2,1,3)
        eri_abba = -eri_abab.transpose(0,1,3,2)
        eri_baab = -eri_abab.transpose(1,0,2,3)
        eri_baba = eri_abab.transpose(1,0,3,2)
        eri_aa = eri_ab = eri_bb = None

        self.oooo = lib.zeros([nocc,nocc,nocc,nocc], sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.ooov = lib.zeros([nocc,nocc,nocc,nvir], sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.oovv = lib.zeros([nocc,nocc,nvir,nvir], sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.ovov = lib.zeros([nocc,nvir,nocc,nvir], sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.ovvo = lib.zeros([nocc,nvir,nvir,nocc], sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.ovvv = lib.zeros([nocc,nvir,nvir,nvir], sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.vvvv = lib.zeros([nvir,nvir,nvir,nvir], sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)

        self.oooo[0,0,0,:nocca,:nocca,:nocca,:nocca] = eri_aaaa[:nocca,:nocca,:nocca,:nocca]
        self.ooov[0,0,0,:nocca,:nocca,:nocca,:nvira] = eri_aaaa[:nocca,:nocca,:nocca,nocca:]
        self.oovv[0,0,0,:nocca,:nocca,:nvira,:nvira] = eri_aaaa[:nocca,:nocca,nocca:,nocca:]
        self.ovov[0,0,0,:nocca,:nvira,:nocca,:nvira] = eri_aaaa[:nocca,nocca:,:nocca,nocca:]
        self.ovvo[0,0,0,:nocca,:nvira,:nvira,:nocca] = eri_aaaa[:nocca,nocca:,nocca:,:nocca]
        self.ovvv[0,0,0,:nocca,:nvira,:nvira,:nvira] = eri_aaaa[:nocca,nocca:,nocca:,nocca:]
        self.vvvv[0,0,0,:nvira,:nvira,:nvira,:nvira] = eri_aaaa[nocca:,nocca:,nocca:,nocca:]

        self.oooo[1,1,1,:noccb,:noccb,:noccb,:noccb] = eri_bbbb[:noccb,:noccb,:noccb,:noccb]
        self.ooov[1,1,1,:noccb,:noccb,:noccb,:nvirb] = eri_bbbb[:noccb,:noccb,:noccb,noccb:]
        self.oovv[1,1,1,:noccb,:noccb,:nvirb,:nvirb] = eri_bbbb[:noccb,:noccb,noccb:,noccb:]
        self.ovov[1,1,1,:noccb,:nvirb,:noccb,:nvirb] = eri_bbbb[:noccb,noccb:,:noccb,noccb:]
        self.ovvo[1,1,1,:noccb,:nvirb,:nvirb,:noccb] = eri_bbbb[:noccb,noccb:,noccb:,:noccb]
        self.ovvv[1,1,1,:noccb,:nvirb,:nvirb,:nvirb] = eri_bbbb[:noccb,noccb:,noccb:,noccb:]
        self.vvvv[1,1,1,:nvirb,:nvirb,:nvirb,:nvirb] = eri_bbbb[noccb:,noccb:,noccb:,noccb:]

        self.oooo[0,1,0,:nocca,:noccb,:nocca,:noccb] = eri_abab[:nocca,:noccb,:nocca,:noccb]
        self.ooov[0,1,0,:nocca,:noccb,:nocca,:nvirb] = eri_abab[:nocca,:noccb,:nocca,noccb:]
        self.oovv[0,1,0,:nocca,:noccb,:nvira,:nvirb] = eri_abab[:nocca,:noccb,nocca:,noccb:]
        self.ovov[0,1,0,:nocca,:nvirb,:nocca,:nvirb] = eri_abab[:nocca,noccb:,:nocca,noccb:]
        self.ovvo[0,1,0,:nocca,:nvirb,:nvira,:noccb] = eri_abab[:nocca,noccb:,nocca:,:noccb]
        self.ovvv[0,1,0,:nocca,:nvirb,:nvira,:nvirb] = eri_abab[:nocca,noccb:,nocca:,noccb:]
        self.vvvv[0,1,0,:nvira,:nvirb,:nvira,:nvirb] = eri_abab[nocca:,noccb:,nocca:,noccb:]

        self.oooo[0,1,1,:nocca,:noccb,:noccb,:nocca] = eri_abba[:nocca,:noccb,:noccb,:nocca]
        self.ooov[0,1,1,:nocca,:noccb,:noccb,:nvira] = eri_abba[:nocca,:noccb,:noccb,nocca:]
        self.oovv[0,1,1,:nocca,:noccb,:nvirb,:nvira] = eri_abba[:nocca,:noccb,noccb:,nocca:]
        self.ovov[0,1,1,:nocca,:nvirb,:noccb,:nvira] = eri_abba[:nocca,noccb:,:noccb,nocca:]
        self.ovvo[0,1,1,:nocca,:nvirb,:nvirb,:nocca] = eri_abba[:nocca,noccb:,noccb:,:nocca]
        self.ovvv[0,1,1,:nocca,:nvirb,:nvirb,:nvira] = eri_abba[:nocca,noccb:,noccb:,nocca:]
        self.vvvv[0,1,1,:nvira,:nvirb,:nvirb,:nvira] = eri_abba[nocca:,noccb:,noccb:,nocca:]

        self.oooo[1,0,0,:noccb,:nocca,:nocca,:noccb] = eri_baab[:noccb,:nocca,:nocca,:noccb]
        self.ooov[1,0,0,:noccb,:nocca,:nocca,:nvirb] = eri_baab[:noccb,:nocca,:nocca,noccb:]
        self.oovv[1,0,0,:noccb,:nocca,:nvira,:nvirb] = eri_baab[:noccb,:nocca,nocca:,noccb:]
        self.ovov[1,0,0,:noccb,:nvira,:nocca,:nvirb] = eri_baab[:noccb,nocca:,:nocca,noccb:]
        self.ovvo[1,0,0,:noccb,:nvira,:nvira,:noccb] = eri_baab[:noccb,nocca:,nocca:,:noccb]
        self.ovvv[1,0,0,:noccb,:nvira,:nvira,:nvirb] = eri_baab[:noccb,nocca:,nocca:,noccb:]
        self.vvvv[1,0,0,:nvirb,:nvira,:nvira,:nvirb] = eri_baab[noccb:,nocca:,nocca:,noccb:]

        self.oooo[1,0,1,:noccb,:nocca,:noccb,:nocca] = eri_baba[:noccb,:nocca,:noccb,:nocca]
        self.ooov[1,0,1,:noccb,:nocca,:noccb,:nvira] = eri_baba[:noccb,:nocca,:noccb,nocca:]
        self.oovv[1,0,1,:noccb,:nocca,:nvirb,:nvira] = eri_baba[:noccb,:nocca,noccb:,nocca:]
        self.ovov[1,0,1,:noccb,:nvira,:noccb,:nvira] = eri_baba[:noccb,nocca:,:noccb,nocca:]
        self.ovvo[1,0,1,:noccb,:nvira,:nvirb,:nocca] = eri_baba[:noccb,nocca:,noccb:,:nocca]
        self.ovvv[1,0,1,:noccb,:nvira,:nvirb,:nvira] = eri_baba[:noccb,nocca:,noccb:,nocca:]
        self.vvvv[1,0,1,:nvirb,:nvira,:nvirb,:nvira] = eri_baba[noccb:,nocca:,noccb:,nocca:]


if __name__ == '__main__':
    from pyscf import gto, scf, cc
    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.verbose=5
    mol.build()
    mf = scf.UHF(mol).run()
    frozen = [0,1,2,3]
    gcc = UCCSD(mf, frozen=frozen)
    gcc.kernel()
