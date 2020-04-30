#!/usr/bin/env python


'''
Mole GCCSD with Numpy
'''

from functools import reduce
import numpy as np
from pyscf import scf
from pyscf import lib as pyscflib
from pyscf import ao2mo
from pyscf.cc import ccsd
import cc_sym.gintermediates as imd
from symtensor import sym
from pyscf.lib.logger import Logger
from cc_sym import rccsd_numpy as rccsd

tensor = sym.tensor

def energy(cc, t1, t2, eris):
    einsum = cc.lib.einsum
    e = einsum('ia,ia', eris.fov, t1)
    e += 0.25*einsum('ijab,ijab', t2, eris.oovv)
    tmp = einsum('ia,ijab->jb', t1, eris.oovv)
    e += 0.5*einsum('jb,jb->', t1, tmp)
    return e

def update_amps(cc, t1, t2, eris):
    einsum = cc.lib.einsum

    fov = eris.fov
    tau = imd.make_tau(t2, t1, t1, eris)

    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)

    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
    Wovvo = imd.cc_Wovvo(t1, t2, eris)

    # Move energy terms to the other side
    Fvv -= eris._fvv
    Foo -= eris._foo

    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*einsum('imef,maef->ia', t2, eris.ovvv)
    t1new += -0.5*einsum('mnae,mnie->ia', t2, eris.ooov)
    t1new += fov.conj()

    # T2 equation
    Ftmp = Fvv - 0.5*einsum('mb,me->be', t1, Fov)
    tmp = einsum('ijae,be->ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0,1,3,2)
    Ftmp = Foo + 0.5*einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab,mj->ijab', t2, Ftmp)
    t2new -= tmp - tmp.transpose(1,0,2,3)
    t2new += eris.oovv.conj()
    t2new += 0.5*einsum('mnab,mnij->ijab', tau, Woooo)
    t2new += 0.5*einsum('ijef,abef->ijab', tau, Wvvvv)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    tmp1 = einsum('ma,mbje->abje', t1, eris.ovov)
    tmp += einsum('ie,abje->ijab', t1, tmp1)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp
    tmp = einsum('ie,jeba->ijab', t1, eris.ovvv.conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma,ijmb->ijab', t1, eris.ooov.conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    t1new /= eris.eia
    t2new /= eris.eijab

    return t1new, t2new

class GCCSD(rccsd.RCCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, SYMVERBOSE=0):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.symlib = None
        self.lib  = sym
        self.SYMVERBOSE = SYMVERBOSE

    energy = energy
    update_amps = update_amps

    def init_amps(self, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        einsum = self.lib.einsum
        log = Logger(self.stdout, self.verbose)
        t1 = eris.fov / eris.eia
        t2 = eris.oovv / eris.eijab
        self.emp2 = 0.25*einsum('ijab,ijab', t2, eris.oovv.conj()).real
        log.info('Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2

    def ao2mo(self, mo_coeff=None):
        if getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            return _PhysicistsERIs(self, mo_coeff)

    def kernel(self, t1=None, t2=None, eris=None):
        return self.ccsd(t1, t2, eris)

    def ccsd(self, t1=None, t2=None, eris=None):
        return rccsd.RCCSD.ccsd(self, t1, t2, eris)


def _eris_common_init(eris, mycc, mo_coeff):
    eris.lib = mycc.lib
    eris.orbspin = None
    eris.mol = mycc.mol
    eris.nocc = mycc.nocc
    if mo_coeff is None:
        mo_coeff = mycc.mo_coeff
    mo_idx = ccsd.get_frozen_mask(mycc)

    if getattr(mo_coeff, 'orbspin', None) is not None:
        eris.orbspin = mo_coeff.orbspin[mo_idx]
        mo_coeff = pyscflib.tag_array(mo_coeff[:,mo_idx], orbspin=eris.orbspin)
    else:
        orbspin = scf.ghf.guess_orbspin(mo_coeff)
        mo_coeff = mo_coeff[:,mo_idx]
        if not np.any(orbspin == -1):
            eris.orbspin = orbspin[mo_idx]
            mo_coeff = pyscflib.tag_array(mo_coeff, orbspin=eris.orbspin)
    eris.mo_coeff = mo_coeff

def _eris_1e_init(eris, mycc, mo_coeff):
    log = Logger(mycc.stdout, mycc.verbose)
    dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
    vhf = mycc._scf.get_veff(mycc.mol, dm)
    fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
    fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
    nocc = eris.nocc
    eris.fock = fock
    mo_e = eris.mo_energy = fock.diagonal().real
    gap = abs(mo_e[:nocc,None] - mo_e[None,nocc:]).min()
    if gap < 1e-5:
        log.warn('HOMO-LUMO gap %s too small for GCCSD', gap)
    eris.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)

class _PhysicistsERIs:
    '''<pq||rs> = <pq|rs> - <pq|sr>'''
    def __init__(self, mycc, mo_coeff=None):
        _eris_common_init(self, mycc, mo_coeff)
        log = Logger(mycc.stdout, mycc.verbose)

        mo_coeff = self.mo_coeff
        _eris_1e_init(self, mycc, mo_coeff)
        nocc = self.nocc
        fock = self.fock
        self.foo = tensor(fock[:nocc,:nocc])
        self.fov = tensor(fock[:nocc,nocc:])
        self.fvv = tensor(fock[nocc:,nocc:])
        self._foo = self.foo.diagonal(preserve_shape=True)
        self._fvv = self.fvv.diagonal(preserve_shape=True)

        mo_e = fock.diagonal().real
        eia  = mo_e[:nocc,None]- mo_e[None,nocc:]
        eijab = eia[:,None,:,None] + eia[None,:,None,:]

        self.eia = tensor(eia)
        self.eijab = tensor(eijab)

        nao, nmo = mo_coeff.shape
        mo_a = mo_coeff[:nao//2]
        mo_b = mo_coeff[nao//2:]
        orbspin = self.orbspin

        if orbspin is None:
            eri  = ao2mo.kernel(mycc._scf._eri, mo_a)
            eri += ao2mo.kernel(mycc._scf._eri, mo_b)
            eri1 = ao2mo.kernel(mycc._scf._eri, (mo_a,mo_a,mo_b,mo_b))
            eri += eri1
            eri += eri1.T
        else:
            mo = mo_a + mo_b
            eri = ao2mo.kernel(mycc._scf._eri, mo)
            if eri.size == nmo**4:  # if mycc._scf._eri is a complex array
                sym_forbid = (orbspin[:,None] != orbspin).ravel()
            else:  # 4-fold symmetry
                sym_forbid = (orbspin[:,None] != orbspin)[np.tril_indices(nmo)]

            eri[sym_forbid,:] = 0
            eri[:,sym_forbid] = 0

        if eri.dtype == np.double:
            eri = ao2mo.restore(1, eri, nmo)

        eri = eri.reshape(nmo,nmo,nmo,nmo)
        eri = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)

        self.oooo = tensor(eri[:nocc,:nocc,:nocc,:nocc].copy())
        self.ooov = tensor(eri[:nocc,:nocc,:nocc,nocc:].copy())
        self.oovv = tensor(eri[:nocc,:nocc,nocc:,nocc:].copy())
        self.ovov = tensor(eri[:nocc,nocc:,:nocc,nocc:].copy())
        self.ovvo = tensor(eri[:nocc,nocc:,nocc:,:nocc].copy())
        self.ovvv = tensor(eri[:nocc,nocc:,nocc:,nocc:].copy())
        self.vvvv = tensor(eri[nocc:,nocc:,nocc:,nocc:].copy())

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()
    mf = scf.addons.convert_to_ghf(mf)

    # Freeze 1s electrons
    frozen = [0,1,2,3]
    gcc = GCCSD(mf, frozen=frozen)
    ecc, t1, t2 = gcc.kernel()

    from pyscf.cc import GCCSD as REFG

    gcc = REFG(mf, frozen=frozen)
    ecc, t1, t2 = gcc.kernel()
