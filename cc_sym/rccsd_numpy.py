#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
Mole RCCSD with numpy
'''

import numpy as np
from functools import reduce
import time

from pyscf.cc import ccsd
from pyscf.lib import logger
from pyscf import ao2mo

from cc_sym.ccsd import CCSDBasics
import cc_sym.rintermediates as imd
from symtensor import sym

tensor = sym.tensor

def init_amps(mycc, eris):
    lib = mycc.lib
    t1 = eris.fov.conj() / eris.eia
    t2 = eris.ovov.transpose(0,2,1,3).conj() / eris.eijab
    mycc.emp2  = 2*lib.einsum('ijab,iajb', t2, eris.ovov)
    mycc.emp2 -=   lib.einsum('ijab,ibja', t2, eris.ovov)
    logger.info(mycc, 'Init t2, MP2 energy = %.15g', mycc.emp2.real)
    return mycc.emp2, t1, t2

def energy(mycc, t1, t2, eris):
    lib = mycc.lib
    e = 2*lib.einsum('ia,ia', eris.fov, t1)
    tau = lib.einsum('ia,jb->ijab',t1,t1)
    tau += t2
    e += 2*lib.einsum('ijab,iajb', tau, eris.ovov)
    e +=  -lib.einsum('ijab,ibja', tau, eris.ovov)
    return e.real

def update_amps(mycc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    lib = mycc.lib
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

    # T2 equation
    t2new = eris.ovov.conj().transpose(0,2,1,3).copy()

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

    return t1new, t2new


class RCCSD(CCSDBasics):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, SYMVERBOSE=0):
        CCSDBasics.__init__(self, mf, frozen=frozen, mo_coeff=mo_coeff, \
                            mo_occ=mo_occ, SYMVERBOSE=SYMVERBOSE)
        self.lib = sym

    init_amps = init_amps
    energy = energy
    update_amps = update_amps

    def amplitudes_to_vector(self, t1, t2):
        vector = np.hstack((t1.array.ravel(), t2.array.ravel()))
        return vector

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

def _eris_common_init(eris, mycc, mo_coeff):
    eris.lib = mycc.lib
    if mo_coeff is None:
        eris.mo_coeff = mo_coeff = ccsd._mo_without_core(mycc, mycc.mo_coeff)
    else:
        eris.mo_coeff = mo_coeff = ccsd._mo_without_core(mycc, mo_coeff)
    dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
    fockao = mycc._scf.get_hcore() + mycc._scf.get_veff(mycc.mol, dm)
    eris.fock = reduce(np.dot, (mo_coeff.T, fockao, mo_coeff))

class _ChemistsERIs:
    def __init__(self, mycc, mo_coeff=None):
        lib = self.lib = mycc.lib
        _eris_common_init(self, mycc, mo_coeff)
        fock = self.fock
        mo_e = fock.diagonal().real
        self.dtype = dtype = fock.dtype
        nocc, nmo = mycc.nocc, mycc.nmo
        nvir = nmo - nocc
        self.foo = tensor(fock[:nocc,:nocc])
        self.fov = tensor(fock[:nocc,nocc:])
        self.fvv = tensor(fock[nocc:,nocc:])
        self._foo = self.foo.diagonal(preserve_shape=True)
        self._fvv = self.fvv.diagonal(preserve_shape=True)
        eia  = mo_e[:nocc,None]- mo_e[None,nocc:]
        eijab = eia[:,None,:,None] + eia[None,:,None,:]
        self.eia = lib.tensor(eia)
        self.eijab = lib.tensor(eijab)
        cput1 = cput0 = (time.clock(), time.time())
        eri1 = ao2mo.incore.full(mycc._scf._eri, mo_coeff)
        eri1 = ao2mo.restore(1, eri1, nmo)
        self.oooo = tensor(eri1[:nocc,:nocc,:nocc,:nocc].copy())
        self.ooov = tensor(eri1[:nocc,:nocc,:nocc,nocc:].copy())
        self.ovov = tensor(eri1[:nocc,nocc:,:nocc,nocc:].copy())
        self.oovv = tensor(eri1[:nocc,:nocc,nocc:,nocc:].copy())
        self.ovvo = tensor(eri1[:nocc,nocc:,nocc:,:nocc].copy())
        self.ovvv = tensor(eri1[:nocc,nocc:,nocc:,nocc:].copy())
        self.vvvv = tensor(eri1[nocc:,nocc:,nocc:,nocc:].copy())
        cput1 = logger.timer(mycc, 'ao2mo transformation', *cput1)

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf.cc.rccsd import RCCSD as REFCCSD

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 5
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    mycc = RCCSD(mf)
    refcc = REFCCSD(mf)
    e0, _, _ = mycc.kernel()
    e1, _, _ = refcc.kernel()
    print(abs(e0-e1))
