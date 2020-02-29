#!/usr/bin/env python


'''
Mole RCCSD with Numpy
'''

from functools import reduce
import time
import numpy as np

from pyscf import lib as pyscflib
from pyscf import ao2mo
from pyscf.cc import ccsd
import cc_sym.rintermediates as imd
from symtensor import sym as lib
from cc_sym.linalg_helper.diis import DIIS
from pyscf.lib.logger import Logger

tensor = lib.tensor

# note MO integrals are treated in chemist's notation

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
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
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

    return t1new, t2new


def energy(cc, t1, t2, eris):
    e = 2*lib.einsum('ia,ia', eris.fov, t1)
    tau = lib.einsum('ia,jb->ijab',t1,t1)
    tau += t2
    e += 2*lib.einsum('ijab,iajb', tau, eris.ovov)
    e +=  -lib.einsum('ijab,ibja', tau, eris.ovov)
    return e.real


class RCCSD(ccsd.CCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])
        self.lib = lib
        self.symlib = None

    @property
    def _backend(self):
        return 'numpy'

    def amplitudes_to_vector(self, t1, t2, out=None):
        vector = np.hstack((t1.array.ravel(), t2.array.ravel()))
        return vector

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        nov = nocc * nvir
        t1 = vec[:nov].reshape(nocc,nvir)
        t2 = vec[nov:].reshape(nocc,nocc,nvir,nvir)
        t1  = tensor(t1)
        t2  = tensor(t2)
        return t1, t2

    def init_amps(self, eris):
        log = Logger(self.stdout, self.verbose)
        t1 = eris.fov.conj() / eris.eia
        t2 = eris.ovov.transpose(0,2,1,3).conj() / eris.eijab
        self.emp2  = 2*lib.einsum('ijab,iajb', t2, eris.ovov)
        self.emp2 -=   lib.einsum('ijab,ibja', t2, eris.ovov)
        log.info('Init t2, MP2 energy = %.15g', self.emp2.real)
        return self.emp2, t1, t2


    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False, cc2=False):
        return self.ccsd(t1, t2, eris, mbpt2, cc2)

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False, cc2=False):
        '''Ground-state CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
            cc2 : bool
                Use CC2 approximation to CCSD.
        '''
        if mbpt2 and cc2:
            raise RuntimeError('MBPT2 and CC2 are mutually exclusive approximations to the CCSD ground state.')
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.eris = eris
        log = Logger(self.stdout, self.verbose)
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

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    def ipccsd(self, nroots=1, koopmans=True, guess=None, left=False,
               eris=None, imds=None, partition=None, kptlist=None,
               dtype=None, **kwargs):
        from cc_sym.eom_rccsd_numpy import EOMIP
        myeom = EOMIP(self)
        return myeom.kernel(nroots, koopmans, guess, left,
                   eris, imds, partition, kptlist,
                   dtype, **kwargs)

    def eaccsd(self, nroots=1, koopmans=True, guess=None, left=False,
               eris=None, imds=None, partition=None, kptlist=None,
               dtype=None, **kwargs):
        from cc_sym.eom_rccsd_numpy import EOMEA
        myeom = EOMEA(self)
        return myeom.kernel(nroots, koopmans, guess, left,
                   eris, imds, partition, kptlist,
                   dtype, **kwargs)

    energy = energy
    update_amps = update_amps


class _ChemistsERIs:
    def __init__(self, cc, mo_coeff=None):
        self.lib = cc.lib
        log = Logger(cc.stdout, cc.verbose)
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, cc.mo_coeff)
        else:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, mo_coeff)
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
        fock = reduce(np.dot, (mo_coeff.T, fockao, mo_coeff))
        mo_e = fock.diagonal().real
        self.dtype = dtype = fock.dtype
        nocc, nmo = cc.nocc, cc.nmo
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
        eri1 = ao2mo.incore.full(cc._scf._eri, mo_coeff)
        eri1 = ao2mo.restore(1, eri1, nmo)
        self.oooo = tensor(eri1[:nocc,:nocc,:nocc,:nocc].copy())
        self.ooov = tensor(eri1[:nocc,:nocc,:nocc,nocc:].copy())
        self.ovov = tensor(eri1[:nocc,nocc:,:nocc,nocc:].copy())
        self.oovv = tensor(eri1[:nocc,:nocc,nocc:,nocc:].copy())
        self.ovvo = tensor(eri1[:nocc,nocc:,nocc:,:nocc].copy())
        self.ovvv = tensor(eri1[:nocc,nocc:,nocc:,nocc:].copy())
        self.vvvv = tensor(eri1[nocc:,nocc:,nocc:,nocc:].copy())
        cput1 = log.timer('ao2mo transformation', *cput1)

class _IMDS:
    def __init__(self, cc):
        self.t1 = cc.t1
        self.t2 = cc.t2
        self.stdout, self.verbose = cc.stdout, cc.verbose
        self.eris = cc.eris
        self.made_ip_imds = False
        self.made_ea_imds = False
        self._made_shared_2e = False

    def _make_shared_1e(self):
        cput0 = (time.clock(), time.time())
        log = Logger(self.stdout, self.verbose)
        t1,t2,eris = self.t1, self.t2, self.eris
        self.Loo = imd.Loo(t1,t2,eris)
        self.Lvv = imd.Lvv(t1,t2,eris)
        self.Fov = imd.cc_Fov(t1,t2,eris)

        log.timer('EOM-CCSD shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (time.clock(), time.time())
        log = Logger(self.stdout, self.verbose)
        t1,t2,eris = self.t1, self.t2, self.eris
        # 2 virtuals
        self.Wovov = imd.Wovov(t1,t2,eris)
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Woovv = eris.ovov.transpose(0,2,1,3)

        log.timer('EOM-CCSD shared two-electron intermediates', *cput0)

    def make_ip(self, ip_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ip_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = Logger(self.stdout, self.verbose)
        t1,t2,eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1,t2,eris)
        self.Wooov = imd.Wooov(t1,t2,eris)
        self.Wovoo = imd.Wovoo(t1,t2,eris)
        self.made_ip_imds = True
        log.timer('EOM-CCSD IP intermediates', *cput0)

    def make_ea(self, ea_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ea_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = Logger(self.stdout, self.verbose)
        t1,t2,eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1,t2,eris)
        if ea_partition == 'mp':
            self.Wvvvo = imd.Wvvvo(t1,t2,eris)
        else:
            self.Wvvvv = imd.Wvvvv(t1,t2,eris)
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,self.Wvvvv)
        self.made_ea_imds = True
        log.timer('EOM-CCSD EA intermediates', *cput0)

    def make_ee(self):
        raise NotImplementedError


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
    mycc.kernel()
    refcc.kernel()
