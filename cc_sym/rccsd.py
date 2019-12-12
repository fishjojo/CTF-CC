#!/usr/bin/env python


'''
Restricted CCSD

Ref: Stanton et al., J. Chem. Phys. 94, 4334 (1990)
Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)
'''

from functools import reduce
import time
import numpy as np

from pyscf import lib as pyscflib
from pyscf import ao2mo
from pyscf.cc import ccsd
import rintermediates as imd

from symtensor import sym

# note MO integrals are treated in chemist's notation

def kernel(mycc, eris, t1, t2):
    if eris is None: eris = mycc.ao2mo(self.mo_coeff)
    if t1 is None: t1, t2 = mycc.init_amps(eris)[1:]
    max_cycle = mycc.max_cycle
    tol = mycc.conv_tol
    tolnormt = mycc.conv_tol_normt

    log = mycc.log
    cput1 = cput0 = (time.clock(), time.time())
    nocc, nvir = t1.shape
    eold = 0
    eccsd = 0

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        normt = (t1new-t1).norm() + (t2new-t2).norm()
        t1, t2 = t1new, t2new
        t1new = t2new = None
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                istep, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2

def update_amps(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    lib, symlib = eris.lib, eris.symlib
    fov = eris.fov.copy()
    foo = eris.foo.copy()
    fvv = eris.fvv.copy()
    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # Move energy terms to the other side
    Foo -= foo.diagonal(preserve_shape=True)
    Fvv -= fvv.diagonal(preserve_shape=True)

    # T1 equation
    t1new = fov.conj().copy()
    tmp    =   lib.einsum('kc,ka->ac', fov, t1, symlib)
    t1new +=-2*lib.einsum('ac,ic->ia', tmp, t1, symlib)
    t1new +=   lib.einsum('ac,ic->ia', Fvv, t1, symlib)
    t1new +=  -lib.einsum('ki,ka->ia', Foo, t1, symlib)
    t1new += 2*lib.einsum('kc,kica->ia', Fov, t2, symlib)
    t1new +=  -lib.einsum('kc,ikca->ia', Fov, t2, symlib)
    tmp    =   lib.einsum('kc,ic->ki', Fov, t1, symlib)
    t1new +=   lib.einsum('ki,ka->ia', tmp, t1, symlib)
    t1new += 2*lib.einsum('kcai,kc->ia', eris.ovvo, t1, symlib)
    t1new +=  -lib.einsum('kiac,kc->ia', eris.oovv, t1, symlib)
    t1new += 2*lib.einsum('kdac,ikcd->ia', eris.ovvv, t2, symlib)
    t1new +=  -lib.einsum('kcad,ikcd->ia', eris.ovvv, t2, symlib)
    tmp    =   lib.einsum('kdac,kd->ac', eris.ovvv, t1, symlib)
    t1new += 2*lib.einsum('ac,ic->ia', tmp, t1, symlib)
    tmp    =   lib.einsum('kcad,kd->ca', eris.ovvv, t1, symlib)
    t1new +=  -lib.einsum('ca,ic->ia', tmp, t1, symlib)
    t1new +=-2*lib.einsum('kilc,klac->ia', eris.ooov, t2, symlib)
    t1new +=   lib.einsum('likc,klac->ia', eris.ooov, t2, symlib)
    tmp    =   lib.einsum('kilc,lc->ki', eris.ooov, t1, symlib)
    t1new +=-2*lib.einsum('ki,ka->ia', tmp, t1, symlib)
    tmp    =   lib.einsum('likc,lc->ik', eris.ooov, t1, symlib)
    t1new +=   lib.einsum('ik,ka->ia', tmp, t1, symlib)

    # T2 equation
    t2new = eris.ovov.conj().transpose(0,2,1,3).copy()
    if cc.cc2:
        Woooo2  = eris.oooo.transpose(0,2,1,3).copy()
        Woooo2 += lib.einsum('kilc,jc->klij', eris.ooov, t1, symlib)
        Woooo2 += lib.einsum('ljkc,ic->klij', eris.ooov, t1, symlib)
        tmp     = lib.einsum('kcld,ic->kild', eris.ovov, t1, symlib)
        Woooo2 += lib.einsum('kild,jd->klij', tmp, t1, symlib)
        tmp     = lib.einsum('klij,ka->alij', Woooo2, t1, symlib)
        t2new  += lib.einsum('alij,lb->ijab', tmp, t1, symlib)
        Wvvvv   = lib.einsum('kcbd,ka->abcd', eris.ovvv, -t1, symlib)
        Wvvvv   = Wvvvv + Wvvvv.transpose(1,0,3,2)
        Wvvvv  += eris.vvvv.transpose(0,2,1,3)
        tmp     = lib.einsum('abcd,ic->abid', Wvvvv, t1, symlib)
        t2new  += lib.einsum('abid,jd->ijab', tmp, t1, symlib)
        Lvv2    = fvv - lib.einsum('kc,ka->ac', fov, t1, symlib)
        Lvv2   -= fvv.diagonal(preserve_shape=True)
        tmp     = lib.einsum('ac,ijcb->ijab', Lvv2, t2, symlib)
        t2new  += (tmp + tmp.transpose(1,0,3,2))
        Loo2    = foo + lib.einsum('kc,ic->ki', fov, t1, symlib)
        Loo2   -= foo.diagonal(preserve_shape=True)
        tmp     = lib.einsum('ki,kjab->ijab', Loo2, t2, symlib)
        t2new  -= (tmp + tmp.transpose(1,0,3,2))
    else:
        Loo = imd.Loo(t1, t2, eris)
        Lvv = imd.Lvv(t1, t2, eris)
        Loo -= foo.diagonal(preserve_shape=True)
        Lvv -= fvv.diagonal(preserve_shape=True)
        Woooo = imd.cc_Woooo(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)
        Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
        tau = t2 + lib.einsum('ia,jb->ijab', t1, t1, symlib)
        t2new += lib.einsum('klij,klab->ijab', Woooo, tau, symlib)
        t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau, symlib)
        tmp = lib.einsum('ac,ijcb->ijab', Lvv, t2, symlib)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('ki,kjab->ijab', Loo, t2, symlib)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp  = 2*lib.einsum('akic,kjcb->ijab', Wvoov, t2, symlib)
        tmp -=   lib.einsum('akci,kjcb->ijab', Wvovo, t2, symlib)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('akic,kjbc->ijab', Wvoov, t2, symlib)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2, symlib)
        t2new -= (tmp + tmp.transpose(1,0,3,2))

    tmp2  = lib.einsum('kibc,ka->abic', eris.oovv, -t1, symlib)
    tmp2 += eris.ovvv.conj().transpose(1,3,0,2)
    tmp = lib.einsum('abic,jc->ijab', tmp2, t1, symlib)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp2  = lib.einsum('kcai,jc->akij', eris.ovvo, t1, symlib)
    tmp2 += eris.ooov.transpose(3,1,2,0).conj()
    tmp = lib.einsum('akij,kb->ijab', tmp2, t1, symlib)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    t1new /= eris.eia
    t2new /= eris.eijab

    return t1new, t2new


def energy(cc, t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    e = 2*lib.einsum('ia,ia', eris.fov, t1, symlib)
    tau = lib.einsum('ia,jb->ijab',t1,t1, symlib)
    tau += t2
    e += 2*lib.einsum('ijab,iajb', tau, eris.ovov, symlib)
    e +=  -lib.einsum('ijab,ibja', tau, eris.ovov, symlib)
    return e.real


class RCCSD(ccsd.CCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])
        self.lib = sym
        self.symlib = None
        self.backend = sym.backend
        self.log = self.backend.Logger(self.stdout, self.verbose)


    def dump_flags(self, verbose=None):
        log = self.log
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

    def init_amps(self, eris):
        lib, symlib = self.lib, self.symlib
        t1 = eris.fov.conj() / eris.eia
        t2 = eris.ovov.transpose(0,2,1,3).conj() / eris.eijab
        self.emp2  = 2*lib.einsum('ijab,iajb', t2, eris.ovov, symlib)
        self.emp2 -=   lib.einsum('ijab,ibja', t2, eris.ovov, symlib)
        self.log.info('Init t2, MP2 energy = %.15g', self.emp2.real)
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
                self.log.info('%s converged', cctyp)
            else:
                self.log.info('%s not converged', cctyp)
        if self._scf.e_tot == 0:
            self.log.note('E_corr = %.16g', self.e_corr)
        else:
            self.log.note('E(%s) = %.16g  E_corr = %.16g',
                        cctyp, self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    energy = energy
    update_amps = update_amps


class _ChemistsERIs:
    def __init__(self, cc, mo_coeff=None):
        self.lib = lib = cc.lib
        self.symlib = cc.symlib
        log = cc.log
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
        self.foo = lib.tensor(fock[:nocc,:nocc])
        self.fov = lib.tensor(fock[:nocc,nocc:])
        self.fvv = lib.tensor(fock[nocc:,nocc:])
        eia  = mo_e[:nocc,None]- mo_e[None,nocc:]
        eijab = eia[:,None,:,None] + eia[None,:,None,:]
        self.eia = lib.tensor(eia)
        self.eijab = lib.tensor(eijab)
        cput1 = cput0 = (time.clock(), time.time())
        eri1 = ao2mo.incore.full(cc._scf._eri, mo_coeff)
        eri1 = ao2mo.restore(1, eri1, nmo)
        self.oooo = lib.tensor(eri1[:nocc,:nocc,:nocc,:nocc].copy())
        self.ooov = lib.tensor(eri1[:nocc,:nocc,:nocc,nocc:].copy())
        self.ovov = lib.tensor(eri1[:nocc,nocc:,:nocc,nocc:].copy())
        self.oovv = lib.tensor(eri1[:nocc,:nocc,nocc:,nocc:].copy())
        self.ovvo = lib.tensor(eri1[:nocc,nocc:,nocc:,:nocc].copy())
        self.ovvv = lib.tensor(eri1[:nocc,nocc:,nocc:,nocc:].copy())
        self.vvvv = lib.tensor(eri1[nocc:,nocc:,nocc:,nocc:].copy())
        cput1 = log.timer('ao2mo transformation', *cput1)

class _IMDS:
    def __init__(self, cc):
        self.t1 = cc.t1
        self.t2 = cc.t2
        self.eris = cc.eris
        self.made_ip_imds = False
        self.made_ea_imds = False
        self._made_shared_2e = False
        self.log = cc.log

    def _make_shared_1e(self):
        cput0 = (time.clock(), time.time())

        t1,t2,eris = self.t1, self.t2, self.eris
        self.Loo = imd.Loo(t1,t2,eris)
        self.Lvv = imd.Lvv(t1,t2,eris)
        self.Fov = imd.cc_Fov(t1,t2,eris)

        self.log.timer('EOM-CCSD shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (time.clock(), time.time())

        t1,t2,eris = self.t1, self.t2, self.eris
        # 2 virtuals
        self.Wovov = imd.Wovov(t1,t2,eris)
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Woovv = eris.ovov.transpose(0,2,1,3)

        self.log.timer('EOM-CCSD shared two-electron intermediates', *cput0)

    def make_ip(self, ip_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ip_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())

        t1,t2,eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1,t2,eris)
        self.Wooov = imd.Wooov(t1,t2,eris)
        self.Wovoo = imd.Wovoo(t1,t2,eris)
        self.made_ip_imds = True
        self.log.timer('EOM-CCSD IP intermediates', *cput0)

    def make_ea(self, ea_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ea_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())

        t1,t2,eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1,t2,eris)
        if ea_partition == 'mp':
            self.Wvvvo = imd.Wvvvo(t1,t2,eris)
        else:
            self.Wvvvv = imd.Wvvvv(t1,t2,eris)
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,self.Wvvvv)
        self.made_ea_imds = True
        self.log.timer('EOM-CCSD EA intermediates', *cput0)

    def make_ee(self):
        raise NotImplementedError


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf.cc.rccsd_slow import RCCSD as REFCCSD

    mol = gto.M()
    nocc, nvir = 5, 12
    nmo = nocc + nvir
    nmo_pair = nmo*(nmo+1)//2
    mf = scf.RHF(mol)
    np.random.seed(12)
    mf._eri = np.random.random(nmo_pair*(nmo_pair+1)//2)
    mf.mo_coeff = np.random.random((nmo,nmo))
    mf.mo_energy = np.arange(0., nmo)
    mf.mo_occ = np.zeros(nmo)
    mf.mo_occ[:nocc] = 2
    vhf = mf.get_veff(mol, mf.make_rdm1())
    cinv = np.linalg.inv(mf.mo_coeff)
    mf.get_hcore = lambda *args: (reduce(np.dot, (cinv.T*mf.mo_energy, cinv)) - vhf)
    mycc = RCCSD(mf)
    lib = mycc.lib
    eris = mycc.ao2mo()
    a = np.random.random((nmo,nmo)) * .1
    a += a.T.conj()

    eris.foo += a[:nocc,:nocc]
    eris.fov += a[:nocc,nocc:]
    eris.fvv += a[nocc:,nocc:]
    mo_occ = eris.foo.diagonal()
    mo_vir = eris.fvv.diagonal()
    eia = mo_occ[:,None] - mo_vir[None,:]
    eris.eia = lib.tensor(eia)
    eris.eijab = lib.tensor(eia[:,None,:,None] + eia[None,:,None,:] )
    #eris.fock += a + a.T.conj()
    t1 = np.random.random((nocc,nvir)) * .1
    t2 = np.random.random((nocc,nocc,nvir,nvir)) * .1
    t2 = t2 + t2.transpose(1,0,3,2)
    t1 = lib.tensor(t1)
    t2 = lib.tensor(t2)

    mycc.cc2 = False
    t1a, t2a = mycc.update_amps(t1, t2, eris)
    print(pyscflib.finger(t1a.array) - -106360.5276951083)
    print(pyscflib.finger(t2a.array) - 66540.100267798145)
    mycc.cc2 = True
    t1a, t2a = mycc.update_amps(t1, t2, eris)
    print(pyscflib.finger(t1a.array) - -106360.5276951083)
    print(pyscflib.finger(t2a.array) - -1517.9391800662809)

    eri1 = np.random.random((nmo,nmo,nmo,nmo)) + np.random.random((nmo,nmo,nmo,nmo))*1j
    eri1 = eri1.transpose(0,2,1,3)
    eri1 = eri1 + eri1.transpose(1,0,3,2).conj()
    eri1 = eri1 + eri1.transpose(2,3,0,1)
    eri1 *= .1
    eri1 = lib.tensor(eri1)
    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
    eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
    eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
    a = np.random.random((nmo,nmo)) * .1j
    a = a + a.T.conj()
    eris.fov = eris.fov + a[:nocc,nocc:]
    eris.foo = eris.foo + a[:nocc,:nocc]
    eris.fvv = eris.fvv + a[nocc:,nocc:]

    mo_occ = eris.foo.diagonal()
    mo_vir = eris.fvv.diagonal()
    eia = mo_occ[:,None] - mo_vir[None,:]
    eris.eia = lib.tensor(eia)
    eris.eijab = lib.tensor(eia[:,None,:,None] + eia[None,:,None,:] )
    #eris.fock = eris.fock + a + a.T.conj()

    t1 = t1 + np.random.random((nocc,nvir)) * .1j
    t2 = t2 + np.random.random((nocc,nocc,nvir,nvir)) * .1j
    t2 = t2 + t2.transpose(1,0,3,2)
    mycc.cc2 = False
    t1a, t2a = mycc.update_amps(t1, t2, eris)
    print(pyscflib.finger(t1a.array) - (-13.32050019680894-1.8825765910430254j))
    print(pyscflib.finger(t2a.array) - (9.2521062044785189+29.999480274811873j))
    mycc.cc2 = True
    t1a, t2a = mycc.update_amps(t1, t2, eris)
    print(pyscflib.finger(t1a.array) - (-13.32050019680894-1.8825765910430254j))
    print(pyscflib.finger(t2a.array) - (-0.056223856104895858+0.025472249329733986j))

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    #mol.basis = '3-21G'
    mol.verbose = 0
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    mycc = RCCSD(mf)
    eris = mycc.ao2mo()
    emp2, t1, t2 = mycc.init_amps(eris)
    print(pyscflib.finger(t2.array) - 0.08551011863965133)
    np.random.seed(1)
    t1 = np.random.random(t1.shape)*.1
    t2 = np.random.random(t2.shape)*.1
    lib = mycc.lib
    t1 = lib.tensor(t1)
    t2 = lib.tensor(t2)
    t2 = t2 + t2.transpose(1,0,3,2)
    t1, t2 = mycc.update_amps(t1, t2, eris)

    print(pyscflib.finger(t1.array) - -0.019600587272652903)
    print(pyscflib.finger(t2.array) - -0.012913260807189797)

    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.21334324674165406)
