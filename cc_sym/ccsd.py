#!/usr/bin/env python

from pyscf.cc import ccsd
from cc_sym.linalg_helper.diis import DIIS
from symtensor.settings import load_lib
from pyscf.lib.logger import Logger
from pyscf.lib import logger
from symtensor import sym
import time


def kernel(mycc, eris, t1, t2):
    if eris is None: mycc.eris = eris = mycc.ao2mo(self.mo_coeff)
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

class CCSDBasics(ccsd.CCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, SYMVERBOSE=0):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])
        self.lib = None
        self.symlib = None
        self.SYMVERBOSE = SYMVERBOSE

    @property
    def _backend(self):
        return 'numpy'

    @property
    def _sym1(self):
        return None

    @property
    def _sym2(self):
        return None

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        nov = nocc * nvir
        t1 = vec[:nov].reshape(nocc,nvir)
        t2 = vec[nov:].reshape(nocc,nocc,nvir,nvir)
        t1  = self.lib.tensor(t1, self._sym1, symlib=self.symlib, verbose=self.SYMVERBOSE)
        t2  = self.lib.tensor(t2, self._sym2, symlib=self.symlib, verbose=self.SYMVERBOSE)
        return t1, t2

    def kernel(self, t1=None, t2=None, eris=None):
        return self.ccsd(t1, t2, eris)

    def ccsd(self, t1=None, t2=None, eris=None):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)
        #if self.verbose >= logger.WARN:
        #    self.check_sanity()
        #exit()
        self.dump_flags()
        if eris is None:
            self.eris = eris = self.ao2mo(self.mo_coeff)

        self.e_hf = getattr(eris, 'e_hf', None)
        if self.e_hf is None:
            self.e_hf = self._scf.e_tot
        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2)

        self._finalize()
        return self.e_corr, self.t1, self.t2

    def init_amps(self, eris):
        raise NotImplementedError

    def update_amps(self, t1, t2, eris):
        raise NotImplementedError

    def energy(cc, t1, t2, eris):
        raise NotImplementedError

    def ao2mo(self, mo_coeff=None):
        raise NotImplementedError

    def amplitudes_to_vector(self, t1, t2):
        raise NotImplementedError
