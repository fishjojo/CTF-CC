#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
#from pyscf.cc.eom_rccsd import EOM
#from pyscf import lib as pyscflib

from pyscf.lib import logger
import numpy as np
import time
import cc_sym.rintermediates as imd
import ctf
import symtensor.sym_ctf as sym
from cc_sym import eom_rccsd_numpy

from cc_sym import mpi_helper

comm = mpi_helper.comm
rank = mpi_helper.rank
size = mpi_helper.size

tensor = sym.tensor

def kernel(eom, nroots=1, koopmans=True, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):

    cput0 = (time.clock(), time.time())
    eom.dump_flags()

    if imds is None:
        imds = eom.make_imds(eris)

    size = eom.vector_size()
    nroots = min(nroots,size)
    from cc_sym.linalg_helper import davidson

    matvec, diag = eom.gen_matvec(imds, left=left, **kwargs)
    user_guess = False
    if guess:
        user_guess = True
        assert len(guess) == nroots
        for g in guess:
            assert g.size == size
    else:
        user_guess = False
        guess = eom.get_init_guess(nroots, koopmans, diag)
    conv, evals, evecs = davidson.davidson(matvec, size, nroots, x0=guess, Adiag=diag, verbose=eom.verbose)
    evals = evals.real
    for n, en, vn in zip(range(nroots), evals, evecs):
        r1, r2 = eom.vector_to_amplitudes(vn)
        qp_weight = r1.norm()**2
        logger.info(eom, 'EOM-CCSD root %d E = %.16g  qpwt = %0.6g', n, en, qp_weight)
    logger.timer(eom, 'EOM-CCSD', *cput0)
    return conv, evals, evecs

def amplitudes_to_vector(eom, r1, r2):
    vector = ctf.hstack((r1.array.ravel(), r2.array.ravel()))
    return vector

def vector_to_amplitudes_ip(eom, vector):
    nocc = eom.nocc
    nvir = eom.nmo - nocc
    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nocc,nocc,nvir)
    r1 = tensor(r1, verbose=eom.SYMVERBOSE)
    r2 = tensor(r2, verbose=eom.SYMVERBOSE)
    return [r1,r2]

def vector_to_amplitudes_ea(eom, vector):
    nocc = eom.nocc
    nvir = eom.nmo - nocc
    r1 = vector[:nvir].copy()
    r2 = vector[nvir:].copy().reshape(nocc,nvir,nvir)
    r1 = tensor(r1)
    r2 = tensor(r2)
    return [r1,r2]

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    lib = eom.lib
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape
    foo = imds.eris.foo
    fvv = imds.eris.fvv

    Hr1 = - imds.Loo.diagonal()
    Hr1 = tensor(Hr1)
    if eom.partition == 'mp':
        ijb = - foo.diagonal().reshape(nocc,1,1)
        ijb = ijb - foo.diagonal().reshape(1,nocc,1)
        ijb = ijb + fvv.diagonal().reshape(1,1,nvir)
        Hr2 = tensor(ijb)
    else:
        wij = ctf.einsum('ijij->ij', imds.Woooo.array)
        wjb = ctf.einsum('jbjb->jb', imds.Wovov.array)
        wjb2 = ctf.einsum('jbbj->jb', imds.Wovvo.array)
        wib = ctf.einsum('ibib->ib', imds.Wovov.array)
        ijb  = imds.Lvv.diagonal().reshape(1,1,nvir)
        ijb = ijb - imds.Loo.diagonal().reshape(nocc,1,1)
        ijb = ijb -imds.Loo.diagonal().reshape(1,nocc,1)
        ijb = ijb + wij.reshape(nocc,nocc,1)
        ijb = ijb - wjb.reshape(1,nocc,nvir)
        ijb = ijb + 2*wjb2.reshape(1,nocc,nvir)
        ijb = ijb - ctf.einsum('ij,jb->ijb', ctf.eye(nocc), wjb2)
        ijb = ijb - wib.reshape(nocc,1,nvir)
        Hr2 = tensor(ijb)
        Hr2 -= 2.*lib.einsum('ijcb,jibc->ijb', t2, imds.Woovv)
        Hr2 += lib.einsum('ijcb,ijbc->ijb', t2, imds.Woovv)
    vector = eom.amplitudes_to_vector(Hr1,Hr2)
    return vector




def eaccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    lib = eom.lib
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape
    foo = imds.eris.foo
    fvv = imds.eris.fvv
    Hr1 = imds.Lvv.diagonal()
    Hr1 = tensor(Hr1)

    if eom.partition == 'mp':
        jab = fvv.diagonal()[None,:,None]
        jab = jab + fvv.diagonal()[None,None,:]
        jab = jab - foo.diagonal()[:,None,None]
        Hr2 = tensor(jab)
    else:
        jab = imds.Lvv.diagonal().reshape(1,nvir,1)
        jab = jab + imds.Lvv.diagonal().reshape(1,1,nvir)
        jab = jab - imds.Loo.diagonal().reshape(nocc,1,1)
        wab = ctf.einsum("abab->ab", imds.Wvvvv.array)
        wjb = ctf.einsum('jbjb->jb', imds.Wovov.array)
        wjb2 = ctf.einsum('jbbj->jb', imds.Wovvo.array)
        wja = ctf.einsum('jaja->ja', imds.Wovov.array)
        jab = jab + wab.reshape(1,nvir,nvir)
        jab = jab - wjb.reshape(nocc,1,nvir)
        jab = jab + 2*wjb2.reshape(nocc,1,nvir)
        jab -= ctf.einsum('jb,ab->jab', wjb2, ctf.eye(nvir))
        jab = jab - wja.reshape(nocc,nvir,1)
        Hr2 = tensor(jab)
        Hr2 -= 2*lib.einsum('ijab,ijab->jab', t2, imds.Woovv)
        Hr2 += lib.einsum('ijab,ijba->jab', t2, imds.Woovv)
    vector = eom.amplitudes_to_vector(Hr1,Hr2)
    return vector

class EOMIP(eom_rccsd_numpy.EOMIP):

    amplitudes_to_vector = amplitudes_to_vector
    vector_to_amplitudes = vector_to_amplitudes_ip

    get_diag = ipccsd_diag
    kernel = kernel
    ipccsd = kernel

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        if koopmans:
            idx = range(self.nocc-nroots, self.nocc)[::-1]
        else:
            if diag is None: diag = self.get_diag()
            idx = diag.to_nparray().argsort()[:nroots]
        guess = ctf.zeros([nroots,size], dtype)
        idx = np.arange(nroots)*size + np.asarray(idx)
        if rank==0:
            guess.write(idx, np.ones(nroots))
        else:
            guess.write([],[])

        return guess


class EOMEA(eom_rccsd_numpy.EOMEA):

    amplitudes_to_vector = amplitudes_to_vector
    vector_to_amplitudes = vector_to_amplitudes_ea
    get_diag = eaccsd_diag
    kernel = kernel
    eaccsd = kernel

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            idx = range(nroots)
        else:
            if diag is None: diag = self.get_diag()
            idx = diag.to_nparray().argsort()[:nroots]
        guess = ctf.zeros([nroots,size], dtype)
        idx = np.arange(nroots)*size + np.asarray(idx)
        if rank==0:
            guess.write(idx, np.ones(nroots))
        else:
            guess.write([], [])
        return guess


class _IMDS:
    def __init__(self, cc, eris=None):
        self.t1 = cc.t1
        self.t2 = cc.t2
        if eris is None:
            if cc.eris is None:
                self.eris = cc.ao2mo()
            else:
                self.eris = cc.eris
        self.made_ip_imds = False
        self.made_ea_imds = False
        self._made_shared_2e = False
        self.symlib = cc.symlib

    def _make_shared_1e(self):
        cput0 = (time.clock(), time.time())
        t1,t2,eris = self.t1, self.t2, self.eris
        self.Loo = imd.Loo(t1,t2,eris)
        self.Lvv = imd.Lvv(t1,t2,eris)
        self.Fov = imd.cc_Fov(t1,t2,eris)

        logger.timer(self, 'EOM-CCSD shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (time.clock(), time.time())

        t1,t2,eris = self.t1, self.t2, self.eris
        # 2 virtuals
        self.Wovov = imd.Wovov(t1,t2,eris)
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Woovv = eris.ovov.transpose(0,2,1,3)

        logger.timer(self, 'EOM-CCSD shared two-electron intermediates', *cput0)

    def make_ip(self, partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())

        t1,t2,eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        if partition != 'mp':
            self.Woooo = imd.Woooo(t1,t2,eris)
        self.Wooov = imd.Wooov(t1,t2,eris)
        self.Wovoo = imd.Wovoo(t1,t2,eris)
        self.made_ip_imds = True
        logger.timer(self, 'EOM-CCSD IP intermediates', *cput0)

    def make_ea(self, partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())

        t1,t2,eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1,t2,eris)
        if partition == 'mp':
            self.Wvvvo = imd.Wvvvo(t1,t2,eris)
        else:
            self.Wvvvv = imd.Wvvvv(t1,t2,eris)
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,self.Wvvvv)
        self.made_ea_imds = True
        logger.timer(self, 'EOM-CCSD EA intermediates', *cput0)

    def make_ee(self):
        raise NotImplementedError

if __name__ == '__main__':
    from pyscf import gto, scf, cc
    from cc_sym.rccsd import RCCSD
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 5
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol)
    if rank==0:
        mf.kernel()

    comm.barrier()
    mf.mo_coeff = comm.bcast(mf.mo_coeff, root=0)
    mf.mo_occ = comm.bcast(mf.mo_occ, root=0)

    mycc = RCCSD(mf)
    mycc.kernel()

    myeom = EOMIP(mycc)
    _, eip, _ = myeom.ipccsd(nroots=2)
    myeom = EOMEA(mycc)
    _, eea, _ = myeom.eaccsd(nroots=2)

    print(np.amax(eip[0]-0.4335604229241659))
    print(np.amax(eip[1]-0.5187659782655635))

    print(np.amax(eea[0]-0.1673788639606518))
    print(np.amax(eea[1]-0.2402762272383755))
