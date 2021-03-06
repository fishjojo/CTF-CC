#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
from pyscf.cc.eom_rccsd import EOM
from pyscf import lib as pyscflib
import numpy as np
import time
import cc_sym.rintermediates as imd
import ctf
import symtensor.sym_ctf as lib
from cc_sym import settings
comm = settings.comm
rank = settings.rank
size = settings.size
Logger = settings.Logger

tensor = lib.tensor

def kernel(eom, nroots=1, koopmans=True, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):

    cput0 = (time.clock(), time.time())
    log = Logger(eom.stdout, eom.verbose)
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
        log.info('EOM-CCSD root %d E = %.16g  qpwt = %0.6g', n, en, qp_weight)
    log.timer('EOM-CCSD', *cput0)
    return conv, evals, evecs

def amplitudes_to_vector(eom, r1, r2):
    vector = ctf.hstack((r1.array.ravel(), r2.array.ravel()))
    return vector

def vector_to_amplitudes_ip(eom, vector):
    nocc = eom.nocc
    nvir = eom.nmo - nocc
    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nocc,nocc,nvir)
    r1 = tensor(r1)
    r2 = tensor(r2)
    return [r1,r2]

def vector_to_amplitudes_ea(eom, vector):
    nocc = eom.nocc
    nvir = eom.nmo - nocc
    r1 = vector[:nvir].copy()
    r2 = vector[nvir:].copy().reshape(nocc,nvir,nvir)
    r1 = tensor(r1)
    r2 = tensor(r2)
    return [r1,r2]

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    r1,r2 = eom.vector_to_amplitudes(vector)

    # 1h-1h block
    Hr1 = -lib.einsum('ki,k->i',imds.Loo,r1)
    #1h-2h1p block
    Hr1 += 2*lib.einsum('ld,ild->i',imds.Fov,r2)
    Hr1 +=  -lib.einsum('kd,kid->i',imds.Fov,r2)
    Hr1 += -2*lib.einsum('klid,kld->i',imds.Wooov,r2)
    Hr1 +=    lib.einsum('lkid,kld->i',imds.Wooov,r2)

    # 2h1p-1h block
    Hr2 = -lib.einsum('kbij,k->ijb',imds.Wovoo,r1)
    # 2h1p-2h1p block
    if eom.partition == 'mp':
        foo = self.eris.foo
        fvv = self.eris.fvv
        Hr2 += lib.einsum('bd,ijd->ijb',fvv,r2)
        Hr2 += -lib.einsum('ki,kjb->ijb',foo,r2)
        Hr2 += -lib.einsum('lj,ilb->ijb',foo,r2)
    elif eom.partition == 'full':
        Hr2 += self._ipccsd_diag_matrix2*r2
    else:
        Hr2 += lib.einsum('bd,ijd->ijb',imds.Lvv,r2)
        Hr2 += -lib.einsum('ki,kjb->ijb',imds.Loo,r2)
        Hr2 += -lib.einsum('lj,ilb->ijb',imds.Loo,r2)
        Hr2 +=  lib.einsum('klij,klb->ijb',imds.Woooo,r2)
        Hr2 += 2*lib.einsum('lbdj,ild->ijb',imds.Wovvo,r2)
        Hr2 +=  -lib.einsum('kbdj,kid->ijb',imds.Wovvo,r2)
        Hr2 +=  -lib.einsum('lbjd,ild->ijb',imds.Wovov,r2) #typo in Ref
        Hr2 +=  -lib.einsum('kbid,kjd->ijb',imds.Wovov,r2)
        tmp = 2*lib.einsum('lkdc,kld->c',imds.Woovv,r2)
        tmp += -lib.einsum('kldc,kld->c',imds.Woovv,r2)
        Hr2 += -lib.einsum('c,ijcb->ijb',tmp,imds.t2)

    vector = eom.amplitudes_to_vector(Hr1,Hr2)
    return vector

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
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
        #print(ijb.shape, wjb.shape)
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


def eaccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    r1,r2 = eom.vector_to_amplitudes(vector)

    # Eq. (30)
    # 1p-1p block
    Hr1 =  lib.einsum('ac,c->a',imds.Lvv,r1)
    # 1p-2p1h block
    Hr1 += lib.einsum('ld,lad->a',2.*imds.Fov,r2)
    Hr1 += lib.einsum('ld,lda->a',  -imds.Fov,r2)
    Hr1 += 2*lib.einsum('alcd,lcd->a',imds.Wvovv,r2)
    Hr1 +=  -lib.einsum('aldc,lcd->a',imds.Wvovv,r2)
    # Eq. (31)
    # 2p1h-1p block
    Hr2 = lib.einsum('abcj,c->jab',imds.Wvvvo,r1)
    # 2p1h-2p1h block
    if eom.partition == 'mp':
        foo = imds.eris.foo
        fvv = imds.eris.fvv
        Hr2 +=  lib.einsum('ac,jcb->jab',fvv,r2)
        Hr2 +=  lib.einsum('bd,jad->jab',fvv,r2)
        Hr2 += -lib.einsum('lj,lab->jab',foo,r2)
    elif eom.partition == 'full':
        Hr2 += eom._eaccsd_diag_matrix2*r2
    else:
        Hr2 +=  lib.einsum('ac,jcb->jab',imds.Lvv,r2)
        Hr2 +=  lib.einsum('bd,jad->jab',imds.Lvv,r2)
        Hr2 += -lib.einsum('lj,lab->jab',imds.Loo,r2)
        Hr2 += 2*lib.einsum('lbdj,lad->jab',imds.Wovvo,r2)
        Hr2 +=  -lib.einsum('lbjd,lad->jab',imds.Wovov,r2)
        Hr2 +=  -lib.einsum('lajc,lcb->jab',imds.Wovov,r2)
        Hr2 +=  -lib.einsum('lbcj,lca->jab',imds.Wovvo,r2)

        Hr2 +=   lib.einsum('abcd,jcd->jab',imds.Wvvvv,r2)
        tmp = (2*lib.einsum('klcd,lcd->k',imds.Woovv,r2)
                -lib.einsum('kldc,lcd->k',imds.Woovv,r2))
        Hr2 += -lib.einsum('k,kjab->jab',tmp,imds.t2)

    vector = eom.amplitudes_to_vector(Hr1,Hr2)
    return vector

def eaccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
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

class EOMIP(EOM):
    def __init__(self, cc):
        EOM.__init__(self,cc)
        self.lib = cc.lib
        self.symlib = cc.symlib
        self._backend = cc._backend
        self.t1, self.t2 = cc.t1, cc.t2

    amplitudes_to_vector = amplitudes_to_vector
    vector_to_amplitudes = vector_to_amplitudes_ip

    matvec = ipccsd_matvec
    get_diag = ipccsd_diag
    kernel = kernel
    ipccsd = kernel

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc + nocc*nocc*nvir

    def make_imds(self, eris=None):
        self.imds = imds = _IMDS(self._cc, eris=eris)
        imds.make_ip(self.partition)
        return imds

    @property
    def eip(self):
        return self.e

    def gen_matvec(self, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        if left:
            matvec = lambda x: self.l_matvec(x, imds, diag)
        else:
            matvec = lambda x: self.matvec(x, imds, diag)
        return matvec, diag

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

    def dump_flags(self, verbose=None):
        if verbose is None: verbose=self.verbose
        logger = Logger(self.stdout, verbose)
        logger.info('')
        logger.info('******** %s ********', self.__class__)
        logger.info('max_space = %d', self.max_space)
        logger.info('max_cycle = %d', self.max_cycle)
        logger.info('conv_tol = %s', self.conv_tol)
        logger.info('partition = %s', self.partition)
        logger.info('max_memory %d MB (current use %d MB)',
                    self.max_memory, pyscflib.current_memory()[0])
        return self

class EOMEA(EOMIP):

    vector_to_amplitudes = vector_to_amplitudes_ea

    matvec = eaccsd_matvec
    get_diag = eaccsd_diag
    eaccsd = kernel

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nvir + nocc*nvir*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_ea(self.partition)
        return imds

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

    @property
    def eea(self):
        return self.e

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
        self.log = Logger(cc.stdout, cc.verbose)

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
        self.log.timer('EOM-CCSD IP intermediates', *cput0)

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
        self.log.timer('EOM-CCSD EA intermediates', *cput0)

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
    myeom.ipccsd(nroots=2)
    myeom = EOMEA(mycc)
    myeom.eaccsd(nroots=2)
