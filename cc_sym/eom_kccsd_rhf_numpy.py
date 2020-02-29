#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

from cc_sym import eom_rccsd_numpy, rccsd_numpy
import numpy as np
import time
from pyscf.pbc.mp.kmp2 import padding_k_idx
import symtensor.sym as lib
from pyscf.lib.logger import Logger


tensor = lib.tensor
zeros = lib.zeros

def kernel(eom, nroots=1, koopmans=True, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):

    cput0 = (time.clock(), time.time())
    eom.dump_flags()

    log = Logger(eom.stdout, eom.verbose)
    if imds is None:
        imds = eom.make_imds(eris)

    size = eom.vector_size()
    nroots = min(nroots,size)
    nkpts = eom.nkpts

    if kptlist is None:
        kptlist = range(nkpts)

    if dtype is None:
        dtype = imds.t1.dtype

    evals = np.zeros((len(kptlist),nroots), np.float)
    evecs = []
    convs = np.zeros((len(kptlist),nroots), dtype)
    from pyscf.pbc.lib.linalg_helper import eigs
    for k, kshift in enumerate(kptlist):
        matvec, diag = eom.gen_matvec(kshift, imds, left=left, **kwargs)
        eom.update_symlib(kshift)
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            user_guess = False
            guess = eom.get_init_guess(kshift, nroots, koopmans, diag)

        conv_k, evals_k, evecs_k = eigs(matvec, size, nroots, x0=guess, Adiag=diag, verbose=eom.verbose)
        evals_k = evals_k.real
        evals[k] = evals_k.real
        evecs.append(evecs_k)
        convs[k] = conv_k
        for n, en, vn in zip(range(nroots), evals_k, evecs_k.T):
            r1, r2 = eom.vector_to_amplitudes(vn, kshift)
            qp_weight = r1.norm()**2
            log.info('EOM-CCSD root %d E = %.16g  qpwt = %0.6g', n, en, qp_weight)
    log.timer('EOM-CCSD', *cput0)
    evecs = np.vstack(tuple(evecs))
    return convs, evals, evecs



def vector_to_amplitudes_ip(eom, vector, kshift):
    nkpts, kpts, nocc = eom.nkpts, eom.kpts, eom.nocc
    nvir = eom.nmo - nocc
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]
    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nkpts,nkpts,nocc,nocc,nvir)
    r1 = tensor(r1, sym1, symlib=eom.symlib)
    r2 = tensor(r2, sym2, symlib=eom.symlib)
    return [r1,r2]

def vector_to_amplitudes_ea(eom, vector, kshift):
    nkpts, kpts, nocc = eom.nkpts, eom.kpts, eom.nocc
    nvir = eom.nmo - nocc
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]
    r1 = vector[:nvir].copy()
    r2 = vector[nvir:].copy().reshape(nkpts,nkpts,nocc,nvir,nvir)
    r1 = tensor(r1,sym1, symlib=eom.symlib)
    r2 = tensor(r2,sym2, symlib=eom.symlib)
    return [r1,r2]

def ipccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    kpts, gvec = eom.kpts, eom._cc._scf.cell.reciprocal_vectors()
    t1, t2 = imds.t1, imds.t2
    dtype = t2.dtype
    nocc, nvir = t1.shape
    nkpts = len(kpts)
    kconserv = eom.kconserv
    Hr1array = -imds.Loo.diagonal()[kshift]
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]
    Hr1 = tensor(Hr1array, sym1, symlib=eom.symlib)
    Hr2 = zeros([nocc,nocc,nvir], dtype, sym2, symlib=eom.symlib)

    if eom.partition == 'mp':
        foo = eom.eris.foo.diagonal()
        fvv = eom.eris.fvv.diagonal()
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki,kshift,kj]
                off = ki * nkpts + kj
                ijb = np.zeros([nocc,nocc,nvir], dtype=dtype)
                ijb += fvv[kb].reshape(1,1,-1)
                ijb -= foo[ki][:,None,None]
                ijb -= foo[kj][None,:,None]
                Hr2.array[ki,kj] = ijb

    else:
        lvv = imds.Lvv.diagonal()
        loo = imds.Loo.diagonal()
        wij = np.einsum('IJIijij->IJij', imds.Woooo.array)
        wjb = np.einsum('JBJjbjb->JBjb', imds.Wovov.array)
        wjb2 = np.einsum('JBBjbbj->JBjb', imds.Wovvo.array)
        wib = np.einsum('IBIibib->IBib', imds.Wovov.array)
        idx = np.arange(nocc)
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki,kshift,kj]
                ijb = np.zeros([nocc,nvir,nvir], dtype=dtype)
                ijb += lvv[kb][None,None,:]
                ijb -= loo[ki][:,None,None]
                ijb -= loo[kj][None,:,None]
                ijb += wij[ki,kj][:,:,None]
                ijb -= wjb[kj,kb][None,:,:]
                ijb += 2*wjb2[kj,kb][None,:,:]
                if ki == kj:
                    ijb[idx,idx] -= wjb2[kj,kb]
                ijb -= wib[ki,kb][:,None,:]
                Hr2.array[ki,kj] = ijb

        Woovvtmp = imds.Woovv.transpose(0,1,3,2)[:,:,kshift]
        Hr2 -= 2.*np.einsum('IJijcb,JIjicb->IJijb', t2[:,:,kshift], Woovvtmp)
        Hr2 += np.einsum('IJijcb,IJijcb->IJijb', t2[:,:,kshift], Woovvtmp)
    return eom.amplitudes_to_vector(Hr1, Hr2)

def ipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    r1,r2 = eom.vector_to_amplitudes(vector, kshift)

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

def eaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    r1,r2 = eom.vector_to_amplitudes(vector, kshift)

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

def eaccsd_diag(eom, kshift, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    kpts, gvec = eom.kpts, eom._cc._scf.cell.reciprocal_vectors()
    t1, t2 = imds.t1, imds.t2
    dtype = t2.dtype
    nocc, nvir = t1.shape
    nkpts = len(kpts)
    kconserv = eom.kconserv
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]

    Hr1array = imds.Lvv.diagonal()[kshift]
    Hr1 = tensor(Hr1array, sym1, symlib=eom.symlib)
    Hr2 = zeros([nocc,nvir,nvir], dtype, sym2, symlib=eom.symlib)
    idx_jab = np.arange(nocc*nvir*nvir)
    if eom.partition == 'mp':
        foo = imds.eris.foo.diagonal()
        fvv = imds.eris.fvv.diagonal()
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kshift]
                jab = np.zeros([nocc,nvir,nvir], dtype=dtype)
                jab += -foo[kj][:,None,None]
                jab += fvv[ka][None,:,None]
                jab += fvv[kb][None,None,:]
                Hr2.array[kj,ka] = jab
    else:
        idx = np.arange(nvir)
        loo = imds.Loo.diagonal()
        lvv = imds.Lvv.diagonal()
        wab = np.einsum("ABAabab->ABab", imds.Wvvvv.array)
        wjb = np.einsum('JBJjbjb->JBjb', imds.Wovov.array)
        wjb2 = np.einsum('JBBjbbj->JBjb', imds.Wovvo.array)
        wja = np.einsum('JAJjaja->JAja', imds.Wovov.array)

        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kj,ka,kshift]
                jab = np.zeros([nocc,nvir,nvir], dtype=dtype)
                jab -= loo[kj][:,None,None]
                jab += lvv[ka][None,:,None]
                jab += lvv[kb][None,None,:]
                jab += wab[ka,kb][None,:,:]
                jab -= wjb[kj,kb][:,None,:]
                jab += 2*wjb2[kj,kb][:,None,:]
                if ka == kb:
                    jab[:,idx,idx] -= wjb2[kj,ka]
                jab -= wja[kj,ka][:,:,None]
                Hr2.array[kj,ka] = jab
        Hr2 -= 2*np.einsum('JAijab,JAijab->JAjab', t2[kshift], imds.Woovv[kshift])
        Woovvtmp = imds.Woovv.transpose(0,1,3,2)[kshift]
        Hr2 += np.einsum('JAijab,JAijab->JAjab', t2[kshift], Woovvtmp)

    return eom.amplitudes_to_vector(Hr1, Hr2)

class EOMIP(eom_rccsd_numpy.EOMIP):
    def __init__(self, cc):
        eom_rccsd_numpy.EOMIP.__init__(self,cc)
        self.kpts = cc.kpts
        self.lib = cc.lib
        self.symlib = cc.symlib
        self.t1, self.t2 = cc.t1, cc.t2
        self.nonzero_opadding, self.nonzero_vpadding = self.get_padding_k_idx(cc)
        self.kconserv = cc.khelper.kconserv

    matvec = ipccsd_matvec
    vector_to_amplitudes = vector_to_amplitudes_ip
    get_diag = ipccsd_diag
    kernel = kernel
    ipccsd = kernel

    def get_padding_k_idx(self, cc):
        return padding_k_idx(cc, kind='split')

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nocc + nkpts**2*nocc*nocc*nvir

    @property
    def nkpts(self):
        return len(self.kpts)

    def update_symlib(self,kshift):
        kpts, gvec = self.kpts, self._cc._scf.cell.reciprocal_vectors()
        sym1 = ['+', [kpts,], kpts[kshift], gvec]
        sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]
        self.symlib.update(sym1,sym2)

    def gen_matvec(self, kshift, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(kshift, imds)
        if left:
            raise NotImplementedError
            matvec = lambda xs: [self.l_matvec(x, kshift, imds, diag) for x in xs]
        else:
            matvec = lambda x: self.matvec(x, kshift, imds, diag)
        return matvec, diag

    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        guess = np.zeros([int(size), nroots], dtype=dtype)
        if koopmans:
            idx = self.nonzero_opadding[kshift][::-1][:nroots]
        else:
            if diag is None:
                diag = self.get_diag(kshift, imds=None)
            idx = diag.argsort()[:nroots]
        for kn, n in enumerate(idx):
            guess[n, kn] = 1.0
        return guess

class EOMEA(eom_rccsd_numpy.EOMEA):
    def __init__(self, cc):
        eom_rccsd_numpy.EOMEA.__init__(self,cc)
        self.kpts = cc.kpts
        self.lib = cc.lib
        self.symlib = cc.symlib
        self.t1, self.t2 = cc.t1, cc.t2
        self.nonzero_opadding, self.nonzero_vpadding = self.get_padding_k_idx(cc)
        self.kconserv = cc.khelper.kconserv

    matvec = eaccsd_matvec
    vector_to_amplitudes = vector_to_amplitudes_ea
    get_diag = eaccsd_diag
    eaccsd = kernel
    kernel = kernel

    def get_padding_k_idx(self, cc):
        return padding_k_idx(cc, kind='split')

    @property
    def nkpts(self):
        return len(self.kpts)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nvir + nkpts**2*nocc*nvir*nvir

    def update_symlib(self,kshift):
        kpts, gvec = self.kpts, self._cc._scf.cell.reciprocal_vectors()
        sym1 = ['+', [kpts,], kpts[kshift], gvec]
        sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]
        self.symlib.update(sym1,sym2)

    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(size, nroots)
        guess = np.zeros([int(size), nroots], dtype=dtype)
        if koopmans:
            idx = self.nonzero_vpadding[kshift][:nroots]
        else:
            idx = diag.argsort()[:nroots]
        for kn, n in enumerate(idx):
            guess[n, kn] = 1.0
        return guess

    def gen_matvec(self, kshift, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(kshift, imds)
        if left:
            raise NotImplementedError
            matvec = lambda xs: [self.l_matvec(x, kshift, imds, diag) for x in xs]
        else:
            matvec = lambda x: self.matvec(x, kshift, imds, diag)
        return matvec, diag


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    from cc_sym.kccsd_rhf_numpy import KRCCSD
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
    cell.mesh = [15,15,15]
    cell.build()

    kpts = cell.make_kpts([1,1,3])
    mf = scf.KRHF(cell,kpts, exxdiv=None)
    mf.kernel()

    mycc = KRCCSD(mf)
    mycc.kernel()

    myeom = EOMIP(mycc)
    myeom.ipccsd(nroots=2, kptlist=[1], koopmans=True)
    myeom = EOMEA(mycc)
    myeom.eaccsd(nroots=2, kptlist=[1], koopmans=True)
