import eom_rccsd
import numpy as np
import time
from pyscf.pbc.mp.kmp2 import padding_k_idx

def kernel(eom, nroots=1, koopmans=True, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):

    cput0 = (time.clock(), time.time())
    log, symlib = eom.log, eom.symlib
    eom.dump_flags()

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
    backend = eom.backend
    from linalg_helper.davidson import eigs

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

        conv_k, evals_k, evecs_k = eigs(matvec, size, nroots, backend=eom._backend, x0=guess, Adiag=diag, verbose=eom.verbose)
        evals_k = evals_k.real
        evals[k] = evals_k.real
        evecs.append(evecs_k)
        convs[k] = conv_k

        for n, en, vn in zip(range(nroots), evals_k, evecs_k):
            r1, r2 = eom.vector_to_amplitudes(vn, kshift)
            qp_weight = r1.norm()**2
            log.info('EOM-CCSD root %d E = %.16g  qpwt = %0.6g', n, en, qp_weight)
    log.timer('EOM-CCSD', *cput0)
    evecs = backend.vstack(tuple(evecs))
    return convs, evals, evecs



def vector_to_amplitudes_ip(eom, vector, kshift):
    nkpts, kpts, nocc = eom.nkpts, eom.kpts, eom.nocc
    nvir = eom.nmo - nocc
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]
    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nkpts,nkpts,nocc,nocc,nvir)
    r1 = eom.lib.tensor(r1,sym1)
    r2 = eom.lib.tensor(r2,sym2)
    return [r1,r2]

def vector_to_amplitudes_ea(eom, vector, kshift):
    nkpts, kpts, nocc = eom.nkpts, eom.kpts, eom.nocc
    nvir = eom.nmo - nocc
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]
    r1 = vector[:nvir].copy()
    r2 = vector[nvir:].copy().reshape(nkpts,nkpts,nocc,nvir,nvir)
    r1 = eom.lib.tensor(r1,sym1)
    r2 = eom.lib.tensor(r2,sym2)
    return [r1,r2]

def ipccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    backend, lib, symlib = eom.backend, eom.lib, eom.symlib
    kpts, gvec = eom.kpts, eom._cc._scf.cell.reciprocal_vectors()
    t1, t2 = imds.t1, imds.t2
    dtype = t2.dtype
    nocc, nvir = t1.shape
    nkpts = len(kpts)
    kconserv = eom.kconserv
    Hr1array = -imds.Loo.diagonal()[kshift]
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]
    Hr1 = lib.tensor(Hr1array, sym1)
    Hr2 = lib.zeros([nocc,nocc,nvir], dtype, sym2)
    tasks, ntasks = eom.gen_tasks(range(nkpts**2))

    idx_ijb = np.arange(nocc*nocc*nvir)
    if eom.partition == 'mp':
        foo = backend.to_nparray(eom.eris.foo.diagonal())
        fvv = backend.to_nparray(eom.eris.fvv.diagonal())
        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2.write([], [])
                continue
            kijb = tasks[itask]
            ki, kj = (kijb).__divmod__(nkpts)
            kb = kconserv[ki,kshift,kj]
            off = ki * nkpts + kj
            ijb = np.zeros([nocc,nocc,nvir], dtype=dtype)
            ijb += fvv[kb].reshape(1,1,-1)
            ijb -= foo[ki][:,None,None]
            ijb -= foo[kj][None,:,None]
            Hr2.write(off*idx_ijb.size+idx_ijb, ijb.ravel())
    else:
        lvv = backend.to_nparray(imds.Lvv.diagonal())
        loo = backend.to_nparray(imds.Loo.diagonal())
        wij = backend.to_nparray(backend.einsum('IJIijij->IJij', imds.Woooo.array))
        wjb = backend.to_nparray(backend.einsum('JBJjbjb->JBjb', imds.Wovov.array))
        wjb2 = backend.to_nparray(backend.einsum('JBBjbbj->JBjb', imds.Wovvo.array))
        wib = backend.to_nparray(backend.einsum('IBIibib->IBib', imds.Wovov.array))
        idx = np.arange(nocc)
        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2.write([], [])
                continue
            kijb = tasks[itask]
            ki, kj = (kijb).__divmod__(nkpts)
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
            off = ki * nkpts + kj
            Hr2.write(off*idx_ijb.size+idx_ijb, ijb.ravel())

        Woovvtmp = imds.Woovv.transpose(0,1,3,2)[:,:,kshift]
        Hr2 -= 2.*backend.einsum('IJijcb,JIjicb->IJijb', t2[:,:,kshift], Woovvtmp)
        Hr2 += backend.einsum('IJijcb,IJijcb->IJijb', t2[:,:,kshift], Woovvtmp)
    return eom.amplitudes_to_vector(Hr1, Hr2)

def ipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    lib, symlib = eom.lib, eom.symlib
    r1,r2 = eom.vector_to_amplitudes(vector, kshift)

    # 1h-1h block
    Hr1 = -lib.einsum('ki,k->i',imds.Loo,r1,symlib)
    #1h-2h1p block
    Hr1 += 2*lib.einsum('ld,ild->i',imds.Fov,r2,symlib)
    Hr1 +=  -lib.einsum('kd,kid->i',imds.Fov,r2,symlib)
    Hr1 += -2*lib.einsum('klid,kld->i',imds.Wooov,r2,symlib)
    Hr1 +=    lib.einsum('lkid,kld->i',imds.Wooov,r2,symlib)

    # 2h1p-1h block
    Hr2 = -lib.einsum('kbij,k->ijb',imds.Wovoo,r1,symlib)
    # 2h1p-2h1p block
    if eom.partition == 'mp':
        foo = self.eris.foo
        fvv = self.eris.fvv
        Hr2 += lib.einsum('bd,ijd->ijb',fvv,r2,symlib)
        Hr2 += -lib.einsum('ki,kjb->ijb',foo,r2,symlib)
        Hr2 += -lib.einsum('lj,ilb->ijb',foo,r2,symlib)
    elif eom.partition == 'full':
        Hr2 += self._ipccsd_diag_matrix2*r2
    else:
        Hr2 += lib.einsum('bd,ijd->ijb',imds.Lvv,r2,symlib)
        Hr2 += -lib.einsum('ki,kjb->ijb',imds.Loo,r2,symlib)
        Hr2 += -lib.einsum('lj,ilb->ijb',imds.Loo,r2,symlib)
        Hr2 +=  lib.einsum('klij,klb->ijb',imds.Woooo,r2,symlib)
        Hr2 += 2*lib.einsum('lbdj,ild->ijb',imds.Wovvo,r2,symlib)
        Hr2 +=  -lib.einsum('kbdj,kid->ijb',imds.Wovvo,r2,symlib)
        Hr2 +=  -lib.einsum('lbjd,ild->ijb',imds.Wovov,r2,symlib) #typo in Ref
        Hr2 +=  -lib.einsum('kbid,kjd->ijb',imds.Wovov,r2,symlib)
        tmp = 2*lib.einsum('lkdc,kld->c',imds.Woovv,r2,symlib)
        tmp += -lib.einsum('kldc,kld->c',imds.Woovv,r2,symlib)
        Hr2 += -lib.einsum('c,ijcb->ijb',tmp,imds.t2,symlib)

    vector = eom.amplitudes_to_vector(Hr1,Hr2)
    return vector

def eaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    lib, symlib = eom.lib, eom.symlib
    r1,r2 = eom.vector_to_amplitudes(vector, kshift)

    # Eq. (30)
    # 1p-1p block
    Hr1 =  lib.einsum('ac,c->a',imds.Lvv,r1,symlib)
    # 1p-2p1h block
    Hr1 += lib.einsum('ld,lad->a',2.*imds.Fov,r2,symlib)
    Hr1 += lib.einsum('ld,lda->a',  -imds.Fov,r2,symlib)
    Hr1 += 2*lib.einsum('alcd,lcd->a',imds.Wvovv,r2,symlib)
    Hr1 +=  -lib.einsum('aldc,lcd->a',imds.Wvovv,r2,symlib)
    # Eq. (31)
    # 2p1h-1p block
    Hr2 = lib.einsum('abcj,c->jab',imds.Wvvvo,r1,symlib)
    # 2p1h-2p1h block
    if eom.partition == 'mp':
        foo = imds.eris.foo
        fvv = imds.eris.fvv
        Hr2 +=  lib.einsum('ac,jcb->jab',fvv,r2,symlib)
        Hr2 +=  lib.einsum('bd,jad->jab',fvv,r2,symlib)
        Hr2 += -lib.einsum('lj,lab->jab',foo,r2,symlib)
    elif eom.partition == 'full':
        Hr2 += eom._eaccsd_diag_matrix2*r2
    else:
        Hr2 +=  lib.einsum('ac,jcb->jab',imds.Lvv,r2,symlib)
        Hr2 +=  lib.einsum('bd,jad->jab',imds.Lvv,r2,symlib)
        Hr2 += -lib.einsum('lj,lab->jab',imds.Loo,r2,symlib)
        Hr2 += 2*lib.einsum('lbdj,lad->jab',imds.Wovvo,r2,symlib)
        Hr2 +=  -lib.einsum('lbjd,lad->jab',imds.Wovov,r2,symlib)
        Hr2 +=  -lib.einsum('lajc,lcb->jab',imds.Wovov,r2,symlib)
        Hr2 +=  -lib.einsum('lbcj,lca->jab',imds.Wovvo,r2,symlib)

        Hr2 +=   lib.einsum('abcd,jcd->jab',imds.Wvvvv,r2,symlib)
        tmp = (2*lib.einsum('klcd,lcd->k',imds.Woovv,r2,symlib)
                -lib.einsum('kldc,lcd->k',imds.Woovv,r2,symlib))
        Hr2 += -lib.einsum('k,kjab->jab',tmp,imds.t2,symlib)

    vector = eom.amplitudes_to_vector(Hr1,Hr2)
    return vector

def eaccsd_diag(eom, kshift, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    backend, lib, symlib = eom.backend, eom.lib, eom.symlib
    kpts, gvec = eom.kpts, eom._cc._scf.cell.reciprocal_vectors()
    t1, t2 = imds.t1, imds.t2
    dtype = t2.dtype
    nocc, nvir = t1.shape
    nkpts = len(kpts)
    kconserv = eom.kconserv
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]

    Hr1array = imds.Lvv.diagonal()[kshift]
    Hr1 = lib.tensor(Hr1array, sym1)
    Hr2 = lib.zeros([nocc,nvir,nvir], dtype, sym2)
    tasks, ntasks = eom.gen_tasks(range(nkpts**2))
    idx_jab = np.arange(nocc*nvir*nvir)
    if eom.partition == 'mp':
        foo = backend.to_nparray(imds.eris.foo.diagonal())
        fvv = backend.to_nparray(imds.eris.fvv.diagonal())
        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2.write([], [])
                continue
            kjab = tasks[itask]
            kj, ka = (kjab).__divmod__(nkpts)
            kb = kconserv[ki,ka,kshift]
            off = kj * nkpts + ka
            jab = np.zeros([nocc,nvir,nvir], dtype=dtype)
            jab += -foo[kj][:,None,None]
            jab += fvv[ka][None,:,None]
            jab += fvv[kb][None,None,:]
            Hr2.write(off*idx_jab.size+idx_jab, jab.ravel())
    else:
        idx = np.arange(nvir)
        loo = backend.to_nparray(imds.Loo.diagonal())
        lvv = backend.to_nparray(imds.Lvv.diagonal())
        wab = backend.to_nparray(backend.einsum("ABAabab->ABab", imds.Wvvvv.array))
        wjb = backend.to_nparray(backend.einsum('JBJjbjb->JBjb', imds.Wovov.array))
        wjb2 = backend.to_nparray(backend.einsum('JBBjbbj->JBjb', imds.Wovvo.array))
        wja = backend.to_nparray(backend.einsum('JAJjaja->JAja', imds.Wovov.array))

        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2.write([], [])
                continue
            kjab = tasks[itask]
            kj, ka = (kjab).__divmod__(nkpts)
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
            off = kj * nkpts + ka
            Hr2.write(off*idx_jab.size+idx_jab, jab.ravel())
        Hr2 -= 2*backend.einsum('JAijab,JAijab->JAjab', t2[kshift], imds.Woovv[kshift])
        Woovvtmp = imds.Woovv.transpose(0,1,3,2)[kshift]
        Hr2 += backend.einsum('JAijab,JAijab->JAjab', t2[kshift], Woovvtmp)

    return eom.amplitudes_to_vector(Hr1, Hr2)

class EOMIP(eom_rccsd.EOMIP):
    def __init__(self, cc):
        eom_rccsd.EOMIP.__init__(self,cc)
        self.kpts = cc.kpts
        self.lib = cc.lib
        self.symlib = cc.symlib
        self.backend =  cc.backend
        self._backend = cc._backend
        self.log = cc.log
        self.t1, self.t2 = cc.t1, cc.t2
        self.nonzero_opadding, self.nonzero_vpadding = self.get_padding_k_idx(cc)
        self.kconserv = cc.khelper.kconserv

    matvec = ipccsd_matvec
    vector_to_amplitudes = vector_to_amplitudes_ip
    get_diag = ipccsd_diag
    kernel = kernel
    ipccsd = kernel

    def gen_tasks(self, jobs):
        if self._backend == 'numpy':
            return jobs, len(jobs)
        elif self._backend=='ctf':
            from symtensor.backend.ctf_funclib import static_partition, comm
            tasks = list(static_partition(jobs))
            ntasks = max(comm.allgather(len(tasks)))
            return tasks, ntasks
        else:
            raise NotImplementedError

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
        backend = self.backend
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        rank = getattr(backend, 'rank', 0)
        guess = backend.zeros([nroots, int(size)], dtype=dtype)
        if koopmans:
            ind = [kn*size+n for kn, n in enumerate(self.nonzero_opadding[kshift][::-1][:nroots])]
        else:
            if diag is None:
                diag = self.get_diag(kshift, imds=None)
            idx = backend.to_nparray(diag).argsort()[:nroots]
            ind = [kn*size+n for kn, n in enumerate(idx)]
        fill = np.ones(nroots)
        if rank==0:
            backend.write(guess,ind,fill)
        else:
            backend.write(guess,[],[])
        return guess

class EOMEA(eom_rccsd.EOMEA):
    def __init__(self, cc):
        eom_rccsd.EOMEA.__init__(self,cc)
        self.kpts = cc.kpts
        self.lib = cc.lib
        self.symlib = cc.symlib
        self.backend =  cc.backend
        self._backend = cc._backend
        self.log = cc.log
        self.t1, self.t2 = cc.t1, cc.t2
        self.nonzero_opadding, self.nonzero_vpadding = self.get_padding_k_idx(cc)
        self.kconserv = cc.khelper.kconserv

    matvec = eaccsd_matvec
    vector_to_amplitudes = vector_to_amplitudes_ea
    get_diag = eaccsd_diag
    eaccsd = kernel
    kernel = kernel

    def gen_tasks(self, jobs):
        if self._backend == 'numpy':
            return jobs, len(jobs)
        elif self._backend=='ctf':
            from symtensor.backend.ctf_funclib import static_partition, comm
            tasks = list(static_partition(jobs))
            ntasks = max(comm.allgather(len(tasks)))
            return tasks, ntasks
        else:
            raise NotImplementedError

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
        backend = self.backend
        nroots = min(nroots, size)
        guess = backend.zeros([nroots, int(size)], dtype=dtype)
        rank = getattr(backend, 'rank', 0)
        if koopmans:
            ind = [kn*size+n for kn, n in enumerate(self.nonzero_vpadding[kshift][:nroots])]
        else:
            idx = backend.to_nparray(diag).argsort()[:nroots]
            ind = [kn*size+n for kn, n in enumerate(idx)]
        fill = np.ones(nroots)
        if rank==0:
            backend.write(guess,ind,fill)
        else:
            backend.write(guess,[],[])
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
    import os
    from kccsd_rhf_ctf import KRCCSDCTF
    from kccsd_rhf import KRCCSD
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
    chkfile = 'dmd_113.chk'
    if os.path.isfile(chkfile):
        mf.__dict__.update(scf.chkfile.load(chkfile, 'scf'))
    else:
        mf.chkfile = chkfile
        mf.kernel()

    mycc = KRCCSDCTF(mf)
    #mycc = KRCCSD(mf)
    mycc.kernel()

    myeom = EOMIP(mycc)
    myeom.ipccsd(nroots=2, kptlist=[1], koopmans=True)

    myeom = EOMEA(mycc)
    myeom.eaccsd(nroots=2, kptlist=[1], koopmans=True)

    refcc = cc.KRCCSD(mf)
    refcc.kernel()

    refeom = cc.eom_kccsd_rhf.EOMIP(refcc)
    refeom.ipccsd(nroots=2, kptlist=[1], koopmans=True)

    refeom = cc.eom_kccsd_rhf.EOMEA(refcc)
    refeom.eaccsd(nroots=2, kptlist=[1], koopmans=False)
