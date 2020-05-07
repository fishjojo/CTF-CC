#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

from cc_sym import eom_kccsd_rhf_numpy
import numpy as np
import time
from pyscf.pbc.mp.kmp2 import padding_k_idx
from pyscf.lib import logger
import ctf
import symtensor.sym_ctf as sym
from cc_sym import mpi_helper

comm = mpi_helper.comm
rank = mpi_helper.rank

tensor = sym.tensor
zeros = sym.zeros

def kernel(eom, nroots=1, koopmans=True, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):

    cput0 = (time.clock(), time.time())
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
    from cc_sym.linalg_helper.davidson import eigs
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

        for n, en, vn in zip(range(nroots), evals_k, evecs_k):
            r1, r2 = eom.vector_to_amplitudes(vn, kshift)
            qp_weight = r1.norm()**2
            logger.info(eom, 'EOM-CCSD root %d E = %.16g  qpwt = %0.6g', n, en, qp_weight)
    logger.timer(eom, 'EOM-CCSD', *cput0)
    evecs = ctf.vstack(tuple(evecs))
    return convs, evals, evecs



def vector_to_amplitudes_ip(eom, vector, kshift):
    nkpts, kpts, nocc = eom.nkpts, eom.kpts, eom.nocc
    nvir = eom.nmo - nocc
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]
    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nkpts,nkpts,nocc,nocc,nvir)
    r1 = tensor(r1, sym1, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r2 = tensor(r2, sym2, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    return [r1,r2]

def vector_to_amplitudes_ea(eom, vector, kshift):
    nkpts, kpts, nocc = eom.nkpts, eom.kpts, eom.nocc
    nvir = eom.nmo - nocc
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]
    r1 = vector[:nvir].copy()
    r2 = vector[nvir:].copy().reshape(nkpts,nkpts,nocc,nvir,nvir)
    r1 = tensor(r1,sym1, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r2 = tensor(r2,sym2, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    return [r1,r2]

def amplitudes_to_vector(eom, r1, r2):
    vector = ctf.hstack((r1.array.ravel(), r2.array.ravel()))
    return vector

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
    Hr1 = tensor(Hr1array, sym1, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    Hr2 = zeros([nocc,nocc,nvir], dtype, sym2, symlib=eom.symlib, verbose=eom.SYMVERBOSE)

    tasks = mpi_helper.static_partition(range(nkpts**2))
    ntasks = max(comm.allgather(len(tasks)))

    idx_ijb = np.arange(nocc*nocc*nvir)
    if eom.partition == 'mp':
        foo = eom.eris.foo.diagonal().to_nparray()
        fvv = eom.eris.fvv.diagonal().to_nparray()
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
        lvv = imds.Lvv.diagonal().to_nparray()
        loo = imds.Loo.diagonal().to_nparray()
        wij = ctf.einsum('IJIijij->IJij', imds.Woooo.array).to_nparray()
        wjb = ctf.einsum('JBJjbjb->JBjb', imds.Wovov.array).to_nparray()
        wjb2 = ctf.einsum('JBBjbbj->JBjb', imds.Wovvo.array).to_nparray()
        wib = ctf.einsum('IBIibib->IBib', imds.Wovov.array).to_nparray()
        idx = np.arange(nocc)
        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2.write([], [])
                continue
            kijb = tasks[itask]
            ki, kj = (kijb).__divmod__(nkpts)
            kb = kconserv[ki,kshift,kj]
            ijb = np.zeros([nocc,nocc,nvir], dtype=dtype)
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
        Hr2 -= 2.*ctf.einsum('IJijcb,JIjicb->IJijb', t2[:,:,kshift], Woovvtmp)
        Hr2 += ctf.einsum('IJijcb,IJijcb->IJijb', t2[:,:,kshift], Woovvtmp)
    return eom.amplitudes_to_vector(Hr1, Hr2)

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

    tasks = mpi_helper.static_partition(range(nkpts**2))
    ntasks = max(comm.allgather(len(tasks)))
    idx_jab = np.arange(nocc*nvir*nvir)
    if eom.partition == 'mp':
        foo = imds.eris.foo.diagonal().to_nparray()
        fvv = imds.eris.fvv.diagonal().to_nparray()
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
        loo = imds.Loo.diagonal().to_nparray()
        lvv = imds.Lvv.diagonal().to_nparray()
        wab = ctf.einsum("ABAabab->ABab", imds.Wvvvv.array).to_nparray()
        wjb = ctf.einsum('JBJjbjb->JBjb', imds.Wovov.array).to_nparray()
        wjb2 = ctf.einsum('JBBjbbj->JBjb', imds.Wovvo.array).to_nparray()
        wja = ctf.einsum('JAJjaja->JAja', imds.Wovov.array).to_nparray()

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
        Hr2 -= 2*ctf.einsum('JAijab,JAijab->JAjab', t2[kshift], imds.Woovv[kshift])
        Woovvtmp = imds.Woovv.transpose(0,1,3,2)[kshift]
        Hr2 += ctf.einsum('JAijab,JAijab->JAjab', t2[kshift], Woovvtmp)

    return eom.amplitudes_to_vector(Hr1, Hr2)

class EOMIP(eom_kccsd_rhf_numpy.EOMIP):

    vector_to_amplitudes = vector_to_amplitudes_ip
    amplitudes_to_vector = amplitudes_to_vector
    get_diag = ipccsd_diag
    kernel = kernel
    ipccsd = kernel


    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        guess = ctf.zeros([nroots, int(size)], dtype=dtype)
        if koopmans:
            ind = [kn*size+n for kn, n in enumerate(self.nonzero_opadding[kshift][::-1][:nroots])]
        else:
            if diag is None:
                diag = self.get_diag(kshift, imds=None)
            idx = diag.to_nparray().argsort()[:nroots]
            ind = [kn*size+n for kn, n in enumerate(idx)]
        fill = np.ones(nroots)
        if rank==0:
            guess.write(ind, fill)
        else:
            guess.write([], [])
        return guess

class EOMEA(eom_kccsd_rhf_numpy.EOMEA):

    vector_to_amplitudes = vector_to_amplitudes_ea
    amplitudes_to_vector = amplitudes_to_vector
    get_diag = eaccsd_diag
    eaccsd = kernel
    kernel = kernel

    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        guess = ctf.zeros([nroots, int(size)], dtype=dtype)
        if koopmans:
            ind = [kn*size+n for kn, n in enumerate(self.nonzero_vpadding[kshift][:nroots])]
        else:
            idx = diag.to_nparray().argsort()[:nroots]
            ind = [kn*size+n for kn, n in enumerate(idx)]
        fill = np.ones(nroots)
        if rank==0:
            guess.write(ind, fill)
        else:
            guess.write([],[])
        return guess


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    from cc_sym.kccsd_rhf import KRCCSD
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
    if rank ==0:
        mf.kernel()

    mf.mo_coeff = comm.bcast(mf.mo_coeff, root=0)
    mf.mo_occ = comm.bcast(mf.mo_occ, root=0)
    mycc = KRCCSD(mf)
    mycc.kernel()

    myeom = EOMIP(mycc)
    _, eip, _ = myeom.ipccsd(nroots=2, kptlist=[1], koopmans=True)
    myeom = EOMEA(mycc)
    _, eea, _ = myeom.eaccsd(nroots=2, kptlist=[1], koopmans=True)

    print(eip[0,1]- -0.5392478826367997)
    print(eea[0,0]- 1.147581866049977)
