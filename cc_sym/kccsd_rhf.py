#!/usr/bin/env python


'''
Restricted CCSD with ctf
'''

from functools import reduce
import time
import numpy as np
from pyscf import lib as pyscflib
import rccsd
from symtensor import sym
from symtensor.symlib import SYMLIB
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)
from pyscf.pbc.cc import kccsd_rhf
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM
from pyscf.pbc.lib import kpts_helper

def energy(cc, t1, t2, eris):
    lib, symlib, log = eris.lib, eris.symlib, cc.log
    nkpts = len(cc.kpts)
    e = 2*lib.einsum('ia,ia', eris.fov, t1, symlib)
    tau = lib.einsum('ia,jb->ijab',t1,t1, symlib)
    tau += t2
    e += 2*lib.einsum('ijab,iajb', tau, eris.ovov, symlib)
    e +=  -lib.einsum('ijab,ibja', tau, eris.ovov, symlib)
    if abs(e.imag)>1e-4:
        log.warn('Non-zero imaginary part found in KRCCSD energy %s', e)
    return e.real / nkpts

class KRCCSD(rccsd.RCCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        rccsd.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.kpts = mf.kpts
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])
        self.lib = sym
        self.backend = sym.backend
        self.log = self.backend.Logger(self.stdout, self.verbose)
        self.symlib = SYMLIB('numpy')
        self.make_symlib()

    @property
    def nkpts(self):
        return len(self.kpts)

    def make_symlib(self):
        kpts = self.kpts
        gvec = self._scf.cell.reciprocal_vectors()
        sym1 = ['+-',[kpts,]*2, None, gvec]
        sym2 = ['++--',[kpts,]*4, None, gvec]
        self.symlib.update(sym1, sym2)

    energy = energy
    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        log, lib, symlib = self.log, self.lib, self.symlib
        nocc = self.nocc
        nvir = self.nmo - nocc
        kpts = self.kpts
        nkpts = self.nkpts
        gvec = self._scf.cell.reciprocal_vectors()
        sym1 = ['+-', [kpts,]*2, None, gvec]
        t1 = lib.zeros([nocc,nvir], eris.dtype, sym1)
        t2 = eris.ovov.transpose(0,2,1,3).conj() / eris.eijab
        self.emp2  = 2*lib.einsum('ijab,iajb', t2, eris.ovov, symlib)
        self.emp2 -=   lib.einsum('ijab,ibja', t2, eris.ovov, symlib)
        self.emp2 = self.emp2.real/nkpts
        log.info('Init t2, MP2 energy (with fock eigenvalue shift) = %.15g', self.emp2)
        log.timer('init mp2', *time0)
        return self.emp2, t1, t2

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)


class _ChemistsERIs:
    def __init__(self, cc, mo_coeff=None,):
        from pyscf.pbc.cc.ccsd import _adjust_occ
        import pyscf.pbc.tools.pbc as tools
        if mo_coeff is None:
            mo_coeff = cc.mo_coeff
        cput0 = (time.clock(), time.time())
        log, lib, backend = cc.log, cc.lib, cc.backend
        self.lib = lib
        self.symlib = cc.symlib
        tensor, zeros = lib.tensor, lib.zeros
        nocc, nmo, nkpts = cc.nocc, cc.nmo, cc.nkpts
        nvir = nmo - nocc
        cell, kpts = cc._scf.cell, cc.kpts
        gvec = cell.reciprocal_vectors()
        sym1 = ['+-', [kpts,]*2, None, gvec]
        sym2 = ['+-+-', [kpts,]*4, None, gvec]
        self.dtype = dtype = np.result_type(*cc.mo_coeff).char
        rank = getattr(backend, 'rank', 0)
        mo_coeff = self.mo_coeff = padded_mo_coeff(cc, mo_coeff)
        nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")
        madelung = tools.madelung(cell, kpts)
        self.foo = zeros([nocc,nocc], dtype, sym1)
        self.fov = zeros([nocc,nvir], dtype, sym1)
        self.fvv = zeros([nvir,nvir], dtype, sym1)
        self.eia = zeros([nocc,nvir], 'float', sym1)
        self._foo = zeros([nocc,nocc], dtype, sym1)
        self._fvv = zeros([nvir,nvir], dtype, sym1)


        if rank ==0:
            dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)

            with pyscflib.temporary_env(cc._scf, exxdiv=None):
                fockao = cc._scf.get_hcore() + cc._scf.get_veff(cell, dm)
            fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                                for k, mo in enumerate(mo_coeff)])
            foo = fock[:,:nocc,:nocc]
            fov = fock[:,:nocc,nocc:]
            fvv = fock[:,nocc:,nocc:]

            mo_energy = [fock[k].diagonal().real for k in range(nkpts)]
            mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                              for k, mo_e in enumerate(mo_energy)]

            mo_e_o = [e[:nocc] for e in mo_energy]
            mo_e_v = [e[nocc:] + cc.level_shift for e in mo_energy]
            foo_ = np.asarray([np.diag(e) for e in mo_e_o])
            fvv_ = np.asarray([np.diag(e) for e in mo_e_v])
            eia = np.zeros([nkpts,nocc,nvir])
            for ki in range(nkpts):
                eia[ki] = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                            [0,nvir,ki,mo_e_v,nonzero_vpadding],
                            fac=[1.0,-1.0])
            self.foo.write(range(foo.size), foo.ravel())
            self.fov.write(range(fov.size), fov.ravel())
            self.fvv.write(range(fvv.size), fvv.ravel())
            self.eia.write(range(eia.size), eia.ravel())
            self._foo.write(range(foo_.size), foo_.ravel())
            self._fvv.write(range(fvv_.size), fvv_.ravel())
        else:
            self.foo.write([],[])
            self.fov.write([],[])
            self.fvv.write([],[])
            self.eia.write([],[])
            self._foo.write([],[])
            self._fvv.write([],[])

        self.oooo = zeros([nocc,nocc,nocc,nocc], dtype, sym2)
        self.ooov = zeros([nocc,nocc,nocc,nvir], dtype, sym2)
        self.ovov = zeros([nocc,nvir,nocc,nvir], dtype, sym2)
        self.oovv = zeros([nocc,nocc,nvir,nvir], dtype, sym2)
        self.ovvo = zeros([nocc,nvir,nvir,nocc], dtype, sym2)
        self.ovvv = zeros([nocc,nvir,nvir,nvir], dtype, sym2)
        self.vvvv = zeros([nvir,nvir,nvir,nvir], dtype, sym2)
        self.eijab = zeros([nocc,nocc,nvir,nvir], 'float', sym2)

        with_df = cc._scf.with_df
        fao2mo = cc._scf.with_df.ao2mo
        kconserv = cc.khelper.kconserv

        idx_oooo = np.arange(nocc*nocc*nocc*nocc)
        idx_ooov = np.arange(nocc*nocc*nocc*nvir)
        idx_ovov = np.arange(nocc*nvir*nocc*nvir)
        idx_oovv = np.arange(nocc*nocc*nvir*nvir)
        idx_ovvo = np.arange(nocc*nvir*nvir*nocc)
        idx_ovvv = np.arange(nocc*nvir*nvir*nvir)
        idx_vvvv = np.arange(nvir*nvir*nvir*nvir)

        tasks, ntasks = self.gen_eri_kpt_tasks(nkpts)
        for itask in range(ntasks):
            if itask >= len(tasks):
                self.oooo.write([], [])
                self.ooov.write([], [])
                self.ovov.write([], [])
                self.oovv.write([], [])
                self.ovvo.write([], [])
                self.ovvv.write([], [])
                self.vvvv.write([], [])
                self.eijab.write([], [])
                continue

            pqr = tasks[itask]
            kp = pqr // nkpts**2
            kq, kr = (pqr % nkpts**2).__divmod__(nkpts)
            ks = kconserv[kp,kq,kr]
            eia = _get_epq([0,nocc,kp,mo_e_o,nonzero_opadding],
                           [0,nvir,kq,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])
            ejb = _get_epq([0,nocc,kr,mo_e_o,nonzero_opadding],
                           [0,nvir,ks,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])
            eijab = eia[:,None,:,None] + ejb[None,:,None,:]

            eri_kpt = fao2mo((mo_coeff[kp],mo_coeff[kq],mo_coeff[kr],mo_coeff[ks]),
                             (cc.kpts[kp],cc.kpts[kq],cc.kpts[kr],cc.kpts[ks]), compact=False)

            eri_kpt = eri_kpt.reshape(nmo,nmo,nmo,nmo) / nkpts

            oooo = eri_kpt[:nocc,:nocc,:nocc,:nocc].ravel()
            ooov = eri_kpt[:nocc,:nocc,:nocc,nocc:].ravel()
            ovov = eri_kpt[:nocc,nocc:,:nocc,nocc:].ravel()
            oovv = eri_kpt[:nocc,:nocc,nocc:,nocc:].ravel()
            ovvo = eri_kpt[:nocc,nocc:,nocc:,:nocc].ravel()
            ovvv = eri_kpt[:nocc,nocc:,nocc:,nocc:].ravel()
            vvvv = eri_kpt[nocc:,nocc:,nocc:,nocc:].ravel()

            off = kp * nkpts**2 + kq * nkpts + kr
            self.oooo.write(off*idx_oooo.size+idx_oooo, oooo)
            self.ooov.write(off*idx_ooov.size+idx_ooov, ooov)
            self.ovov.write(off*idx_ovov.size+idx_ovov, ovov)
            self.oovv.write(off*idx_oovv.size+idx_oovv, oovv)
            self.ovvo.write(off*idx_ovvo.size+idx_ovvo, ovvo)
            self.ovvv.write(off*idx_ovvv.size+idx_ovvv, ovvv)
            self.vvvv.write(off*idx_vvvv.size+idx_vvvv, vvvv)
            off = kp * nkpts**2 + kr * nkpts + kq
            self.eijab.write(off*idx_oovv.size+idx_oovv, eijab.ravel())
            log.debug1('_ERIS pqr %d', pqr)

    def gen_eri_kpt_tasks(self, nkpts):
        tasks = range(nkpts**3)
        ntasks = nkpts**3
        return tasks, ntasks

def _get_epq(pindices,qindices,fac=[1.0,1.0],large_num=LARGE_DENOM):
    '''Create a denominator

        fac[0]*e[kp,p0:p1] + fac[1]*e[kq,q0:q1]

    where padded elements have been replaced by a large number.

    Args:
        pindices (5-list of object):
            A list of p0, p1, kp, orbital values, and non-zero indicess for the first
            denominator indices.
        qindices (5-list of object):
            A list of q0, q1, kq, orbital values, and non-zero indicess for the second
            denominator element.
        fac (3-list of float):
            Factors to multiply the first and second denominator elements.
        large_num (float):
            Number to replace the padded elements.
    '''
    def get_idx(x0,x1,kx,n0_p):
        return np.logical_and(n0_p[kx] >= x0, n0_p[kx] < x1)

    assert(all([len(x) == 5 for x in [pindices,qindices]]))
    p0,p1,kp,mo_e_p,nonzero_p = pindices
    q0,q1,kq,mo_e_q,nonzero_q = qindices
    fac_p, fac_q = fac

    epq = large_num * np.ones((p1-p0,q1-q0), dtype=mo_e_p[0].dtype)
    idxp = get_idx(p0,p1,kp,nonzero_p)
    idxq = get_idx(q0,q1,kq,nonzero_q)
    n0_ovp_pq = np.ix_(nonzero_p[kp][idxp], nonzero_q[kq][idxq])
    epq[n0_ovp_pq] = pyscflib.direct_sum('p,q->pq', fac_p*mo_e_p[kp][p0:p1],
                                               fac_q*mo_e_q[kq][q0:q1])[n0_ovp_pq]
    return epq



if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    import os
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

    mycc = KRCCSD(mf)
    mycc.max_cycle=100
    eris = mycc.ao2mo()
    _, t1, t2 = mycc.init_amps(eris)
    t10, t20 = mycc.update_amps(t1, t2, eris)

    refcc = cc.KRCCSD(mf)

    erisref = refcc.ao2mo()
    _, t1r, t2r= refcc.init_amps(erisref)
    t1r0, t2r0= refcc.update_amps(t1r, t2r, erisref)

    print((eris.oooo.transpose(0,2,1,3)-erisref.oooo).norm())
    print((eris.ooov.transpose(0,2,1,3)-erisref.ooov).norm())
    print((eris.ovov.transpose(0,2,1,3)-erisref.oovv).norm())
    print((eris.oovv.transpose(0,2,1,3)-erisref.ovov).norm())
    print((eris.ovvo.transpose(2,0,3,1)-erisref.voov).norm())
    print((eris.ovvv.transpose(2,0,3,1)-erisref.vovv).norm())
    print((eris.vvvv.transpose(0,2,1,3)-erisref.vvvv).norm())

    #print((eris.eijab-eijab).norm())
    print((t2-t2r).norm())
    print("t1", (t10-t1r0).norm())
    print("t2", (t20-t2r0).norm())

    #mycc.kernel()
    ecc, t1r, t2r = refcc.kernel()
    t1.array = t1r
    t2.array = t2r
    mycc.kernel(t1, t2)
