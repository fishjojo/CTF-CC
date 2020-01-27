from pyscf.pbc import df
import numpy as np
from symtensor.backend.ctf_funclib import static_partition, rank, comm, size
import ctf
from symtensor import sym_ctf, sym
from pyscf.pbc.lib.kpts_helper import (is_zero, gamma_point, member, unique,
                                       KPT_DIFF_TOL)
import time
from pyscf.pbc.df.df_jk import zdotCN, zdotNN, zdotNC
from pyscf import lib as pyscflib
from pyscf.pbc.df.df import LINEAR_DEP_THR
from pyscf.pbc.df import ft_ao
import scipy
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv

def get_member(kpti, kptj, kptij_lst):
    kptij = np.vstack((kpti,kptj))
    ijid = member(kptij, kptij_lst)
    dagger = False
    if len(ijid) == 0:
        kptji = np.vstack((kptj,kpti))
        ijid = member(kptji, kptij_lst)
        dagger = True
    return ijid, dagger

def ao2mo(mydf, mo_coeffs, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_general_compact', True)):
    if mydf.j3c is None: mydf.build()
    log = mydf.log
    cell = mydf.cell
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    if isinstance(mo_coeffs, np.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    if not _iskconserv(cell, kptijkl):
        log.warn('df_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return np.zeros([mo.shape[1] for mo in mo_coeffs])

    ijid, ijdagger = get_member(kpti, kptj, mydf.kptij_lst)
    klid, kldagger = get_member(kptk, kptl, mydf.kptij_lst)
    aux_idx = np.arange(mydf.j3c.size).reshape(mydf.j3c.shape)
    nao, naux = mydf.j3c.shape[1], mydf.j3c.shape[-1]
    ijid = aux_idx[ijid].ravel()
    klid = aux_idx[klid].ravel()

    ijL = mydf.j3c.read(ijid).reshape(nao,nao,naux)
    if ijdagger:
        ijL = ijL.transpose(1,0,2).conj()

    klL = mydf.j3c.read(klid).reshape(nao,nao,naux)
    if kldagger:
        klL = klL.transpose(1,0,2).conj()

    pvL = np.dot(mo_coeffs[0].conj().T, ijL.transpose(1,0,2))
    pqL = np.dot(mo_coeffs[1].T, pvL).transpose(1,0,2)
    pvL = ijL = None
    rvL = np.dot(mo_coeffs[2].conj().T, klL.transpose(1,0,2))
    rLs = np.dot(mo_coeffs[3].T, rvL).transpose(1,2,0)
    rvL = klL = None
    eri = np.dot(pqL,rLs)

    return eri

def get_eri(mydf, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_general_compact', True)):
    if mydf.j3c is None: mydf.build()
    log = mydf.log
    cell = mydf.cell
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    if not _iskconserv(cell, kptijkl):
        log.warn('df_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return np.zeros([mo.shape[1] for mo in mo_coeffs])

    ijid, ijdagger = get_member(kpti, kptj, mydf.kptij_lst)
    klid, kldagger = get_member(kptk, kptl, mydf.kptij_lst)
    aux_idx = np.arange(mydf.j3c.size).reshape(mydf.j3c.shape)
    nao, naux = mydf.j3c.shape[1], mydf.j3c.shape[-1]
    ijid = aux_idx[ijid].ravel()
    klid = aux_idx[klid].ravel()

    ijL = mydf.j3c.read(ijid).reshape(nao,nao,naux)
    if ijdagger:
        ijL = ijL.transpose(1,0,2).conj()

    klL = mydf.j3c.read(klid).reshape(nao,nao,naux)
    if kldagger:
        klL = klL.transpose(1,0,2).conj()

    eri = np.dot(ijL,klL.transpose(0,2,1))
    return eri

def dump_to_file(mydf, cderi_file):
    import h5py
    if rank==0:
        feri = h5py.File(cderi_file,'w')
        feri['j3c-kptij'] = mydf.kptij_lst
    else:
        feri = None
    nao, naux = mydf.j3c.shape[-2:]
    idx_j3c = np.arange(mydf.j3c.size).reshape(mydf.j3c.shape)
    for i in range(len(mydf.kptij_lst)):
        idx = idx_j3c[i].ravel()
        if rank!=0:
            tmp = mydf.j3c.read([])
        else:
            kpti_kptj = mydf.kptij_lst[i]
            is_real = is_zero(kpti_kptj[0]-kpti_kptj[1])
            tmp = mydf.j3c.read(idx).reshape(nao,nao,naux)
            if is_real:
                tmp = pyscflib.pack_tril(tmp, axis=0).T
                if is_zero(kpti_kptj[0]):
                    tmp = tmp.real
            else:
                tmp = tmp.reshape(-1,naux).T
            feri['j3c/%d/0'%i] = tmp

    if rank==0:
        feri.close()
    mydf._cderi = cderi_file
    return mydf



def _make_j3c(mydf, cell, auxcell, kptij_lst):
    t1 = (time.clock(), time.time())
    max_memory = max(2000, mydf.max_memory-pyscflib.current_memory()[0])
    fused_cell, fuse = df.df.fuse_auxcell(mydf, auxcell)
    log, backend = mydf.log, mydf.backend
    nao, nfao = cell.nao_nr(), fused_cell.nao_nr()
    jobs = np.arange(fused_cell.nbas)
    tasks = list(static_partition(jobs))
    ntasks = max(comm.allgather(len(tasks)))
    j3c_junk = backend.zeros([len(kptij_lst), nao**2, nfao], dtype=np.complex128)
    t1 = (time.clock(), time.time())

    for itask in range(ntasks):
        if itask >= len(tasks):
            backend.write(j3c_junk, [], [])
            continue
        shls_slice = (0, cell.nbas, 0, cell.nbas, tasks[itask], tasks[itask]+1)
        bstart, bend = fused_cell.ao_loc_nr()[tasks[itask]], fused_cell.ao_loc_nr()[tasks[itask]+1]
        idx = np.arange(len(kptij_lst)*nao**2) * nfao
        idx = idx[:,None] + np.arange(bstart,bend)[None,:]
        idx = idx.ravel()
        tmp = df.incore.aux_e2(cell, fused_cell, intor='int3c2e', aosym='s2', kptij_lst=kptij_lst, shls_slice=shls_slice)
        
        print("rank=%i, j3c_junk size=%i"%(rank, idx.size))
        nao_pair = nao**2
        if tmp.shape[-2] != nao_pair and tmp.ndim == 2:
            tmp = pyscflib.unpack_tril(tmp, axis=0).reshape(nao_pair,-1)
        backend.write(j3c_junk, idx, tmp.ravel())
    t1 = log.timer('j3c_junk', *t1)

    naux = auxcell.nao_nr()
    mesh = mydf.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    b = cell.reciprocal_vectors()
    gxyz = pyscflib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    mydf.kptij_lst = kptij_lst
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)

    jobs = np.arange(len(uniq_kpts))
    tasks = list(static_partition(jobs))
    ntasks = max(comm.allgather(len(tasks)))

    blksize = max(2048, int(max_memory*.5e6/16/fused_cell.nao_nr()))
    log.debug2('max_memory %s (MB)  blocksize %s', max_memory, blksize)
    j2c  = backend.zeros([len(uniq_kpts),naux,naux], dtype=np.complex128)

    a = cell.lattice_vectors() / (2*np.pi)
    def kconserve_indices(kpt):
        '''search which (kpts+kpt) satisfies momentum conservation'''
        kdif = np.einsum('wx,ix->wi', a, uniq_kpts + kpt)
        kdif_int = np.rint(kdif)
        mask = np.einsum('wi->i', abs(kdif - kdif_int)) < KPT_DIFF_TOL
        uniq_kptji_ids = np.where(mask)[0]
        return uniq_kptji_ids

    def cholesky_decomposed_metric(j2c_kptij):
        j2c_negative = None
        try:
            j2c_kptij = scipy.linalg.cholesky(j2c_kptij, lower=True)
            j2ctag = 'CD'
        except scipy.linalg.LinAlgError as e:
            w, v = scipy.linalg.eigh(j2c_kptij)
            log.debug('cond = %.4g, drop %d bfns',
                      w[-1]/w[0], np.count_nonzero(w<mydf.linear_dep_threshold))
            v1 = np.zeros(v.T.shape, dtype=v.dtype)
            v1[w>mydf.linear_dep_threshold,:] = v[:,w>mydf.linear_dep_threshold].conj().T
            v1[w>mydf.linear_dep_threshold,:] /= np.sqrt(w[w>mydf.linear_dep_threshold]).reshape(-1,1)
            j2c_kptij = v1
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                idx = np.where(w < -mydf.linear_dep_threshold)[0]
                if len(idx) > 0:
                    j2c_negative = np.zeros(v1.shape, dtype=v1.dtype)
                    j2c_negative[idx,:] = (v[:,idx]/np.sqrt(-w[idx])).conj().T

            w = v = None
            j2ctag = 'eig'
        return j2c_kptij, j2c_negative, j2ctag

    for itask in range(ntasks):
        if itask >= len(tasks):
            j2c.write([],[])
            continue
        k = tasks[itask]
        kpt = uniq_kpts[k]
        j2ctmp = np.asarray(fused_cell.pbc_intor('int2c2e', hermi=1, kpts=kpt))
        coulG = mydf.weighted_coulG(kpt, False, mesh)
        for p0, p1 in pyscflib.prange(0, ngrids, blksize):
            aoaux = ft_ao.ft_ao(fused_cell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
            if is_zero(kpt):
                j2ctmp[naux:] -= np.dot(aoaux[naux:].conj()*coulG[p0:p1].conj(), aoaux.T).real
                j2ctmp[:naux,naux:] = j2ctmp[naux:,:naux].T
            else:
                j2ctmp[naux:] -= np.dot(aoaux[naux:].conj()*coulG[p0:p1].conj(), aoaux.T)
                j2ctmp[:naux,naux:] = j2ctmp[naux:,:naux].T.conj()

        tmp = fuse(fuse(j2ctmp).T).T
        idx = k * naux**2 + np.arange(naux**2)
        j2c.write(idx, tmp.ravel())
        j2ctmp = tmp = None

    coulG = None
    t1 = log.timer('j2c', *t1)

    j3c = ctf.zeros([len(kpt_ji),nao,nao,naux], dtype=np.complex128)
    jobs = np.arange(len(kpt_ji))
    tasks = list(static_partition(jobs))
    ntasks = max(comm.allgather(len(tasks)))

    for itask in range(ntasks):
        if itask >= len(tasks):
            j2c_ji = j2c.read([])
            j3ctmp = j3c_junk.read([])
            j3c.write([],[])
            continue
        idx_ji = tasks[itask]
        kpti, kptj = kptij_lst[idx_ji]
        idxi, idxj = member(kpti, mydf.kpts), member(kptj, mydf.kpts)
        uniq_idx = uniq_inverse[idx_ji]
        kpt = uniq_kpts[uniq_idx]
        id_eq = kconserve_indices(-kpt)
        id_conj = kconserve_indices(kpt)
        id_conj = np.asarray([i for i in id_conj if i not in id_eq], dtype=int)
        id_full = np.hstack((id_eq, id_conj))
        map_id, conj = min(id_full), np.argmin(id_full) >=len(id_eq)
        j2cidx = map_id * naux**2 + np.arange(naux**2)
        j2c_ji = j2c.read(j2cidx).reshape(naux, naux) # read to be added
        j2c_ji, j2c_negative, j2ctag = cholesky_decomposed_metric(j2c_ji)
        if conj: j2c_ji = j2c_ji.conj()
        shls_slice= (auxcell.nbas, fused_cell.nbas)
        Gaux = ft_ao.ft_ao(fused_cell, Gv, shls_slice, b, gxyz, Gvbase, kpt)
        wcoulG = mydf.weighted_coulG(kpt, False, mesh)
        Gaux *= wcoulG.reshape(-1,1)
        j3c_id = idx_ji * nao**2*nfao + np.arange(nao**2*nfao)
        j3ctmp = j3c_junk.read(j3c_id).reshape(nao**2, fused_cell.nao_nr()).T
        if is_zero(kpt):  # kpti == kptj
            if cell.dimension == 3:
                vbar = fuse(mydf.auxbar(fused_cell))
                ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kptj)
                for i in np.where(vbar != 0)[0]:
                    j3ctmp[i] -= vbar[i] * ovlp.reshape(-1)

        aoao = ft_ao._ft_aopair_kpts(cell, Gv, None, 's1', b, gxyz, Gvbase, kpt, kptj)[0].reshape(len(Gv),-1)
        j3ctmp[naux:] -= np.dot(Gaux.T.conj(), aoao)
        j3ctmp = fuse(j3ctmp)
        if j2ctag == 'CD':
            v = scipy.linalg.solve_triangular(j2c_ji, j3ctmp, lower=True, overwrite_b=True)
        else:
            v = np.dot(j2c_ji, j3ctmp)
        v = v.T.reshape(nao,nao,naux)
        j3c_id = idx_ji * nao**2*naux + np.arange(nao**2*naux)
        j3c.write(j3c_id, v.ravel())

    mydf.j3c = j3c
    return None


class MPIGDF(df.GDF):
    def __init__(self, cell, kpts=np.zeros((1,3)), backend='numpy'):
        if backend=='numpy':
            self.lib = sym
        elif backend =='ctf':
            self.lib = sym_ctf
        else:
            raise NotImplementedError
        self._backend = backend
        self.backend =  self.lib.backend
        self.verbose = 5
        self.j3c = None
        self.log = self.backend.Logger(self.stdout, self.verbose)
        self.kptij_lst = None
        df.GDF.__init__(self, cell, kpts)
        #del self._cderi_to_save

    def dump_flags(self, verbose=None):
        log = self.log
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        if self.auxcell is None:
            log.info('auxbasis = %s', self.auxbasis)
        else:
            log.info('auxbasis = %s', self.auxcell.basis)
        log.info('eta = %s', self.eta)
        log.info('exp_to_discard = %s', self.exp_to_discard)
        log.info('len(kpts) = %d', len(self.kpts))
        log.debug1('    kpts = %s', self.kpts)
        if self.kpts_band is not None:
            log.info('len(kpts_band) = %d', len(self.kpts_band))
            log.debug1('    kpts_band = %s', self.kpts_band)
        return self

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        log = self.log
        if self.kpts_band is not None:
            self.kpts_band = np.reshape(self.kpts_band, (-1,3))
        if kpts_band is not None:
            kpts_band = np.reshape(kpts_band, (-1,3))
            if self.kpts_band is None:
                self.kpts_band = kpts_band
            else:
                self.kpts_band = unique(np.vstack((self.kpts_band,kpts_band)))[0]

        self.check_sanity()
        self.dump_flags()

        self.auxcell = df.df.make_modrho_basis(self.cell, self.auxbasis,
                                         self.exp_to_discard)

        if self.kpts_band is None:
            kpts = self.kpts
            kband_uniq = np.zeros((0,3))
        else:
            kpts = self.kpts
            kband_uniq = [k for k in self.kpts_band if len(member(k, kpts))==0]
        if j_only is None:
            j_only = self._j_only
        if j_only:
            kall = np.vstack([kpts,kband_uniq])
            kptij_lst = np.hstack((kall,kall)).reshape(-1,2,3)
        else:
            kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i+1)]
            kptij_lst.extend([(ki, kj) for ki in kband_uniq for kj in kpts])
            kptij_lst.extend([(ki, ki) for ki in kband_uniq])
            kptij_lst = np.asarray(kptij_lst)

        t1 = (time.clock(), time.time())
        self._make_j3c(self.cell, self.auxcell, kptij_lst)
        t1 = log.timer('j3c', *t1)
        return self

    dump_to_file = dump_to_file
    _make_j3c = _make_j3c
    get_eri = get_eri
    ao2mo = ao2mo

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc, df
    import os
    cell = gto.Cell()
    cell.atom='''
    H 0.000000000000   0.000000000000   0.000000000000
    H 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-tzvp'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()

    kpts = cell.make_kpts([1,1,3])
    mydf = MPIGDF(cell, kpts, backend='ctf')
    mydf.mesh = [5,5,5]
    mydf.build()

    if rank==0:
        mf = scf.KRHF(cell, kpts)
        mydf.dump_to_file('erimpi.int')
        mf.with_df = mydf # USING MPIGDF ERI FILE for SCF
        mf.kernel()

        mf= scf.KRHF(cell,kpts)
        mf.with_df = df.GDF(cell,kpts) # USING SERIAL GDF ERI
        mf.with_df.mesh = [5,5,5]
        mf.kernel()

    comm.Barrier()
    nao, nkpts = cell.nao_nr(), len(kpts)
    mo_coeff = np.random.random([nkpts, nao, nao])
    from pyscf.pbc.lib import kpts_helper
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    for k0 in range(nkpts):
        for k1 in range(nkpts):
            for k2 in range(nkpts):
                k3 = kconserv[k0,k1,k2]
                eri_mo = mydf.ao2mo((mo_coeff[k0], mo_coeff[k1], mo_coeff[k2], mo_coeff[k3]), (kpts[k0], kpts[k1], kpts[k2], kpts[k3]), compact=False)
                eri_ao = mydf.get_eri((kpts[k0], kpts[k1], kpts[k2], kpts[k3]), compact=False)
                eri_mo2 = mf.with_df.ao2mo((mo_coeff[k0], mo_coeff[k1], mo_coeff[k2], mo_coeff[k3]), (kpts[k0], kpts[k1], kpts[k2], kpts[k3]), compact=False).reshape(nao,nao,nao,nao)
                eri_ao2 = mf.with_df.get_eri((kpts[k0], kpts[k1], kpts[k2], kpts[k3]), compact=False).reshape(nao,nao,nao,nao)
                print(k0,k1,k2,k3)
                print(np.linalg.norm(eri_ao-eri_ao2), np.linalg.norm(eri_mo-eri_mo2))
