#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
Mole RCCSD with CTF
'''


import time
import numpy as np
import ctf

from pyscf import lib as pyscflib
from pyscf import ao2mo, gto
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo

from cc_sym import mpi_helper
from cc_sym import rccsd_numpy

from symtensor import sym_ctf as sym

comm = mpi_helper.comm
rank = mpi_helper.rank
size = mpi_helper.size
tensor = sym.tensor
zeros = sym.zeros

class RCCSD(rccsd_numpy.RCCSD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, SYMVERBOSE=0):
        rccsd_numpy.RCCSD.__init__(self, mf, frozen=frozen, mo_coeff=mo_coeff, \
                                   mo_occ=mo_occ, SYMVERBOSE=SYMVERBOSE)
        self.lib = sym

    @property
    def _backend(self):
        return 'ctf'

    def amplitudes_to_vector(self, t1, t2):
        vector = ctf.hstack((t1.array.ravel(), t2.array.ravel()))
        return vector

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

class _ChemistsERIs:
    def __init__(self, mycc, mo_coeff=None):
        self.lib = mycc.lib
        nocc, nmo = mycc.nocc, mycc.nmo
        nvir = nmo - nocc
        cput1 = cput0 = (time.clock(), time.time())
        self.fock = self.mo_coeff = None
        if rank==0:
            rccsd_numpy._eris_common_init(self, mycc, mo_coeff)
        comm.barrier()
        fock = comm.bcast(self.fock, root=0)
        mo_coeff = self.mo_coeff = comm.bcast(self.mo_coeff, root=0)
        self.dtype = dtype = np.result_type(fock)

        self.foo = zeros([nocc,nocc], dtype, verbose=mycc.SYMVERBOSE)
        self.fov = zeros([nocc,nvir], dtype, verbose=mycc.SYMVERBOSE)
        self.fvv = zeros([nvir,nvir], dtype, verbose=mycc.SYMVERBOSE)
        self.eia = zeros([nocc,nvir], verbose=mycc.SYMVERBOSE)

        if rank==0:
            mo_e = fock.diagonal().real
            eia  = mo_e[:nocc,None]- mo_e[None,nocc:]
            self.foo.write(range(nocc*nocc), fock[:nocc,:nocc].ravel())
            self.fov.write(range(nocc*nvir), fock[:nocc,nocc:].ravel())
            self.fvv.write(range(nvir*nvir), fock[nocc:,nocc:].ravel())
            self.eia.write(range(nocc*nvir), eia.ravel())
        else:
            self.foo.write([], [])
            self.fov.write([], [])
            self.fvv.write([], [])
            self.eia.write([], [])
        cput1 = logger.timer(mycc, "Writing Fock", *cput1)
        self._foo = self.foo.diagonal(preserve_shape=True)
        self._fvv = self.fvv.diagonal(preserve_shape=True)
        eijab = self.eia.array.reshape(nocc,1,nvir,1) + self.eia.array.reshape(1,nocc,1,nvir)
        self.eijab = tensor(eijab, verbose=mycc.SYMVERBOSE)

        ppoo, ppov, ppvv = _make_ao_ints(mycc.mol, mo_coeff, nocc, dtype)
        cput1 = logger.timer(mycc, 'making ao integrals', *cput1)
        mo = ctf.astensor(mo_coeff)
        orbo, orbv = mo[:,:nocc], mo[:,nocc:]

        tmp = ctf.einsum('uvmn,ui->ivmn', ppoo, orbo.conj())
        oooo = ctf.einsum('ivmn,vj->ijmn', tmp, orbo)
        ooov = ctf.einsum('ivmn,va->mnia', tmp, orbv)

        tmp = ctf.einsum('uvma,vb->ubma', ppov, orbv)
        ovov = ctf.einsum('ubma,ui->ibma', tmp, orbo.conj())
        tmp = ctf.einsum('uvma,ub->mabv', ppov, orbv.conj())
        ovvo = ctf.einsum('mabv,vi->mabi', tmp, orbo)

        tmp = ctf.einsum('uvab,ui->ivab', ppvv, orbo.conj())
        oovv = ctf.einsum('ivab,vj->ijab', tmp, orbo)

        tmp = ctf.einsum('uvab,vc->ucab', ppvv, orbv)
        ovvv = ctf.einsum('ucab,ui->icab', tmp, orbo.conj())
        vvvv = ctf.einsum('ucab,ud->dcab', tmp, orbv.conj())

        self.oooo = tensor(oooo, verbose=mycc.SYMVERBOSE)
        self.ooov = tensor(ooov, verbose=mycc.SYMVERBOSE)
        self.ovov = tensor(ovov, verbose=mycc.SYMVERBOSE)
        self.oovv = tensor(oovv, verbose=mycc.SYMVERBOSE)
        self.ovvo = tensor(ovvo, verbose=mycc.SYMVERBOSE)
        self.ovvv = tensor(ovvv, verbose=mycc.SYMVERBOSE)
        self.vvvv = tensor(vvvv, verbose=mycc.SYMVERBOSE)
        logger.timer(mycc, 'ao2mo transformation', *cput0)

def _make_ao_ints(mol, mo_coeff, nocc, dtype):
    NS = ctf.SYM.NS
    SY = ctf.SYM.SY

    ao_loc = mol.ao_loc_nr()
    mo = np.asarray(mo_coeff, order='F')
    nao, nmo = mo.shape
    nvir = nmo - nocc

    ppoo = ctf.tensor((nao,nao,nocc,nocc), sym=[SY,NS,NS,NS], dtype=dtype)
    ppov = ctf.tensor((nao,nao,nocc,nvir), sym=[SY,NS,NS,NS], dtype=dtype)
    ppvv = ctf.tensor((nao,nao,nvir,nvir), sym=[SY,NS,SY,NS], dtype=dtype)
    intor = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    blksize = int(max(4, min(nao/3, nao/size**.5, 2000e6/8/nao**3)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, blksize)
    tasks = []
    for k, (ish0, ish1, di) in enumerate(sh_ranges):
        for jsh0, jsh1, dj in sh_ranges[:k+1]:
            tasks.append((ish0,ish1,jsh0,jsh1))

    sqidx = np.arange(nao**2).reshape(nao,nao)
    trilidx = sqidx[np.tril_indices(nao)]
    vsqidx = np.arange(nvir**2).reshape(nvir,nvir)
    vtrilidx = vsqidx[np.tril_indices(nvir)]

    subtasks = list(mpi_helper.static_partition(tasks))
    ntasks = max(comm.allgather(len(subtasks)))
    for itask in range(ntasks):
        if itask >= len(subtasks):
            ppoo.write([], [])
            ppov.write([], [])
            ppvv.write([], [])
            continue

        shls_slice = subtasks[itask]
        ish0, ish1, jsh0, jsh1 = shls_slice
        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
        di = i1 - i0
        dj = j1 - j0
        if i0 != j0:
            eri = gto.moleintor.getints4c(intor, mol._atm, mol._bas, mol._env,
                                          shls_slice=shls_slice, aosym='s2kl',
                                          ao_loc=ao_loc, cintopt=ao2mopt._cintopt)
            idx = sqidx[i0:i1,j0:j1].ravel()

            eri = _ao2mo.nr_e2(eri.reshape(di*dj,-1), mo, (0,nmo,0,nmo), 's2kl', 's1')
        else:
            eri = gto.moleintor.getints4c(intor, mol._atm, mol._bas, mol._env,
                                          shls_slice=shls_slice, aosym='s4',
                                          ao_loc=ao_loc, cintopt=ao2mopt._cintopt)
            eri = _ao2mo.nr_e2(eri, mo, (0,nmo,0,nmo), 's4', 's1')
            idx = sqidx[i0:i1,j0:j1][np.tril_indices(i1-i0)]

        ooidx = idx[:,None] * nocc**2 + np.arange(nocc**2)
        ovidx = idx[:,None] * (nocc*nvir) + np.arange(nocc*nvir)
        vvidx = idx[:,None] * nvir**2 + vtrilidx
        eri = eri.reshape(-1,nmo,nmo)
        ppoo.write(ooidx.ravel(), eri[:,:nocc,:nocc].ravel())
        ppov.write(ovidx.ravel(), eri[:,:nocc,nocc:].ravel())
        ppvv.write(vvidx.ravel(), pyscflib.pack_tril(eri[:,nocc:,nocc:]).ravel())
        idx = eri = None
    return ppoo, ppov, ppvv

if __name__ == '__main__':
    from pyscf import scf
    import os

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 4
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol)
    if rank==0:
        mf.kernel()

    comm.barrier()
    mf.mo_coeff  = comm.bcast(mf.mo_coeff, root=0)
    mf.mo_occ = comm.bcast(mf.mo_occ, root=0)

    mycc = RCCSD(mf)
    mycc.kernel()
