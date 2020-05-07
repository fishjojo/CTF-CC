#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
Mole GCCSD with CTF
'''

import time
import numpy as np
import ctf

from pyscf import lib as pyscflib
from pyscf import ao2mo, gto
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo

from cc_sym import mpi_helper
from cc_sym import gccsd_numpy, rccsd

from symtensor import sym_ctf as sym

comm = mpi_helper.comm
rank = mpi_helper.rank
size = mpi_helper.size
tensor = sym.tensor
zeros = sym.zeros

class GCCSD(gccsd_numpy.GCCSD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, SYMVERBOSE=0):
        gccsd_numpy.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ, SYMVERBOSE)
        self.lib  = sym

    @property
    def _backend(self):
        return 'ctf'

    def amplitudes_to_vector(self, t1, t2):
        vector = ctf.hstack((t1.array.ravel(), t2.array.ravel()))
        return vector

    def ao2mo(self, mo_coeff=None):
        if getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            return _PhysicistsERIs(self, mo_coeff)

class _PhysicistsERIs:
    '''<pq||rs> = <pq|rs> - <pq|sr>'''
    def __init__(self, mycc, mo_coeff=None):
        cput1 = cput0 = (time.clock(), time.time())
        gccsd_numpy._eris_common_init(self, mycc, mo_coeff)
        mo_coeff = self.mo_coeff = comm.bcast(self.mo_coeff, root=0)
        if rank==0:
            gccsd_numpy._eris_1e_init(self, mycc, mo_coeff)
        comm.barrier()
        fock = self.fock = comm.bcast(self.fock, root=0)
        self.e_hf = comm.bcast(self.e_hf, root=0)

        self.dtype = dtype = np.result_type(self.fock)
        nocc, nmo = mycc.nocc, mycc.nmo
        nvir = nmo - nocc

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

        self._foo = self.foo.diagonal(preserve_shape=True)
        self._fvv = self.fvv.diagonal(preserve_shape=True)
        cput1 = logger.timer(mycc, "Writing Fock", *cput1)
        eijab = self.eia.array.reshape(nocc,1,nvir,1) + self.eia.array.reshape(1,nocc,1,nvir)
        self.eijab = tensor(eijab, verbose=mycc.SYMVERBOSE)

        assert(dtype==np.double)

        nao = mo_coeff.shape[0]
        mo_a = mo_coeff[:nao//2]
        mo_b = mo_coeff[nao//2:]

        def _force_sym(eri, sym_forbid_ij, sym_forbid_kl):
            eri_size = eri.size
            nij = sum(sym_forbid_ij)
            nkl = sum(sym_forbid_kl)
            task_ij = mpi_helper.static_partition(np.arange(nij))
            task_kl = mpi_helper.static_partition(np.arange(nkl))

            assert(eri_size==sym_forbid_ij.size*sym_forbid_kl.size)
            off_ij = np.arange(sym_forbid_ij.size)[sym_forbid_ij] * sym_forbid_kl.size
            off_kl = np.arange(sym_forbid_ij.size) * sym_forbid_kl.size

            idx_ij = off_ij[task_ij][:,None] + np.arange(sym_forbid_kl.size)
            idx_kl = off_kl[:,None] + np.arange(sym_forbid_kl.size)[sym_forbid_kl][task_kl]
            idx_full = np.append(idx_ij.ravel(), idx_kl.ravel())

            eri.write(idx_full, np.zeros(idx_full.size))
            return eri

        ppoo, ppov, ppvv = rccsd._make_ao_ints(mycc.mol, mo_a+mo_b, nocc, dtype)
        orbspin = self.orbspin
        occspin = orbspin[:nocc]
        virspin = orbspin[nocc:]

        oosym_forbid = (occspin[:,None] != occspin).ravel()
        ovsym_forbid = (occspin[:,None] != virspin).ravel()
        vosym_forbid = (virspin[:,None] != occspin).ravel()
        vvsym_forbid = (virspin[:,None] != virspin).ravel()

        cput1 = logger.timer(mycc, 'making ao integrals', *cput1)
        mo = ctf.astensor(mo_a+mo_b)
        orbo, orbv = mo[:,:nocc], mo[:,nocc:]


        tmp = ctf.einsum('uvmn,ui->ivmn', ppoo, orbo)
        oooo = ctf.einsum('ivmn,vj->ijmn', tmp, orbo)
        _force_sym(oooo, oosym_forbid, oosym_forbid)
        oooo = oooo.transpose(0,2,1,3) - oooo.transpose(0,2,3,1)

        ooov = ctf.einsum('ivmn,va->mnia', tmp, orbv)
        _force_sym(ooov, oosym_forbid, ovsym_forbid)
        ooov = ooov.transpose(0,2,1,3) - ooov.transpose(2,0,1,3)

        tmp = ctf.einsum('uvma,vb->ubma', ppov, orbv)
        ovov = ctf.einsum('ubma,ui->ibma', tmp, orbo)
        _force_sym(ovov, ovsym_forbid, ovsym_forbid)
        oovv = ovov.transpose(0,2,1,3) - ovov.transpose(0,2,3,1)
        del ppoo, ovov, tmp

        tmp = ctf.einsum('uvma,ub->mabv', ppov, orbv)
        _ovvo = ctf.einsum('mabv,vi->mabi', tmp, orbo)
        _force_sym(_ovvo, ovsym_forbid, vosym_forbid)
        tmp = ctf.einsum('uvab,ui->ivab', ppvv, orbo)
        _oovv = ctf.einsum('ivab,vj->ijab', tmp, orbo)
        _force_sym(_oovv, oosym_forbid, vvsym_forbid)

        ovov = _oovv.transpose(0,2,1,3) - _ovvo.transpose(0,2,3,1)
        ovvo = _ovvo.transpose(0,2,1,3) - _oovv.transpose(0,2,3,1)
        del _ovvo, _oovv, ppov, tmp

        tmp = ctf.einsum('uvab,vc->ucab', ppvv, orbv)
        ovvv = ctf.einsum('ucab,ui->icab', tmp, orbo)
        _force_sym(ovvv, ovsym_forbid, vvsym_forbid)
        ovvv = ovvv.transpose(0,2,1,3) - ovvv.transpose(0,2,3,1)

        vvvv = ctf.einsum('ucab,ud->dcab', tmp, orbv)
        _force_sym(vvvv, vvsym_forbid, vvsym_forbid)
        vvvv = vvvv.transpose(0,2,1,3) - vvvv.transpose(0,2,3,1)
        del ppvv, tmp

        self.oooo = tensor(oooo)
        self.ooov = tensor(ooov)
        self.oovv = tensor(oovv)
        self.ovov = tensor(ovov)
        self.ovvo = tensor(ovvo)
        self.ovvv = tensor(ovvv)
        self.vvvv = tensor(vvvv)
        logger.timer(mycc, 'ao2mo transformation', *cput0)

if __name__ == '__main__':

    from pyscf import gto, scf
    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.verbose=4
    mol.build()

    mf = scf.UHF(mol)
    if rank==0: mf.kernel()
    comm.barrier()
    mf.mo_coeff = comm.bcast(mf.mo_coeff, root=0)
    mf.mo_occ = comm.bcast(mf.mo_occ, root=0)
    mf.mo_energy = comm.bcast(mf.mo_occ, root=0)

    mf = scf.addons.convert_to_ghf(mf)


    # Freeze 1s electrons
    frozen = [0,1,2,3]
    gcc = GCCSD(mf, frozen=frozen)
    ecc0, t1, t2 = gcc.kernel()

    from pyscf.cc import GCCSD as REFG
    if rank==0:
        gcc = REFG(mf, frozen=frozen)
        ecc, t1, t2 = gcc.kernel()
        print(ecc0, ecc)
