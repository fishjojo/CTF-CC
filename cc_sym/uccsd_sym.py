#!/usr/bin/env python


'''
Mole UCCSD with CTF
'''

import numpy as np
import time
from functools import reduce
import ctf

from pyscf.lib import logger
import pyscf.lib as pyscflib
from pyscf import ao2mo, gto
from pyscf.ao2mo import _ao2mo
from pyscf.lib.parameters import LARGE_DENOM
from cc_sym import uccsd_sym_numpy, rccsd
from cc_sym import mpi_helper

from symtensor import sym_ctf as sym

comm = mpi_helper.comm
rank = mpi_helper.rank
size = mpi_helper.size
tensor = sym.tensor
zeros = sym.zeros

class UCCSD(uccsd_sym_numpy.UCCSD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, SYMVERBOSE=0):
        uccsd_sym_numpy.UCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ, SYMVERBOSE)
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

def _make_eris(mol, mo_a, mo_b, nocca, noccb):
    NS = ctf.SYM.NS
    SY = ctf.SYM.SY
    dtype = np.result_type(mo_a, mo_b)
    ao_loc = mol.ao_loc_nr()
    nao = mol.nao_nr()
    nmoa, nmob = mo_a.shape[1], mo_b.shape[1]
    nvira, nvirb = nmoa - nocca, nmob - noccb
    nocc = max(nocca, noccb)
    nvir = max(nvira, nvirb)
    nmo = nocc + nvir
    moa = np.zeros([nao, nmo], dtype=mo_a.dtype)
    moa[:,:nocca] = mo_a[:,:nocca]
    moa[:,nocc:nocc+nvira] = mo_a[:,nocca:]
    mob = np.zeros([nao, nmo], dtype=mo_b.dtype)
    mob[:,:noccb] = mo_b[:,:noccb]
    mob[:,nocc:nocc+nvirb] = mo_b[:,noccb:]
    moa = np.asarray(moa, order='F')
    mob = np.asarray(mob, order='F')

    ppoo = ctf.tensor((2,nao,nao,nocc,nocc), sym=[NS,SY,NS,SY,NS], dtype=dtype)
    ppov = ctf.tensor((2,nao,nao,nocc,nvir), sym=[NS,SY,NS,NS,NS], dtype=dtype)
    ppvv = ctf.tensor((2,nao,nao,nvir,nvir), sym=[NS,SY,NS,SY,NS], dtype=dtype)

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
    osqidx = np.arange(nocc**2).reshape(nocc,nocc)
    otrilidx = osqidx[np.tril_indices(nocc)]
    vsqidx = np.arange(nvir**2).reshape(nvir,nvir)
    vtrilidx = vsqidx[np.tril_indices(nvir)]


    subtasks = list(mpi_helper.static_partition(tasks))
    ntasks = max(comm.allgather(len(subtasks)))
    for itask in range(ntasks):
        if itask >= len(subtasks):
            ppoo.write([],[])
            ppov.write([],[])
            ppvv.write([],[])
            ppoo.write([],[])
            ppov.write([],[])
            ppvv.write([],[])
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

            eri_aa = _ao2mo.nr_e2(eri.reshape(di*dj,-1), moa, (0,nmo,0,nmo), 's2kl', 's1')
            eri_bb = _ao2mo.nr_e2(eri.reshape(di*dj,-1), mob, (0,nmo,0,nmo), 's2kl', 's1')
        else:
            eri = gto.moleintor.getints4c(intor, mol._atm, mol._bas, mol._env,
                                          shls_slice=shls_slice, aosym='s4',
                                          ao_loc=ao_loc, cintopt=ao2mopt._cintopt)
            eri_aa = _ao2mo.nr_e2(eri, moa, (0,nmo,0,nmo), 's4', 's1')
            eri_bb = _ao2mo.nr_e2(eri, mob, (0,nmo,0,nmo), 's4', 's1')
            idx = sqidx[i0:i1,j0:j1][np.tril_indices(i1-i0)]

        ooidx = idx[:,None] * nocc**2 + otrilidx
        ovidx = idx[:,None] * (nocc*nvir) + np.arange(nocc*nvir)
        vvidx = idx[:,None] * nvir**2 + vtrilidx
        eri_aa = eri_aa.reshape(-1,nmo,nmo)
        eri_bb = eri_bb.reshape(-1,nmo,nmo)
        ppoo.write(ooidx.ravel(), pyscflib.pack_tril(eri_aa[:,:nocc,:nocc]).ravel())
        ppov.write(ovidx.ravel(), eri_aa[:,:nocc,nocc:].ravel())
        ppvv.write(vvidx.ravel(), pyscflib.pack_tril(eri_aa[:,nocc:,nocc:]).ravel())

        ppoo.write(ooidx.ravel()+nao**2*nocc**2, pyscflib.pack_tril(eri_bb[:,:nocc,:nocc]).ravel())
        ppov.write(ovidx.ravel()+nao**2*nocc*nvir, eri_bb[:,:nocc,nocc:].ravel())
        ppvv.write(vvidx.ravel()+nao**2*nvir**2, pyscflib.pack_tril(eri_bb[:,nocc:,nocc:]).ravel())

    mo = ctf.astensor(np.stack((moa,mob)))
    orbo = mo[:,:,:nocc]
    orbv = mo[:,:,nocc:]

    mniv = ctf.einsum('Muvmn,Iui->MImniv', ppoo, orbo.conj())

    oooo = ctf.einsum('MImniv,Ivj->MIMminj', mniv, orbo)
    oooo -= oooo.transpose(1,0,2,4,3,5,6)

    ooov = ctf.einsum('MImniv,Iva->MIMmina', mniv, orbv) #ooov
    ooov-= ooov.transpose(1,0,2,4,3,5,6)

    oovv = ctf.einsum('Iuvia,Juj,Jvb->IJIijab', ppov, orbo.conj(), orbv)
    oovv-= oovv.transpose(1,0,2,4,3,5,6)

    ijab = ctf.einsum('Iuvij,Aua,Avb->IAijab', ppoo, orbv.conj(), orbv) #oovv
    iabj = ctf.einsum('Iuvia,Jub,Jvj->IJiabj', ppov, orbv.conj(), orbo) #ovvo

    ovov = ctf.einsum('IAijab->IAIiajb', ijab)
    ovov-= ctf.einsum('IAibaj->IAAiajb', iabj)

    ovvo = ctf.einsum('IAibaj->IAIiabj', iabj)
    ovvo-= ctf.einsum('IAijab->IAAiabj', ijab)

    del ppoo, ppov, ijab, iabj, mniv

    pdab = ctf.einsum('Auvab,Dvd->ADabud', ppvv, orbv)

    del ppvv

    iabc = ctf.einsum('BIbcua,Iui->IBiabc', pdab, orbo.conj()) #ovvv
    ovvv = ctf.einsum('IAibac->IAIiabc', iabc)
    ovvv-= ctf.einsum('IAicab->IAAiabc', iabc)
    del iabc

    vvvv = ctf.einsum('ABacud,Bub->ABAabcd', pdab, orbv.conj()) #vvvv
    vvvv -= vvvv.transpose(1,0,2,4,3,5,6)

    del pdab

    return oooo, ooov, oovv, ovov, ovvo, ovvv, vvvv

class _PhysicistsERIs:
    '''<pq||rs> = <pq|rs> - <pq|sr>'''
    def __init__(self, mycc, mo_coeff=None):
        lib = self.lib = mycc.lib
        if rank==0:
            uccsd_sym_numpy._eris_common_init(self, mycc, mo_coeff)
        comm.barrier()
        self.focka = focka = comm.bcast(self.focka, root=0)
        self.fockb = fockb = comm.bcast(self.fockb, root=0)

        nao = mycc.mol.nao_nr()
        nocca, noccb = mycc.nocc
        nmoa, nmob = mycc.nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb
        nocc, nvir = mycc._no, mycc._nv
        mo_coeff = self.mo_coeff

        def get_idx(string):
            dict_partial = {'o':nocca, 'O':noccb, 'v':nvira, 'V':nvirb}
            dict_full = {'o':nocc, 'O':nocc, 'v':nvir, 'V':nvir}
            off = 0
            for ks, s in enumerate(string[:-1][::-1]):
                if s == s.upper(): off += 2**ks
            off *= np.prod([dict_full[i] for i in string])

            shape = [dict_full[i] for i in string]
            full_size = np.prod(shape)
            idx_full = np.arange(full_size).reshape(shape)
            tab = np.ix_(*[np.arange(dict_partial[i]) for i in string])
            idx = idx_full[tab] + off
            return idx.ravel()

        self.foo = zeros([nocc,nocc], sym=mycc._sym1, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.fov = zeros([nocc,nvir], sym=mycc._sym1, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.fvv = zeros([nvir,nvir], sym=mycc._sym1, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        eia = ctf.ones([2,nocc,nvir]) * LARGE_DENOM
        self.eia = tensor(eia, sym=mycc._sym1, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        eijab = ctf.ones([2,2,2,nocc,nocc,nvir,nvir]) * LARGE_DENOM
        self.eijab = tensor(eijab, sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        if rank==0:
            mo_ea = focka.diagonal().real
            mo_eb = fockb.diagonal().real
            eia_a  = mo_ea[:nocca,None]- mo_ea[None,nocca:]
            eia_b  = mo_eb[:noccb,None]- mo_eb[None,noccb:]

            idx_oo = get_idx('oo')
            idx_OO = get_idx('OO')
            idx_vv = get_idx('vv')
            idx_VV = get_idx('VV')
            idx_ov = get_idx('ov')
            idx_OV = get_idx('OV')

            self.foo.write(idx_oo, focka[:nocca,:nocca].ravel())
            self.foo.write(idx_OO, fockb[:noccb,:noccb].ravel())
            self.fov.write(idx_ov, focka[:nocca,nocca:].ravel())
            self.fov.write(idx_OV, fockb[:noccb,noccb:].ravel())
            self.fvv.write(idx_vv, focka[nocca:,nocca:].ravel())
            self.fvv.write(idx_VV, fockb[noccb:,noccb:].ravel())
            self.eia.write(idx_ov, eia_a.ravel())
            self.eia.write(idx_OV, eia_b.ravel())

            e_ijab = eia_a[:,None,:,None] + eia_a[None,:,None,:]
            self.eijab.write(get_idx('oovv'), e_ijab.ravel())
            e_ijab = eia_b[:,None,:,None] + eia_b[None,:,None,:]
            self.eijab.write(get_idx('OOVV'), e_ijab.ravel())

            e_ijab = eia_a[:,None,:,None] + eia_b[None,:,None,:]

            self.eijab.write(get_idx('oOvV'), e_ijab.ravel())
            e_ijab = e_ijab.transpose(0,1,3,2)
            self.eijab.write(get_idx('oOVv'), e_ijab.ravel())
            e_ijab = e_ijab.transpose(1,0,3,2)
            self.eijab.write(get_idx('OovV'), e_ijab.ravel())
            e_ijab = e_ijab.transpose(0,1,3,2)
            self.eijab.write(get_idx('OoVv'), e_ijab.ravel())

        else:
            self.foo.write([],[],repeat=2)
            self.fov.write([],[],repeat=2)
            self.fvv.write([],[],repeat=2)
            self.eia.write([],[],repeat=2)
            self.eijab.write([], [],repeat=6)


        self._foo = self.foo.diagonal(preserve_shape=True)
        self._fvv = self.fvv.diagonal(preserve_shape=True)


        oooo, ooov, oovv, ovov, ovvo, ovvv, vvvv = \
                    _make_eris(mycc.mol, mo_coeff[0], mo_coeff[1], nocca, noccb)

        self.oooo = tensor(oooo, sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.ooov = tensor(ooov, sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.oovv = tensor(oovv, sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.ovov = tensor(ovov, sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.ovvo = tensor(ovvo, sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.ovvv = tensor(ovvv, sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.vvvv = tensor(vvvv, sym=mycc._sym2, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)


if __name__ == '__main__':
    from pyscf import gto, scf, cc
    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.verbose=5
    mol.build()
    mf = scf.UHF(mol).run()
    frozen = [0,1,2,3]
    frozen = None
    gcc = UCCSD(mf, frozen=frozen)
    gcc.kernel()

    gmf = scf.addons.convert_to_ghf(mf)
    from cc_sym.gccsd import GCCSD
    gcc = GCCSD(gmf)
    gcc.kernel()
