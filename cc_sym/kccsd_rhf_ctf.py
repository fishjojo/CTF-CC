#!/usr/bin/env python

import time
import numpy as np
from pyscf import lib as pyscflib
import rccsd
from symtensor import sym_ctf
from symtensor.symlib import SYMLIB
from pyscf.pbc.cc import kccsd_rhf
from pyscf.pbc.lib import kpts_helper
import kccsd_rhf
from symtensor.backend.ctf_funclib import static_partition, rank, comm, size
import ctf


class KRCCSDCTF(kccsd_rhf.KRCCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        rccsd.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.kpts = mf.kpts
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])
        self.lib = sym_ctf
        self.backend = sym_ctf.backend
        self.log = self.backend.Logger(self.stdout, self.verbose)
        self.symlib = SYMLIB('ctf')
        self.make_symlib()

    @property
    def _backend(self):
        return 'ctf'


    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)


class _ChemistsERIs(kccsd_rhf._ChemistsERIs):
    def gen_tasks(self, jobs):
        tasks = list(static_partition(jobs))
        ntasks = max(comm.allgather(len(tasks)))
        return tasks, ntasks


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
    mycc = KRCCSDCTF(mf)
    mycc.max_cycle=100
    eris = mycc.ao2mo()
    _, t1, t2 = mycc.init_amps(eris)
    t10, t20 = mycc.update_amps(t1, t2, eris)

    refcc = cc.KRCCSD(mf)

    erisref = refcc.ao2mo()
    _, t1r, t2r= refcc.init_amps(erisref)
    t1r0, t2r0= refcc.update_amps(t1r, t2r, erisref)

    print(np.linalg.norm(eris.oooo.transpose(0,2,1,3).array.to_nparray()-erisref.oooo))
    print(np.linalg.norm(eris.ooov.transpose(0,2,1,3).array.to_nparray()-erisref.ooov))
    print(np.linalg.norm(eris.ovov.transpose(0,2,1,3).array.to_nparray()-erisref.oovv))
    print(np.linalg.norm(eris.oovv.transpose(0,2,1,3).array.to_nparray()-erisref.ovov))
    print(np.linalg.norm(eris.ovvo.transpose(2,0,3,1).array.to_nparray()-erisref.voov))
    print(np.linalg.norm(eris.ovvv.transpose(2,0,3,1).array.to_nparray()-erisref.vovv))
    print(np.linalg.norm(eris.vvvv.transpose(0,2,1,3).array.to_nparray()-erisref.vvvv))

    print("t1", np.linalg.norm(t10.array.to_nparray()-t1r0))
    print("t2", np.linalg.norm(t20.array.to_nparray()-t2r0))

    mycc.kernel()
    refcc.kernel()
