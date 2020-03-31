#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

"""
Diamond DZVP with 27 kpts, >400GB memory required
"""
import ctf
from pyscf.pbc import gto, scf
from cc_sym import mpigdf, kccsd_rhf, settings
import sys, os
rank = settings.rank
size = settings.size
comm = settings.comm

if rank!=0:
    sys.stdout = open(os.devnull, "w")
else:
    sys.stdout = open("dmd333_%i.dat"%size, "w")

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

kpts = cell.make_kpts([3,3,3])
mf = scf.KRHF(cell, kpts, exxdiv=None)

mydf = mpigdf.GDF(cell, kpts)
mydf.build()
mydf.dump_to_file()
mf.with_df = mydf

if rank==0:
    mf.kernel()

comm.barrier()
mf.mo_coeff = comm.bcast(mf.mo_coeff, root=0)
mf.mo_occ = comm.bcast(mf.mo_occ, root=0)

mycc = kccsd_rhf.KRCCSD(mf, SYMVERBOSE=2)
mycc.max_cycle = 1
mycc.kernel()

#mycc.ipccsd(nroots=2, kptlist=[0], koopmans=True)
#mycc.eaccsd(nroots=2, kptlist=[0], koopmans=True)
