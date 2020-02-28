#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
Using MPIGDF to generate ERIs for pyscf.pbc
Usage: mpirun -np 4 python 04-mpigdf.py
'''
from pyscf.pbc import scf, gto
from cc_sym import mpigdf, settings

rank = settings.rank
comm = settings.comm

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
mf = scf.KRHF(cell, kpts, exxdiv=None)

mydf = mpigdf.GDF(cell, kpts)
mydf.build() #building eris incore
# dump eris to disk so pyscf can use it
# a cderi_file can be specified so you can run a OMP-threaded SCF in a separate job.
mydf.dump_to_file(cderi_file=None)

mf.with_df = mydf

if rank==0:
    mf.kernel()
