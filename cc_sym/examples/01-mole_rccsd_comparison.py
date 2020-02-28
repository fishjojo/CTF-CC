#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
A simple example to compare ctf RCCSD with serial RCCSD on PYSCF
Usage: mpirun -np 4 python 01-mole_rccsd_comparison.py
'''
from pyscf import scf, gto, cc
from cc_sym import rccsd, settings

rank = settings.rank
comm = settings.comm

mol = gto.Mole(atom = 'H 0 0 0; F 0 0 1.1',
               basis = 'ccpvdz',
               verbose= 4)

mf = scf.RHF(mol)

if rank==0: 
    mf.kernel()
    refcc = cc.RCCSD(mf)
    refcc.kernel()

comm.barrier()
mf.mo_coeff = comm.bcast(mf.mo_coeff, root=0)
mf.mo_occ = comm.bcast(mf.mo_occ, root=0)

mycc = rccsd.RCCSD(mf)
mycc.kernel()

if rank==0:
    refcc.ipccsd(nroots=2)

comm.barrier()
mycc.ipccsd(nroots=2)

if rank==0:
    refcc.eaccsd(nroots=2)
comm.barrier()
mycc.eaccsd(nroots=2)
