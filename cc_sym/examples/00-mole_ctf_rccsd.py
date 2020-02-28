#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
A simple example to run molecular RCCSD with ctf parallelization
Usage: mpirun -np 4 python 00-mole_ctf_rccsd.py
'''
from pyscf import scf, gto
from cc_sym import rccsd, settings

rank = settings.rank
comm = settings.comm

mol = gto.Mole(atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
               basis = 'ccpvdz',
               verbose= 5)

mf = scf.RHF(mol)

if rank==0: #SCF only needs to run once
    mf.kernel()

comm.barrier()
mf.mo_coeff = comm.bcast(mf.mo_coeff, root=0)
mf.mo_occ = comm.bcast(mf.mo_occ, root=0)

mycc = rccsd.RCCSD(mf)
mycc.kernel()

mycc.ipccsd(nroots=2, koopmans=True)
mycc.eaccsd(nroots=2, koopmans=True)
