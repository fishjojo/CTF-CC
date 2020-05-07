#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import sys, os
if rank!=0:
    sys.stdout = open(os.devnull, 'w')    # supress printing

def static_partition(tasks):
    segsize = (len(tasks)+size-1) // size
    start = rank * segsize
    stop = min(len(tasks), start+segsize)
    return tasks[start:stop]
