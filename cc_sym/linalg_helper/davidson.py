from functools import reduce
import time
import numpy as np
import scipy.linalg
from pyscf import lib
from symtensor.settings import load_lib
import sys
import ctf
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def eigs(matvec, vecsize, nroots, x0=None, Adiag=None, guess=False, verbose=4):
    '''Davidson diagonalization method to solve A c = E c
    when A is not Hermitian.
    '''

    # We don't pass args
    def matvec_args(vec, args=None):
        return matvec(vec)

    nroots = min(nroots, vecsize)
    conv, e, c = davidson(matvec, vecsize, nroots, x0, Adiag, verbose)
    return conv, e, c


def davidson(mult_by_A, N, neig, x0=None, Adiag=None, verbose=4):
    """Diagonalize a matrix via non-symmetric Davidson algorithm.

    mult_by_A() is a function which takes a vector of length N
        and returns a vector of length N.
    neig is the number of eigenvalues requested
    """

    if rank==0:
        log = lib.logger.Logger(sys.stdout, verbose)
    else:
        log = lib.logger.Logger(sys.stdout, 0)

    cput1 = (time.clock(), time.time())

    Mmin = min(neig,N)
    Mmax = min(N,2000)
    tol = 1e-6

    def mult(arg):
        return mult_by_A(arg)

    if Adiag is None:
        Adiag = np.zeros(N,np.complex)
        for i in range(N):
            test = np.zeros(N,np.complex)
            test[i] = 1.0
            Adiag[i] = mult(test)[i]
    else:
        Adiag = Adiag.to_nparray()

    idx = Adiag.argsort()
    lamda_k_old = 0
    lamda_k = 0
    target = 0
    conv = False
    if x0 is not None:
        assert x0.shape == (Mmin,N)
        b = x0.copy()
        Ab = tuple([mult(b[m,:]) for m in range(Mmin)])
        Ab = ctf.vstack(Ab).transpose()

    evals = np.zeros(neig,dtype=np.complex)
    evecs = []
    for istep,M in enumerate(range(Mmin,Mmax+1)):
        if M == Mmin:
            b = ctf.zeros((N,M))
            if rank==0:
                ind = [i*M+m for m,i in zip(range(M),idx)]
                fill = np.ones(len(ind))
                b.write(ind, fill)
            else:
                b.write([],[])
            Ab = tuple([mult(b[:,m]) for m in range(M)])
            Ab = ctf.vstack(Ab).transpose()

        else:

            Ab = ctf.hstack((Ab, mult(b[:,M-1]).reshape(N,-1)))

        Atilde = ctf.dot(b.conj().transpose(),Ab)
        Atilde = Atilde.to_nparray()

        lamda, alpha = diagonalize_asymm(Atilde)
        lamda_k_old, lamda_k = lamda_k, lamda[target]
        alpha_k = ctf.astensor(alpha[:,target])
        if M == Mmax:
            break

        q = ctf.dot( Ab-lamda_k*b, alpha_k)
        qnorm = ctf.norm(q)
        log.info('davidson istep = %d  root = %d  E = %.15g  dE = %.9g  residual = %.6g',
                 istep, target, lamda_k.real, (lamda_k - lamda_k_old).real, qnorm)
        cput1 = log.timer('davidson iter', *cput1)

        if ctf.norm(q) < tol:
            evecs.append(ctf.dot(b,alpha_k))
            evals[target] = lamda_k
            if target == neig-1:
                conv = True
                break
            else:
                target += 1
        eps = 1e-10
        xi = q/(lamda_k-Adiag+eps)
        bxi,R = ctf.qr(ctf.hstack((b,xi.reshape(N,-1))))
        nlast = bxi.shape[-1] - 1
        b = ctf.hstack((b,bxi[:,nlast].reshape(N,-1))) #can not replace nlast with -1, (inconsistent between numpy and ctf)
    evecs = ctf.vstack(tuple(evecs))

    return conv, evals, evecs

def diagonalize_asymm(H):
    E,C = np.linalg.eig(H)
    idx = E.real.argsort()
    E = E[idx]
    C = C[:,idx]

    return E,C

if __name__ == '__main__':
    N = 200
    neig = 2
    backend = 'numpy'
    func= load_lib(backend)
    A = np.zeros((N,N))
    k = N/2
    for ii in range(N):
        i = ii+1
        for jj in range(N):
            j = jj+1
            if j <= k:
                A[ii,jj] = i*(i==j)-(i-j-k**2)
            else:
                A[ii,jj] = i*(i==j)+(i-j-k**2)

    A = func.astensor(A)
    def matvec(x):
        return func.dot(A,x)

    conv, e,c = eigs(matvec,N,neig,backend=backend,Adiag=func.diag(A))
    if rank ==0:
        print("# davidson evals =", e)

    e,c = diagonalize_asymm(func.to_nparray(A))
    if rank==0:
        print("# numpy evals =", e.real[:neig])
