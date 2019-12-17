from functools import reduce
import time
import numpy as np
import scipy.linalg
from pyscf import lib
from symtensor.settings import load_lib
import sys

def eigs(matvec, size, nroots, backend = 'numpy', x0=None, Adiag=None, guess=False, verbose=4):
    '''Davidson diagonalization method to solve A c = E c
    when A is not Hermitian.
    '''

    # We don't pass args
    def matvec_args(vec, args=None):
        return matvec(vec)

    nroots = min(nroots, size)
    conv, e, c = davidson(matvec, size, nroots, backend, x0, Adiag, verbose)
    return conv, e, c


def davidson(mult_by_A, N, neig, backend='numpy', x0=None, Adiag=None, verbose=4):
    """Diagonalize a matrix via non-symmetric Davidson algorithm.

    mult_by_A() is a function which takes a vector of length N
        and returns a vector of length N.
    neig is the number of eigenvalues requested
    """


    func = load_lib(backend)
    log = func.Logger(sys.stdout, verbose)
    rank = getattr(func,'rank',0)
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
        Adiag = func.to_nparray(Adiag)



    idx = Adiag.argsort()
    lamda_k_old = 0
    lamda_k = 0
    target = 0
    conv = False
    if x0 is not None:
        assert x0.shape == (Mmin,N)
        b = x0.copy()
        Ab = tuple([mult(b[m,:]) for m in range(Mmin)])
        Ab = func.vstack(Ab).transpose()

    evals = np.zeros(neig,dtype=np.complex)
    evecs = []
    for istep,M in enumerate(range(Mmin,Mmax+1)):
        if M == Mmin:
            b = func.zeros((N,M))
            ind = [i*M+m for m,i in zip(range(M),idx)]
            fill = np.ones(len(ind))
            func.write(b, ind, fill)
            Ab = tuple([mult(b[:,m]) for m in range(M)])
            Ab = func.vstack(Ab).transpose()

        else:

            Ab = func.hstack((Ab, mult(b[:,M-1]).reshape(N,-1)))

        Atilde = func.dot(b.conj().transpose(),Ab)
        Atilde = func.to_nparray(Atilde)
        lamda, alpha = diagonalize_asymm(Atilde)
        lamda_k_old, lamda_k = lamda_k, lamda[target]
        alpha_k = func.astensor(alpha[:,target])

        if M == Mmax:
            break

        q = func.dot( Ab-lamda_k*b, alpha_k)
        log.info('davidson istep = %d  root = %d  E = %.15g  dE = %.9g  residual = %.6g',
                 istep, target, lamda_k.real, (lamda_k - lamda_k_old).real, func.norm(q))
        cput1 = log.timer('davidson iter', *cput1)
        if func.norm(q) < tol:
            evecs.append(func.dot(b,alpha_k))
            evals[target] = lamda_k
            if target == neig-1:
                conv = True
                break
            else:
                target += 1
        eps = 1e-10
        xi = q/(lamda_k-Adiag+eps)
        bxi,R = func.qr(func.hstack((b,xi.reshape(N,-1))))
        nlast = bxi.shape[-1] - 1
        b = func.hstack((b,bxi[:,nlast].reshape(N,-1))) #can not replace nlast with -1, (inconsistent between numpy and ctf)

    evecs = func.vstack(tuple(evecs))

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
