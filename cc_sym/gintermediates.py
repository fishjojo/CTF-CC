#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

# Ref: Gauss and Stanton, J. Chem. Phys. 103, 3561 (1995) Table III

# Section (a)

def make_tau(t2, t1a, t1b, eris, fac=1, out=None):
    lib = eris.lib
    t1t1 = lib.einsum('ia,jb->ijab', fac*0.5*t1a, t1b)
    t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
    tau1 = t1t1 - t1t1.transpose(0,1,3,2)
    tau1 += t2
    return tau1

def cc_Fvv(t1, t2, eris):
    lib = eris.lib
    fov = eris.fov
    fvv = eris.fvv
    tau_tilde = make_tau(t2, t1, t1, eris, fac=0.5)
    Fae = fvv - 0.5*lib.einsum('me,ma->ae',fov, t1)
    Fae += lib.einsum('mf,mafe->ae', t1, eris.ovvv)
    Fae -= 0.5*lib.einsum('mnaf,mnef->ae', tau_tilde, eris.oovv)
    return Fae

def cc_Foo(t1, t2, eris):
    lib = eris.lib
    fov = eris.fov
    foo = eris.foo
    tau_tilde = make_tau(t2, t1, t1, eris, fac=0.5)
    Fmi = ( foo + 0.5*lib.einsum('me,ie->mi',fov, t1)
            + lib.einsum('ne,mnie->mi', t1, eris.ooov)
            + 0.5*lib.einsum('inef,mnef->mi', tau_tilde, eris.oovv) )
    return Fmi

def cc_Fov(t1, t2, eris):
    lib = eris.lib
    fov = eris.fov
    Fme = fov + lib.einsum('nf,mnef->me', t1, eris.oovv)
    return Fme

def cc_Woooo(t1, t2, eris):
    lib = eris.lib
    tau = make_tau(t2, t1, t1, eris)
    tmp = lib.einsum('je,mnie->mnij', t1, eris.ooov)
    Wmnij = eris.oooo + tmp - tmp.transpose(0,1,3,2)
    Wmnij += 0.25*lib.einsum('ijef,mnef->mnij', tau, eris.oovv)
    return Wmnij

def cc_Wvvvv(t1, t2, eris):
    lib = eris.lib
    tau = make_tau(t2, t1, t1, eris)
    tmp = lib.einsum('mb,mafe->bafe', t1, eris.ovvv)
    Wabef = eris.vvvv - tmp + tmp.transpose(1,0,2,3)
    Wabef += lib.einsum('mnab,mnef->abef', tau, 0.25*eris.oovv)
    return Wabef

def cc_Wovvo(t1, t2, eris):
    lib = eris.lib
    eris_ovvo = -eris.ovov.transpose(0,1,3,2)
    Wmbej  = lib.einsum('jf,mbef->mbej', t1, eris.ovvv)
    Wmbej += lib.einsum('nb,mnje->mbej', t1, eris.ooov)

    Wmbej -= 0.5*lib.einsum('jnfb,mnef->mbej', t2, eris.oovv)
    tmp = lib.einsum('nb,mnef->mbef', t1, eris.oovv)
    Wmbej -= lib.einsum('jf,mbef->mbej', t1, tmp)
    Wmbej += eris_ovvo
    return Wmbej

### Section (b)

def Fvv(t1, t2, eris):
    lib = eris.lib
    ccFov = cc_Fov(t1, t2, eris)
    Fae = cc_Fvv(t1, t2, eris) - 0.5*lib.einsum('ma,me->ae', t1,ccFov)
    return Fae

def Foo(t1, t2, eris):
    lib = eris.lib
    ccFov = cc_Fov(t1, t2, eris)
    Fmi = cc_Foo(t1, t2, eris) + 0.5*lib.einsum('ie,me->mi', t1,ccFov)
    return Fmi

def Fov(t1, t2, eris):
    Fme = cc_Fov(t1, t2, eris)
    return Fme

def Woooo(t1, t2, eris):
    lib = eris.lib
    tau = make_tau(t2, t1, t1, eris)
    Wmnij = 0.25*lib.einsum('ijef,mnef->mnij', tau, eris.oovv)
    Wmnij += cc_Woooo(t1, t2, eris)
    return Wmnij

def Wvvvv(t1, t2, eris):
    lib = eris.lib
    tau = make_tau(t2, t1, t1, eris)
    Wabef = cc_Wvvvv(t1, t2, eris)
    Wabef += lib.einsum('mnab,mnef->abef', tau, .25*eris.oovv)
    return Wabef

def Wovvo(t1, t2, eris):
    lib = eris.lib
    Wmbej = -0.5*lib.einsum('jnfb,mnef->mbej', t2, eris.oovv)
    Wmbej += cc_Wovvo(t1, t2, eris)
    return Wmbej

def Wooov(t1, t2, eris):
    lib = eris.lib
    Wmnie = lib.einsum('if,mnfe->mnie', t1, eris.oovv)
    Wmnie += eris.ooov
    return Wmnie

def Wvovv(t1, t2, eris):
    lib = eris.lib
    Wamef = lib.einsum('na,nmef->amef', -t1, eris.oovv)
    Wamef -= eris.ovvv.transpose(1,0,2,3)
    return Wamef

def Wovoo(t1, t2, eris):
    lib = eris.lib
    tmp1 = lib.einsum('mnie,jnbe->mbij', eris.ooov, t2)
    tmp2 = -lib.einsum('ie,mbje->mbij', t1, eris.ovov)
    tmp = lib.einsum('ie,mnef->mnif', t1, eris.oovv)
    tmp2 -= lib.einsum('mnif,njbf->mbij', tmp, t2)

    FFov = Fov(t1, t2, eris)
    WWoooo = Woooo(t1, t2, eris)
    tau = make_tau(t2, t1, t1, eris)
    Wmbij = lib.einsum('me,ijbe->mbij', -FFov, t2)
    Wmbij -= lib.einsum('nb,mnij->mbij', t1, WWoooo)
    Wmbij += 0.5 * lib.einsum('mbef,ijef->mbij', eris.ovvv, tau)
    Wmbij += tmp1 - tmp1.transpose(0,1,3,2)
    Wmbij += tmp2 - tmp2.transpose(0,1,3,2)
    Wmbij += eris.ooov.conj().transpose(2,3,0,1)
    return Wmbij

def Wvvvo(t1, t2, eris):
    lib = eris.lib
    eris_vvvo = -eris.ovvv.transpose(2,3,1,0).conj()
    tmp1 = lib.einsum('mbef,miaf->abei', eris.ovvv, t2)
    tmp2 = lib.einsum('ma,mbie->abei', t1, -eris.ovov)

    tmp = lib.einsum('ma,mnef->anef', t1, eris.oovv)
    tmp2 -= lib.einsum('nibf,anef->abei', t2, tmp)
    FFov = Fov(t1, t2, eris)
    tau = make_tau(t2, t1, t1, eris)
    Wabei  = 0.5 * lib.einsum('mnie,mnab->abei', -eris.ooov, tau)
    Wabei -= lib.einsum('me,miab->abei', FFov, t2)
    Wabei += eris_vvvo
    Wabei -= tmp1 - tmp1.transpose(1,0,2,3)
    Wabei -= tmp2 - tmp2.transpose(1,0,2,3)
    nocc,nvir = t1.shape
    _Wvvvv = Wvvvv(t1, t2, eris)
    Wabei += lib.einsum('abef,if->abei', _Wvvvv, t1)
    return Wabei
