#!/usr/bin/env python
def cc_Foo(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Fki  = 2*lib.einsum('kcld,ilcd->ki', eris.ovov, t2, symlib)
    Fki -= lib.einsum('kdlc,ilcd->ki', eris.ovov, t2, symlib)
    tmp  = lib.einsum('kcld,ld->kc', eris.ovov, t1, symlib)
    Fki += 2*lib.einsum('kc,ic->ki', tmp, t1, symlib)
    tmp  = lib.einsum('kdlc,ld->kc', eris.ovov, t1, symlib)
    Fki -= lib.einsum('kc,ic->ki', tmp, t1, symlib)
    Fki += eris.foo
    return Fki

def cc_Fvv(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Fac  =-2*lib.einsum('kcld,klad->ac', eris.ovov, t2, symlib)
    Fac +=   lib.einsum('kdlc,klad->ac', eris.ovov, t2, symlib)
    tmp  =   lib.einsum('kcld,ld->kc', eris.ovov, t1, symlib)
    Fac -= 2*lib.einsum('kc,ka->ac', tmp, t1, symlib)
    tmp  =   lib.einsum('kdlc,ld->kc', eris.ovov, t1, symlib)
    Fac +=   lib.einsum('kc,ka->ac', tmp, t1, symlib)
    Fac +=   eris.fvv
    return Fac

def cc_Fov(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Fkc  = 2*lib.einsum('kcld,ld->kc', eris.ovov, t1, symlib)
    Fkc -=   lib.einsum('kdlc,ld->kc', eris.ovov, t1, symlib)
    Fkc +=   eris.fov
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Lki = cc_Foo(t1, t2, eris) + lib.einsum('kc,ic->ki',eris.fov, t1, symlib)
    Lki += 2*lib.einsum('kilc,lc->ki', eris.ooov, t1, symlib)
    Lki -=   lib.einsum('likc,lc->ki', eris.ooov, t1, symlib)
    return Lki

def Lvv(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Lac = cc_Fvv(t1, t2, eris) - lib.einsum('kc,ka->ac', eris.fov, t1, symlib)
    Lac += 2*lib.einsum('kdac,kd->ac', eris.ovvv, t1, symlib)
    Lac -=   lib.einsum('kcad,kd->ac', eris.ovvv, t1, symlib)
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Wklij = lib.einsum('kilc,jc->klij', eris.ooov, t1, symlib)
    Wklij += lib.einsum('ljkc,ic->klij', eris.ooov, t1, symlib)

    Wklij += lib.einsum('kcld,ijcd->klij', eris.ovov, t2, symlib)
    tmp    = lib.einsum('kcld,ic->kild', eris.ovov, t1, symlib)
    Wklij += lib.einsum('kild,jd->klij', tmp, t1, symlib)
    Wklij += eris.oooo.transpose(0,2,1,3)
    return Wklij

def cc_Wvvvv(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Wabcd  = lib.einsum('kdac,kb->abcd', eris.ovvv,-t1, symlib)
    Wabcd -= lib.einsum('kcbd,ka->abcd', eris.ovvv, t1, symlib)
    Wabcd += eris.vvvv.transpose(0,2,1,3)
    return Wabcd

def cc_Wvoov(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Wakic  = lib.einsum('kcad,id->akic', eris.ovvv, t1, symlib)
    Wakic -= lib.einsum('likc,la->akic', eris.ooov, t1, symlib)
    Wakic += eris.ovvo.transpose(2,0,3,1)

    Wakic -= 0.5*lib.einsum('ldkc,ilda->akic', eris.ovov, t2, symlib)
    Wakic -= 0.5*lib.einsum('lckd,ilad->akic', eris.ovov, t2, symlib)
    tmp    = lib.einsum('ldkc,id->likc', eris.ovov, t1, symlib)
    Wakic -= lib.einsum('likc,la->akic', tmp, t1, symlib)
    Wakic += lib.einsum('ldkc,ilad->akic', eris.ovov, t2, symlib)
    return Wakic

def cc_Wvovo(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Wakci  = lib.einsum('kdac,id->akci', eris.ovvv, t1, symlib)
    Wakci -= lib.einsum('kilc,la->akci', eris.ooov, t1, symlib)
    Wakci += eris.oovv.transpose(2,0,3,1)
    Wakci -= 0.5*lib.einsum('lckd,ilda->akci', eris.ovov, t2, symlib)
    tmp    = lib.einsum('lckd,la->ackd', eris.ovov, t1)
    Wakci -= lib.einsum('ackd,id->akci', tmp, t1)
    return Wakci

def Wooov(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Wklid  = lib.einsum('ic,kcld->klid', t1, eris.ovov, symlib)
    Wklid += eris.ooov.transpose(0,2,1,3)
    return Wklid

def Wvovv(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Walcd  = lib.einsum('ka,kcld->alcd',-t1, eris.ovov, symlib)
    Walcd += eris.ovvv.transpose(2,0,3,1)
    return Walcd

def W1ovvo(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Wkaci  = 2*lib.einsum('kcld,ilad->kaci', eris.ovov, t2, symlib)
    Wkaci +=  -lib.einsum('kcld,liad->kaci', eris.ovov, t2, symlib)
    Wkaci +=  -lib.einsum('kdlc,ilad->kaci', eris.ovov, t2, symlib)
    Wkaci += eris.ovvo.transpose(0,2,1,3)
    return Wkaci

def W2ovvo(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Wkaci = lib.einsum('la,lkic->kaci',-t1, Wooov(t1, t2, eris), symlib)
    Wkaci += lib.einsum('kcad,id->kaci', eris.ovvv, t1, symlib)
    return Wkaci

def Wovvo(t1, t2, eris):
    Wkaci = W1ovvo(t1, t2, eris) + W2ovvo(t1, t2, eris)
    return Wkaci

def W1ovov(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Wkbid = -lib.einsum('kcld,ilcb->kbid', eris.ovov, t2, symlib)
    Wkbid += eris.oovv.transpose(0,2,1,3)
    return Wkbid

def W2ovov(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Wkbid = lib.einsum('klid,lb->kbid', Wooov(t1, t2, eris),- t1, symlib)
    Wkbid += lib.einsum('kcbd,ic->kbid', eris.ovvv, t1, symlib)
    return Wkbid

def Wovov(t1, t2, eris):
    return W1ovov(t1, t2, eris) + W2ovov(t1, t2, eris)

def Woooo(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Wklij  = lib.einsum('kcld,ijcd->klij', eris.ovov, t2, symlib)
    tmp    = lib.einsum('kcld,ic->kild', eris.ovov, t1, symlib)
    Wklij += lib.einsum('kild,jd->klij', tmp, t1, symlib)
    Wklij += lib.einsum('kild,jd->klij', eris.ooov, t1, symlib)
    Wklij += lib.einsum('ljkc,ic->klij', eris.ooov, t1, symlib)
    Wklij += eris.oooo.transpose(0,2,1,3)
    return Wklij

def Wvvvv(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Wabcd  = lib.einsum('kcld,klab->abcd', eris.ovov, t2, symlib)
    tmp    = lib.einsum('kcld,ka->acld', eris.ovov, t1)
    Wabcd += lib.einsum('acld,lb->abcd', tmp, t1)
    Wabcd += eris.vvvv.transpose(0,2,1,3)
    Wabcd -= lib.einsum('ldac,lb->abcd', eris.ovvv, t1, symlib)
    Wabcd -= lib.einsum('kcbd,ka->abcd', eris.ovvv, t1, symlib)
    return Wabcd

def Wvvvo(t1, t2, eris, Wvvvv=None):
    lib, symlib = eris.lib, eris.symlib
    Wabcj  =  -lib.einsum('alcj,lb->abcj', W1ovov(t1, t2, eris).transpose(1,0,3,2), t1, symlib)
    Wabcj +=  -lib.einsum('kbcj,ka->abcj', W1ovvo(t1, t2, eris), t1, symlib)
    Wabcj += 2*lib.einsum('ldac,ljdb->abcj', eris.ovvv, t2, symlib)
    Wabcj +=  -lib.einsum('ldac,ljbd->abcj', eris.ovvv, t2, symlib)
    Wabcj +=  -lib.einsum('lcad,ljdb->abcj', eris.ovvv, t2, symlib)
    Wabcj +=  -lib.einsum('kcbd,jkda->abcj', eris.ovvv, t2, symlib)
    Wabcj +=   lib.einsum('ljkc,lkba->abcj', eris.ooov, t2, symlib)
    tmp    =   lib.einsum('ljkc,lb->kcbj', eris.ooov, t1)
    Wabcj +=   lib.einsum('kcbj,ka->abcj', tmp, t1)
    Wabcj +=  -lib.einsum('kc,kjab->abcj', cc_Fov(t1, t2, eris), t2, symlib)
    Wabcj += eris.ovvv.transpose(3,1,2,0).conj()
    if Wvvvv is None:
        Wvvvv = Wvvvv(t1, t2, eris)
    Wabcj += lib.einsum('abcd,jd->abcj', Wvvvv, t1, symlib)
    return Wabcj

def Wovoo(t1, t2, eris):
    lib, symlib = eris.lib, eris.symlib
    Wkbij  =   lib.einsum('kbid,jd->kbij', W1ovov(t1, t2, eris), t1, symlib)
    Wkbij +=  -lib.einsum('klij,lb->kbij', Woooo(t1, t2, eris), t1, symlib)
    Wkbij +=   lib.einsum('kbcj,ic->kbij', W1ovvo(t1, t2, eris), t1, symlib)
    Wkbij += 2*lib.einsum('kild,ljdb->kbij', eris.ooov, t2, symlib)
    Wkbij +=  -lib.einsum('kild,jldb->kbij', eris.ooov, t2, symlib)
    Wkbij +=  -lib.einsum('likd,ljdb->kbij', eris.ooov, t2, symlib)
    Wkbij +=   lib.einsum('kcbd,jidc->kbij', eris.ovvv, t2, symlib)
    tmp    =   lib.einsum('kcbd,ic->kibd', eris.ovvv, t1, symlib)
    Wkbij +=   lib.einsum('kibd,jd->kbij', tmp, t1, symlib)
    Wkbij +=  -lib.einsum('ljkc,libc->kbij', eris.ooov, t2, symlib)
    Wkbij +=   lib.einsum('kc,ijcb->kbij', cc_Fov(t1, t2, eris), t2, symlib)
    Wkbij += eris.ooov.transpose(1,3,0,2).conj()
    return Wkbij
