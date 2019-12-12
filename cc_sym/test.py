from pyscf import scf
from pyscf import gto
from rccsd_slow import RCCSD as REFCC
import numpy as np
from rccsd import RCCSD
from pyscf import lib as pyscflib
from functools import reduce


mol = gto.M()
nocc, nvir = 5, 12
nmo = nocc + nvir
nmo_pair = nmo*(nmo+1)//2
mf = scf.RHF(mol)
np.random.seed(12)
mf._eri = np.random.random(nmo_pair*(nmo_pair+1)//2)
mf.mo_coeff = np.random.random((nmo,nmo))
mf.mo_energy = np.arange(0., nmo)
mf.mo_occ = np.zeros(nmo)
mf.mo_occ[:nocc] = 2
vhf = mf.get_veff(mol, mf.make_rdm1())
cinv = np.linalg.inv(mf.mo_coeff)
mf.get_hcore = lambda *args: (reduce(np.dot, (cinv.T*mf.mo_energy, cinv)) - vhf)

mycc = REFCC(mf)
eris = mycc.ao2mo()
a = np.random.random((nmo,nmo)) * .1
eris.fock += a + a.T.conj()
t1 = np.random.random((nocc,nvir)) * .1
t2 = np.random.random((nocc,nocc,nvir,nvir)) * .1
t2 = t2 + t2.transpose(1,0,3,2)

mycc.cc2 = False
t1a, t2a = mycc.update_amps( t1, t2, eris)
print(pyscflib.finger(t1a) - -106360.5276951083)
print(pyscflib.finger(t2a) - 66540.100267798145)
mycc.cc2 = True
t1b, t2b = mycc.update_amps( t1, t2, eris)
print(pyscflib.finger(t1b) - -106360.5276951083)
print(pyscflib.finger(t2b) - -1517.9391800662809)



mycct = RCCSD(mf)
lib = mycct.lib
erist = mycct.ao2mo()
moe = eris.fock.diagonal()
eia = moe[:nocc,None] - moe[None,nocc:]
erist.eia = lib.tensor(eia)
erist.eijab = eia[:,None,:,None] + eia[None,:,None,:]
erist.foo = lib.tensor(eris.fock[:nocc,:nocc].copy())
erist.fov = lib.tensor(eris.fock[:nocc,nocc:].copy())
erist.fvv = lib.tensor(eris.fock[nocc:,nocc:].copy())

t1t = lib.tensor(t1)
t2t = lib.tensor(t2)

mycct.cc2 = False
t1at, t2at = mycct.update_amps( t1t, t2t, erist)
print(pyscflib.finger(t1at.array) - pyscflib.finger(t1a))
print(pyscflib.finger(t2at.array) - pyscflib.finger(t2a))
mycct.cc2 = True
t1bt, t2bt = mycct.update_amps( t1t, t2t, erist)
print(pyscflib.finger(t1bt.array) - pyscflib.finger(t1b))
print(pyscflib.finger(t2bt.array) - pyscflib.finger(t2b))


mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = 'cc-pvdz'
#mol.basis = '3-21G'
mol.verbose = 0
mol.spin = 0
mol.build()
mf = scf.RHF(mol).run(conv_tol=1e-14)

mycc = REFCC(mf)
eris = mycc.ao2mo()
emp2, t1, t2 = mycc.init_amps(eris)
print(pyscflib.finger(t2) - 0.08551011863965133)
np.random.seed(1)
t1 = np.random.random(t1.shape)*.1
t2 = np.random.random(t2.shape)*.1
t2 = t2 + t2.transpose(1,0,3,2)
t1, t2 = mycc.update_amps( t1, t2, eris)
print(pyscflib.finger(t1) - -0.01960058727265309)
print(pyscflib.finger(t2) - -0.012913260807190019)

ecc, t1, t2 = mycc.kernel()
print(ecc - -0.21334326214236796)

mycct = RCCSD(mf)
erist = mycct.ao2mo()
moe = eris.fock.diagonal()
eia = moe[:nocc,None] - moe[None,nocc:]
erist.eia = lib.tensor(eia)
erist.eijab = eia[:,None,:,None] + eia[None,:,None,:]
erist.foo = lib.tensor(eris.fock[:nocc,:nocc].copy())
erist.fov = lib.tensor(eris.fock[:nocc,nocc:].copy())
erist.fvv = lib.tensor(eris.fock[nocc:,nocc:].copy())
emp2, t1t, t2t = mycct.init_amps(erist)
print(pyscflib.finger(t2t.array) - 0.08551011863965133)
