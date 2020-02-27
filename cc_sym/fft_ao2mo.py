import numpy as np
import ctf
from cc_sym import rccsd
import symtensor.sym_ctf as lib
from pyscf.pbc import tools
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_plain, _contract_compact
from pyscf import lib as pyscflib
from pyscf.ao2mo.incore import iden_coeffs
import time

rank = rccsd.rank
comm = rccsd.comm
size = rccsd.size
tensor = lib.tensor
Logger = rccsd.Logger
static_partition = rccsd.static_partition

def ao2mo(mycc, mo_coeff=None):
    mydf = mycc._scf.with_df
    if mo_coeff is None: mo_coeff = mycc.mo_coeff
    kpts = mycc.kpts
    logger = Logger(mycc.stdout, mycc.verbose)
    cell = mydf.cell
    gvec = cell.reciprocal_vectors()
    nao = cell.nao_nr()
    coords = cell.gen_uniform_grids(mydf.mesh)
    ngrids = len(coords)
    nkpts = len(kpts)

    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    cput1 = cput0 = (time.clock(), time.time())
    ijG = ctf.zeros([nkpts,nkpts,nocc,nocc,ngrids], dtype=np.complex128)
    iaG = ctf.zeros([nkpts,nkpts,nocc,nvir,ngrids], dtype=np.complex128)
    abG = ctf.zeros([nkpts,nkpts,nvir,nvir,ngrids], dtype=np.complex128)

    ijR = ctf.zeros([nkpts,nkpts,nocc,nocc,ngrids], dtype=np.complex128)
    iaR = ctf.zeros([nkpts,nkpts,nocc,nvir,ngrids], dtype=np.complex128)
    aiR = ctf.zeros([nkpts,nkpts,nvir,nocc,ngrids], dtype=np.complex128)
    abR = ctf.zeros([nkpts,nkpts,nvir,nvir,ngrids], dtype=np.complex128)

    jobs = []
    for ki in range(nkpts):
        for kj in range(ki,nkpts):
            jobs.append([ki,kj])

    tasks = list(static_partition(jobs))
    ntasks = max(comm.allgather(len(tasks)))
    idx_ooG = np.arange(nocc*nocc*ngrids)
    idx_ovG = np.arange(nocc*nvir*ngrids)
    idx_vvG = np.arange(nvir*nvir*ngrids)

    for itask in range(ntasks):
        if itask >= len(tasks):
            ijR.write([], [])
            ijR.write([], [])
            iaR.write([], [])
            iaR.write([], [])
            aiR.write([], [])
            aiR.write([], [])
            abR.write([], [])
            abR.write([], [])

            ijG.write([], [])
            ijG.write([], [])
            iaG.write([], [])
            iaG.write([], [])
            abG.write([], [])
            abG.write([], [])
            continue
        ki, kj = tasks[itask]
        kpti, kptj = kpts[ki], kpts[kj]
        ao_kpti = mydf._numint.eval_ao(cell, coords, kpti)[0]
        ao_kptj = mydf._numint.eval_ao(cell, coords, kptj)[0]
        q = kptj - kpti
        coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
        wcoulG = coulG * (cell.vol/ngrids)
        fac = np.exp(-1j * np.dot(coords, q))
        mo_kpti = np.dot(ao_kpti, mo_coeff[ki]).T
        mo_kptj = np.dot(ao_kptj, mo_coeff[kj]).T
        mo_pairs = np.einsum('ig,jg->ijg', mo_kpti.conj()*fac, mo_kptj)
        mo_pairs_G = tools.fft(mo_pairs.reshape(-1,ngrids), mydf.mesh)
        mo_pairs = None
        mo_pairs_G*= wcoulG



        v = tools.ifft(mo_pairs_G, mydf.mesh)
        v *= fac.conj()
        v = v.reshape(nmo,nmo,ngrids)

        mo_pairs = np.einsum('ig,jg->ijg', mo_kpti.conj(), mo_kptj)
        mo_pairs_G = tools.fft(mo_pairs.reshape(-1,ngrids)*fac, mydf.mesh)

        off = ki * nkpts + kj
        ijR.write(off*idx_ooG.size+idx_ooG, mo_pairs[:nocc,:nocc].ravel())
        iaR.write(off*idx_ovG.size+idx_ovG, mo_pairs[:nocc,nocc:].ravel())
        aiR.write(off*idx_ovG.size+idx_ovG, mo_pairs[nocc:,:nocc].ravel())
        abR.write(off*idx_vvG.size+idx_vvG, mo_pairs[nocc:,nocc:].ravel())

        off = kj * nkpts + ki
        mo_pairs = mo_pairs.transpose(1,0,2).conj()
        ijR.write(off*idx_ooG.size+idx_ooG, mo_pairs[:nocc,:nocc].ravel())
        iaR.write(off*idx_ovG.size+idx_ovG, mo_pairs[:nocc,nocc:].ravel())
        aiR.write(off*idx_ovG.size+idx_ovG, mo_pairs[nocc:,:nocc].ravel())
        abR.write(off*idx_vvG.size+idx_vvG, mo_pairs[nocc:,nocc:].ravel())

        mo_pairs = None
        mo_pairs_G*= wcoulG
        v = tools.ifft(mo_pairs_G, mydf.mesh)
        v *= fac.conj()
        v = v.reshape(nmo,nmo,ngrids)

        off = ki * nkpts + kj
        ijG.write(off*idx_ooG.size+idx_ooG, v[:nocc,:nocc].ravel())
        iaG.write(off*idx_ovG.size+idx_ovG, v[:nocc,nocc:].ravel())
        abG.write(off*idx_vvG.size+idx_vvG, v[nocc:,nocc:].ravel())

        off = kj * nkpts + ki
        v = v.transpose(1,0,2).conj()
        ijG.write(off*idx_ooG.size+idx_ooG, v[:nocc,:nocc].ravel())
        iaG.write(off*idx_ovG.size+idx_ovG, v[:nocc,nocc:].ravel())
        abG.write(off*idx_vvG.size+idx_vvG, v[nocc:,nocc:].ravel())



    cput1 = logger.timer("Generating ijG", *cput1)
    sym1 = ["+-+", [kpts,]*3, None, gvec]
    sym2 = ["+--", [kpts,]*3, None, gvec]

    ooG = tensor(ijG, sym1)
    ovG = tensor(iaG, sym1)
    vvG = tensor(abG, sym1)

    ooR = tensor(ijR, sym2)
    ovR = tensor(iaR, sym2)
    voR = tensor(aiR, sym2)
    vvR = tensor(abR, sym2)

    mycc.oooo = lib.einsum('ijg,klg->ijkl', ooG, ooR)/ nkpts
    mycc.ooov = lib.einsum('ijg,kag->ijka', ooG, ovR)/ nkpts
    mycc.oovv = lib.einsum('ijg,abg->ijab', ooG, vvR)/ nkpts
    ooG = ooR = ijG = ijR = None
    mycc.ovvo = lib.einsum('iag,bjg->iabj', ovG, voR)/ nkpts
    mycc.ovov = lib.einsum('iag,jbg->iajb', ovG, ovR)/ nkpts
    ovR = iaR = voR = aiR = None
    mycc.ovvv = lib.einsum('iag,bcg->iabc', ovG, vvR)/ nkpts
    ovG = iaG = None
    mycc.vvvv = lib.einsum('abg,cdg->abcd', vvG, vvR)/ nkpts
    cput1 = logger.timer("integral transformation", *cput1)
    logger.timer("ao2mo transformation", *cput0)
    return mycc

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    import os

    from pyscf.pbc.mp.kmp2 import padded_mo_coeff, padding_k_idx
    from cc_sym.kccsd_rhf import KRCCSD, Logger
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
    cell.verbose = 0
    cell.mesh = [5,5,5]
    cell.build()

    kpts = cell.make_kpts([1,1,3])
    mf = scf.KRHF(cell,kpts, exxdiv=None)
    chkfile = 'dmd_113.chk'
    if os.path.isfile(chkfile):
        mf.__dict__.update(scf.chkfile.load(chkfile, 'scf'))
    else:
        mf.chkfile = chkfile
        mf.kernel()

    kpts = cell.make_kpts([3,3,3])
    nkpts = len(kpts)
    nao = cell.nao_nr()
    mf = scf.KRHF(cell, kpts, exxdiv=None)
    mf.mo_coeff = np.random.random([nkpts, nao, nao])
    mf.mo_occ = np.zeros([nkpts,nao])
    mf.mo_occ[:,:4] = 2
    #mf.exxdiv= None
    #mf.with_df.exxdiv=None
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mf.verbose=5
    import time
    mycc = cc.KRCCSD(mf)
    mycc.verbose=5
    mycc = ao2mo(mycc)
    #cput1 = logger.timer("new scheme", *cput1)

    ref = KRCCSD(mf)
    ref.verbose=5
    eris = ref.ao2mo()
    #print((eris.oooo-mycc.oooo).norm())
    #print((eris.ooov-mycc.ooov).norm())
    #print((eris.oovv-mycc.oovv).norm())
    #print((eris.ovvv-mycc.ovvv).norm())
    #print((eris.vvvv-mycc.vvvv).norm())
    #print((eris.ovov-mycc.ovov).norm())
    #print((eris.ovvo-mycc.ovvo).norm())
