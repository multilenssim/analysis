import matplotlib.patches as patches
import matplotlib.pyplot as plt
import h5py, glob, argparse
import numpy as np
import time
import multiprocessing

def process_timer(funct):
    import time

    def wrapper(*args,**kwargs):
        start_time = time.time()
        result = funct(*args,**kwargs)
        print 'function %s executed in %.2f'%(funct.__name__,time.time() - start_time)
        return result

    return wrapper


def remove_nan(dist,ofst_diff,drct):
    #if np.isnan(dist).any():
    idx = np.where(np.isnan(dist))[0]
    dist[idx] = 0	#np.absolute(np.cross(ofst_diff[idx],drct[idx]))/np.linalg.norm(drct[idx],axis=1).reshape(-1,1)
    return dist

def syst_solve(drct,r_drct,ofst_diff):
    s_a = np.einsum('ij,ij->i',drct,drct)
    s_b = np.einsum('ij,ij->i',r_drct,r_drct)
    d_dot = np.einsum('ij,ij->i',drct,r_drct)
    q1 = np.einsum('ij,ij->i',-ofst_diff,drct)
    q2 = np.einsum('ij,ij->i',-ofst_diff,r_drct)
    matr = np.stack((np.vstack((s_a,-d_dot)).T,np.vstack((d_dot,-s_b)).T),axis=1)

    if any(np.linalg.det(matr)==0):
        matr[np.linalg.det(matr)==0] = np.identity(2)

    qt = np.vstack((q1,q2)).T
    return np.linalg.solve(matr,qt)

def roll_funct(ofst,drct,sgm,i,half=False,outlier=False):
    pt = []

    if not any(sgm):
        sgm = np.ones(len(drct))

    r_ofst = np.roll(ofst,i,axis=0)
    r_drct = np.roll(drct,i,axis=0)
    r_sgm = np.roll(sgm,i,axis=0)

    if half:
        ofst = ofst[:i]
        drct = drct[:i]
        sgm = sgm[:i]
        r_ofst = r_ofst[:i]
        r_drct = r_drct[:i]
        r_sgm = r_sgm[:i]

    ofst_diff = ofst - r_ofst
    b_drct = np.cross(drct,r_drct)
    norm_d = b_drct/np.linalg.norm(b_drct,axis=1).reshape(-1,1)
    dist = np.absolute(np.einsum('ij,ij->i',ofst_diff,norm_d))
    dist = remove_nan(dist,ofst_diff,drct)
    sm = np.stack((sgm,r_sgm),axis=1)
    multp = syst_solve(drct,r_drct,ofst_diff)
    multp[np.where(multp==0)] = 1
    sigmas = np.linalg.norm(np.einsum('ij,ij->ij',sm,multp),axis=1)

    if outlier:
        r = np.einsum('ij,i->ij',drct,multp[:,0]) + ofst
        s = np.einsum('ij,i->ij',r_drct,multp[:,1]) + r_ofst
        c_point = np.mean(np.asarray([r,s]),axis=0)
        idx_arr = np.where(np.linalg.norm(c_point,axis=1)>7000)[0]
        dist = np.delete(dist,idx_arr)
        sigmas = np.delete(sigmas,idx_arr)

    return dist,sigmas

@process_timer
def track_dist(ofst,drct,sgm=False,outlier=False,dim_len=0):
    half = ofst.shape[0]/2
    arr_dist, arr_sgm = [], []

    for i in range(1,(ofst.shape[0]-1)/2+1):
        dist,sigmas = roll_funct(ofst,drct,sgm,i,half=False,outlier=outlier)
        arr_dist.extend(dist)
        arr_sgm.extend(sigmas)

    if ofst.shape[0] & 0x1: pass						#condition removed if degeneracy is kept

    else:
        dist,sigmas = roll_funct(ofst,drct,sgm,half,half=False,outlier=outlier)
        arr_dist.extend(dist)
        arr_sgm.extend(sigmas)

    if not np.array_equal(sgm,False):
        return arr_dist,(np.asarray(arr_sgm)+dim_len)

    else:
        return arr_dist,np.full(len(arr_dist),dim_len)


def make_hist(bn_arr,arr,c_wgt=None,norm=True):

    if np.array_equal(c_wgt,None): c_wgt = np.ones(len(arr))

    if norm: return np.histogram(arr,bn_arr,weights=c_wgt,density=True)[0]

    else: return np.histogram(arr,bn_arr,weights=c_wgt)[0]


def track_util(f_name,ev,tag):
    ks_par = []
    i_idx = 0

    with h5py.File(f_name,'r') as f:

        for i in xrange(ev):
            f_idx = int(f['idx'][i])
            hit_pos = f['coord'][0,i_idx:f_idx,:]
            means = f['coord'][1,i_idx:f_idx,:]
            sigmas = f['sigma'][i_idx:f_idx]
            i_idx = f_idx
            tr_dist, er_dist = track_dist(hit_pos,means,sgm=sigmas,outlier=outlier,dim_len=f['r_lens'][()])
            err_dist = 1./np.asarray(er_dist)

            if tag == 'avg':
                ks_par.append(np.average(tr_dist,weights=err_dist))

            elif tag == 'chi2':
                ks_par.append(make_hist(bn_arr,tr_dist,c_wgt=err_dist))

        print f_name

    return ks_par

	
def chi2(bkg_hist,chi2h):
    chi2h = np.asarray(chi2h)
    c2 = np.sum(np.square((chi2h - bkg_hist[0])/bkg_hist[1]),axis=1)/(len(bkg_hist[0])-1)
    return c2


def use_chi2(f_name,ev,n_null):
    chi2_arr = track_util(f_name,ev,'chi2')

    if f_name == ssite:
        n_null = [np.mean(chi2_arr,axis=0),np.std(chi2_arr,axis=0)]
        c2 = chi2(n_null,chi2_arr)
        return n_null,c2

    else:
        c2 = chi2(n_null,chi2_arr)
        return c2


def find_cl(ss_site,ms_site,cl):
    ix = np.abs(ms_site-1+cl).argmin()
    return ss_site[ix]

def thread_solve(ofs, i, sort_drct, sort_offset, sort_sigma, mask, sgm, dim_len):
    mask_dir = sort_drct[i:][mask[i:]].reshape(-1, 3)
    cross_dir = np.cross(sort_drct[i], mask_dir)
    norm_d = np.einsum('ij,i->ij', cross_dir, 1.0 / np.linalg.norm(cross_dir, axis=1))
    ofst_diff = ofs - sort_offset[i:][mask[i:]].reshape(-1, 3)
    dist = np.absolute(np.einsum('ij,ij->i', ofst_diff, norm_d))

    if not np.array_equal(sgm, False):
        dct_stack = np.tile(sort_drct[i], mask_dir.shape[0]).reshape(-1, 3)
        multp = syst_solve(dct_stack, mask_dir, ofst_diff)
        sig = np.stack((np.repeat(sort_sigma[i], mask_dir.shape[0]), sort_sigma[i:][mask[i:, 0]]), axis=1)
        sigma = np.linalg.norm(np.einsum('ij,ij->ij', sig, multp), axis=1)
        sigma = sigma + dim_len
    else:
        sigma = 1 * len(dist)
    return {'dist':dist, 'sigma':sigma}

THREADED = True

@process_timer
def new_track_dist(ofst,drct,sgm=False,outlier=False,dim_len=0):
    start = time.time()
    arr_dist, arr_sgm = [], []
    _, lns_idx  = np.unique(ofst,return_inverse=True,axis=0)
    sort_idx    = np.argsort(lns_idx)
    sort_offset = ofst[sort_idx]
    sort_drct   = drct[sort_idx]

    if not np.array_equal(sgm,False): sort_sigma = sgm[sort_idx]

    fist_offset = sort_offset[0]
    mask        = sort_offset != fist_offset 

    if THREADED:
        pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)
        print('Thread pool: %d' % (multiprocessing.cpu_count() // 2))
        thread_results = []

    for i,ofs in enumerate(sort_offset):
        if np.array_equal(ofs,fist_offset):
            pass

        else:
            fist_offset = ofs
            mask = sort_offset != ofs

        if THREADED:
            result = pool.apply_async(thread_solve, (ofs, i, sort_drct, sort_offset, sort_sigma, mask, sgm, dim_len))
            thread_results.append(result)

        else:
            dist, sgm = thread_solve(ofs, i, sort_drct, sort_offset, sort_sigma, mask, sgm, dim_len)
            arr_dist.extend(dist)
            arr_sgm.extend(sgm)

        if i % 1000 == 0:
            print('Time at %d is %f secs' % (i, time.time() - start))

    if THREADED:
        childs = multiprocessing.active_children()
        print('Child count: ' + str(len(childs)))
        pool.close()
        pool.join()
        childs = multiprocessing.active_children()
        print('Child count: ' + str(len(childs)))
        print('Result count: %d' % len(thread_results))
        for result in thread_results:
            ds_dict = result.get()
            arr_dist.extend(ds_dict['dist'])
            arr_sgm.extend(ds_dict['sigma'])

    return arr_dist,arr_sgm


if __name__ == '__main__':
    max_val = 2000
    bin_width = 10
    n_bin = max_val/bin_width
    step = 2
    sample = 250
    sgm = True
    outlier = False
    bn_arr = np.linspace(0,max_val,n_bin)
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='insert path-to-file with seed location')
    args = parser.parse_args()
    path = args.path
    ssite = path+'s-site.h5'
    n_null,c2_sgn = use_chi2(ssite,sample,None)
    val_avg,val_c2 = [],[]

    for fname in sorted(glob.glob(path+'*cm.h5')):
        c2_bkg = use_chi2(fname,step*sample,n_null)

        for spt_c2 in np.split(c2_bkg,step):
            f_bin_c2 = np.amax([c2_sgn,spt_c2])
            bin_c2 = np.linspace(0,f_bin_c2,200)
            b = find_cl(1-np.cumsum(make_hist(bin_c2,spt_c2)),np.cumsum(make_hist(bin_c2,c2_sgn)),0.2)
            val_c2.append(b)

    path = os.path.join(os.path.split(path)[0][:-8],os.path.split(path)[1])
    np.savetxt(path+'datapoints',val_c2)
