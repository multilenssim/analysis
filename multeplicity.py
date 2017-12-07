import iter_analysis as ia
import h5py,glob,argparse
import numpy as np
import os
import time

def track_hist(fname,ev):
    with h5py.File(fname,'r') as f:
        ks_par = []
        i_idx = 0
        for ix in xrange(ev):
            f_idx = f['idx_tr'][ix]
            hit_pos = f['coord'][0,i_idx:f_idx,:]
            means = f['coord'][1,i_idx:f_idx,:]
            sigmas = f['sigma'][i_idx:f_idx]
            i_idx = f_idx
            tr_dist, er_dist = ia.track_dist_threaded(hit_pos, means, sigmas, False, f['r_lens'][()])
            err_dist = 1./np.asarray(er_dist)
            ks_par.append(ia.make_hist(bn_arr, tr_dist, err_dist))
            print('Event ' + str(ix) + ' of ' + str(ev))
    return ks_par


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='insert path-to-file with seed location')
    parser.add_argument("-f", "--file", help="use file instead of path", action="store_true")
    args = parser.parse_args()
    path = args.path
    n_ev = 10
    max_val = 2000
    bin_width = 10
    n_bin = max_val/bin_width
    bn_arr = np.linspace(0,max_val,n_bin)
    start_time = time.time()
    if args.file:
        print path
        chi2_arr = track_hist(path, n_ev)
        n_null = [np.mean(chi2_arr, axis=0), np.std(chi2_arr, axis=0)]
        hist = ia.chi2(n_null, chi2_arr)
        np.savetxt('one-particle',(hist))
    else:
        for fname in sorted(glob.glob(path+'*sim.h5')):
            print fname
            chi2_arr = track_hist(fname,n_ev)
            n_null = [np.mean(chi2_arr, axis=0), np.std(chi2_arr, axis=0)]
            if fname[len(path)] == 'e':
                e_hist = ia.chi2(n_null,chi2_arr)
            elif fname[len(path)] == 'g':
                g_hist = ia.chi2(n_null,chi2_arr)		# n_null will not exist??
        new_path = os.path.join(os.path.split(path)[0],os.path.split(path)[1])
        np.savetxt(new_path+'-electron-gammac2',(e_hist,g_hist))
    print('Time: ' + str(time.time() - start_time))
