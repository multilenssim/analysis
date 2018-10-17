import sys, h5py, glob, argparse, pickle, os, time
sys.path.insert(0, '/home/jacopo/simulation/')
import iter_analysis as ia
import numpy as np
import paths

def track_hist(fname,ev):
    ks_par = []
    i_idx = 0

    for ix in xrange(ev):
        if ix%50 == 0: print ix

        with h5py.File(fname,'r') as f:
            start_time = time.time()
            f_idx = f['idx_tr'][ix]
            hit_pos = f['coord'][0,i_idx:f_idx,:]
            means = f['coord'][1,i_idx:f_idx,:]
            sigmas = f['sigma'][i_idx:f_idx]
            r_lens = f['r_lens'][()]
        n_ph = f_idx - i_idx
        i_idx = f_idx
        print 'opening the file %.2f with %i photons'%(time.time() - start_time,n_ph)
        tr_dist, er_dist = ia.new_track_dist(hit_pos, means, sigmas, False, r_lens)
        start_time = time.time()
        err_dist = 1./np.asarray(er_dist)
        ks_par.append(ia.make_hist(bn_arr, tr_dist, err_dist))
        print 'making the histogram %.2f'%(time.time() - start_time)
    return ks_par


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('key', help='insert the key to the dictonary')
    args = parser.parse_args()
    ix = int(args.key)
    n_ev = 300
    max_val = 2000
    bin_width = 10
    n_bin = max_val/bin_width
    bn_arr = np.linspace(0,max_val,n_bin)
    configs_pickle_file = '%sconf_file_obj.pickle' % paths.detector_config_path

    with open(configs_pickle_file,'r') as f:
        config_list = pickle.load(f)

    key = config_list.keys()[ix]
    print key
    path = paths.get_data_file_path(key)
    f_list = sorted(os.listdir(path))
    first = True

    for fl in [f_list[:len(f_list)/2],f_list[len(f_list)/2:]]:
        electron = path+fl[0]
        gamma = path+fl[1]
        print electron, gamma
        e_chi2_arr = track_hist(electron,n_ev)
        g_chi2_arr = track_hist(gamma,n_ev)
        n_null = [np.mean(e_chi2_arr,axis=0),np.std(e_chi2_arr,axis=0)]
        e_hist = ia.chi2(n_null,e_chi2_arr)
        g_hist = ia.chi2(n_null,g_chi2_arr)

        if first:
            seed = 'r0-1'
            first = False

        else:
            seed = 'r3-4'

        np.savetxt('%s%selectron-gamma_perf_c2'%(paths.get_data_file_path_no_raw(key),seed),(e_hist,g_hist))
