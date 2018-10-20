import sys, h5py, glob, argparse, pickle, os, time
sys.path.insert(0, '/home/users/wkwells/Development/simulation/')
import iter_analysis as ia
import numpy as np
import psutil
import gc
import multiprocessing

import paths
from logger_lfd import logger

THREADED = True

# TODO: we are opening the file many times - but I don't think there is a way to avoid that... with threading
def process_event_thread(fname, ix, i_idx, bn_arr):
    start_time = time.time()
    with h5py.File(fname,'r') as f:
        f_idx = f['idx_tr'][ix]
        hit_pos = f['coord'][0,i_idx:f_idx,:]
        means = f['coord'][1,i_idx:f_idx,:]
        sigmas = f['sigma'][i_idx:f_idx]
        r_lens = f['r_lens'][()]
    n_ph = f_idx - i_idx
    i_idx = f_idx
    logger.info('Run %d.  Opening the file %s took %.2f secs with %i photons'%(ix, fname, time.time() - start_time, n_ph))
    tr_dist, er_dist = ia.new_track_dist(hit_pos, means, sigmas, False, r_lens)
    err_dist = 1./np.asarray(er_dist)
    hist = ia.make_hist(bn_arr, tr_dist, err_dist)
    logger.info('Run %d.  Histogram took' % (ix, fname, time.time() - start_time, n_ph))
    return hist

def track_hist(fname, ev, bn_arr):
    ks_par = []

    if THREADED:
        threads_allowed = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(threads_allowed)
        logger.info('Thread pool size: %d' % threads_allowed)
        thread_results = []

    indices = [0]
    with h5py.File(fname,'r') as f:
        indices.extend(f['idx_tr'][()])

    for ix in xrange(5): # ev):
        if ix%50 == 0:
            logger.info('Index: %d' % ix)

        start_time = time.time()
        total_time = start_time
        if THREADED:
            result = pool.apply_async(process_event_thread, (fname, ix, indices[ix], bn_arr))
            thread_results.append(result)

        else:
            histogram = process_event_thread(fname, ix, indices[ix], bn_arr)
            ks_par.append(histogram)

        #logger.info('Post histogram time:  %.2f'%(time.time() - total_time))
        #logger.info('Memory size: %d MB' % (process.memory_info().rss // 1000000))
        #tr_dist = None
        #er_dist = None
        #gc.collect()
        #logger.info('Memory size after gc(): %d MB' % (process.memory_info().rss // 1000000))
        #logger.info('Post GC time:  %.2f'%(time.time() - total_time))

        #logger.info('Making the histogram took %.2f secs'%(time.time() - start_time))
        #logger.info('Total time:  %.2f'%(time.time() - total_time))

    if THREADED:
        childs = multiprocessing.active_children()
        print('Child count: ' + str(len(childs)))
        pool.close()
        logger.info('Pool closed')
        pool.join()
        childs = multiprocessing.active_children()
        print('Child count: ' + str(len(childs)))
        print('Result count: %d' % len(thread_results))
        for result in thread_results:
            ks_par.append(result.get())

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
    _bn_arr = np.linspace(0,max_val,n_bin)
    configs_pickle_file = '%sconf_file_obj.pickle' % paths.detector_config_path

    logger.info('Loading config file: %s' % configs_pickle_file)

    with open(configs_pickle_file,'r') as f:
        config_list = pickle.load(f)

    key = config_list.keys()[ix]
    logger.info('Key: %s' % key)
    path = paths.get_data_file_path(key)
    f_list = sorted(os.listdir(path))
    first = True

    process = psutil.Process(os.getpid())

    for fl in [f_list[:len(f_list)/2],f_list[len(f_list)/2:]]:
        electron = path+fl[0]
        gamma = path+fl[1]
        logger.info('Files: %s %s' % (electron, gamma))

        logger.info('1 Memory size: %d MB' % (process.memory_info().rss // 1000000))

        e_chi2_arr = track_hist(electron,n_ev,_bn_arr)

        logger.info('2 Memory size: %d MB' % (process.memory_info().rss // 1000000))
        gc.collect()
        logger.info('2 Memory size after gc(): %d MB' % (process.memory_info().rss // 1000000))

        g_chi2_arr = track_hist(gamma,n_ev,_bn_arr)

        logger.info('3 Memory size: %d MB' % (process.memory_info().rss // 1000000))
        gc.collect()
        logger.info('3 Memory size after gc(): %d MB' % (process.memory_info().rss // 1000000))

        n_null = [np.mean(e_chi2_arr,axis=0),np.std(e_chi2_arr,axis=0)]
        e_hist = ia.chi2(n_null,e_chi2_arr)
        g_hist = ia.chi2(n_null,g_chi2_arr)

        if first:
            seed = 'r0-1'
            first = False

        else:
            seed = 'r3-4'

        np.savetxt('%s%selectron-gamma_perf_c2'%(paths.get_data_file_path_no_raw(key),seed),(e_hist,g_hist))
