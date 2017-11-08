import iter_analysis as ia
import h5py,glob,argparse
import numpy as np
import time

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, TimeoutError
import multiprocessing		# Just for CPU count

def clusterize(fname, ev):
	lb = []
	from sklearn.cluster import DBSCAN
	with h5py.File(fname,'r') as f:
		i_idx = 0
		for ix in xrange(ev):
			f_idx = f['idx_depo'][ix]
			vert = f['en_depo'][i_idx:f_idx,:]
			i_idx = f_idx
			db = DBSCAN(eps=3, min_samples=10).fit(vert)
			labels =  db.labels_
			labels = labels[labels!=-1]
			lb.append(max(labels))
	return lb

def plot_cluster():
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(vert[:,0],vert[:,1],vert[:,2],'.',color='red')
	plt.show()
	plt.close()
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(set(labels)))]
	for lb,col in zip(set(labels),colors):
		ax.plot(vert[labels==lb][:,0],vert[labels==lb][:,1],vert[labels==lb][:,2],'.',markerfacecolor=tuple(col))
	plt.show()

def pair_tracks(fname, i_idx, f_idx):
	print("pair_tracks parameters: " + fname + ' ' + str(i_idx) + ' ' + str(f_idx))
	try:
		with h5py.File(fname, 'r') as f:		# Expensive to open this each time, but I think it's the only way with processes
			hit_pos = f['coord'][0, i_idx:f_idx, :]
			means = f['coord'][1, i_idx:f_idx, :]
			sigmas = f['sigma'][i_idx:f_idx]
			tr_dist, er_dist = ia.track_dist(hit_pos, means, sigmas, False, f['r_lens'][()])
			err_dist = 1. / np.asarray(er_dist)
	except Exception as e:
		print('Exception pairing tracks: ' + str(e))
	return ia.make_hist(bn_arr, tr_dist, err_dist)


def track_hist(fname, ev, pool, result_list):
	with h5py.File(fname, 'r') as f:
		indices = f['idx_tr'][()]  # See: https://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py
	start = time.time()
	ks_par = []
	for ix in xrange(ev):
		i_idx = 0 if ix == 0 else indices[ix-1]
		result = pool.apply_async(pair_tracks, (fname, i_idx, indices[ix]))
		result_list.append(result)
		# ks_par.append(hist)
	return ks_par		# This is now unused


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('path', help='insert path-to-file with seed location')
	args = parser.parse_args()
	path = args.path
	n_ev = 5 # 500
	max_val = 2000
	bin_width = 10
	n_bin = max_val/bin_width
	bn_arr = np.linspace(0,max_val,n_bin)

	pool = Pool(multiprocessing.cpu_count())
	results_e = []
	results_gamma = []
	for fname in sorted(glob.glob(path + '*sim.h5')):
		print fname
		result_list = results_e if fname[len(path)] == 'e' else results_gamma
		chi2_arr = track_hist(fname, n_ev, pool, result_list)
	e_chi = []
	gamma_chi = []
	for result in results_e:
		chi2_arr = result.get()
		e_chi.append(chi2_arr)
	for result in results_gamma:
		chi2_arr = result.get()
		gamma_chi.append(chi2_arr)

	n_null = [np.mean(e_chi, axis=0), np.std(e_chi, axis=0)]
	e_hist = ia.chi2(n_null, e_chi)
	g_hist = ia.chi2(n_null, gamma_chi)
	np.savetxt(path + 'electron-gammac2', (e_hist, g_hist))

	pool.close()
	pool.join()