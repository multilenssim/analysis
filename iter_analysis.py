import h5py, glob, argparse
import numpy as np

#from numba import jit
import time

from Queue import Queue
from Queue import Empty
from threading import Thread
import multiprocessing
import os
#import line_profiler

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, TimeoutError

# See: https://www.metachris.com/2016/04/python-threadpool/
# And: https://stackoverflow.com/questions/3033952/threading-pool-similar-to-the-multiprocessing-pool/7257510#7257510 [Copied this one]
# And: https://stackoverflow.com/questions/1886090/return-value-from-thread

# Using built in thread pools:  https://stackoverflow.com/questions/26104512/how-to-obtain-the-results-from-a-pool-of-threads-in-python

class Worker(Thread):
	"""Thread executing tasks from a given tasks queue"""
	def __init__(self, tasks, results):
		Thread.__init__(self)
		self.tasks = tasks
		self.results = results
		self.daemon = True
		self.start()

	def run(self):
		while True:
			func, args, kargs = self.tasks.get()
			try:
				result = func(*args, **kargs)
				self.results.put(result)		# Put the results onto the queue
			except Exception as e:
				print e
			finally:
				self.tasks.task_done()

class ThreadPool:
	"""Pool of threads consuming tasks from a queue"""
	def __init__(self, num_threads):
		self.tasks = Queue(num_threads)
		self.results = Queue()				# Thsi is really a misuse of queues!!  I think
		for _ in range(num_threads): Worker(self.tasks, self.results)

	def add_task(self, func, *args, **kargs):
		"""Add a task to the queue"""
		self.tasks.put((func, args, kargs))

	def get_results_queue(self):
		return self.results

	def wait_completion(self):
		"""Wait for completion of all the tasks in the queue"""
		self.tasks.join()

#@jit(nopython=True)
def my_any(iterable):
	for it in iterable:
		if it:
			return True
	return False

# Uses a different distance computation for Nan = parallel lines
def remove_nan(dist,ofst_diff,drct):
	if np.isnan(dist).any():
		idx = np.where(np.isnan(dist))[0]
		im_05 = np.cross(ofst_diff[idx],drct[idx])
		# print('NaN distance vectors: ' + str(ofst_diff[idx]) + ' ' + str(drct[idx]) + ' ' + str(im_05))
		im_1 = np.linalg.norm(im_05, axis=1)
		im_2 = np.linalg.norm(drct[idx],axis=1)
		im_3 = im_1 / im_2
		dist[idx] = im_3
	return dist
	'''
	if np.isnan(dist).any():
		idx = np.where(np.isnan(dist))[0]
		cross = np.cross(ofst_diff[idx],drct[idx])
		norm = np.linalg.norm(drct[idx],axis=1).reshape(-1,1)
		dist_er = cross / norm
		dist[int(idx)] = dist_er
	return dist
	'''

#@jit(nopython=True, nogil=True)		# Numba can only solve 2-D arrays
def solver(matrix, qt):
	return np.linalg.solve(matrix,qt)

#@jit(nopython=True, nogil=True)		# Jit performance is a little worse - 15 secs - non jit is horrendous - 10x worse!!!
def zerodet(matrix):
	return np.linalg.det(matrix) == 0

# Baseline perf is about 6-8 seconds
#@profile
#@jit()
def syst_solve(drct,r_drct,ofst_diff):
	s_a = np.einsum('ij,ij->i', drct, drct)
	s_b = np.einsum('ij,ij->i', r_drct, r_drct)
	d_dot = np.einsum('ij,ij->i', drct, r_drct)
	q1 = np.einsum('ij,ij->i', -ofst_diff, drct)
	q2 = np.einsum('ij,ij->i', -ofst_diff, r_drct)
	matr = np.stack((np.vstack((s_a,-d_dot)).T,np.vstack((d_dot,-s_b)).T),axis=1)
	'''
	for index, submat in enumerate(matr):
		if zerodet(submat):
		#if np.linalg.det(submat) == 0:
			matr[index] = np.identity(2)
	# det_replacer(matr)
	'''

	#dets = np.linalg.det(matr)
	# This is the baseline - seems like it computes the all the determinants twice??
	if any(np.linalg.det(matr) == 0):
		matr[np.linalg.det(matr) == 0] = np.identity(2)

	'''
	dets = np.linalg.det(matr)
	for it in (dets==0):
			matr[np.linalg.det(matr)==0] = np.identity(2)
			break
	'''
	qt = np.vstack((q1,q2)).T
	return solver(matr,qt)

#@jit()
#@profile
def roll_funct(ofst,drct,sgm,i,half=False,outlier=False):
	#print('.... Rolling')
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
	#print('...... Solving')
	multp = syst_solve(drct,r_drct,ofst_diff)
	#print('...... Solved')
	multp[np.where(multp==0)] = 1
	sigmas = np.linalg.norm(np.einsum('ij,ij->ij',sm,multp),axis=1)
	if outlier:
		r = np.einsum('ij,i->ij',drct,multp[:,0]) + ofst
		s = np.einsum('ij,i->ij',r_drct,multp[:,1]) + r_ofst
		c_point = np.mean(np.asarray([r,s]),axis=0)
		idx_arr = np.where(np.linalg.norm(c_point,axis=1)>7000)[0]
		dist = np.delete(dist,idx_arr)
		sigmas = np.delete(sigmas,idx_arr)
	#print('.... Rolled')
	# KW mods
	idx = np.where(dist < (sigmas / 5.))
	relevant_dists = dist[idx]
	print('Sizes: ' + str(len(ofst)) + ' ' + str(len(dist)) + ' ' + str(len(relevant_dists)))
	print('Distances: ' + str(relevant_dists))
	print('Sigmas: ' + str(sigmas[idx]))
	return dist,sigmas

#@jit()
#@profile
# 			tr_dist, er_dist = ia.track_dist(hit_pos, means, sigmas, False, f['r_lens'][()])
'''
ofst:	detector hit position matrix
drct:	detector direction matrix
sgm:	sigmas
outlier:	
dim_len:	lens radius
'''
def track_dist(ofst,drct,sgm=False,outlier=False,dim_len=0):
	half = ofst.shape[0]/2
	arr_dist, arr_sgm,plot_test = [], [], []
	count = (ofst.shape[0]-1)/2+1
	start_time = time.time()
	for i in range(1,(ofst.shape[0]-1)/2+1):
		print('Pairing hit: ' + str(i) + ' of ' + str(count))
		dist,sigmas = roll_funct(ofst,drct,sgm,i,half=False,outlier=outlier)
		arr_dist.extend(dist)
		arr_sgm.extend(sigmas)
		print(time.time() - start_time)
	if ofst.shape[0] & 0x1:
		pass						#condition removed if degeneracy is kept
	else:
		dist,sigmas = roll_funct(ofst,drct,sgm,half,half=False,outlier=outlier)
		arr_dist.extend(dist)
		arr_sgm.extend(sigmas)
	if any(sgm):
		return arr_dist,(np.asarray(arr_sgm)+dim_len)
	else:
		return arr_dist

def track_dist_threaded(ofst,drct,sgm=False,outlier=False,dim_len=0):
	half = ofst.shape[0]/2
	arr_dist, arr_sgm,plot_test = [], [], []

	pool = Pool(multiprocessing.cpu_count())
	results = []
	for i in range(1,(ofst.shape[0]-1)/2+1):
		result = pool.apply_async(roll_funct, (ofst,drct,sgm,i)) # ,half=False,outlier=outlier))
		results.append(result)
	childs = multiprocessing.active_children()
	print('Child count: ' + str(len(childs)))
	pool.close()
	childs = multiprocessing.active_children()
	print('Child count: ' + str(len(childs)))
	pool.join()
	childs = multiprocessing.active_children()
	print('Child count: ' + str(len(childs)))
	for result in results:
		result_value = result.get()
		arr_dist.extend(result_value[0])
		arr_sgm.extend(result_value[1])

	if ofst.shape[0] & 0x1:
		pass						#condition removed if degeneracy is kept
	else:
		dist,sigmas = roll_funct(ofst,drct,sgm,half,half=False,outlier=outlier)
		arr_dist.extend(dist)
		arr_sgm.extend(sigmas)
	if any(sgm):
		return arr_dist,(np.asarray(arr_sgm)+dim_len)
	else:
		return arr_dist

def make_hist(bn_arr,arr,c_wgt=None,norm=True):
	if c_wgt is None:
		c_wgt = np.ones(len(arr))
	wgt = []
	np_double = np.asarray(arr)
	bn_wdt = bn_arr[1] - bn_arr[0]
	for bn in bn_arr:
		wgt.extend(np.dot([(np_double>=bn) & (np_double<(bn+bn_wdt))],c_wgt))
	if norm:
		return np.asarray(wgt)/abs(sum(wgt))
	else:
		return np.asarray(wgt)

def track_util(f_name,ev,tag):
	print('File start: ' + f_name)
	start = time.time()
	ks_par = []
	i_idx = 0
	with h5py.File(f_name,'r') as f:
		print(str(ev) + ' sub events to process')
		for i in xrange(ev):
			f_idx = int(f['idx'][i])
			hit_pos = f['coord'][0,i_idx:f_idx,:]
			means = f['coord'][1,i_idx:f_idx,:]
			sigmas = f['sigma'][i_idx:f_idx]
			i_idx = f_idx
			#print('Computing distance')
			tr_dist, er_dist = track_dist(hit_pos,means,sgm=sigmas,outlier=outlier,dim_len=f['r_lens'][()])
			#print(' .. computed')
			err_dist = 1./np.asarray(er_dist)
			if tag == 'avg':
				ks_par.append(np.average(tr_dist,weights=err_dist))
			elif tag == 'chi2':
				ks_par.append(make_hist(bn_arr,tr_dist,c_wgt=err_dist))
			print(str('Sub-event ' + str(i) + ' ' + str(time.time() - start)))

	print('File done: ' + f_name)
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

def compute_distro(dir, filename, step, sample, c2_sgn, n_null):		# These params are a mess - clean up
	c2_bkg = use_chi2(dir+filename, step * sample, n_null)
	value_c2 = []
	for spt_c2 in np.split(c2_bkg, step):
		f_bin_c2 = np.amax([c2_sgn, spt_c2])
		bin_c2 = np.linspace(0, f_bin_c2, 200)
		b = find_cl(1 - np.cumsum(make_hist(bin_c2, spt_c2)), np.cumsum(make_hist(bin_c2, c2_sgn)), 0.2)
		value_c2.append(b)
	return value_c2
	#np.savetxt(dir + 'datapoints'+'-'+ filename, value_c2)

if __name__ == '__main__':
	max_val = 2000
	bin_width = 10
	n_bin = max_val/bin_width
	step = 2
	sample = 2  #250
	sgm = True
	outlier = False
	bn_arr = np.linspace(0,max_val,n_bin)
	parser = argparse.ArgumentParser()
	parser.add_argument('path', help='insert path-to-file with seed location')
	args = parser.parse_args()
	_path = args.path
	ssite = _path+'s-site.h5'
	n_null,c2_sgn = use_chi2(ssite,sample,None)
	val_avg,val_c2 = [],[]

	# See: https://stackoverflow.com/questions/1886090/return-value-from-thread
	# Not sure adding results to a queue is rally the right thing??  Queues are largely for jabs
	pool = ThreadPool(multiprocessing.cpu_count())
	results = []
	threadit = False
	for _fname in sorted(glob.glob(_path+'*cm.h5')):
		filepath = os.path.normpath(_fname)
		filename = os.path.basename(filepath)
		dir = os.path.dirname(filepath) + '/'

		if threadit:
			pool.add_task(compute_distro, dir, filename, step, sample, c2_sgn, n_null)
		else:
			result = compute_distro(dir, filename, step, sample, c2_sgn, n_null)
			results.append(result)
	''' Jacopo's (changed) code:
	for fname in sorted(glob.glob(path+'*cm.h5')):
		c2_bkg = use_chi2(fname,step*sample,n_null)
		for spt_c2 in np.split(c2_bkg,step):
			f_bin_c2 = np.amax([c2_sgn,spt_c2])
			bin_c2 = np.linspace(0,f_bin_c2,200)
			b = find_cl(1-np.cumsum(make_hist(bin_c2,spt_c2)),np.cumsum(make_hist(bin_c2,c2_sgn)),0.2)
			val_c2.append(b)
	path = os.path.join(os.path.split(path)[0][:-8],os.path.split(path)[1])
	np.savetxt(path+'datapoints',val_c2)
	'''

	if threadit:
		pool.wait_completion()
		# Results need to be ordered!!
		resultq = pool.get_results_queue()
		while True:
			try:
				result = resultq.get(False)		# Not sure if this is the best way to hack this in
				results.append(result)
			except Empty as qe:
				break		# Is this the best way to deal with queues?
	np.savetxt(dir + 'datapoints', results)

	# Aggregate result in non threaded case
