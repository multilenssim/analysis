import matplotlib.patches as patches
import matplotlib.pyplot as plt
import h5py, glob, argparse
import numpy as np


def remove_nan(dist,ofst_diff,drct):
	if np.isnan(dist).any():
		idx = np.where(np.isnan(dist))[0]
		dist[idx] = np.absolute(np.cross(ofst_diff[idx],drct[idx]))/np.linalg.norm(drct[idx],axis=1).reshape(-1,1)
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

def track_dist(ofst,drct,sgm=False,outlier=False,dim_len=0):
	half = ofst.shape[0]/2
	arr_dist, arr_sgm,plot_test = [], [], []
	for i in range(1,(ofst.shape[0]-1)/2+1):
		dist,sigmas = roll_funct(ofst,drct,sgm,i,half=False,outlier=outlier)
		arr_dist.extend(dist)
		arr_sgm.extend(sigmas)
	if ofst.shape[0] & 0x1: pass						#condition removed if degeneracy is kept
	else:
		dist,sigmas = roll_funct(ofst,drct,sgm,half,half=False,outlier=outlier)
		arr_dist.extend(dist)
		arr_sgm.extend(sigmas)
	if any(sgm):
		return arr_dist,(np.asarray(arr_sgm)+dim_len)
	else:
		return arr_dist

def make_hist(bn_arr,arr,c_wgt=None,norm=True):
	if c_wgt == None:
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

def track_util(f_name,e_idx,tag,counter,reindex):
	ks_par = []
	lg_arr = len(e_idx)
	if reindex:
		lg_arr = sample
	for i in range(lg_arr):
		i_idx = e_idx[counter*sample+i-1]
		f_idx = e_idx[counter*sample+i]
		if i == 0 and counter == 0:
			i_idx = 0
		with h5py.File(f_name,'r') as f:
			hit_pos = f['coord'][0,i_idx:f_idx,:]
			means = f['coord'][1,i_idx:f_idx,:]
			sigmas = f['sigma'][i_idx:f_idx]
			tr_dist, er_dist = track_dist(hit_pos,means,sgm=sigmas,outlier=outlier,dim_len=f['r_lens'][()])
			err_dist = 1./np.asarray(er_dist)
			if tag == 'avg':
				ks_par.append(np.average(tr_dist,weights=err_dist))
			elif tag == 'chi2':
				ks_par.append(make_hist(bn_arr,tr_dist,c_wgt=err_dist))
	return ks_par
	
def chi2(bkg_hist,chi2h):
	chi2h = np.asarray(chi2h)
	c2 = np.sum(np.square((chi2h - bkg_hist[0])/bkg_hist[1]),axis=1)/(len(bkg_hist[0])-1)
	return c2

def use_chi2(f_name,e_idx,n_null,counter,reindex):
	chi2_arr = track_util(f_name,e_idx,'chi2',counter,reindex)
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


if __name__ == '__main__':
	max_val = 2000
	bin_width = 10
	n_bin = max_val/bin_width
	step = 5
	sample = 60
	sgm = True
	outlier = False
	bn_arr = np.linspace(0,max_val,n_bin)
        parser = argparse.ArgumentParser()
        parser.add_argument('path', help='insert path-to-file with seed location')
        args = parser.parse_args()
        path = args.path
	ssite = path+'s-site.h5'
	with h5py.File(ssite,'r') as f:
		b_idx = f['idx'][:60]
	avg_sgn = track_util(ssite,b_idx,'avg',0,False)
	n_null,c2_sgn = use_chi2(ssite,b_idx,0,0,False)
	val_avg,val_c2 = [],[]
	for fname in sorted(glob.glob(path+'*cm.h5')):
		print fname
		with h5py.File(fname,'r') as f:
			e_index = f['idx'][:step*sample]
		for i in range(step):
			avg_bkg = track_util(fname,e_index,'avg',i,True)
			i_bin_avg = min(min(avg_sgn),min(avg_bkg))
			f_bin_avg = max(max(avg_sgn),max(avg_bkg))
			bin_avg = np.linspace(i_bin_avg,f_bin_avg,50)
			c2_bkg = use_chi2(fname,e_index,n_null,i,True)
			f_bin_c2 = np.amax([c2_sgn,c2_bkg])
			bin_c2 = np.linspace(0,f_bin_c2,50)
			a = find_cl(np.cumsum(make_hist(bin_avg,avg_sgn)),np.cumsum(make_hist(bin_avg,avg_bkg)),0.95)
			b = find_cl(np.cumsum(make_hist(bin_c2,c2_sgn)),np.cumsum(make_hist(bin_c2,c2_bkg)),0.95)
			val_avg.append(a)
			val_c2.append(b)
	np.savetxt(path+'datapoints',val_c2)
