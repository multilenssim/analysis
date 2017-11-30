from sklearn.cluster import DBSCAN
import numpy as np
import h5py

def e_clusterize(fname, ev, perc):
        arr_dist = []
        with h5py.File(fname,'r') as f:
                i_idx = 0
		avg_ptn = np.mean(np.diff(f['idx_depo'][:]))
                for ix in xrange(ev):
                        f_idx = f['idx_depo'][ix]
                        vert = f['en_depo'][i_idx:f_idx,:]
                        i_idx = f_idx
			arr_dist.append(max(np.linalg.norm(vert-np.mean(vert,axis=0),axis=1)))
	ks_bin = np.linspace(0,max(arr_dist),10000)
	hist = np.histogram(arr_dist,bins=ks_bin)[0].astype(float)
	mask = np.cumsum(hist/sum(hist))>perc
	ks_bin = ks_bin[:-1]
	return ks_bin[mask][0], np.sqrt(avg_ptn)

def g_clusterize(fname, ev, e_clust, sigma_e):
        i = 0
	arr = []
        from sklearn.cluster import DBSCAN
        with h5py.File(fname,'r') as f:
                i_idx = 0
		arr_ptn = np.append(f['idx_depo'][0],np.diff(f['idx_depo'][:]))
                for ix,en in zip(xrange(ev),arr_ptn):
                        f_idx = f['idx_depo'][ix]
                        vert = f['en_depo'][i_idx:f_idx,:]
                        i_idx = f_idx
                        db = DBSCAN(eps=3, min_samples=10).fit(vert)
                        label =  db.labels_
                        labels = label[label!=-1]
                        vert = vert[label!=-1]
                        unique, counts = np.unique(labels, return_counts=True)
                        main_cluster = vert[labels==unique[np.argmax(counts)],:]
			vert_incl = vert[np.linalg.norm(vert-np.mean(main_cluster,axis=0),axis=1)<e_clust]
			arr.append(vert_incl.shape[0]/float(en))
			if not (vert_incl.shape[0]+sigma_e)/float(en)>1:
				i += 1
        return float(i)/float(ev),np.mean(arr)

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

n_ev = 500
for perc in [0.8,0.9,0.99]:
	rad,sgm = e_clusterize('/farmshare/user_data/jdalmass/cfSam1_K4_8/raw_data/r0-1_2e-_sim.h5',n_ev,perc)
	print '%ith percentile electron energy deposit radius (r_e): %.2fmm'%(int(perc*100),rad)
	frc,e_dp = g_clusterize('/farmshare/user_data/jdalmass/cfSam1_K4_8/raw_data/r0-1_2gamma_sim.h5',n_ev,rad,sgm)
	print 'fraction of gamma depositing more than one sigma of their total energy outside r_e: %.2f'%frc
	print 'average energy of the gamma deposited in r_e (fraction): %.2f'%e_dp
	print '---------------------------------------------------------------------------------------------------------------------------------------'
