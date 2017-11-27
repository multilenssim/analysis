import matplotlib.patches as patches
import argparse, os, pickle, glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
import iter_analysis as ia
import numpy as np

def set_style():
        rcParams['lines.markersize'] = 4
        rcParams['font.size'] = 20.0
        rcParams['figure.figsize'] = (12, 9)


def tick_function(X):
        V = np.asarray(map(str,100000/X.astype(int)))
        V[11::2] = ''
        return V

def norm_hist(ks_bin,arr):
	return np.histogram(arr,bins=ks_bin)[0].astype(float)/sum(np.histogram(arr,bins=ks_bin)[0])

def binning(c_sgn,c_bkg):
        last_bin = np.amax([c_sgn,c_bkg])
        n_bin = int(last_bin*125)
        return np.linspace(0,last_bin,n_bin)

def read_conf(path):
	with open(path+'/raw_data/conf.pkl','r') as infile:
		conf = pickle.load(infile)
		return conf


def calc_steps(x_value,y_value,detector_r,base_pixel):
        x_coord = np.asarray([x_value,np.roll(x_value,-1)]).T[:-1]
        y_coord = np.asarray([y_value,np.roll(y_value,-1)]).T[:-1]
        lat_area = 2*np.pi*detector_r*(y_coord[:,0]-y_coord[:,1])
        n_step = (lat_area/lat_area[-1]*base_pixel).astype(int)
        return n_step


def curved_surface2(detector_r=2.0, diameter = 2.5, nsteps=20,base_pxl=4):
    '''Builds a curved surface based on the specified radius. Origin is center of surface.'''
    if (detector_r < diameter/2.0):
        raise Exception('The Radius of the curved surface must be larger than diameter/2.0')
    shift1 = np.sqrt(detector_r**2 - (diameter/2.0)**2)
    theta1 = np.arctan(shift1/(diameter/2.0))
    angles1 = np.linspace(theta1, np.pi/2, nsteps)
    x_value = abs(detector_r*np.cos(angles1))
    y_value = detector_r-detector_r*np.sin(angles1)
    return calc_steps(x_value,y_value,detector_r,base_pxl)


def plot_cl(path):
        data = np.loadtxt(path+'electron-gammac2')
        c_sgn = data[0]
        c_bkg = data[1]
	ks_bin = binning(c_sgn,c_bkg)
	x_max = 4
	x_lim = [0,x_max]
	y_lim = [0,1]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        csgn = np.cumsum(norm_hist(ks_bin,c_sgn))
        cbkg = np.cumsum(norm_hist(ks_bin,c_bkg))
        ax1.plot(ks_bin, csgn,'r-')
        ax2.plot(ks_bin, 1-cbkg, 'b-')
        ax1.text(x_max/2,0.55,'efficiency at 95% bkg. rejection: '+str(np.around(ia.find_cl(csgn,cbkg,0.95),decimals=2))[:4]+', $\~\chi^2$<'+str(ia.find_cl(ks_bin,cbkg,0.95))[:4],fontsize=15,bbox={'facecolor':'green','alpha':0.3})
        ax1.text(x_max/2,0.45,'efficiency at 80% bkg. rejection: '+str(np.around(ia.find_cl(csgn,cbkg,0.8),decimals=2))[:4]+', $\~\chi^2$<'+str(ia.find_cl(ks_bin,cbkg,0.8))[:4],fontsize=15,bbox={'facecolor':'green','alpha':0.15})
        ax1.add_patch(patches.Rectangle((0,0),ia.find_cl(ks_bin,cbkg,0.95),1,facecolor='green',alpha=0.15))
        ax1.add_patch(patches.Rectangle((0,0),ia.find_cl(ks_bin,cbkg,0.8),1,facecolor='green',alpha=0.15))
	ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
        ax2.set_xlim(x_lim)
	ax2.set_ylim(y_lim)
        ax1.set_xlabel('$\~\chi^2$')
        ax1.set_ylabel('signal efficiency (e$^-$-like events retained)', color='r')
        ax2.set_ylabel('background rejection ($\gamma$-like events discarded)', color='b')
        plt.title(path.split('/')[4]+': discrimination power e$^-$/$\gamma$')
        plt.show()
        plt.close()

def est_vals(method,c_sgn,c_bkg,s_eff):
	if method == 'bootstrap':
		bootstrp_arr = []
		for i in xrange(20):
			idx_s = np.random.choice(c_sgn.shape[0],c_sgn.shape[0])
			idx_b = np.random.choice(c_bkg.shape[0],c_bkg.shape[0])
			boot_s = c_sgn[idx_s]
			boot_b = c_bkg[idx_b]
			ks_bin = binning(boot_s,boot_b)
			bootstrp_arr.append(ia.find_cl(1-np.cumsum(norm_hist(ks_bin,boot_b)),np.cumsum(norm_hist(ks_bin,boot_s)),1-s_eff))
		return np.mean(bootstrp_arr),np.std(bootstrp_arr)
	elif method == 'delta':
		ks_bin = binning(c_sgn,c_bkg)
		cdf_s = np.cumsum(norm_hist(ks_bin,c_sgn))
		cdf_b = 1-np.cumsum(norm_hist(ks_bin,c_bkg))
		idx = np.abs(cdf_s-s_eff).argmin()
		der_s = np.mean(np.gradient(cdf_s,ks_bin[1])[idx-5:idx+5])
		der_b = np.mean(np.gradient(cdf_b,ks_bin[1])[idx-5:idx+5])
		return cdf_b[idx], np.sqrt(1.0/der_s*(s_eff)*(1-s_eff)/float(c_sgn.shape[0])*np.square(der_b)+(cdf_b[idx])*(1-cdf_b[idx])/float(c_bkg.shape[0]))


def compare_bkgrej(seed_loc):
	line11,line12,line21,line22 = [],[],[],[]
	EPDR_flag = False
	ext_rad_flag = False
	ax1 = plt.subplot()
	ax2 = ax1.twiny()
	ax1.set_xscale('log')
	ax2.set_xscale('log')
	for sl in seed_loc:
		for root in glob.glob('/farmshare/user_data/jdalmass/*'):#/%selectron-gammac2'%sl):
			fname = '%s/%selectron-gammac2'%(root,sl)
			try:
	       			if read_conf(root)['lens_system_name'] == 'Sam1' and root[37] == 'K' and read_conf(root)['EPD_ratio'] != 0.5:
        	     			data = np.loadtxt(fname)
			        	c_sgn = data[0]
			        	c_bkg = data[1]
					n = read_conf(root)['base']
					max_diam = read_conf(root)['edge_length']/(n+np.sqrt(3)-1)
					px_per_lens = sum(curved_surface2(read_conf(root)['detector_r'],max_diam,read_conf(root)['nsteps'],read_conf(root)['b_pixel']))
					av_conf_lev,ste_conf_lev = est_vals('delta',c_sgn,c_bkg,0.8)
					#ax1.errorbar(px_per_lens, av_conf_lev, yerr=ste_conf_lev,fmt='o',markersize=3)

					if read_conf(root)['EPD_ratio'] == 1.0 and sl[:4] == 'r0-1':
						line11.append([px_per_lens, av_conf_lev, ste_conf_lev])
              		        	if read_conf(root)['EPD_ratio'] == 0.8 and sl[:4] == 'r0-1':
						EPDR_flag = True
                                		line12.append([px_per_lens, av_conf_lev, ste_conf_lev])
                      			if read_conf(root)['EPD_ratio'] == 1.0 and sl[:4] == 'r3-4':
						ext_rad_flag = True
                        		        line21.append([px_per_lens, av_conf_lev, ste_conf_lev])
                        		if read_conf(root)['EPD_ratio'] == 0.8 and sl[:4] == 'r3-4':
                        		        line22.append([px_per_lens, av_conf_lev, ste_conf_lev])
			except IOError:
				continue

	line11 = np.asarray(sorted(line11,key=lambda x: x[0]))
	line12 = np.asarray(sorted(line12,key=lambda x: x[0]))
	line21 = np.asarray(sorted(line21,key=lambda x: x[0]))
	line22 = np.asarray(sorted(line22,key=lambda x: x[0]))
	#ax1.plot(line11[:,0],line11[:,1],linestyle='-',color='green')
	ax1.errorbar(line11[:,0],line11[:,1],line11[:,2],color='green',ms=8,fmt='o',linestyle='-')
	if EPDR_flag:
		#ax1.plot(line12[:,0],line12[:,1],linestyle='-',color='red')
		ax1.errorbar(line12[:,0],line12[:,1],line12[:,2],color='red',ms=8,fmt='o',linestyle='-')
		if ext_rad_flag:
			ax1.errorbar(line22[:,0],line22[:,1],line22[:,2],color='red',ms=8,fmt='o')
			#ax1.plot(line22[:,0],line22[:,1],linestyle=':',color='red')
			ax1.fill_between(line12[:,0],line12[:,1],line22[:,1],facecolor='none',linestyle='--',edgecolor='red',hatch='///',label='$R_{pupil}/R_{lens}$ = 0.8')
	if ext_rad_flag:
		ax1.errorbar(line21[:,0],line21[:,1],line21[:,2],color='green',ms=8,fmt='o')
		#ax1.plot(line21[:,0],line21[:,1],linestyle=':',color='green')
		ax1.fill_between(line11[:,0],line11[:,1],line21[:,1],facecolor='none',linestyle='--',edgecolor='green',hatch='\\\\\\',label='$R_{pupil}/R_{lens}$ = 1.0')
	ax1.grid()
	ax2.set_xlabel('Total number of lens assemblies')
	ax1Ticks = ax1.get_xticks(minor=True)
	ax2Ticks = ax1Ticks
	ax2.set_xticks(ax2Ticks)
	ax2.set_xbound(ax1.get_xbound())
	ax2.set_xticklabels(tick_function(ax2Ticks),minor=True)
	ax1.set_xlabel('Pixels per lens assembly')#('$R_{pupil}/R_{lens}$')('pixel per lens')
	ax1.set_ylabel('Bkg rejection at 0.8 signal efficiency')
	ax1.set_ylim([0.1,0.57])
	ax1.legend()
	plt.show()


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
	parser.add_argument('path', help='select path-to-file or type compare')
        args = parser.parse_args()
        path = args.path
	set_style()
	if path == 'compare':
		compare_bkgrej(['r0-1_2','r3-4_2'])
	else:
		path = path + seed_loc
		plot_cl(path)
