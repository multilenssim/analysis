import matplotlib.patches as patches
import matplotlib.pyplot as plt
import argparse, os, pickle
import iter_analysis as ia
import numpy as np

def norm_hist(ks_bin,arr):
	return np.histogram(arr,bins=ks_bin)[0].astype(float)/sum(np.histogram(arr,bins=ks_bin)[0])

def binning(c_sgn,c_bkg):
        last_bin = np.amax([c_sgn,c_bkg])
        n_bin = int(last_bin*125)
        return np.linspace(0,last_bin,n_bin)

def read_conf(path):
	with open(path+'/conf.pkl','r') as infile:
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


def compare_bkgrej(seed_loc):
	line11,line12,line21,line22 = [],[],[],[]
	spl = 2
	energy = 2
	#f1, ax1 = plt.subplots()
	#f2, ax2 = plt.subplots()
	#ax3 = ax2.twinx()
	ax2 = plt.subplot()
	for sl in seed_loc:
		for root, _, files in os.walk('/farmshare/user_data/jdalmass/'):
		    for fl in files:
        		if fl.endswith(sl+'electron-gammac2') and read_conf(root)['lens_system_name'] == 'Sam1' and root[35] == 'k':
				EPDR_flag = False
        	     		data = np.loadtxt(os.path.join(root, fl))
			        c_sgn = data[0]
			        c_bkg = data[1]
				n = read_conf(root)['base']
				max_diam = read_conf(root)['edge_length']/(n+np.sqrt(3)-1)
				px_per_lens = sum(curved_surface2(read_conf(root)['detector_r'],max_diam,read_conf(root)['nsteps'],read_conf(root)['b_pixel']))
			        ks_bin = binning(c_sgn,c_bkg)
				conf_lev = [ia.find_cl(1-np.cumsum(norm_hist(ks_bin,c_b)),np.cumsum(norm_hist(ks_bin,c_s)),0.2) for c_s,c_b in zip(np.split(c_sgn,spl),np.split(c_bkg,spl))]
				av_conf_lev = np.mean(conf_lev)
				#e_hist = np.cumsum(ia.make_hist(ks_bin,c_s))
				#g_hist = 1-np.cumsum(ia.make_hist(ks_bin,c_b))
				ax2.errorbar(px_per_lens, av_conf_lev, yerr=np.std(conf_lev)/np.sqrt(spl) ,fmt='o')

				if read_conf(root)['EPD_ratio'] == 1.0 and sl[:4] == 'r0-1':
					#ax1.plot(e_hist,g_hist,label=root.split('/')[4]+', base lenses: %i'%n)
					line11.append([px_per_lens, av_conf_lev])
              		        if read_conf(root)['EPD_ratio'] == 0.8 and sl[:4] == 'r0-1':
					EPDR_flag = True
					#ax1.plot(e_hist,g_hist,label=root.split('/')[4]+', base lenses: %i'%n)
                                	line12.append([px_per_lens, av_conf_lev])
                       		if read_conf(root)['EPD_ratio'] == 1.0 and sl[:4] == 'r3-4':
                        	        line21.append([px_per_lens, av_conf_lev])
                        	if read_conf(root)['EPD_ratio'] == 0.8 and sl[:4] == 'r3-4':
                        	        line22.append([px_per_lens, av_conf_lev])

	line11 = np.asarray(sorted(line11,key=lambda x: x[0]))
	line12 = np.asarray(sorted(line12,key=lambda x: x[0]))
	line21 = np.asarray(sorted(line21,key=lambda x: x[0]))
	line22 = np.asarray(sorted(line22,key=lambda x: x[0]))
	ax2.plot(line11[:,0],line11[:,1],linestyle='-',color='green')
	ax2.plot(line21[:,0],line21[:,1],linestyle=':',color='green')
	ax2.fill_between(line11[:,0],line11[:,1],line21[:,1],facecolor='green',alpha=0.5, label='EPD ratio = 1.0')
	if EPDR_flag:
		ax2.plot(line12[:,0],line12[:,1],linestyle='-',color='red')
		ax2.plot(line22[:,0],line22[:,1],linestyle=':',color='red')
		ax2.fill_between(line12[:,0],line12[:,1],line22[:,1],facecolor='red', alpha=0.5, label='EPD ratio = 0.8')	
	#ax1.grid(linestyle='--', linewidth=0.5)
	#ax1.set_xlabel('signal efficiency')
	#ax1.set_ylabel('background rejection')
	#ax1.set_title('comparison of different configuration')
	ax2.set_title('comparison of different configuration')
	ax2.set_xlabel('pixel per lens')
	ax2.set_xscale('log')
	ax2.set_ylabel('bkg rejection at 0.8 signal efficiency')
	#ax1.axis('equal')
	#ax1.legend()
	ax2.legend()
	plt.show()


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
	parser.add_argument('path', help='select path-to-file or type compare')
        args = parser.parse_args()
        path = args.path
	if path == 'compare':
		compare_bkgrej(['r0-1_2','r3-4_2'])
	else:
		path = path + seed_loc
		plot_cl(path)
