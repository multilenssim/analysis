import h5py, glob, sys, os
sys.path.insert(0, '/home/jacopo/simulation/')

import matplotlib.pyplot as plt
import numpy as np
import g4_plot
import paths


g4_plot.set_style()
#ax1 = plt.subplot()
#ax2 = ax1.twiny()
#ax1.set_xscale('log')
#ax2.set_xscale('log')

def gather_data(path):
    fpath = '%sraw_data/'%path
    flist = sorted(os.listdir(fpath))[::2]

    with h5py.File(fpath+flist[0],'r') as f:
        e_index = np.diff(f['idx_tr'][:])
        tot_number = np.mean(np.diff(f['idx_depo'][:]))
    center_value = np.mean(e_index/tot_number)
    center_err = np.std(e_index/tot_number)/np.sqrt(len(e_index))

    with h5py.File(fpath+flist[1],'r') as f:
        e_index = np.diff(f['idx_tr'][:])
        tot_number = np.mean(np.diff(f['idx_depo'][:]))
    o_value = np.mean(e_index/tot_number)
    o_err = np.std(e_index/tot_number)/np.sqrt(len(e_index))
    return center_value, center_err, o_value, o_err

y_label = 'Collection efficiency'
g4_plot.compare_bkgrej(gather_data,y_label,[0.0,0.3])



'''
for sl in seed_loc:
    for cfg in sorted(glob.glob('/farmshare/user_data/jdalmass/cfSam1_K*/')):
		read_conf = g4_plot.read_conf(cfg[:-1])
                n = read_conf['base']
		max_diam = read_conf['edge_length']/(read_conf['base']+np.sqrt(3)-1) 
                px_per_lens = sum(g4_plot.curved_surface2(read_conf['detector_r'],max_diam,read_conf['nsteps'],read_conf['b_pixel']))
		if read_conf['EPD_ratio'] == 0.5:
			continue
		try:
			with h5py.File('%sraw_data/%se-_sim.h5'%(cfg,sl),'r') as f:
				e_index = np.diff(f['idx_tr'][:])
				tot_index = np.diff(f['idx_depo'][:])
				ax1.errorbar(px_per_lens,np.mean(e_index/np.mean(tot_index)),np.std(e_index/np.mean(tot_index))/np.sqrt(len(e_index)),color='black')#fmt='o'
                                if read_conf['EPD_ratio'] == 1.0 and sl == seed_loc[0]:
                                        line11.append([px_per_lens,np.mean(e_index/np.mean(tot_index)),np.pi/(2*np.sqrt(3))*(n*(n+1))/np.square(n+np.sqrt(3)-1)])
                                if read_conf['EPD_ratio'] == 0.8 and sl == seed_loc[0]:
                                        line12.append([px_per_lens,np.mean(e_index/np.mean(tot_index))])
                                if read_conf['EPD_ratio'] == 1.0 and sl == seed_loc[1]:
                                        line21.append([px_per_lens,np.mean(e_index/np.mean(tot_index))])
                                if read_conf['EPD_ratio'] == 0.8 and sl == seed_loc[1]:
                                        line22.append([px_per_lens,np.mean(e_index/np.mean(tot_index))])

		except IOError:
			pass
line11 = np.asarray(sorted(line11,key=lambda x: x[0]))
line12 = np.asarray(sorted(line12,key=lambda x: x[0]))
line21 = np.asarray(sorted(line21,key=lambda x: x[0]))
line22 = np.asarray(sorted(line22,key=lambda x: x[0]))
ax1.plot(line11[:,0],line11[:,1],linestyle='-',color='green')
ax1.plot(line12[:,0],line12[:,1],linestyle='-',color='red')
ax1.plot(line21[:,0],line21[:,1],linestyle=':',color='green')
ax1.plot(line22[:,0],line22[:,1],linestyle=':',color='red')
#ax2.plot(line11[:,0],line11[:,2],linestyle='-',linewidth=0.5,color='blue',label='EPD ratio = 1.0')
#ax2.plot(line11[:,0],line11[:,2]*np.square(0.8),linestyle=':',linewidth=0.5,color='blue',label='EPD ratio = 0.8')
ax1.fill_between(line11[:,0],line11[:,1],line21[:,1],facecolor='green',alpha=0.5, label='$R_{pupil}/R_{lens}$ = 1.0')
ax1.fill_between(line12[:,0],line12[:,1],line22[:,1],facecolor='red', alpha=0.5, label='$R_{pupil}/R_{lens}$ = 0.8')
ax1.set_ylim([0,0.25])
ax2.set_xlabel('Total number of lens assemblies')
ax1.grid(True)
ax1Ticks = ax1.get_xticks(minor=True)   
ax2Ticks = ax1Ticks
ax2.set_xticks(ax2Ticks)
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(tick_function(ax2Ticks),minor=True)
#ax2.set_ylabel('geometrical efficiency',color='blue')
ax1.set_ylabel('Collection efficiency',color='black')
ax1.set_xlabel('Pixels per lens assembly')
ax1.legend(loc='upper right')
plt.show()'''
