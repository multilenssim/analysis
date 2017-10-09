import matplotlib.pyplot as plt
import numpy as np
import h5py, glob
import g4_plot

ax1 = plt.subplot()
ax2 = ax1.twinx()
seed_loc = ['r0-1','r3-4']
line11,line12,line21,line22 = [],[],[],[]
for sl in seed_loc:
	for cfg in sorted(glob.glob('/farmshare/user_data/jdalmass/S*/')):
		read_conf = g4_plot.read_conf(cfg[:-1])
                n = read_conf['base']
		max_diam = read_conf['edge_length']/(read_conf['base']+np.sqrt(3)-1) 
                px_per_lens = sum(g4_plot.curved_surface2(read_conf['detector_r'],max_diam,read_conf['nsteps'],read_conf['b_pixel']))
		try:
			with h5py.File(cfg+sl+'e-_sim.h5','r') as f:
				e_index = np.diff(f['idx_tr'][:])
				tot_index = np.diff(f['idx_depo'][:])
				ax2.errorbar(px_per_lens,np.mean(e_index/np.mean(tot_index)),np.std(e_index/np.mean(tot_index)),fmt='o')

                                if read_conf['EPD_ratio'] == 1.0 and sl == 'r0-1':
                                        line11.append([px_per_lens,np.mean(e_index/np.mean(tot_index)),np.pi/(2*np.sqrt(3))*(n*(n+1))/np.square(n+np.sqrt(3)-1)])
                                if read_conf['EPD_ratio'] == 0.8 and sl == 'r0-1':
                                        line12.append([px_per_lens,np.mean(e_index/np.mean(tot_index))])
                                if read_conf['EPD_ratio'] == 1.0 and sl == 'r3-4':
                                        line21.append([px_per_lens,np.mean(e_index/np.mean(tot_index))])
                                if read_conf['EPD_ratio'] == 0.8 and sl == 'r3-4':
                                        line22.append([px_per_lens,np.mean(e_index/np.mean(tot_index))])

				#data_arr.append([px_per_lens,np.mean(e_index/np.mean(tot_index)),np.std(e_index/np.mean(tot_index))])
		except IOError:
			pass
			#data_arr.append([px_per_lens,0,0])
line11 = np.asarray(sorted(line11,key=lambda x: x[0]))
line12 = np.asarray(sorted(line12,key=lambda x: x[0]))
line21 = np.asarray(sorted(line21,key=lambda x: x[0]))
line22 = np.asarray(sorted(line22,key=lambda x: x[0]))
ax2.plot(line11[:,0],line11[:,1],linestyle='-',color='blue',label='EPD ratio = 1.0, seed location 0<r<1 meters')
ax2.plot(line12[:,0],line12[:,1],linestyle='-.',color='blue',label='EPD ratio = 0.8, seed location 0<r<1 meters')
ax2.plot(line21[:,0],line21[:,1],linestyle=':',color='blue',label='EPD ratio = 1.0, seed location 3<r<4 meters')
ax2.plot(line22[:,0],line22[:,1],linestyle='--',color='blue',label='EPD ratio = 0.8, seed location 3<r<4 meters')
ax1.plot(line11[:,0],line11[:,2],linestyle='-',linewidth=0.5,color='red',label='EPD ratio = 0.8')
ax1.plot(line11[:,0],line11[:,2]*np.square(0.8),linestyle='-.',linewidth=0.5,color='red',label='EPD ratio = 1.0')
ax1.set_ylabel('geometrical efficiency',color='red')
ax2.set_ylabel('collection efficiency',color='blue')
ax1.set_xlabel('pixel per lens')
ax1.set_xscale('log')
plt.xlabel('pixel per lens')
plt.legend()
plt.show()
