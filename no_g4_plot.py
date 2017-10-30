import glob, argparse, pickle
import matplotlib.pyplot as plt
import numpy as np
    

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('se', help='seed location and energy')
	args = parser.parse_args()
	se = args.se
	distance = []
	path_file = ['/farmshare/user_data/jdalmass/%s/%s'%(cfg,se) for cfg in ['Sam1_K6_10','Sam1_M10_10']]
	for fname in sorted(glob.glob(path_file[0]+'*cm.h5')):
		dst = fname.split('/')[-1][12:]
		distance.append(int(dst[:-5]))
	lg = len(distance)
	first = True
	lab = '100k'
	for path in path_file:
		val_c2 = np.loadtxt(path+'datapoints')
		val_c2 = np.asarray((val_c2[::2],val_c2[1::2]))
		plt.errorbar(distance,np.mean(val_c2,axis=0),yerr=np.std(val_c2,axis=0),fmt='^',label='%s'%lab)
		if first:
			lab = '1M'
		first = False
	plt.xlabel('relative distance [cm]')
	plt.ylabel('bkg rej at 80% of sgn eff')
	plt.title('$\chi^2$ performance with event seeded in %s meter shell'%path.split('/')[-1][1:4])
	plt.legend(loc='upper left')
	plt.xlim(0,48)
	plt.ylim(-0.1,1.1)
	plt.show()
