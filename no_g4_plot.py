import glob, re, argparse, pickle
import matplotlib.pyplot as plt
import numpy as np
    

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('path', help='select path-to-file with seed location')
	args = parser.parse_args()
	path = args.path
	distance = []
	for fname in sorted(glob.glob(path+'*cm.h5')):
		rgx = re.compile(r'\d+')
		distance.append(int(rgx.findall(fname)[4]))
	lg = len(distance)
	data = np.loadtxt(path+'datapoints')
	a_lg = data.shape[1]
	#val_avg = data[0]
	val_c2 = data[0]
	#plt.errorbar(distance,np.mean(np.reshape(val_avg,(lg,a_lg/lg)),axis=1),yerr=np.std(np.reshape(val_avg,(lg,a_lg/lg)),axis=1),fmt='v',label='weighted average')
	plt.errorbar(distance,np.mean(np.reshape(val_c2,(lg,a_lg/lg)),axis=1),yerr=np.std(np.reshape(val_c2,(lg,a_lg/lg)),axis=1),fmt='^',label='$\chi^2$')
	plt.xlabel('relative distance [cm]')
	plt.ylabel('signal efficiency at 95% background rejection')
	plt.title(fname.split('/')[4]+' events seeded in 1m radius sph.')
	plt.legend(loc='upper left')
	plt.xlim(0,72)
	plt.ylim(-0.1,1.1)
	plt.show()



#'Sam1_1'
#'Jiani3_4'
#'Jiani3_2'
