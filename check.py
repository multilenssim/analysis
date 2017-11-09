import glob,h5py,argparse
import numpy as np


def check_status(path):
	f_err = False
	for fname in glob.glob('%s*'%path):
		try:
			with h5py.File(fname,'r') as f:
				print 'collection efficiency averaged over the events in file %s: %0.2f'%(fname,np.mean(np.diff(f['idx_tr'][:]))/np.mean(np.diff(f['idx_depo'][:])))
				print 'events: %i'%len(f['idx_tr'][:])
				print '----------------------------------------------------------------------------------'
		except KeyError:
			f_err = True
			print '%s corrupted'%fname
			print '----------------------------------------------------------------------------------'
			continue

	        except IOError:
			f_err = True
			print '%s corrupted'%fname
			print '----------------------------------------------------------------------------------'
			continue
	return f_err

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('path', help='insert path-to-file with seed location')
	args = parser.parse_args()
	path = args.path
	check_status(path)
