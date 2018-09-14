import sys, h5py, glob, argparse, pickle, os
sys.path.insert(0, '/home/jacopo/simulation/')

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import rcParams
import iter_analysis as ia
import detectorconfig
import numpy as np
import paths


def set_style():
    rcParams['lines.markersize'] = 4
    rcParams['font.size'] = 20.0
    rcParams['figure.figsize'] = (12, 9)

def norm_hist(ks_bin,arr):
    return np.histogram(arr,bins=ks_bin)[0].astype(float)/sum(np.histogram(arr,bins=ks_bin)[0])

def binning(c_sgn,c_bkg):
    last_bin = np.amax([c_sgn,c_bkg])
    n_bin = int(last_bin*125)
    return np.linspace(0,last_bin,n_bin)

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

def gather_data(path):
    flist = sorted(os.listdir(path))
    center = path+flist[1]
    off_center = path+flist[4]
    center_data = np.loadtxt(center)
    off_center_data = np.loadtxt(off_center)
    c_av_conf_lev,c_ste_conf_lev = est_vals('delta',center_data[0],center_data[1],0.8)
    o_av_conf_lev,o_ste_conf_lev = est_vals('delta',off_center_data[0],off_center_data[1],0.8)
    return c_av_conf_lev, c_ste_conf_lev, o_av_conf_lev, o_ste_conf_lev


def compare_bkgrej(gather_data,y_label,lim):
    ax1 = plt.subplot()
    ax2 = ax1.twiny()
    tot_lens = []
    ax1.set_xscale('log')
    ax2.set_xscale('log')

    configs_pickle_file = '%sconf_file_obj.pickle' % paths.detector_config_path

    with open(configs_pickle_file,'r') as f:
        config_list = pickle.load(f)

    for pupil,color,hatch in zip([0.8,1.0],['red','green'],['///','\\\\\\']):
        line11,line22 = [],[]

        for key in config_list:
            path = paths.get_data_file_path_no_raw(key)
            dc = detectorconfig.get_detector_config(key)
            epd = dc.EPD_ratio

            if not pupil == epd: continue

            try:
                center_val,center_err,o_val,o_err = gather_data(path)
                line11.append([dc.half_EPD/(dc.EPD_ratio*10), center_val, center_err])
                line22.append([dc.half_EPD/(dc.EPD_ratio*10), o_val, o_err])
                tot_lens.append(dc.lens_count)

            except IndexError: pass

        line11 = np.asarray(sorted(line11,key=lambda x: x[0]))
        line22 = np.asarray(sorted(line22,key=lambda x: x[0]))
        ax1.errorbar(line11[:,0],line11[:,1],line11[:,2],color=color,ms=8,fmt='o',linestyle='-')
        ax1.errorbar(line22[:,0],line22[:,1],line22[:,2],color=color,ms=8,fmt='o')
        ax1.fill_between(line11[:,0],line11[:,1],line22[:,1],facecolor='none',linestyle='--',edgecolor=color,hatch=hatch,label='$R_{pupil}/R_{lens}$ = %s'%pupil)

    ax1.grid()
    ax2.set_xticklabels(sorted(np.unique(tot_lens),reverse=True),minor=False)
    ax1.set_xlabel('Lens radius [cm]')#('$R_{pupil}/R_{lens}$')('pixel per lens')
    ax2.set_xlabel('Total number of lens assemblies')
    ax1.set_ylabel(y_label)
    ax1.set_ylim(lim)
    ax1.legend()
    plt.show()
    #plt.savefig('chroma-data/simulations/finite_res_perf_lens.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='select path-to-file or type compare')
    args = parser.parse_args()
    path = args.path
    set_style()

    if path == 'compare':
        y_label = 'Bkg rejection at 0.8 signal efficiency'
        compare_bkgrej(gather_data,y_label,[0.0,1])

    else:
        path = path + seed_loc
        plot_cl(path)
