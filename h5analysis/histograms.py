import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from lv17analysis import lv17data
from os import path, makedirs
import glob
import datetime
from IPython.display import display, clear_output


def golden_figsize(fwidth=8):
    golden_ratio = 1.618034
    fheight = fwidth / golden_ratio
    return (fwidth, fheight)


def read_data(run, data_dir='/reg/data/ana01/tmo/tmolv1720/hdf5/smalldata/',
              postfix='epix'):
    data_fname = 'r{:03d}_{}.h5'.format(run, postfix)
    if not path.exists(data_dir + data_fname):
        return None

    data_keys = ['epix_lit', 'epix_sum', 'gmd', 'photon_energy',
                 'source_delay', 'source_on', 'source_temp',
                 'timestamp', 'tof_abs_sum', 'tof_sum', 'xgmd']
    data = {}
    with h5py.File(data_dir + data_fname, 'r') as f:
        for key in data_keys:
            data[key] = np.copy(f[key])
    data['run'] = run
    return data


def data_filter(data, source='on',
                gmd_min=0.0, gmd_max=np.inf,
                xgmd_min=0.0, xgmd_max=np.inf):
    source_on = data['source_on']
    source_filt = np.array(source_on if source == 'on' else ~source_on)
    gmd_filt = np.array(data['gmd'] >= 0) & \
        np.array(data['gmd'] <= np.inf)
    xgmd_filt = np.array(data['xgmd'] >= 0) & \
        np.array(data['xgmd'] <= np.inf)
    return source_filt & gmd_filt & xgmd_filt


def plot_data(data, dkey='epix_lit', bins=100, norm=None,
              figsize=(8, 5), ptype='plot', yscale='linear',
              fig=None):
    dfilt = data_filter(data)

    if norm is None:
        plotdata = data[dkey]
    else:
        plotdata = data[dkey] / data[norm]

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ptype == 'plot':
        plt.scatter(data['timestamp'][dfilt],
                    plotdata[dfilt], 3, label='source on',
                    marker='.')
        plt.scatter(data['timestamp'][~dfilt],
                    plotdata[~dfilt], 3, label='source off',
                    marker='.')
    elif ptype == 'hist':
        plt.hist(plotdata[dfilt], bins=100, label='source on', alpha=0.7)
        plt.hist(plotdata[~dfilt], bins=100, label='source off', alpha=0.7)
    plt.yscale(yscale)
    plt.grid()
    plt.legend(title='run {:03d}, {}\n{}'.format(
        data['run'], lv17data.run_type(data['run']), dkey))
    return fig


def save_plot(fig, save_name, save_dir='/reg/d/psdm/tmo/tmolv1720/results/run_hist/'):
    spath = path.join(save_dir, str(datetime.date.today()))
    makedirs(spath, exist_ok=True)
    fname = path.join(spath, save_name)
    plt.savefig(fname)


def plot_and_save(data, dkey='epix_lit', bins=100, norm=None, figsize=(8, 5),
                  ptype='plot', closefig=False, yscale='linear', fig=None,
                  save_dir='/reg/d/psdm/tmo/tmolv1720/results/run_hist/'):
    fig = plot_data(data=data, dkey=dkey, bins=bins, norm=norm, fig=fig,
                    figsize=figsize, ptype=ptype, yscale=yscale)
    save_name = "r{:3d}_{}_{}.png".format(data['run'], dkey, ptype)
    save_plot(fig, save_name, save_dir=save_dir)
    if closefig:
        plt.close(fig=fig)


def create_histograms(run_list=[180],
                      data_dir='/reg/data/ana01/tmo/tmolv1720/hdf5/smalldata/',
                      save_dir='/reg/d/psdm/tmo/tmolv1720/results/run_hist/',
                      postfix='epix'):
    if run_list is None:
        run_list = lv17data.epix_runs
    if np.ndim(run_list) == 0:
        run_list = [run_list]
    print('Creating histogram plots for runs: {}'.format(run_list))

    for run in run_list:
        print('Creating histogram plot for run {:03d}'.format(run))

        data = read_data(run=run, data_dir=data_dir, postfix=postfix)
        if data is None:
            return
        for dkey in ['epix_lit', 'epix_sum', 'tof_abs_sum', 'gmd', 'xgmd']:
            fig = plt.figure(figsize=(16, 5))
            for iplt, ptype in enumerate(['plot', 'hist']):
                fig.add_subplot(1, 2, iplt + 1)
                if ptype == 'hist' and dkey not in ['gmd', 'xgmd']:
                    yscale = 'log'
                else:
                    yscale = 'linear'
                fig = plot_data(data=data, dkey=dkey, fig=fig, ptype=ptype, yscale=yscale)

                # plot_and_save(data, dkey=dkey, ptype=ptype, closefig=False,
                #               yscale=yscale, save_dir=save_dir, fig=fig)
            save_name = "r{:3d}_{}_{}.png".format(data['run'], dkey, ptype)
            save_plot(fig, save_name, save_dir=save_dir)
            plt.close(fig=fig)
        clear_output(wait=True)
        print('Saved histograms for run {:03d} in {}'.format(run, save_dir))


if __name__ == "__main__":
    # Create plots and histograms for all epix runs
    run_list = lv17data.epix_runs
    create_histograms(run_list=run_list)
