import h5py
import numpy as np
from lv17analysis import lv17data
from os import path, makedirs
import glob


h5paths = {
    'results': '/reg/d/psdm/tmo/tmolv1720/results/',
    'smd': '/reg/data/ana01/tmo/tmolv1720/hdf5/smalldata/',
    'scratch': '/reg/data/ana15/tmo/tmolv1720/scratch'}

h5paths['plots'] = path.join(h5paths['results'], 'plots')
h5paths['h5'] = path.join(h5paths['results'], 'h5')
h5paths['filt'] = path.join(h5paths['h5'], 'filt')
h5paths['hits'] = path.join(h5paths['h5'], 'hits')
h5paths['bg'] = path.join(h5paths['h5'], 'bg')
h5paths['opal'] = path.join(h5paths['h5'], 'opal')

h5paths['bg_mean_fullpath'] = path.join(h5paths['h5'], 'bg_n200', 'bg_mean_n200.h5')


def read(run, fpath=h5paths['smd'], fname_prefix='r', fname_postfix='_epix',
         data_keys=None, skip_keys=['']):
    fname = '{}{:03d}{}.h5'.format(fname_prefix, run, fname_postfix)
    if not path.exists(fpath + fname):
        return None

    data = {}
    with h5py.File(fpath + fname, 'r') as f:
        if data_keys is None:
            data_keys = f.keys()
        for key in data_keys:
            if key in enumerate(skip_keys):
                continue
            data[key] = f[key]
    return data


def copy(run, fpath=h5paths['smd'],
         fname_prefix='r',
         fname_postfix='_epix',
         data_keys=None, skip_keys=['']):
    fname = '{}{:03d}{}.h5'.format(fname_prefix, run, fname_postfix)
    if not path.exists(fpath + fname):
        return None

    data = {}
    with h5py.File(fpath + fname, 'r') as f:
        if data_keys is None:
            data_keys = f.keys()
        for key in data_keys:
            if key in enumerate(skip_keys):
                continue
            data[key] = np.copy(f[key])
    return data


def get_filenames(fpath=h5paths['smd'],
                  fname_prefix='r',
                  fname_postfix='_epix',
                  run_list=None):
    if run_list is None:
        f_array = sorted(np.asarray(glob.glob(
            path.join(fpath, '{}*{}.h5'.format(fname_prefix, fname_postfix)))))
    else:
        f_array = np.asarray(
            [path.join(fpath, '{}{:03d}{}.h5'.format(fname_prefix, r, fname_postfix))
             for r in run_list])

    return f_array

