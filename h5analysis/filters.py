import h5py
import numpy as np
from lv17analysis import lv17data
from h5analysis import h5data
from os import path, makedirs, remove
import glob
import time
import datetime
from IPython.display import display, clear_output


h5paths = h5data.h5paths

excl_keys = ['run', 'tof_times']
new_keys = ['n_entries', 'lit_thresh', 'n_max', 'n_brightest']

ranges_dict = {
    'gmd': np.asarray([0.02, 2.5]),
    'xgmd': np.asarray([0.02, 1.5]),
    'epix_lit': np.asarray([15000, 540672]),
    'epix_sum': np.asarray([0., np.inf]),
    'tof_sum': np.asarray([-np.inf, np.inf]),
    'tof_abs_sum': np.asarray([0., np.inf])}


def filt_range(flt, data_dict, ranges_dict=ranges_dict):
    for key in ranges_dict.keys():
        flt *= filt_entry(data_dict, ranges_dict, key)
    return flt


def filt_entry(data_dict, ranges_dict, key):
    data = np.asarray(data_dict[key])
    dmin, dmax = ranges_dict[key]
    return filt_array(data, dmin, dmax)


def filt_array(data, dmin, dmax):
    d = np.asarray(data)
    return (d >= dmin) * (d <= dmax)


def hits(data_dict,
         gmd_min=0.02, gmd_max=2.5,
         xgmd_min=0.02, xgmd_max=1.5,
         lit_min=15000, lit_max=540672,
         epix_sum_min=0, epix_sum_max=np.inf,
         tof_abs_sum_min=0, tof_abs_sum_max=np.inf):
    return None


def filter_and_write_run(run,
                         fpath=h5paths['smd'],
                         fname_prefix='r',
                         fname_postfix='_epix',
                         spath=h5paths['filt'],
                         sname_prefix='r',
                         sname_postfix='_filt',
                         ranges_dict=ranges_dict,
                         lit_filter_factor=2,
                         source_on=None,
                         overwrite=False,
                         n_max=0,
                         n_brightest=0):

    print("run {:03d} started at {}".format(run, datetime.datetime.now()))
    fn = path.join(fpath, "{}{:03d}{}.h5".format(
        fname_prefix, run, fname_postfix))
    if not path.exists(fn):
        print('File {} does not exist.'.format(fn))
        return None

    if n_max > 0:
        spath = spath + "_n{:.0f}".format(n_max)
    if n_brightest > 0:
        spath = spath + "_brightest{:.0f}".format(n_brightest)
    makedirs(spath, exist_ok=True)
    sn = path.join(spath, "{}{:03d}{}.h5".format(
        sname_prefix, run, sname_postfix))

    if path.exists(sn):
        if not overwrite:
            print("'{}' already exists".format(fn))
            return sn
        remove(sn)

    print("  -> Reading: '{}'".format(fn))
    t_0 = time.time()

    with h5py.File(fn, 'r') as f:

        print("  -> Calculating filter ...")
        rdict = ranges_dict.copy()
        if n_brightest > 0:
            rdict['epix_lit'] = [0., np.inf]
        lit_thresh = rdict['epix_lit'][0]
        if lv17data.run_type(run) == "fullbeam":
            lit_thresh *= lit_filter_factor

        flt = np.ones_like(f['gmd'], dtype=bool)
        flt = filt_range(flt, f, ranges_dict=rdict)
        flt = filt_source(flt, f, source_on=source_on)
        flt = filt_n_brightest(flt, f, n_brightest)
        flt = filt_n_max(flt, n_max)

        n_entries = np.sum(flt)
        nkeys = ['n_entries', 'lit_thresh', 'n_max', 'n_brightest']
        nvals = [np.sum(flt), lit_thresh, n_max, n_brightest]

        sn = write_filtered_h5(f, sn, flt,
                               excl_keys=['run', 'tof_times'],
                               new_keys=nkeys,
                               new_vals=nvals)

    clear_output(wait=True)
    t_diff = datetime.timedelta(seconds=int(time.time() - t_0))
    print("  -> run {:03d} finished, Time: {:}, {:d} entries".format(
        run, str(t_diff), n_entries))

    return sn


def write_filtered_h5(f, sn, flt,
                      excl_keys=['run', 'tof_times'],
                      new_keys=['n_entries', 'lit_thresh'],
                      new_vals=[0, 0]):

    if path.exists(sn):
        print("  -> Destination file exists, skipping run {:03d}.".format(f['run']))
        return sn

    print("  -> Writing: '{}'".format(sn))
    with h5py.File(sn, 'a') as s:

        for key in f.keys():
            if key in new_keys:
                continue
            if key in excl_keys:
                s[key] = np.asarray(f[key])
                continue
            s[key] = np.asarray(f[key][flt])

        for i_new, key in enumerate(new_keys):
            s[key] = np.asarray(new_vals[i_new])

    return sn


def filt_source(flt, f, source_on=None):
    if source_on is not None:
        flt *= ~np.logical_xor(source_on, np.asarray(f['source_on'], dtype=bool))
    return flt


def filt_n_brightest(flt, f, n_brightest=0):
    if n_brightest > 0:
        flt = get_bool_n_largest(f['epix_sum'] * flt, n=n_brightest)
    return flt


def filt_n_max(flt, n_max=0):
    if n_max > 0:
        flt[np.asarray(np.where(flt))[0][n_max:]] = False
    return flt


def get_indices_n_largest(arr, n=200):
    sorted_index_array = np.argsort(arr)
    return sorted_index_array[-n:]


def get_bool_n_largest(arr, n=200):
    largest_indices = get_indices_n_largest(arr, n=n)
    largest_bool = np.zeros(np.shape(arr), dtype=bool)
    largest_bool[largest_indices] = True
    return largest_bool


def filter_for_ranges_dict(run_list=lv17data.neon_runs):
    for run in run_list:
        filter_and_write_run(run)
    return


def filter_for_hits(run_list=lv17data.neon_runs,
                    fpath=h5paths['smd'], fname_prefix='r', fname_postfix='_epix',
                    spath=h5paths['hits'], sname_prefix='r', sname_postfix='_hits',
                    ranges_dict=ranges_dict.copy(),
                    n_max=0, n_brightest=0):
    for run in run_list:
        filter_and_write_run(run,
                             fpath=fpath, fname_prefix=fname_prefix, fname_postfix=fname_postfix,
                             spath=spath, sname_prefix=sname_prefix, sname_postfix=sname_postfix,
                             source_on=True, n_max=n_max, n_brightest=n_brightest)
    return


def filter_for_bg(run_list=lv17data.neon_runs, n_max=200):
    rdict = ranges_dict.copy()
    rdict['epix_lit'] = np.asarray([0.0, np.inf])
    rdict['epix_sum'] = np.asarray([0.0, np.inf])
    for run in run_list:
        filter_and_write_run(run,
                             fpath=h5paths['smd'], fname_prefix='r', fname_postfix='_epix',
                             spath=h5paths['bg'], sname_prefix='r', sname_postfix='_bg',
                             ranges_dict=rdict,
                             source_on=False,
                             n_max=n_max)
    return


if __name__ == "__main__":
    import sys

    nargs = len(sys.argv)

    if nargs > 1:
        if nargs > 2:
            run_list = list(range(int(sys.argv[1]), int(sys.argv[2]) + 1))
        else:
            run_list = [int(sys.argv[1])]
    else:
        run_list = lv17data.epix_runs

    filter_for_hits(run_list=run_list)
