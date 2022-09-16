'''
Anatoli Ulmer, Linos Hecht, 2022
'''
import psana as ps
import numpy as np
import h5py
import os
import time
import datetime
from lv17analysis import detectors, lv17data, epix
import glob
import sys


exclude_data_dict = {'full': [''],
                     'epix': ['spectrometer'],
                     'tof': ['epix', 'spectrometer'],
                     'smd': ['epix', 'spectrometer', 'tof_trace']}

exclude_det_dict = {'full': [''],
                    'epix': ['xtcav', 'pcav', 'ebeam', 'tmo_opal3', 'epicsinfo'],
                    'tof': ['xtcav', 'pcav', 'ebeam', 'tmo_opal3', 'epicsinfo'],
                    'smd': ['xtcav', 'pcav', 'ebeam', 'tmo_opal3', 'epicsinfo']}


def populate_data_dict(evt, datasets_list, crucial_datasets=['']):
    data_dict = {"timestamp": evt.timestamp}
    for data_str in datasets_list:
        data = detectors.get(data_str, evt)
        if data is None:
            if data_str in crucial_datasets:
                return None
            else:
                data = -99999
        data_dict[data_str] = data
    return data_dict


def xtc_exists(run):
    xtc_files = glob.glob(os.path.join(lv17data.path, 'xtc',
                                       'tmolv1720-r{:04d}*.xtc2'.format(run)))
    if len(xtc_files) == 0:
        print('No files for Run ' + str(run))
        return False
    return True


def get_filename(run, data_scope, overwrite=False):
    fname = os.path.join(lv17data.h5path, 'r{:03d}_{}.h5'.format(run, data_scope))
    if os.path.exists(fname):
        if not overwrite:
            print("File {} already exists. Continuing ...".format(fname))
            return None
        print("File {} already exists.\n\tDeleting it before continuing".format(fname))
        os.remove(fname)
        print("\tDeleted file {}".format(fname))
    print("Creating file {}".format(fname))
    return fname


def produce_h5_files(run_list, data_scope='epix', overwrite=False, i_max=None):
    if np.ndim(run_list) == 0:
        run_list = [run_list]
    n_runs = len(run_list)

    exclude_detectors = exclude_det_dict[data_scope]
    detector_list = detectors.detector_list
    for excl_det in exclude_detectors:
        if excl_det in detector_list:
            detector_list.remove(excl_det)
    crucial_datasets = ['gmd', 'xgmd']
    exclude_datasets = exclude_data_dict[data_scope]
    datasets_list = detectors.detectors2datasets(detector_list)

    for i_run, run in enumerate(run_list):

        t_start = time.time()

        print('run. ' + str(run) + ' started (' + str(i_run + 1) + '/' + str(len(run_list)) + ')')

        # check if data exists
        if not xtc_exists(run):
            continue

        fname = get_filename(run, data_scope, overwrite=overwrite)
        if fname is None:
            continue

        ds = ps.DataSource(exp=lv17data.exp, run=run)
        ds_run = next(ds.runs())

        # # get detector info
        # detector_list = detectors.get_active_detectors(ds_run)
        # for excl_det in exclude_detectors:
        #     if excl_det in detector_list:
        #         detector_list.remove(excl_det)
        # datasets_list = detectors.detectors2datasets(detector_list)

        # only process if run contains epix
        if run not in lv17data.epix_runs:
            continue

        # check for completeness of crucial datasets
        if any([cr_data not in datasets_list for cr_data in crucial_datasets]):
            continue

        # create h5 file

        smd = ds.smalldata(filename=fname, batch_size=5)

        i_success = 0

        # iterate through events
        for i_evt, evt in enumerate(ds_run.events()):
            if i_evt % 1000 == 0:
                t_diff = datetime.timedelta(seconds=int(time.time() - t_start))
                print("run {:04d} ({:d}/{:d}), Time: {:}, {:d} entries".format(
                    run, i_run + 1, n_runs, str(t_diff), i_evt))

            if i_max is not None and i_success >= i_max:
                break

            # create and populate data dictionary for current shot
            data_dict = populate_data_dict(evt, datasets_list, crucial_datasets)
            if data_dict is None:
                continue

            if 'epix' in data_dict.keys():
                lit_thresh = 0.5 * lv17data.photon_energy_kev(ds_run.runnum)
                data_dict['epix_sum'] = np.nansum(data_dict['epix'])
                data_dict['epix_lit'] = epix.count_lit(data_dict['epix'], lit_thresh=lit_thresh)

            if 'tof_trace' in data_dict.keys():
                data_dict['tof_sum'] = np.nansum(data_dict['tof_trace'])
                data_dict['tof_abs_sum'] = np.nansum(np.abs(data_dict['tof_trace']))

            # filter and write file
            for excl_data in exclude_datasets:
                if excl_data in data_dict.keys():
                    data_dict.pop(excl_data)
            smd.event(evt, data_dict)

            i_success += 1

        smd.done()  # close file

        # add tof_times to h5 file if tof_trace was also written
        with h5py.File(fname, 'r+') as f:
            if 'tof_trace' in f.keys():
                f['tof_times'] = detectors.get_tof_times(evt)

        t_diff = datetime.timedelta(seconds=int(time.time() - t_start))
        print("Time for run {:04d}: {}, processed {:d} good entries.".format(
            run, str(t_diff), i_success))


###################################################################################################
if __name__ == "__main__":
    '''
    Run script with 'mpirun -n 1 python lv17analysis/h5convert.py'

    Anatoli Ulmer, Linos Hecht, 2022
    '''
    os.environ['PS_SRV_NODES'] = '1'

    print(sys.argv)

    # python ~/lv17/lv17analysis/lv17analysis/h5convert.py
    # start script: python lv17analysis/h5convert.py <start_run> <end_run> <data_scope>
    # mpirun -n 6 python lv17analysis/h5convert.py run_start run_end
    nargs = len(sys.argv)

    # choose data scope between 'full', 'tof' and 'smd'
    data_scope = sys.argv[3] if nargs > 3 else 'epix'

    if nargs > 1:
        if nargs > 2:
            run_list = list(range(int(sys.argv[1]), int(sys.argv[2])+1))
        else:
            run_list = [int(sys.argv[1])]
    else:
        run_list = lv17data.epix_runs

    produce_h5_files(run_list)
