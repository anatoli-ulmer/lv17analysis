'''
This is a workaround to get all photon energies and source temperatures as floats. Unfortunetaly
they were saved as integers in the first analysis run.
'''
import numpy as np
import psana as ps
from lv17analysis import lv17data, detectors
from h5analysis import h5data
import h5py
import os
# from IPython.display import display, clear_output


def write_hv_to_new_h5():
    run_list = lv17data.epix_runs
    data_scope = 'epix'

    for run in run_list:
        # clear_output(wait=True)
        fname = os.path.join(lv17data.h5path, 'r{:03d}_{}.h5'.format(run, data_scope))
        sname = os.path.join(lv17data.h5path, 'r{:03d}_{}.h5'.format(run, 'hv'))
        if os.path.exists(sname):
            continue

        with h5py.File(fname, 'r') as f:
            tstamps = np.asarray(f['timestamp'])

        numel = len(tstamps)
        print("run {:03d} has {:d} entries".format(run, numel))

        hv = np.zeros_like(tstamps, dtype=np.single)
        st = np.zeros_like(tstamps, dtype=np.single)

        ds = ps.DataSource(exp=lv17data.exp, run=run)
        # detectors=['photon_energy_welch', 'source_valve_temp'])
        ds_run = next(ds.runs())

        idx = 0

        for i_evt, evt in enumerate(ds_run.events()):
            if evt.timestamp not in tstamps:
                continue
            hv[idx] = detectors.get('photon_energy', evt)
            st[idx] = detectors.get('source_temp', evt)
            idx += 1

            if idx % 1000 == 0:
                print("run {:03d}, entry {:d}/{:d}".format(run, idx, numel))

        with h5py.File(sname, 'w') as f:
            f['timestamp'] = np.asarray(tstamps)
            f['photon_energy'] = np.asarray(hv)
            f['source_temp'] = np.asarray(st)
    return


def update_hv_in_existing_h5(run_list=lv17data.epix_runs,
                             fpath=lv17data.h5path,
                             hvpath=lv17data.h5path,
                             data_scope='epix'):
    for run in run_list:
        fname = os.path.join(fpath, 'r{:03d}_{}.h5'.format(run, data_scope))
        hvname = os.path.join(hvpath, 'r{:03d}_{}.h5'.format(run, 'hv'))
        with h5py.File(hvname, 'r') as hvf:
            hv_t = np.asarray(hvf['timestamp'])
            hv_hv = np.asarray(hvf['photon_energy'])
            hv_st = np.asarray(hvf['source_temp'])

            with h5py.File(fname, 'r+') as f:
                f_t = np.asarray(f['timestamp'])
                f_hv = np.asarray(f['photon_energy'], dtype=np.single)
                f_st = np.asarray(f['source_temp'], dtype=np.single)
                for i, t in enumerate(f_t):
                    idx = np.where(hv_t == t)
                    f_hv[i] = hv_hv[idx]
                    f_st[i] = hv_st[idx]

                print('r{:03d} replacing existing entries...'.format(run))
                del f['photon_energy']
                f['photon_energy'] = f_hv
                del f['source_temp']
                f['source_temp'] = f_st
                print('done')

    return


if __name__ == "__main__":
    # for data_scope in
    # update_hv_in_existing_h5(run_list=lv17data.epix_runs,
    #                          fpath=h5data.h5paths['filt'],
    #                          data_scope)
