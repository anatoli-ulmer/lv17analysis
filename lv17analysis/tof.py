import os
import numpy as np
import psana as ps
import matplotlib.pyplot as plt
import warnings
import pickle as pkl

import math
from pylab import polyfit, plot, show


def get_trace(evt, tof_channel=None, ranges_bg=[0, 10000], n_digitizers_per_channel=4):
    '''
    Parameters:
        evt : element of ds_run.events()
            psana evt object
        tof_channel : int, optional
            ADC channel
        ranges_bg : 2 element array, optional
            element range for offset subtraction
        n_digitizers_per_channel : int, optional
             Number of digitizers per tof_channel. Relevant for subtracting offset
             from each of the digitizers independently. Is typically a power of 2.
    Returns:
        tof_trace : array
            ToF trace with subtracted offset.
        tof_times : array
            ToF times

    Anatoli Ulmer, 2022
    '''
    hsd = evt.run().Detector('hsd')
    hsd_data = hsd.raw.waveforms(evt)

    if hsd_data is None:
        return None, None

    if tof_channel is None:
        # If no channel is specified and only one hsd channel is active, it will be chosen.
        # If multiple channels are active, the default channel 3 will be chosen.
        if len(hsd_data.keys()) == 1:
            tof_channel = list(hsd_data.keys())[0]  # the digitzer channel the tof is on
        else:
            tof_channel = 3  # the default is channel 3

    tof_data = hsd_data[tof_channel]

    if tof_data is None:
        return None, None

    tof_times = tof_data['times']
    tof_trace = np.asarray(tof_data[0], dtype=float)  # the actual tof data
    tof_trace = bg_correction(tof_trace, ranges_bg=ranges_bg, nchannels=n_digitizers_per_channel)

    # convert to mV
    _hsd_fs_range_vpp = hsd.raw._seg_configs()[tof_channel].config.expert.fs_range_vpp
    tof_trace *= _hsd_to_mv(_hsd_fs_range_vpp)
    return tof_times, tof_trace


def bg_correction(trace, ranges_bg=[0, 10000], nchannels=4):
    for i in np.arange(nchannels):
        bg = np.mean(trace[i + ranges_bg[0]:ranges_bg[1]:nchannels])
        trace[i::nchannels] = trace[i::nchannels] - bg
    return trace


def write_run_mean(exp='tmolv1720', run=[6],
                   tof_channel=None, ranges_bg=[0, 10000], nchannels=4,
                   gmd_min=0.0, gmd_max=np.inf, gmd_norm=False,
                   xgmd_min=0.0, xgmd_max=np.inf, xgmd_norm=False,
                   tof_dir='/cds/data/psdm/tmo/tmolv1720/results/shared/tof_run_mean'):

    ds = ps.DataSource(exp=exp, run=run, detectors=['hsd', 'gmd', 'xgmd'])

    for ds_run in ds.runs():
        if isinstance(ds_run, ps.psexp.null_ds.NullRun):
            sleep(0.1)
            print('Nullrunning! Continuing...')
            continue

        hsd = ds_run.Detector('hsd')
        gmd = ds_run.Detector('gmd')
        xgmd = ds_run.Detector('xgmd')

        num_traces = 0

        for i, evt in enumerate(ds_run.events()):

            # gmd
            gmd_energy = gmd.raw.energy(evt)
            xgmd_energy = xgmd.raw.energy(evt)
            if gmd_energy is None or gmd_energy < gmd_min or gmd_energy > gmd_max:
                continue

            if xgmd_energy is None or xgmd_energy < xgmd_min or xgmd_energy > xgmd_max:
                continue

            # ToF
            hsd_data = hsd.raw.waveforms(evt)
            if tof_channel is None:
                if not len(hsd_data.keys()) == 1:
                    _default_tof_channel = 3
                    warnings.warn('Multiple hsd channels in run {:d}.\
                                  Using default tof channel {:d}.'.
                                  format(ds_run.runnum, _default_tof_channel))
                    tof_channel = _default_tof_channel
                else:
                    tof_channel = list(hsd_data.keys())[0]  # the digitzer channel the tof is on

            tof_data = hsd_data[tof_channel]
            tof_times = tof_data['times']  # the times
            tof_trace = np.asarray(tof_data[0], dtype=float)  # the actual tof data

            tof_trace = bg_correction(tof_trace)
            _hsd_fs_range_vpp = hsd.raw._seg_configs()[3].config.expert.fs_range_vpp
            tof_trace = tof_trace * _hsd_to_mv(_hsd_fs_range_vpp)

            if gmd_norm is True:
                tof_trace = tof_trace / gmd_energy

            if xgmd_norm is True:
                tof_trace = tof_trace / xgmd_energy

            if num_traces == 0:
                tof_sum = tof_trace
            else:
                tof_sum += tof_trace

            num_traces += 1

        tof_mean = tof_sum / num_traces
        tof_save = [tof_times, tof_mean]
        filename = _get_filename(ds_run.runnum, tof_channel, gmd_min, gmd_max, gmd_norm,
                                 xgmd_min, xgmd_max, xgmd_norm, postfix='pkl')
        fullfilename = os.path.join(tof_dir, filename)

        # np.savetxt(fullfilename, tof_save, delimiter=",")
        _pkl_save(fullfilename, tof_save)


def read_run_mean(exp='tmolv1720', run=6, tof_channel=3, ranges_bg=[0, 10000], nchannels=4,
                  gmd_min=0.0, gmd_max=np.inf, gmd_norm=False,
                  xgmd_min=0.0, xgmd_max=np.inf, xgmd_norm=False,
                  tof_dir='/cds/data/psdm/tmo/tmolv1720/results/shared/tof_run_mean'):
    ''' reads and returns tof trace for a single run. trace has to be
    written by write_run_mean function prior to readout. If file does
    not exist, it will be generated with write_run_mean. '''

    if isinstance(run, list):
        raise Exception("read_run_mean expects a single run number as input and not a list!")

    filename = _get_filename(run, tof_channel, gmd_min, gmd_max, gmd_norm,
                             xgmd_min, xgmd_max, xgmd_norm, postfix='pkl')
    fullfilename = os.path.join(tof_dir, filename)

    if not os.path.exists(fullfilename):
        write_run_mean(exp=exp, run=run, tof_dir=tof_dir, tof_channel=tof_channel,
                       gmd_min=gmd_min, gmd_max=gmd_max, gmd_norm=gmd_norm,
                       xgmd_min=xgmd_min, xgmd_max=xgmd_max, xgmd_norm=xgmd_norm,
                       ranges_bg=ranges_bg, nchannels=nchannels)

    # tof_data = np.loadtxt(fullfilename, delimiter=",")
    tof_data = _pkl_load(fullfilename)

    tof_times = tof_data[0]
    tof_trace = tof_data[1]
    # tof_trace = bg_correction(tof_data[1], ranges_bg=ranges_bg, nchannels=nchannels)

    return tof_times, tof_trace


def fast_plot(exp='tmolv1720', run=[6], t_unit='µs', t_zero=0.0, t_min=None, t_max=None,
              time_axis=True, figsize=(25, 5), tof_channel=3,
              gmd_min=0.0, gmd_max=np.inf, gmd_norm=False,
              xgmd_min=0.0, xgmd_max=np.inf, xgmd_norm=False,
              tof_dir='/cds/data/psdm/tmo/tmolv1720/results/shared/tof_run_mean'):

    plt.figure(figsize=figsize)

    if t_unit == 'µs' or t_unit == 'us':
        _t_scale = 1e6
    elif t_unit == 'ns':
        _t_scale = 1e9
    elif t_unit == 'ms':
        _t_scale == 1e3
    else:
        _t_scale = 1

    for irun in run:
        tof_times, tof_trace = read_run_mean(run=irun, tof_channel=tof_channel, gmd_min=gmd_min,
                                             gmd_max=gmd_max, gmd_norm=gmd_norm, xgmd_min=xgmd_min,
                                             xgmd_max=xgmd_max, xgmd_norm=xgmd_norm)

        if time_axis is True:
            plot_x = tof_times * _t_scale - t_zero
        else:
            plot_x = np.arange(len(tof_trace))

        plt.plot(plot_x, tof_trace, '-', label='run {:n}'.format(irun))

    if t_min is None:
        t_min = plot_x[0]

    if t_max is None:
        t_max = plot_x[-1]

    plt.legend()
    plt.xlabel('flight time [{}]'.format(t_unit) if time_axis else 'digitizer units')
    plt.ylabel('tof signal [mV]')
    plt.xlim(t_min, t_max)
    plt.grid(which='both')


def _get_filename(run, tof_channel, gmd_min, gmd_max, gmd_norm, xgmd_min, xgmd_max, xgmd_norm,
                  postfix='pkl'):
    return 'r{:04n}_cnl{:d}_gmd{:.2f}to{:.2f}norm{:d}_xgmd{:.2f}to{:.2f}norm{:d}_tof_mean.{}'\
        .format(run, tof_channel, gmd_min, gmd_max, gmd_norm,
                xgmd_min, xgmd_max, xgmd_norm, postfix)


def _pkl_save(filename, data):
    with open(filename, 'wb') as f:
        pkl.dump(data, f)
    print("saved file {}\r".format(filename))


def _pkl_load(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)
    print("read file {}\r".format(filename))
    return data


def get_time_indices(tof_times, times):
    return np.searchsorted(tof_times, times)


def _hsd_to_mv(_hsd_fs_range_vpp):
    return (400 + _hsd_fs_range_vpp * (1040 - 480) / (65535 - 8192)) / 4096


def _get_peaks_filename(run, peaks_dir):
    return os.path.join(peaks_dir, 'r{:04d}_peak_indices.pkl'.format(run))


def write_peak_indices(run=[42], peaks=None,
                       peaks_dir='/cds/data/psdm/tmo/tmolv1720/results/shared/tof_peaks'):

    if not isinstance(run, list):
        run = [run]

    for irun in run:
        if peaks is None:
            peaks = {
                1: [0, 10],
                2: [41700, 41850],
                3: [40620, 40770],
                4: [39960, 40100],
                5: [39520, 39670],
                6: [39190, 39340],
                7: [38930, 39080],
                8: [38730, 38880],
                9: [38410, 38560],
                10: [0, 10],
                11: [0, 10],
                12: [0, 10],
                13: [0, 10],
                14: [0, 10],
                15: [0, 10],
                16: [0, 10],
                17: [0, 10],
                18: [0, 10],
                19: [0, 10],
                20: [0, 10]
            }

        _fullfilename = _get_peaks_filename(irun, peaks_dir)
        _pkl_save(_fullfilename, peaks)


def read_peak_indices(run=42, peaks_dir='/cds/data/psdm/tmo/tmolv1720/results/shared/tof_peaks'):

    _fullfilename = _get_peaks_filename(run, peaks_dir=peaks_dir)
    if not os.path.exists(_fullfilename):
        print('File {} does not exist. Creating new file with standard peak positions.'
              .format(_fullfilename))
        write_peak_indices(run=run, peaks_dir=peaks_dir)
    return _pkl_load(_fullfilename)


def calc_peak_ratios(run=[34, 35, 36, 37, 38, 39, 40, 41, 42],
                     charge_state_array=[[7, 2], [7, 3], [7, 4],
                                         [8, 2], [8, 3], [8, 4],
                                         [9, 2], [9, 3], [9, 4]],
                     exp='tmolv1720', tof_channel=3, ranges_bg=[0, 10000], nchannels=4,
                     gmd_min=0.0, gmd_max=np.inf, gmd_norm=False,
                     xgmd_min=0.0, xgmd_max=np.inf, xgmd_norm=False,
                     tof_dir='/cds/data/psdm/tmo/tmolv1720/results/shared/tof_run_mean'):

    peak_sum_ratio = np.zeros([len(run), np.shape(charge_state_array)[0]])

    for i, irun in enumerate(run):
        tof_times, tof_trace = read_run_mean(run=irun, tof_channel=tof_channel, gmd_min=gmd_min,
                                             gmd_max=gmd_max, gmd_norm=gmd_norm, xgmd_min=xgmd_min,
                                             xgmd_max=xgmd_max, xgmd_norm=xgmd_norm)
        peaks = read_peak_indices(run=irun)

        for j, charge_states in enumerate(charge_state_array):
            peak_sum1 = np.sum(np.abs(tof_trace[peaks[charge_states[0]][0]:
                                                peaks[charge_states[0]][1]]))
            peak_sum2 = np.sum(np.abs(tof_trace[peaks[charge_states[1]][0]:
                                                peaks[charge_states[1]][1]]))
            peak_sum_ratio[i, j] = peak_sum1 / peak_sum2

    return peak_sum_ratio


def calc_peak_sums(run=[34, 35, 36, 37, 38, 39, 40, 41, 42], peak_range=[35800, 35870],
                   exp='tmolv1720', tof_channel=3, ranges_bg=[0, 10000], nchannels=4,
                   gmd_min=0.0, gmd_max=np.inf, gmd_norm=False,
                   xgmd_min=0.0, xgmd_max=np.inf, xgmd_norm=False,
                   tof_dir='/cds/data/psdm/tmo/tmolv1720/results/shared/tof_run_mean'):

    peak_sums = np.zeros(len(run))

    for i, irun in enumerate(run):
        tof_times, tof_trace = read_run_mean(run=irun, tof_channel=tof_channel, gmd_min=gmd_min,
                                             gmd_max=gmd_max, gmd_norm=gmd_norm, xgmd_min=xgmd_min,
                                             xgmd_max=xgmd_max, xgmd_norm=xgmd_norm)

        peak_sums[i] = np.sum(np.abs(tof_trace[peak_range[0]:peak_range[1]]))

    return peak_sums


def draw_charged_states(argon_states=[7.027, 6.844, 6.734, 6.66, 6.605, 6.562,
                                      6.527, 6.5, 6.476, 6.455, 6.436],
                        exp='tmolv1720', run=[61], t_zero=0.0, t_min=6, t_max=8, figsize=(30, 10),
                        gmd_min=0.5, gmd_max=np.inf, gmd_norm=True, time_axis=True, t_units='µs',
                        water=True, nitrogen=True, carbon=True, m_q=[]):
    tau = list(argon_states)
    mq = []
    for i in range(2, (2 + len(tau))):
        mq.append(math.sqrt(39.948 / i))
    x = np.asarray(mq)
    y = np.asarray(tau)

    m, b = polyfit(x, y, 1)

    plot(x, y, 'yo', x, (m * x) + b, '--k')
    show()
    print('Slope= ' + str(m) + ', t0= ' + str(b))
    fast_plot(exp=exp, run=run, t_zero=t_zero, t_min=t_min, t_max=t_max, figsize=figsize,
              gmd_min=gmd_min, gmd_max=gmd_max, gmd_norm=gmd_norm,
              time_axis=time_axis, t_units=t_units)

    for i in range(1, 18):
        plt.axvline(x=(m * math.sqrt(39.948 / i) + b), ymin=-40, ymax=15, color='red',
                    linestyle='--')
    plt.axvline(x=(m * math.sqrt(39.948) + b), ymin=-40, ymax=15, color='red',
                linestyle='--', label='Ar')
    plt.axvline(x=b, ymin=-40, ymax=15, color='green', linestyle='--', label='t0')
    if water is True:
        plt.axvline(x=(m * math.sqrt(1 / 1) + b), ymin=-40, ymax=15, color='lawngreen',
                    linestyle='--', label='H+')
        plt.axvline(x=(m * math.sqrt(18 / 1) + b), ymin=-40, ymax=15,
                    color='lightgreen',
                    linestyle='--', label='H2O+')
        plt.axvline(x=(m * math.sqrt(17 / 1) + b), ymin=-40, ymax=15,
                    color='aquamarine', linestyle='--', label='OH+')
    if nitrogen is True:
        plt.axvline(x=(m * math.sqrt(14 / 1) + b), ymin=-40, ymax=15, color='orange',
                    linestyle='--', label='N+')
    if carbon is True:
        plt.axvline(x=(m * math.sqrt(12.011) + b), ymin=-40, ymax=15, color='grey',
                    linestyle='--', label='C+')
    for i in m_q:
        plt.axvline(x=(m * math.sqrt(i) + b), ymin=-40, ymax=15, color='yellow',
                    linestyle='--', label='additional')
    plt.legend()
