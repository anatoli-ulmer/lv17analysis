import numpy as np
import psana as ps
import matplotlib.pyplot as plt


def run_hist(exp='tmolv1720', run=[13, 14, 15], oneplot=True, figsize=(20, 5), alpha=0.5,
             xlim=None, detector='gmd', normalized=True, histtype='step'):

    ''' Creates histograms for run list for either detector='gmd' or detector='xgmd' '''

    ds = ps.DataSource(exp=exp, run=run, detectors=[detector])

    if oneplot:
        plt.figure(figsize=figsize)

    for ds_run in ds.runs():
        if type(ds_run) == ps.psexp.null_ds.NullRun:
            sleep(0.1)
            print('Nullrunning! Continuing...')
            continue

        gmd = ds_run.Detector(detector).raw.energy
        gmd_array = []

        for i, evt in enumerate(ds_run.events()):

            # gmd
            gmd_energy = gmd(evt)
            if gmd_energy is None:
                continue

            gmd_array.append(gmd_energy)

        if not oneplot:
            plt.figure(figsize=figsize)

        if histtype == 'step':
            alpha = 1

        plt.hist(gmd_array, alpha=alpha, histtype=histtype, density=normalized,
                 label='run {:n}'.format(ds_run.runnum))

        if xlim is None:
            if detector == 'gmd':
                xlim = [0, 2.5]
            else:
                xlim = [0, 1.2]

        plt.xlim(xlim)
        plt.legend()

        if normalized is True:
            prefix = 'normalized '
        else:
            prefix = ''
        plt.title('{}{} histogram'.format(prefix, detector))
