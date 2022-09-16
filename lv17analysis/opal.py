import numpy as np
from psana import DataSource


def run_mean(exp='tmolv1720', run=75, nmax=2000):

    ds = DataSource(exp=exp, run=run)
    ds_run = next(ds.runs())

    tmo_opal3 = ds_run.Detector('tmo_opal3')
    gmd = ds_run.Detector('gmd')
    xgmd = ds_run.Detector('xgmd')
    ebeam = ds_run.Detector('ebeam')

    opal_sum = np.zeros((1024, 1024), dtype=float)
    ncollected = 0

    for i, evt in enumerate(ds_run.events()):

        if ncollected > nmax:
            break

        opal_img = tmo_opal3.raw.image(evt)
        gmd_energy = gmd.raw.energy(evt)
        xgmd_energy = xgmd.raw.energy(evt)
        L3_val = ebeam.raw.ebeamL3Energy(evt)

        if gmd_energy is None:
            continue

        if opal_img is None:
            continue

        opal_sum += np.asarray(opal_img, dtype=float) / gmd_energy
        ncollected += 1

    return opal_sum / ncollected


def run_mean_spec(exp='tmolv1720', run=75, nmax=2000):
    opal_run_mean = run_mean(exp=exp, run=run, nmax=nmax)
    return spec_from_img(opal_run_mean)


def roi_from_img(opal_img, pad_pix=3, bg_pix_offset=100):
    bright_pix_pos = np.argmax(np.sum(opal_img[:, :], axis=1))
    return opal_img[bright_pix_pos-pad_pix+1: bright_pix_pos+pad_pix, :]


def spec_from_img(opal_img, pad_pix=3, bg_pix_offset=100):
    bright_pix_pos = np.argmax(np.sum(opal_img[:, :], axis=1))
    dark_pix_pos = bright_pix_pos - bg_pix_offset
    spec_bg = np.mean(opal_img[dark_pix_pos-pad_pix+1: dark_pix_pos+pad_pix, :], axis=0)
    img_roi = roi_from_img(opal_img, pad_pix=pad_pix, bg_pix_offset=bg_pix_offset)
    spec_roi = np.mean(img_roi, axis=0)
    return spec_roi - spec_bg
