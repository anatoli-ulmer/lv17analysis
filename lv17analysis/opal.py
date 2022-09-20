import numpy as np
from psana import DataSource


def run_mean(run=75, exp='tmolv1720', nmax=2000):

    ds = DataSource(exp=exp, run=run, detectors=['tmo_opal3', 'gmd', 'xgmd'])
    ds_run = next(ds.runs())

    tmo_opal3 = ds_run.Detector('tmo_opal3')
    gmd = ds_run.Detector('gmd')
    xgmd = ds_run.Detector('xgmd')

    opal_sum = np.zeros((1024, 1024), dtype=float)
    ncollected = 0

    for i, evt in enumerate(ds_run.events()):

        if ncollected > nmax:
            break

        opal_img = tmo_opal3.raw.raw(evt)
        gmd_energy = gmd.raw.energy(evt)
        xgmd_energy = xgmd.raw.energy(evt)

        if gmd_energy is None:
            continue

        if opal_img is None:
            continue

        opal_sum += np.asarray(opal_img, dtype=float) / gmd_energy
        ncollected += 1

    return opal_sum / ncollected


def run_mean_spec(run=75, exp='tmolv1720', nmax=2000):
    opal_run_mean = run_mean(exp=exp, run=run, nmax=nmax)
    return spec_from_img(opal_run_mean)


def get_center_px(opal_img):
    return np.argmax(np.nansum(opal_img[:, :], axis=1))


def roi_from_img(opal_img, center_px=None, pad_px=3):
    if center_px is None:
        center_px = get_center_px(opal_img)
    pad = [np.max([0, center_px-pad_px]), np.min([1023, center_px+pad_px+1])]
    return opal_img[pad[0]:pad[1], :]


def spec_from_img(opal_img, center_px=None, pad_px=3, bg_offset=100):
    if center_px is None:
        center_px = get_center_px(opal_img)
    bg_px_pos = center_px - bg_offset * (-1)**(center_px > 512)
    spec_roi = np.nanmean(roi_from_img(opal_img, center_px=center_px, pad_px=pad_px), axis=0)
    spec_bg = np.nanmean(opal_img[bg_px_pos-pad_px: bg_px_pos+pad_px+1, :], axis=0)
    return spec_roi - spec_bg