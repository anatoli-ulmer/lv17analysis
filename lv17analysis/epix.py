import numpy as np
import psana as ps
import pickle as pkl
import os
from lv17analysis.helpers import *
from lv17analysis import lv17data


parameter = {
    'px': 50e-6,
    'distance': 0.395,
    'size': np.asarray([352, 384])}

# center position for run 219 from slack channel
center = np.asarray([403, 677])


def get_img(evt, img_selector='image', cm=True, masked=False, mask=None):
    '''
    Parameters:
        evt : element of ds_run.events()
            psana evt object
        img_selector : 'raw', 'calib' or 'image', optional
            Choose which image to ask from the psana detector calibration function.
            Choosing 'image' (default) is recommended.
        cm : bool, optional
            Apply common mode correction? Default = True
        masked : bool, optional
            Apply masking? Default = True
    Returns:
        img : 2D array
            corrected and masked epix100 detector image

    Anatoli Ulmer, 2022
    '''
    det = evt.run().Detector('epix100')

    if img_selector == 'image':
        img = det.raw.image(evt)
    elif img_selector == 'calib':
        img_raw = det.raw.calib(evt)
        img = img_raw[0].transpose()
    elif img_selector == 'raw':
        img_raw = det.raw.raw(evt)
        img = np.asarray(img_raw[0], dtype=np.float64).transpose()
    else:
        raise Exception("img_selector has to be one of the following options:\
                        'raw', 'calib' or 'image' (default)")

    if img is not None:
        if cm:
            cm_thresh = 0.5 * lv17data.photon_energy_kev(evt.run().runnum)
            img = cm_correction(img, cm_thresh=cm_thresh)
        if masked:
            img = masked_img(img, mask=mask)

    return img


def nan_cross(img):
    img[:, 352:356] = np.nan
    img[384:388, :] = np.nan
    return img


def count_lit(img, lit_thresh=0.5):
    if np.shape(img) == ():
        return -99999
    return np.nansum(img.flatten() >= lit_thresh)


def get_mask(img):
    mask = np.ones(img.shape, dtype=bool)
    mask[:, 352:356] = False
    mask[384:388, :] = False
    mask[196:442, 610:] = False
    mask[1*96-1, :] = False
    mask[2*96-1, :] = False
    mask[3*96-1, :] = False
    mask[5*96+5-1, :] = False
    mask[6*96+5-1, :] = False
    mask[7*96+5-1, :] = False
    # mask[:, 351:357] = False
    # mask[383:389, :] = False
    # mask[196:442, 610:] = False
    # mask[1*96-1: 1*96, :] = False
    # mask[2*96-1: 2*96, :] = False
    # mask[3*96-1: 3*96, :] = False
    # mask[5*96+5-1: 5*96+5, :] = False
    # mask[6*96+5-1: 6*96+5, :] = False
    # mask[7*96+5-1: 7*96+5, :] = False
    mask[593:597, 662:665] = False
    mask[543:547, 356:653] = False

    mask[545:550, 0:352] = False
    # mask[541:547, 644:649] = False
    mask[541:544, 642:650, ] = False
#     mask[539:541, 643:649] = False
#     mask[543, 641:651] = False
#     mask[546, 644:653] = False
#     mask[617, 560:562] = False
#     mask[681:691, 629:640] = False
#      # mask[534, 662] = False
#     mask[540:543, 427:431] = False

    return mask


def masked_img(img, mask=None):
    if mask is None or len(mask) == 0:
        mask = get_mask(img)
    # ret = np.array(img, dtype=np.float64)
    # ret[~mask] = np.nan
    return np.ma.array(img, mask=~mask, dtype=np.float64)


def offset_correction(img, offs_thresh=0.5, x_chunks=2, y_chunks=8):
    ny, nx = img.shape
    nx_chunk = (nx-5)/x_chunks
    ny_chunk = (ny-5)/y_chunks

    for ix in range(x_chunks):
        for iy in range(y_chunks):
            img_tile = img[int(iy*ny_chunk):int((iy+1)*ny_chunk-1),
                           int(ix*nx_chunk):int((ix+1)*nx_chunk-1)]
            img_tile = img_tile - np.nanmedian(img_tile[img_tile < offs_thresh])

    return img


def cm_correction(img, axis=None, cm_thresh=0.5):
    nrows, ncols = img.shape
    ny, nx = np.int16(nrows/2), np.int16(ncols/2)

    img_cm = np.copy(img)
    img_cm[img_cm > cm_thresh] = np.nan

    if axis is None:
        img = cm_correction(img, axis=0, cm_thresh=cm_thresh)
        img = cm_correction(img, axis=1, cm_thresh=cm_thresh)
    elif axis == 0:
        img[:, :(nx-1)] -= np.nanmedian(img_cm[:, :(nx-1)], axis=1).reshape(nrows, 1)
        img[:, -(nx-1):] -= np.nanmedian(img_cm[:, -(nx-1):], axis=1).reshape(nrows, 1)
    elif axis == 1:
        img[:(ny-1), :] -= np.nanmedian(img_cm[:(ny-1), :], axis=0).reshape(1, ncols)
        img[-(ny-1):, :] -= np.nanmedian(img_cm[-(ny-1):, :], axis=0).reshape(1, ncols)

    return img


def calc_sum_lit(exp='tmolv1720', run=[170], lit_thresh=0.5,
                 save_dir='/cds/data/psdm/tmo/tmolv1720/results/shared/epix100'):

    img_sum = []
    img_lit = []
    delay = []

    ds = ps.DataSource(exp=exp, run=run)

    for ds_run in ds.runs():

        print('processing run {:d} ...'.format(ds_run.runnum))

        epix100_det = ds_run.Detector('epix100')
        timing_det = ds_run.Detector('evr_ch0_delay')
        ebeam_det = ds_run.Detector("ebeam")  # electron beam parameters
        gmd_det = ds_run.Detector('gmd')  # gas monitoring detector
        xgmd_det = ds_run.Detector('xgmd')  # gas monitoring detector
        hsd_det = ds_run.Detector('hsd')  # gas monitoring detector

        for ievt, evt in enumerate(ds_run.events()):
            if np.mod(ievt, 1000) == 0:
                print('\t ... event {:d}'.format(ievt))

            epix100_data = epix100_det.raw.image(evt)
            timing_data = timing_det(evt)
            ebeam_data = ebeam_det.raw.ebeamPhotonEnergy(evt)
            gmd_data = gmd_det.raw.energy(evt)
            xgmd_data = xgmd_det.raw.energy(evt)
            hsd_data = hsd_det.raw.waveforms(evt)

            if gmd_data is None or xgmd_data is None:
                continue

            if epix100_data is None:
                continue

            if timing_data is None:
                continue

            epix_img = nan_cross(epix100_data)
            epix_zero = np.nanmedian(epix_img)
            epix_img = epix_img-epix_zero

            epix_sum = np.nansum(epix_img.flatten())
            lit_pix = count_lit(epix_img, lit_thresh=lit_thresh)

            img_sum.append(epix_sum)
            img_lit.append(lit_pix)
            delay.append(timing_data)

    _pkl_save(os.path.join(save_dir, 'r{:04d}_delay.pkl'.format(ds_run.runnum)))
    _pkl_save(os.path.join(save_dir, 'r{:04d}_sums.pkl'.format(ds_run.runnum)))
    _pkl_save(os.path.join(save_dir, 'r{:04d}_lit.pkl'.format(ds_run.runnum)))
    return delay, img_sum, img_lit


def read_quantum_efficiency_csv(filename="Nonesmoothed.csv"):
    '''
    Choose between the following filter files:
        'None_smoothed.csv' (default)
        'None300um.csv'
        'None300um_smoothed.csv'
        'Al25um.csv'
        'Al25um_smoothed.csv'
        'Be25um.csv'
        'Be25um_smoothed.csv'
        'Kapton30um.csv'
        'Kapton30um_smoothed.csv'

    Returns:
        ev - photon energy in electronvolts
        qe - quantum efficiency for given filter

    Anatoli Ulmer, 2022
    '''
    # abs_filepath = os.path.dirname(__file__) + rel_filepath
    # dir_list = os.listdir(abs_filepath)
    # print(dir_list)

    rel_filepath = "../calibration_data/epix100/quantum_efficiency/"
    abs_filepath = os.path.join(os.path.dirname(__file__), rel_filepath, filename)

    data = np.loadtxt(abs_filepath, delimiter=',')
    ev = data[:, 0]
    qe = data[:, 1]

    return ev, qe


def quantum_efficiency(photon_energy_ev, filename="None_smoothed.csv"):
    '''
    Anatoli Ulmer, 2022
    '''
    ev, qe = read_quantum_efficiency_csv(filename=filename)
    quantum_efficiency = np.interp(photon_energy_ev, ev, qe)

    return quantum_efficiency

