import os
import numpy as np
import psana as ps
from lv17analysis import epix, lv17data


exp_name = 'tmolv1720'
run_list = [296]
detector = ['gmd', 'xgmd', 'ebeam', 'hsd', 'epix100']


def get_detector_info(run=296):
    # if np.ndim(run) == 0:
    #     run_list = [run]
    # else:
    #     run_list = run

    ds = ps.DataSource(exp=exp_name, run=[run])

    for ds_run in ds.runs():

        detector_info = ds_run.detinfo
        # print(detector_info)

        epix100_det = ds_run.Detector('epix100')
        timing_det = ds_run.Detector('evr_ch0_delay')
        ebeam_det = ds_run.Detector("ebeam")  # electron beam parameters
        gmd_det = ds_run.Detector('gmd')  # gas monitoring detector
        xgmd_det = ds_run.Detector('xgmd')  # gas monitoring detector
        hsd_det = ds_run.Detector('hsd')  # gas monitoring detector
        evr_det = ds_run.Detector('timing')
    
    return detector_info


def get_detector_keys(run):
    detector_info = get_detector_info(run)
    detector_keys = [key for key, val in detector_info.items()]
    return detector_keys