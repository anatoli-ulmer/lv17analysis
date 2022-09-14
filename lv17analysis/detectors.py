import os
import numpy as np
import psana as ps
from lv17analysis import epix, lv17data, tof


exp_name = 'tmolv1720'
run_list = [296]
detector_list = ['epix100', 'hsd', 'tmo_opal3',  'gmd', 'xgmd', 'ebeam', 'pcav', 
                 'xtcav', 'timing', 'evr_ch0_delay', 'source_valve_temp', 'epicsinfo']

data_list = ['epix', 'tof', 'spectrometer', 'gmd', 'xgmd', 'photon_energy', 'pcav',
             'xtcav',  'source_on', 'source_delay', 'source_temp', 'epicsinfo']

# get data set name from detector name
detector_dict = {'epix100' : 'epix',
                 'hsd' : 'tof',
                 'tmo_opal3' : 'spectrometer',
                 'gmd' : 'gmd',
                 'xgmd' : 'xgmd',
                 'ebeam' : 'photon_energy',
                 'pcav' : 'pcav',
                 'xtcav' : 'xtcav',
                 'timing': 'source_on',
                 'evr_ch0_delay' : 'source_delay',
                 'source_valve_temp' : 'source_temp',
                 'epicsinfo' : 'epicsinfo'
                }

# get detector name from data set name
data_dict = {'epix' : 'epix100',
             'tof' : 'hsd',
             'spectrometer' : 'tmo_opal3',
             'gmd' : 'gmd',
             'xgmd' : 'xgmd',
             'photon_energy' : 'ebeam',
             'pcav' : 'pcav',
             'xtcav' : 'xtcav',
             'source_on' : 'timing',
             'source_delay' : 'evr_ch0_delay',
             'source_temp' : 'source_valve_temp',
             'epicsinfo' : 'epicsinfo'
            }


'''
relevant detector entries:

hsd.raw.padded(evt)
hsd.raw.peak_times(evt)
hsd.raw.peaks(evt)
hsd.raw.waveforms(evt)
pcav.raw.charge1(evt)
pcav.raw.charge2(evt)
pcav.raw.fitTime1(evt)
pcav.raw.fitTime2(evt)
xgmd.raw.avgIntensity(evt)
xgmd.raw.electron1BkgNoiseAvg(evt)
xgmd.raw.electron2BkgNoiseAvg(evt)
xgmd.raw.energy(evt)
xgmd.raw.rmsElectronSum(evt)
xgmd.raw.xpos(evt)
xgmd.raw.ypos(evt)
gmd.raw.avgIntensity(evt)
gmd.raw.electron1BkgNoiseAvg(evt)
gmd.raw.electron2BkgNoiseAvg(evt)
gmd.raw.energy(evt)
gmd.raw.rmsElectronSum(evt)
gmd.raw.xpos(evt)
gmd.raw.ypos(evt)
ebeam.raw.damageMask(evt)
ebeam.raw.ebeamCharge(evt)
ebeam.raw.ebeamDumpCharge(evt)
ebeam.raw.ebeamEnergyBC1(evt)
ebeam.raw.ebeamEnergyBC2(evt)
ebeam.raw.ebeamL3Energy(evt)
ebeam.raw.ebeamLTU250(evt)
ebeam.raw.ebeamLTU450(evt)
ebeam.raw.ebeamLTUAngY(evt)
ebeam.raw.ebeamLTUPosX(evt)
ebeam.raw.ebeamLTUPosY(evt)
ebeam.raw.ebeamLUTAngX(evt)
ebeam.raw.ebeamPhotonEnergy(evt)
ebeam.raw.ebeamPkCurrBC1(evt)
ebeam.raw.ebeamPkCurrBC2(evt)
ebeam.raw.ebeamUndAngX(evt)
ebeam.raw.ebeamUndAngY(evt)
ebeam.raw.ebeamUndPosX(evt)
ebeam.raw.ebeamUndPosY(evt)
ebeam.raw.ebeamXTCAVAmpl(evt)
ebeam.raw.ebeamXTCAVPhase(evt)
timing.raw.eventcodes(evt)
xtcav.raw.value(evt)
epix100.raw.calib(evt)
epix100.raw.image(evt)
epix100.raw.raw(evt)
tmo_opal3.raw.calib(evt)
tmo_opal3.raw.image(evt)
tmo_opal3.raw.raw(evt)
'''


relevant_detectors = {
    ('hsd', 'raw'): ['waveforms'],
    ('pcav', 'raw'): ['charge1', 'charge2', 'fitTime1', 'fitTime2'],
    ('xgmd', 'raw'): ['avgIntensity', 'electron1BkgNoiseAvg', 'electron2BkgNoiseAvg',
                      'energy', 'rmsElectronSum', 'xpos', 'ypos'],
    ('gmd', 'raw'): ['avgIntensity', 'electron1BkgNoiseAvg', 'electron2BkgNoiseAvg',
                     'energy', 'rmsElectronSum', 'xpos', 'ypos'],
    ('ebeam', 'raw'): ['damageMask', 'ebeamCharge', 'ebeamDumpCharge', 'ebeamEnergyBC1',
                       'ebeamEnergyBC2', 'ebeamL3Energy', 'ebeamLTU250', 'ebeamLTU450',
                       'ebeamLTUAngY', 'ebeamLTUPosX', 'ebeamLTUPosY', 'ebeamLUTAngX',
                       'ebeamPhotonEnergy', 'ebeamPkCurrBC1', 'ebeamPkCurrBC2',
                       'ebeamUndAngX', 'ebeamUndAngY', 'ebeamUndPosX', 'ebeamUndPosY',
                       'ebeamXTCAVAmpl', 'ebeamXTCAVPhase'],
    ('timing', 'raw'): ['eventcodes'],
    ('xtcav', 'raw'): ['value'],
    ('epix100', 'raw'): ['calib', 'image', 'raw'],
    ('tmo_opal3', 'raw'): ['calib', 'image', 'raw'],
    'evr_ch0_delay': [],
    'source_valve_temp': []
}


def get_detector_info(run=296):
    # if np.ndim(run) == 0:
    #     run_list = [run]
    # else:
    #     run_list = run
    ds = ps.DataSource(exp=exp_name, run=[run])

    for ds_run in ds.runs():
        detector_info = ds_run.detinfo

    return detector_info


def get_detector_keys(run):
    detector_info = get_detector_info(run)
    detector_keys = [key for key, val in detector_info.items()]
    return detector_keys


def get_detector_active(ds_run):
    det_info = get_detector_info(ds_run.runnum)
    det_active = []
    for key, val in det_info.items():
        if len(val) > 0:
            det_active.append(key[0])
    return det_active
    

def get_epix(ds_run, evt):
    return epix.get_img(ds_run, evt, img_selector='image', cm=True, masked=True)


def get_tof(ds_run, evt):
    return tof.get_trace(ds_run.Detector('hsd'), evt)


def get_spectrometer(ds_run, evt):
    return ds_run.Detector('tmo_opal3').raw.image(evt)


def get_gmd(ds_run, evt):
    return ds_run.Detector('gmd').raw.energy(evt)


def get_xgmd(ds_run, evt):
    return ds_run.Detector('xgmd').raw.energy(evt)


def get_photon_energy(ds_run, evt):
    return ds_run.Detector('ebeam').raw.ebeamPhotonEnergy(evt)


def get_evt_codes(ds_run, evt):
    return ds_run.Detector('timing').raw.eventcodes(evt)


def get_evt_code_71(ds_run, evt):
    return ds_run.Detector('timing').raw.eventcodes(evt)[71]


def get_pcav(ds_run, evt):
    charge1 = ds_run.Detector('pcav').raw.charge1(evt)
    charge2 = ds_run.Detector('pcav').raw.charge2(evt)
    fitTime1 = ds_run.Detector('pcav').raw.fitTime1(evt)
    fitTime2 = ds_run.Detector('pcav').raw.fitTime2(evt)
    return charge1, charge2, fitTime1, fitTime2


def get_xtcav(ds_run, evt):
    return ds_run.Detector('xtcav').raw.value(evt)


def get_source_on(ds_run, evt):
    return get_evt_code_71(ds_run, evt)


def get_source_delay(ds_run, evt):
    return ds_run.Detector('evr_ch0_delay')(evt)


def get_source_temp(ds_run, evt):
    return ds_run.Detector('source_valve_temp')(evt)


def get_epicsinfo(ds_run, evt):
    return ds_run.Detector('epicsinfo')(evt)

