import os
import numpy as np
import psana as ps
from lv17analysis import epix, lv17data, tof

###################################################################################################

exp_name = 'tmolv1720'
run_list = [296]
detector_list = ['epix100', 'hsd', 'tmo_opal3', 'gmd', 'xgmd', 'ebeam', 'pcav', 'xtcav',
                 'timing', 'evr_ch0_delay', 'source_valve_temp', 'epicsinfo',
                 'photon_energy_welch']

data_list = ['epix', 'tof_trace', 'spectrometer', 'gmd', 'xgmd', 'ebeam', 'pcav', 'xtcav',
             'source_on', 'source_delay', 'source_temp', 'epicsinfo', 'photon_energy']

# get data set name from detector name
detector_dict = {'epix100': 'epix',
                 'hsd': 'tof_trace',
                 'tmo_opal3': 'spectrometer',
                 'gmd': 'gmd',
                 'xgmd': 'xgmd',
                 'ebeam': 'ebeam',
                 'photon_energy_welch': 'photon_energy',
                 'pcav': 'pcav',
                 'xtcav': 'xtcav',
                 'timing': 'source_on',
                 'evr_ch0_delay': 'source_delay',
                 'source_valve_temp': 'source_temp',
                 'epicsinfo': 'epicsinfo'}

# get detector name from data set name
data_dict = {'epix': 'epix100',
             'tof_trace': 'hsd',
             'spectrometer': 'tmo_opal3',
             'gmd': 'gmd',
             'xgmd': 'xgmd',
             'ebeam': 'ebeam',
             'photon_energy': 'photon_energy_welch',
             'pcav': 'pcav',
             'xtcav': 'xtcav',
             'source_on': 'timing',
             'source_delay': 'evr_ch0_delay',
             'source_temp': 'source_valve_temp',
             'epicsinfo': 'epicsinfo'}


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


all_detectors = {
    ('hsd', 'raw'): ['padded', 'peak_times', 'peaks', 'waveforms'],
    ('epicsinfo', 'epicsinfo'): [],
    ('pcav', 'raw'): ['charge1', 'charge2', 'fitTime1', 'fitTime2'],
    ('xgmd', 'raw'): ['avgIntensity', 'electron1BkgNoiseAvg', 'electron2BkgNoiseAvg',
                      'energy', 'rmsElectronSum', 'xpos', 'ypos'],
    ('gmd', 'raw'): ['avgIntensity', 'electron1BkgNoiseAvg', 'electron2BkgNoiseAvg', 'energy',
                     'rmsElectronSum', 'xpos', 'ypos'],
    ('ebeam', 'raw'): ['damageMask', 'ebeamCharge', 'ebeamDumpCharge', 'ebeamEnergyBC1',
                       'ebeamEnergyBC2', 'ebeamL3Energy', 'ebeamLTU250', 'ebeamLTU450',
                       'ebeamLTUAngY', 'ebeamLTUPosX', 'ebeamLTUPosY', 'ebeamLUTAngX',
                       'ebeamPhotonEnergy', 'ebeamPkCurrBC1', 'ebeamPkCurrBC2', 'ebeamUndAngX',
                       'ebeamUndAngY', 'ebeamUndPosX', 'ebeamUndPosY', 'ebeamXTCAVAmpl',
                       'ebeamXTCAVPhase'],
    ('timing', 'raw'): ['eventcodes'],
    ('xtcav', 'raw'): ['value'],
    ('epix100', 'raw'): ['calib', 'image', 'raw'],
    ('tmo_opal3', 'raw'): ['calib', 'image', 'raw'],
    'evr_ch0_delay': [],
    'source_valve_temp': [],
    'photon_energy_welch': []
}


relevant_detectors = {
    ('hsd', 'raw'): ['waveforms'],
    ('xgmd', 'raw'): ['energy'],
    ('gmd', 'raw'): ['energy'],
    ('timing', 'raw'): ['eventcodes'],
    ('xtcav', 'raw'): ['value'],
    ('epix100', 'raw'): ['image'],
    ('tmo_opal3', 'raw'): ['image'],
    'evr_ch0_delay': [],
    'source_valve_temp': [],
    'photon_energy_welch': []
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


def get_active_detectors(ds_run):
    det_info = get_detector_info(ds_run.runnum)
    det_active = []
    for key, val in det_info.items():
        if len(val) > 0:
            det_active.append(key[0])
    return det_active


def detectors2datasets(detectors_list):
    return [detector_dict[det_str] for det_str in detectors_list]


def get(data_str, evt):
    return eval("get_{}(evt)".format(data_str))


def get_epix(evt):
    return epix.get_img(evt, img_selector='image', cm=True, masked=False)


def get_tof(evt):
    return tof.get_trace(evt)


def get_tof_trace(evt):
    times, trace = tof.get_trace(evt)
    return trace


def get_tof_times(evt):
    times, trace = tof.get_trace(evt)
    return times


def get_spectrometer(evt):
    return evt.run().Detector('tmo_opal3').raw.image(evt)


def get_gmd(evt):
    return evt.run().Detector('gmd').raw.energy(evt)


def get_xgmd(evt):
    return evt.run().Detector('xgmd').raw.energy(evt)


def get_photon_energy(evt):
    return evt.run().Detector('photon_energy_welch')(evt)


def get_evt_codes(evt):
    return evt.run().Detector('timing').raw.eventcodes(evt)


def get_evt_code_71(evt):
    return evt.run().Detector('timing').raw.eventcodes(evt)[71]


def get_pcav(evt):
    charge1 = evt.run().Detector('pcav').raw.charge1(evt)
    charge2 = evt.run().Detector('pcav').raw.charge2(evt)
    fitTime1 = evt.run().Detector('pcav').raw.fitTime1(evt)
    fitTime2 = evt.run().Detector('pcav').raw.fitTime2(evt)
    return np.array([charge1, charge2, fitTime1, fitTime2])


def get_xtcav(evt):
    return evt.run().Detector('xtcav').raw.value(evt)


def get_ebeam(evt):
    data = []
    for det in all_detectors[('ebeam', 'raw')]:
        data.append(eval("evt.run().Detector('ebeam').raw.{}(evt)".format(det)))
    return data


def get_source_on(evt):
    return get_evt_code_71(evt)


def get_source_delay(evt):
    return evt.run().Detector('evr_ch0_delay')(evt)


def get_source_temp(evt):
    return evt.run().Detector('source_valve_temp')(evt)


def get_epicsinfo(evt):
    return evt.run().Detector('epicsinfo')(evt)

