# Anatoli Ulmer, 2022
import numpy as np


# energies from eLog data table in electron volts (eV)
energy_elog_ev = np.array([
    874.65, 876.17, 875.22, 876.75, 874.55, 875.11, 875.33, 875.03, 804.51, 804.81,
    806.58, 806.40, 807.16, 806.81, 807.03, 806.87, 807.22, 806.98, 806.62, 806.48,
    805.72, 806.79, 806.68, 806.67, 806.76, 806.95, 806.27, 806.81, 806.63, 806.96,
    807.02, 808.10, 806.81, 806.74, 806.34, 806.69, 807.07, 806.83, 806.40, 806.53,
    806.69, 806.63, 806.58, 806.75, 806.95, 806.85, 806.61, 806.97, 807.22, 806.47,
    807.29, 807.12, 806.13, 807.03, 807.26, 806.71, 806.73, 799.12, 868.32, 867.60,
    872.88, 880.85, 880.83, 879.83, 880.55, 880.58, 880.69, 880.63, 880.56, 880.74,
    880.69, 880.71, 880.66, 880.89, 881.50, 881.64, 881.31, 863.19, 863.25, 900.53,
    900.51, 900.64, 900.67, 900.74, 900.56, 900.34, 900.65, 900.72, 900.79, 900.87,
    900.78, 900.68, 900.71, 900.69, 900.61, 900.75, 900.75, 900.72, 900.73, 900.74,
    900.65, 900.80, 900.69, 900.66, 900.70, 900.77, 900.69, 900.27, 900.77, 900.76,
    900.75, 900.69, 900.68, 900.78, 900.69, 900.73, 900.74, 900.80, 900.80, 900.59,
    900.78, 900.75, 900.67, 900.64, 900.49, 900.67, 900.68, 900.73, 900.68, 900.60,
    900.76, 900.78, 900.49, 900.69, 900.76, 900.65, 900.50, 900.21, 899.53, 899.81,
    899.50, 899.84, 900.00,	861.22, 910.46, 910.06, 910.79, 911.31, 910.77, 910.99,
    910.36, 910.79, 910.70, 910.68, 911.40, 911.03, 911.43, 911.54, 910.76, 911.09,
    911.23, 910.82, 911.63, 910.86, 911.69, 910.88, 911.07, 911.19, 911.50, 911.11,
    911.71, 911.73, 910.75, 885.75, 881.99, 891.15, 885.99, 906.62, 890.07, 889.46,
    888.91, 889.39, 889.25, 888.98, 909.47, 908.74, 909.08, 909.12, 908.10, 908.68,
    853.76, 854.33, 854.00,	854.20, 853.96, 879.61, 879.52, 879.71, 878.76, 878.99,
    878.61, 884.44, 889.79, 869.48, 869.72, 890.32, 890.62, 889.39, 910.60, 910.37,
    909.98, 910.00,	910.62, 909.96, 910.19, 910.19, 910.56, 909.89, 910.88, 911.09,
    910.23, 910.09, 910.69, 910.54, 926.24, 925.57, 925.79, 924.88, 925.39, 925.74,
    844.18, 845.00,	844.09, 844.95, 844.40, 844.67, 844.90, 859.57, 859.62, 859.81,
    859.86, 859.74, 859.62, 859.73, 870.17, 870.02, 870.28, 869.80, 870.19, 869.24,
    875.37, 875.45, 876.20, 875.23, 874.90, 875.00,	875.12, 874.50, 879.59, 880.42,
    880.60, 880.48, 881.01, 879.19, 880.00,	885.47, 885.74, 885.13, 885.19, 885.01,
    885.93, 884.74, 885.00,	885.56, 890.58, 890.55, 890.20, 890.34, 889.92, 889.20,
    890.19, 890.65, 890.81, 906.08, 903.22, 902.10, 906.14, 906.06, 903.23, 906.19,
    906.39, 904.97, 906.64, 906.36, 926.19, 925.77, 925.60, 965.86, 968.23, 967.08,
    966.37, 967.03, 966.63, 1017.4, 1016.32, 1017.68, 1017.67, 1017.65, 829.31, 829.74,
    829.45, 829.30, 829.10, 870.31, 870.55, 890.82, 901.56, 900.27, 903.66, 903.66,
    903.66, 889.91, 890.19, 889.28, 890.23, 889.06, 889.84, 890.43, 890.44, 890.27,
    889.99, 890.58, 889.99, 882.70, 878.91, 879.64, 879.72, 869.61, 869.62, 869.44,
    890.11, 890.06, 890.06, 884.93, 884.79, 884.81, 885.01, 884.76, 884.75, 884.62,
    884.82, 885.03, 884.79, 884.79, 884.93, 884.81, 885.13, 884.85, 884.67, 884.67,
    885.20, 884.76, 884.90, 864.42, 864.41, 1211.1, 1211.6, 1211.78, 1211.78, 1211.84,
    1211.42, 1211.91, 1211.53, 1211.67, 1211.62, 1211.6, 1211.39, 1211.85, 1211.75, 1211.8,
    1211.62, 1211.44, 827.94, 827.95, 827.93, 828.06, 827.91, 828.07, 827.92, 827.69,
    858.32, 858.34, 858.37, 858.49, 858.27, 858.46, 858.46, 858.51, 843.12, 843.10,
    843.18, 843.07, 843.08, 843.08, 843.22, 842.85, 868.64, 868.65, 868.66, 868.78,
    868.72, 868.52, 873.83, 873.66, 873.60, 873.45, 873.58, 873.57, 873.32, 878.63,
    878.72, 878.33, 878.52, 878.71, 883.33, 883.46, 883.70, 883.67, 883.70, 883.61,
    883.71, 883.69, 888.72, 888.75, 888.74, 888.43, 888.59, 888.79, 888.81, 888.89,
    903.13, 903.23, 903.40, 903.32, 903.34, 903.03, 923.36, 923.64, 923.52, 923.72,
    923.63, 963.83, 964.22, 964.10, 964.18, 964.03, 964.24, 963.91, 973.37, 973.38,
    973.51, 973.34, 973.49, 973.45, 826.82, 826.64, 826.75, 826.79, 826.68, 826.79,
    827.04, 827.09, 826.84, 868.05, 868.40, 868.36, 868.34, 799.62, 799.57, 903.71,
    903.24, 903.64, 903.57, 903.42, 903.64, 903.67, 924.06, 923.89, 890.57, 827.91,
    827.87, 827.72, 890.33, 923.83, 923.93, 853.17, 903.57, 898.64, 999.95, 999.85,
    999.93, 999.69, 999.83, 999.89, 999.86, 1099.8, 1099.43, 1099.42, 1099.41, 1311.22,
    1298.7, 650.37, 651.00, 650.88, 702.71, 702.73, 702.61, 857.93, 856.73, 856.49,
    929.85, 926.30])


run_type_array = np.array([
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
    "void", "dark", "", "", "", "", "", "", "", "", "", "", "data", "",
    "", "spectrometer", "spectrometer", "spectrometer", "spectrometer",
    "spectrometer", "", "", "", "", "", "", "", "", "", "tof scan",
    "tof scan", "tof scan", "tof scan", "tof scan", "tof scan",
    "tof scan", "tof scan", "tof scan", "tof scan", "tof scan",
    "tof scan", "tof scan", "tof scan", "tof scan", "tof scan",
    "tof scan", "tof scan", "tof scan", "tof scan", "tof scan",
    "tof scan", "tof scan", "tof scan", "tof scan", "tof scan",
    "tof scan", "tof scan", "tof scan", "tof scan", "tof scan",
    "tof scan", "tof scan", "tof scan", "tof scan", "tof scan",
    "tof scan", "tof scan", "tof scan", "tof scan", "tof scan",
    "tof scan", "void", "void", "background", "", "", "", "background",
    "", "", "", "", "", "", "", "", "", "tof scan", "tof scan",
    "tof scan", "tof scan", "tof scan", "tof scan", "", "tof scan",
    "tof scan", "tof scan", "tof scan", "tof scan", "tof scan",
    "tof scan", "", "", "", "timing scan", "timing scan", "timing scan",
    "", "timing scan", "timing scan", "", "", "", "", "energy scan",
    "energy scan", "void", "dark", "skimmer open", "skimmer open",
    "skimmer open", "skimmer open", "skimmer open", "skimmer open",
    "skimmer open", "skimmer closed", "skimmer closed", "skimmer closed",
    "skimmer closed", "skimmer open", "skimmer closed", "skimmer closed",
    "skimmer closed", "skimmer closed", "skimmer open", "skimmer open",
    "skimmer open", "skimmer open", "skimmer open", "", "dark",
    "tof scan", "spectrometer", "spectrometer", "spectrometer",
    "spectrometer", "dark", "background", "dark", "playing", "playing",
    "playing", "playing", "playing", "skimmer open", "skimmer closed",
    "skimmer closed", "skimmer closed", "skimmer closed",
    "skimmer closed", "data", "data", "spectrometer", "skimmer closed",
    "skimmer closed", "skimmer closed", "skimmer closed",
    "skimmer closed", "skimmer closed", "skimmer closed",
    "skimmer closed", "skimmer closed", "skimmer closed",
    "skimmer open", "spectrometer", "dark", "skimmer closed",
    "skimmer closed", "skimmer closed", "skimmer closed", "skimmer open",
    "spectrometer", "dark", "spectrometer", "skimmer open",
    "skimmer closed", "skimmer closed", "skimmer closed",
    "skimmer closed", "dark", "spectrometer", "void", "skimmer open",
    "skimmer closed", "skimmer closed", "skimmer closed",
    "skimmer closed", "dark", "spectrometer", "skimmer open",
    "skimmer closed", "skimmer closed", "skimmer closed",
    "skimmer closed", "dark", "spectrometer", "skimmer open",
    "skimmer closed", "skimmer closed", "skimmer closed",
    "skimmer closed", "skimmer closed", "dark", "spectrometer",
    "skimmer open", "skimmer open", "skimmer closed", "skimmer closed",
    "skimmer closed", "skimmer closed", "skimmer closed", "spectrometer",
    "spectrometer", "void", "dark", "skimmer open", "skimmer open",
    "dark", "skimmer closed", "skimmer closed", "skimmer closed",
    "skimmer closed", "spectrometer", "dark", "skimmer open", "void",
    "dark", "skimmer open", "skimmer closed", "skimmer closed",
    "skimmer closed", "skimmer closed", "skimmer open", "skimmer closed",
    "skimmer closed", "skimmer closed", "skimmer closed", "skimmer open",
    "skimmer closed", "skimmer closed", "skimmer closed", "dark",
    "tof scan", "tof scan", "tof scan", "tof scan", "tof scan",
    "tof scan", "tof scan", "other", "other", "other", "other", "other",
    "other", "other", "other", "other", "other", "other", "other",
    "dark", "dark", "void", "void", "void", "void", "spectrometer",
    "spectrometer", "void", "spectrometer", "spectrometer",
    "spectrometer", "void", "void", "spectrometer", "void",
    "spectrometer", "tof scan", "tof scan", "tof scan", "tof scan",
    "tof scan", "tof scan", "tof scan", "tof scan", "void", "tof scan",
    "tof scan", "void", "dark", "dark", "dark", "dark", "fullbeam",
    "fullbeam", "fullbeam", "fullbeam", "background", "void", "void",
    "void", "dark", "fullbeam", "fullbeam", "fullbeam", "att beam",
    "att beam", "att beam", "dark", "att beam", "att beam", "att beam",
    "att beam", "att beam", "fullbeam", "fullbeam", "fullbeam", "dark",
    "fullbeam", "fullbeam", "att beam", "att beam", "att beam",
    "att beam", "spectrometer", "dark", "fullbeam", "fullbeam",
    "fullbeam", "fullbeam", "att beam", "att beam", "att beam", "dark",
    "att beam", "att beam", "att beam", "fullbeam", "fullbeam", "dark",
    "fullbeam", "fullbeam", "att beam", "att beam", "att beam",
    "att beam", "spectrometer", "fullbeam", "fullbeam", "dark",
    "att beam", "att beam", "dark", "background", "att beam", "att beam",
    "att beam", "fullbeam", "fullbeam", "spectrometer", "dark", "dark",
    "att beam", "att beam", "fullbeam", "fullbeam", "fullbeam",
    "backgroundXTCAV", "dark", "att beam", "att beam", "fullbeam",
    "fullbeam", "spectrometer", "dark", "att beam", "att beam",
    "fullbeam", "fullbeam", "dark", "backgroundXTCAV", "fullbeam",
    "fullbeam", "att beam", "att beam", "other", "backgroundXTCAV",
    "void", "att beam", "att beam", "fullbeam", "fullbeam",
    "backgroundXTCAV", "att beam", "att beam", "att beam", "att beam",
    "att beam", "fullbeam", "fullbeam", "dark", "att beam", "att beam",
    "att beam", "fullbeam", "playing", "playing", "backgroundXTCAV",
    "Brunos wishlist", "fullbeam", "fullbeam", "fullbeam", "fullbeam",
    "att beam", "void", "energy scan", "energy scan", "energy scan",
    "energy scan", "energy scan", "energy scan", "energy scan", "void",
    "energy scan", "energy scan", "playing", "playing", "playing",
    "playing", "playing", "playing", "data", "data", "dark", "data",
    "data", "data", "data", "data", "data", "data", "data", "data",
    "data", "", "data", "data", "", "void", "other"])


def photon_energy(run):
    '''
    Returns calibrated energy for a given run number in electron volts (eV).

    Anatoli Ulmer, 2022
    '''
    return energy_elog_ev[run-1]


def photon_energy_kev(run):
    '''
    Returns calibrated energy for a given run number in kilo electron volts (keV).

    Anatoli Ulmer, 2022
    '''
    return photon_energy(run)/1000


def pulse_length(run):
    '''
    Returns pulse length in seconds for a given run number.

    Anatoli Ulmer, 2022
    '''
    if run <= 332:
        tau = 1e-15
    else:
        tau = 20e-15

    return tau


def run_type(run):
    '''
    Returns run type string for a given run number.

    Anatoli Ulmer, 2022
    '''
    return run_type_array[run-1]

