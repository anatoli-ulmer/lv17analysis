import numpy as np
import pickle as pkl


def fftshift(sig):
    return np.fft.fftshift(sig)


def fft(sig):
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(sig)))


def ifft(sig):
    return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(sig)))


def conv(f, g):
    return np.real(ifft(fft(f)*fft(g)))


def pkl_save(filename, data):
    with open(filename, 'wb') as f:
        pkl.dump(data, f)
    print("saved file {}\r".format(filename))


def pkl_load(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)
    print("read file {}\r".format(filename))
    return data

