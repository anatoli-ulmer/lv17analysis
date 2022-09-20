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


def addcolorbar(ax, im, pos='right', size='5%', pad=0.05, orientation='vertical',
                stub=False, max_ticks=None, label=None):
    '''
    add a colorbar to a matplotlib image.

    ax -- the axis object the image is drawn in
    im -- the image (return value of ax.imshow(...))

    When changed, please update:
    https://gist.github.com/skuschel/85f0645bd6
    e37509164510290435a85a

    Stephan Kuschel, 2018
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad)
    if stub:
        cax.set_visible(False)
        return cax

    cb = plt.colorbar(im, cax=cax, orientation=orientation)
    if max_ticks is not None:
        from matplotlib import ticker
        tick_locator = ticker.MaxNLocator(nbins=max_ticks)
        cb.locator = tick_locator
        cb.update_ticks()
    if label is not None:
        cb.set_label(label)
    return cax


def bindata(data, bins=50):
    '''
    takes data points `xx, yy = data`.
    returns data points with errorbars: `(x, deltax, y, deltay)`.
    '''
    sargs = np.argsort(data[0])
    sdata = data[:, sargs]  # sortiert

    def onepoint(b):
        subset = sdata[:, int(b / (bins + 1) * len(sargs)):int((b + 1) / (bins + 1) * len(sargs))]
        x, y = np.mean(subset, axis=-1)
        dx, dy = np.std(subset, axis=-1)
        return x, dx, y, dy
    ret = np.asarray([onepoint(b) for b in range(bins)]).T
    return ret


def movmean(dlist, k=5):
    kk = np.arange(k)
    return np.convolve(dlist, np.ones_like(kk), 'valid') / k

