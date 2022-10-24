import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import display
from IPython.display import clear_output as clear_out
import datetime
import traceback
from glob import glob
import os.path


def fftshift(sig):
    return np.fft.fftshift(sig)


def fft(sig):
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(sig)))


def ifft(sig):
    return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(sig)))


def conv(f, g):
    return np.real(ifft(fft(f) * fft(g)))


def pkl_save(filename, data):
    with open(filename, "wb") as f:
        pkl.dump(data, f)
    print("saved file {}\r".format(filename))


def pkl_load(filename):
    with open(filename, "rb") as f:
        data = pkl.load(f)
    print("read file {}\r".format(filename))
    return data


def addcolorbar(
    ax,
    im,
    pos="right",
    size="5%",
    pad=0.05,
    orientation="vertical",
    stub=False,
    max_ticks=None,
    label=None,
):
    """
    add a colorbar to a matplotlib image.

    ax -- the axis object the image is drawn in
    im -- the image (return value of ax.imshow(...))

    When changed, please update:
    https://gist.github.com/skuschel/85f0645bd6
    e37509164510290435a85a

    Stephan Kuschel, 2018
    """
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
    """
    takes data points `xx, yy = data`.
    returns data points with errorbars: `(x, deltax, y, deltay)`.
    """
    sargs = np.argsort(data[0])
    sdata = data[:, sargs]  # sortiert

    def onepoint(b):
        subset = sdata[
            :, int(b / (bins + 1) * len(sargs)): int((b + 1) / (bins + 1) * len(sargs))
        ]
        x, y = np.mean(subset, axis=-1)
        dx, dy = np.std(subset, axis=-1)
        return x, dx, y, dy

    ret = np.asarray([onepoint(b) for b in range(bins)]).T
    return ret


def movmean(y, k=5):
    kk = np.arange(k)
    return np.convolve(y, np.ones_like(kk), "valid") / k


def clear():
    clear_out(wait=True)
    return


def clear_output(wait=True):
    clear_out(wait=wait)
    return


def clc(wait=False):
    clear_output(wait=wait)
    return


def today():
    return str(datetime.date.today())


def now():
    return str(datetime.datetime.now())


class Tfig:

    curr_ax = np.asarray(0, dtype=int)
    curr_ax_pos = np.asarray((0, 0), dtype=int)
    cmap = "magma"

    def __init__(self, rows=1, cols=1, *args, **kwargs):
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)
        self.fig, self.ax = plt.subplots(rows, cols, squeeze=False, *args, **kwargs)
        # self.ax_shape = np.shape()
        self._args = args
        self._kwargs = kwargs

    def imshow(self, img, title=None, *args, **kwargs):

        ax = self.gca()
        im = ax.imshow(img, cmap=self.cmap)
        ax.axis("image")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.025)
        self.fig.colorbar(im, cax=cax, orientation="vertical")

        if title is not None:
            ax.title(title)

        self.curr_ax += 1

        return im

    def gca(self, pos=None):
        if pos is not None:
            if len(np.shape((pos))) == 0:
                self.curr_ax = pos
            else:
                self.curr_ax = pos[0] * self.ax.shape[1] + pos[1]

        self.curr_ax_pos = divmod(self.curr_ax, self.ax.shape[1])
        return self.ax[self.curr_ax_pos]


def nargout(*args):
    callInfo = traceback.extract_stack()
    callLine = str(callInfo[-3].line)
    split_equal = callLine.split("=")
    split_comma = split_equal[0].split(",")
    num = len(split_comma)
    return args[0:num] if num > 1 else args[0]


def ring_data(radii=[75, 150], center=[403, 677]):
    th = np.append(np.linspace(0, 2 * np.pi, 100), np.nan)
    th = np.expand_dims(th, -1)
    th = np.repeat(th, len(radii), axis=1)
    x = np.cos(th) * np.asarray(radii) + center[1]
    y = np.sin(th) * np.asarray(radii) + center[0]
    return x, y


def plot_rings(
    radii=[75, 150],
    center=[403, 677],
    ax=None,
    ls="--",
    lw=1,
    c="tab:red",
    alpha=0.75,
    label=None,
    keep_lims=False,
):
    x, y = ring_data(radii=radii, center=center)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if keep_lims:
        lims = ax.get_xlim(), ax.get_ylim()
    pt = ax.plot(x, y, c=c, ls=ls, lw=lw, label=label, alpha=alpha)
    if keep_lims:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
    if label is not None:
        ax.legend()
    return nargout(pt)


def get_filenames(fp):
    return np.sort(glob(os.path.join(fp, "r*.h5")))


def fn2run(fn):
    return [int(f.split("/r")[-1].split("_")[0]) for f in fn]

