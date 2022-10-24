from h5analysis import h5data
from lv17analysis import lv17data, detectors, epix, helpers
from lv17analysis.helpers import nargout

import numpy as np
import numpy.ma as ma
import scipy.optimize as so
import numba as nb


def ev2wavelength(hv):
    return 1.2398e-06 / hv


def px2q(px, hv, det_dist=0.395, px_size=50e-6):
    theta = np.arctan(px * px_size / det_dist)
    wavelength = ev2wavelength(hv)
    q = 4 * np.pi / wavelength * np.sin(theta / 2)
    return nargout(q, theta)


def q2px(q, hv, det_dist=0.395, px_size=50e-6):
    wavelength = ev2wavelength(hv)
    theta = 2 * np.arcsin(q * wavelength / 4 / np.pi)
    px = det_dist / px_size * np.arctan(theta)
    return nargout(px, theta)


def qr_zero(qr=[0.0, 100.0]):
    zero_vals = np.pi * np.asarray(
        [
            1.43033977696463,
            2.45912761698905,
            3.47104486433979,
            4.47755078951219,
            5.48151021013007,
            6.48451469154001,
            7.48656423374200,
            8.48797714980537,
            9.48939006586874,
            10.4904846688628,
            11.4912609587875,
            12.4920372487123,
            13.4924952255677,
            14.4929532024231,
            15.4934111792785,
            16.4938691561340,
            17.4943271329894,
            18.4944667967755,
            19.4949247736309,
            20.4950644374170,
            21.4952041012031,
            22.4953437649892,
            23.4958017418447,
            24.4959414056308,
            25.4960810694169,
            26.4962207332030,
            27.4963603969891,
            28.4965000607752,
            29.4966397245613,
            30.4967793883474,
            31.4969190521335,
            32.5,
        ]
    )
    return zero_vals[(zero_vals >= np.nanmin(qr)) * (zero_vals <= np.nanmax(qr))]


def plot_qr_zero_rings(
    qr=[0.0, 100.0],
    center=[403, 677],
    ax=None,
    ls="--",
    lw=2,
    c="tab:red",
    alpha=0.75,
    label=None,
    keep_lims=False,
):
    radii = qr_zero(qr)
    pt = helpers.plot_rings(
        radii=radii,
        center=center,
        ax=ax,
        ls=ls,
        lw=lw,
        c=c,
        label=label,
        alpha=alpha,
        keep_lims=keep_lims,
    )
    return pt


def guinier(q, I0, r):
    qr = ma.masked_array(q * r, mask=(q == 0))
    return I0 * (9 * (np.sin(qr) / qr**2 - np.cos(qr) / qr) ** 2) / qr**2


def guinier0c(q, I0, r, c):
    qr = ma.masked_array(q * r, mask=(q == 0))
    return I0 * (9 * (np.sin(qr) / qr**2 - np.cos(qr) / qr) ** 2) / qr**2 + c


def guinier1(q, I0, r):
    qr = ma.masked_array(q * r, mask=(q == 0))
    return I0 * (9 * (np.sin(qr) / qr**2 - np.cos(qr) / qr) ** 2) / r**2 / q


def guinier2(q, I0, r):
    qr = ma.masked_array(q * r, mask=(q == 0))
    return I0 * (9 * (np.sin(qr) / qr**2 - np.cos(qr) / qr) ** 2) / r**2


def guinier3(q, I0, r):
    qr = ma.masked_array(q * r, mask=(q == 0))
    return I0 * (9 * (np.sin(qr) / qr**2 - np.cos(qr) / qr) ** 2) / r**2 * q


def guinier4(q, I0, r):
    qr = ma.masked_array(q * r, mask=(q == 0))
    return I0 * (9 * (np.sin(qr) / qr**2 - np.cos(qr) / qr) ** 2) / r**2 * q**2


def guinierI0(qr, I0):
    qr = ma.masked_array(qr, mask=(qr == 0))
    return I0 * (9 * (np.sin(qr) / qr**2 - np.cos(qr) / qr) ** 2) / qr**2


def guinierI0c(qr, I0, c):
    qr = ma.masked_array(qr, mask=(qr == 0))
    return I0 * (9 * (np.sin(qr) / qr**2 - np.cos(qr) / qr) ** 2) / qr**2 + c


def radial_profile(data, center=None, rrange=None, returnmask=False):
    """
    Nans are ignored. data can be a masked array.

    Center is assumed to be center if not given.
    If given, is allowed to lay outside.

    returns the radial profile rprof.
    r values are always np.arange(len(rprof))

    Stephan Kuschel, 2022
    """
    # if center is None:
    #     center = data.shape[0] // 2, data.shape[1] // 2
    # y, x = np.indices(data.shape)
    # r = np.rint((np.sqrt((y - center[0])**2 + (x - center[1])**2))).astype(int)
    sy, sx = np.shape(data)
    if center is None:
        center = [sy / 2, sx / 2]
    y, x = np.ogrid[0:sy, 0:sx]
    r = np.hypot(y - center[0], x - center[1]).astype(np.int16)
    nanmask = ~np.isnan(data)
    if rrange is not None:
        nanmask *= r > rrange[0]
        nanmask *= r < rrange[1]
    rr = np.ravel(r[nanmask])
    nr = np.bincount(rr)
    tbin = np.bincount(rr, data[nanmask].ravel())
    nr = np.ma.array(nr)
    tbin[nr == 0] = np.ma.masked
    nr[nr == 0] = np.ma.masked
    radialprofile = tbin / nr
    # radialprofile[nr == 0] = np.nan
    if returnmask:
        return radialprofile, nanmask
    else:
        return nargout(radialprofile, nr)


def fit_guinier(q, rprof, I0=None, r0=50e-9, qpow=0, c=0.0):
    m = ma.masked_invalid(rprof)
    amp = rprof * (q**qpow) if qpow > 0 else rprof
    m.mask[0] = True  # center must always be maskedImax = np.nanmax(rprof)
    Imax = np.nanmax(rprof[~m.mask])
    I0 = 10 * Imax if I0 is None else I0
    # I0 = np.nanmean(rprof[~m.mask] * q[~m.mask]**4)
    I_bo = np.asarray([0.0, np.inf])
    r_bo = np.asarray([10e-9, 500e-9])
    c_bo = np.asarray([0.0, 1e-3])
    if qpow == 0:
        popt = so.curve_fit(guinier, q[~m.mask], amp[~m.mask], p0=(I0, r0))
    elif qpow == 1:
        popt = so.curve_fit(guinier1, q[~m.mask], amp[~m.mask], p0=(I0, r0))
    elif qpow == 2:
        popt = so.curve_fit(guinier2, q[~m.mask], amp[~m.mask], p0=(I0, r0))
    elif qpow == 3:
        popt = so.curve_fit(guinier3, q[~m.mask], amp[~m.mask], p0=(I0, r0))
    elif qpow == 4:
        popt = so.curve_fit(guinier4, q[~m.mask], amp[~m.mask], p0=(I0, r0))
    elif qpow == -1:
        popt = so.curve_fit(guinierI0, q[~m.mask] * r0, amp[~m.mask], p0=(I0))
    elif qpow == -2:
        popt = so.curve_fit(
            guinierI0c,
            q[~m.mask] * r0,
            amp[~m.mask],
            p0=(I0, c),
            bounds=([I_bo[0], c_bo[0]], [I_bo[1], c_bo[1]]),
        )
    return popt


def iterative_fit_guinier(q, rprof, radial_range=(10, 300), n_steps=30, qpow=0):
    pbest = (np.asarray([0.0, 0.0]), np.asarray([[np.inf, np.inf], [np.inf, np.inf]]))
    errbest = np.asarray([np.inf, np.inf])
    r_array = np.linspace(radial_range[0] * 1e-9, radial_range[1] * 1e-9, n_steps)
    r_array = np.logspace(np.log10(radial_range[0]*1e-9),
                          np.log10(radial_range[1]*1e-9), n_steps)
    for ir, r0 in enumerate(r_array):
        try:
            popt = fit_guinier(q, rprof, 10 * np.nanmax(rprof), r0, qpow=qpow)
        except BaseException:
            continue
        perr = np.sqrt(np.diag(popt[1]))
        if errbest[1] > perr[1]:
            pbest = popt
            errbest = perr
    return pbest


def iterative_fit_guinier2(rprof, q, radial_range=(10, 300), n_steps=30, qpow=0):
    p_arr = np.zeros((n_steps, 2), dtype="float")
    e_arr = np.zeros((n_steps, 2, 2), dtype="float")
    pbest = (np.asarray([0.0, 0.0]), np.asarray([[np.inf, np.inf], [np.inf, np.inf]]))
    errbest = np.asarray([np.inf, np.inf])
    for ir, r0 in enumerate(
        np.linspace(radial_range[0] * 1e-9, radial_range[1] * 1e-9, n_steps)
    ):
        try:
            p_arr[ir], e_arr[ir] = fit_guinier(q, rprof, None, r0, qpow=qpow)
        except BaseException:
            p_arr[ir] = np.inf
            e_arr[ir] = np.inf
            continue
    r_err_arr = [np.sqrt(np.diag(d))[1] for d in e_arr]
    i = np.nanargmin(r_err_arr)
    return (p_arr[i], e_arr[i])


if __name__ == "__main__":

    # python ~/lv17analysis/lv17analysis/sizing.py

    import numpy as np
    import numpy.ma as ma
    import h5py
    from h5analysis import h5data
    from lv17analysis import lv17data, detectors, epix
    from lv17analysis import helpers as hp
    from lv17analysis import sizing as sz
    import os
    import tarfile
    import glob
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import scipy.optimize as so
    import diffractionimaging.clustersize as dc

    fp = os.path.join(h5data.h5paths["h5"], "hits_brightest500")
    # spath = os.path.join(h5data.h5paths['scratch'], 'plots', 'radial_fitting', str(hp.today()))
    spath = os.path.join(
        h5data.h5paths["scratch"], "plots", "radial_fitting", "2022-10-18"
    )

    fnames = hp.get_filenames(fp)
    run_list = hp.fn2run(fnames)
    # run_list = lv17data.neon_runs
    mask = epix.get_mask()

    rrange = [75, 384]
    rrange = [75, 500]
    yrange = [1e-5, 1e3]
    crange = [0.75, 700]
    # cmap = 'viridis'
    # cmap = 'plasma'
    cmap = "magma"
    plt_norm = LogNorm(vmin=crange[0], vmax=crange[1])
    plt_ratio = 2
    plt_size = 4
    # plt_ratio = .25
    # plt_size = 40
    save_dpi = 200

    show_plots = True
    save_plots = True
    save_fitresults = True
    test_mode = False

    if save_plots:
        os.makedirs(spath, exist_ok=True)

    for i_run, run in enumerate(run_list):
        if save_fitresults:
            ft_sname = os.path.join(spath, "fitresults", f"r{run:03d}_fitresults.h5")
            if os.path.exists(ft_sname):
                continue

        bg = epix.load_bg(run)

        with h5py.File(fnames[i_run], "r") as f:
            fkeys = f.keys()
            hv_mean = np.nanmean(f["photon_energy"])
            hv_array = np.asarray(f["photon_energy"])
            hv_array[np.isnan(hv_array)] = np.nanmean(f["photon_energy"])
            ts_array = np.asarray(f["timestamp"])

            n_imgs = np.shape(f["epix"])[0]
            rprof_dict = {
                "rprof": np.zeros([n_imgs, rrange[1]]),
                "fit_rprof": np.zeros([n_imgs, rrange[1]]),
                "q": np.zeros([n_imgs, rrange[1]]),
                "hv": np.zeros([n_imgs, 1]),
                "fit_i0": np.zeros([n_imgs, 2]),
                "fit_r": np.zeros([n_imgs, 2]),
                "run": np.zeros([n_imgs, 1]),
                "timestamp": np.zeros([n_imgs, 1]),
            }

            fitresults = {
                "i0": np.zeros([n_imgs, 2]),
                "r": np.zeros([n_imgs, 2]),
                "run": np.zeros([n_imgs, 1]),
                "timestamp": np.zeros([n_imgs, 1]),
            }

            i_sorted = np.argsort(f["epix_sum"])
            for i_img, i in enumerate(i_sorted):
                img = f["epix"][i] - bg
                # hv = f['photon_energy'][i] if ~np.isnan(f['photon_energy'][i]) else hv_mean
                timestamp = ts_array[i]
                hv = hv_array[i]
                # if np.isnan(hv):
                #     hv = hv_mean

                img_masked = np.asarray(img * 1000 / hv)

                img_masked[~mask] = np.nan
                img_masked[img > 700] = np.nan

                # img_masked = epix.offset_correction(img_masked, offs_thresh=0.75)
                # img_masked = epix.cm_correction(img_masked)
                # #             img_masked[img_masked < 0.75] = 0.0
                # #             img_masked = np.round(img_masked)

                center = epix.center[[1, 0]]
                center = [404, 682]
                # center = dc.find_center_auto_data(img_masked, center=center, n=4)
                rprof = sz.radial_profile(img_masked, center=center, rrange=rrange)
                pxl = np.arange(rrange[1])
                q = sz.px2q(pxl, hv)

                qpow = 3
                best = sz.iterative_fit_guinier(q, rprof, n_steps=10, qpow=qpow)
                i0, rval = best[0]
                i0err, rerr = np.sqrt(np.diag(best[1]))

                # rprof_qpow = rprof * q**qpow
                # if qpow == 0:
                #     rprof_fit = sz.guinier(q, i0, rval)
                # elif qpow == 1:
                #     rprof_fit = sz.guinier1(q, i0, rval)
                # elif qpow == 2:
                #     rprof_fit = sz.guinier2(q, i0, rval)
                # elif qpow == 3:
                #     rprof_fit = sz.guinier3(q, i0, rval)
                # elif qpow == 4:
                #     rprof_fit = sz.guinier4(q, i0, rval)

                # rprof_fit_qpow = np.copy(rprof_fit)

                popt = sz.fit_guinier(q, rprof, best[0][0], best[0][1], qpow=-1)
                i0 = popt[0][0]
                i0err = np.sqrt(popt[1][0][0])
                # i0_qpow, rval_qpow, i0err_qpow, rerr_qpow = i0, rval, i0err, rerr

                # popt = sz.fit_guinier(q, rprof, i0, rval, qpow=0)
                # i02, rval2 = popt[0]
                # i0err2, rerr2 = np.sqrt(np.diag(popt[1]))

                # popt = sz.fit_guinier(q, rprof, best[0][0], best[0][1], qpow=0)
                # i0, rval = popt[0]
                # i0err, rerr = np.sqrt(np.diag(popt[1]))

                rprof_fit = sz.guinier(q, i0, rval)
                # rprof_fit2 = sz.guinier(q, i02, rval2)
                fitresults["i0"][i] = [i0, i0err]
                fitresults["r"][i] = [rval, rerr]
                fitresults["run"][i] = run
                fitresults["timestamp"][i] = timestamp

                rprof_dict["run"][i] = run
                rprof_dict["rprof"][i] = rprof
                rprof_dict["hv"][i] = hv
                rprof_dict["q"][i] = q
                rprof_dict["fit_rprof"][i] = rprof_fit
                rprof_dict["fit_i0"][i] = [i0, i0err]
                rprof_dict["fit_r"][i] = [rval, rerr]
                rprof_dict["timestamp"][i] = timestamp

                if show_plots:

                    # hp.clear()

                    fig, ax = plt.subplots(
                        1,
                        2,
                        figsize=((plt_ratio + 1.1) * plt_size, plt_size),
                        gridspec_kw={"width_ratios": [1.1, plt_ratio]},
                    )
                    im = ax[0].matshow(img_masked, norm=plt_norm, cmap=cmap)
                    ax[0].xaxis.tick_bottom()
                    # mmm = np.asarray(~mask, dtype=np.float64) * 100
                    # mmm[mask] = np.nan
                    # ax[0].matshow(mmm, alpha=.5, cmap=cmap)
                    # ax[0].invert_yaxis()
                    hp.addcolorbar(ax[0], im)
                    zrads = sz.q2px(sz.qr_zero(q * rval) / rval, hv)
                    hp.plot_rings(
                        radii=zrads,
                        center=center,
                        ax=ax[0],
                        keep_lims=True,
                        lw=1.5,
                        c="tab:red",
                    )

                    ax[1].semilogy(rprof, ".")

                    ax[1].semilogy(
                        pxl,
                        rprof_fit,
                        "--",
                        label=r"$R = ({:.2f} \pm {:.2f})\,$nm".format(
                            rval * 1e9, rerr * 1e9
                        ) +
                        "\n" +
                        r"$I_0 = ${:.0f}$ \pm {:.2g}$".format(i0, i0err),
                    )
                    # ax[1].semilogy(pxl, rprof_fit2, ":", label=
                    #                r'$R = ({:.2f} \pm {:.2f})\,$nm'.format(rval2*1e9, rerr2*1e9)
                    #                + '\n'
                    #                + r"$I_0 = ${:.0f}$ \pm {:.3g}$".format(i02, i0err2))
                    # ax[1].semilogy(rmask)
                    ax[1].set_xlim(0, rrange[1])
                    ax[1].set_ylim(yrange)
                    i0_pe = i0err / i0 * 100
                    r_pe = rerr / rval * 100

                    ax[1].legend(title=f"r{run:03d}, " +
                                 f"$\\delta I_0 = {i0_pe:.2f}\\%$, " +
                                 f"$\\delta R = {r_pe:.2f}\\%$")
                    ax[1].grid()

                    # ax[2].semilogy(
                    #     pxl,
                    #     rprof_fit,
                    #     "--",
                    #     label=r"$R = ({:.2f} \pm {:.2f})\,$nm".format(
                    #         rval_qpow * 1e9, rerr_qpow * 1e9
                    #     ) +
                    #     "\n" +
                    #     r"$I_0 = ${:.0f}$ \pm {:.2g}$".format(i0_qpow, i0err_qpow),
                    # )
                    # ax[2].semilogy(pxl[50:], rprof_fit_qpow[50:], "--")
                    # # ax[1].semilogy(rmask)
                    # ax[2].set_xlim(0,rrange[1])
                    # # ax[1].set_ylim(yrange)
                    # ax[2].legend(title=r"run {:03d}, ".format(run)
                    #              + r"$\delta I_0 = {:.2f} \%$,".format(i0err_qpow/i0_qpow*100)
                    #              + r"$\delta R = {:.2f} \%$".format(rerr_qpow/rval_qpow*100))
                    # ax[1].legend(title=
                    #              r"run {:03d}, $\Delta I_0/I_0 = {:.3f}$,".format(run, i0err/i0)
                    #              + "$\Delta R/R = {:.3f}\,$".format(rerr/rval))
                    # ax[2].grid()

                    # plt.show()

                    if save_plots:
                        png_spath = os.path.join(spath, "png")
                        os.makedirs(png_spath, exist_ok=True)
                        png_sname = os.path.join(
                            png_spath,
                            "r{:03d}.{:04d}_{:}.png".format(run, i_img + 1, timestamp),
                        )
                        # print(png_sname)
                        fig.savefig(png_sname, dpi=save_dpi, bbox_inches="tight")
                        plt.close()
                        hp.clear()
                        print("run {:03d} ({:d}/{:d})".format(run, i_img + 1, n_imgs))
                        print("saved in '{}'".format(png_sname))

                if test_mode and i >= 0:
                    break

            if save_fitresults:
                ft_spath = os.path.join(spath, "fitresults")
                os.makedirs(ft_spath, exist_ok=True)
                ft_sname = os.path.join(ft_spath, "r{:03d}_fitresults.h5".format(run))
                with h5py.File(ft_sname, "w") as ft:
                    for key, val in fitresults.items():
                        ft[key] = np.asarray(val)
                print("saved fit results in: {}".format(ft_sname))

                rp_spath = os.path.join(spath, "profiles")
                os.makedirs(rp_spath, exist_ok=True)
                rp_sname = os.path.join(rp_spath, "r{:03d}_profiles.h5".format(run))
                with h5py.File(rp_sname, "w") as frp:
                    for key, val in rprof_dict.items():
                        frp[key] = np.asarray(val)
                print("saved fit results in: {}".format(rp_sname))

            if save_plots:
                # Create a tar file from png's
                tar_spath = os.path.join(spath, "tar")
                os.makedirs(tar_spath, exist_ok=True)
                tar_sname = os.path.join(
                    tar_spath, "r{:03d}_radial_plots.tar".format(run)
                )
                ftar = tarfile.open(tar_sname, "w")
                print("saving tar file: {}".format(tar_sname))
                png_files = np.asarray(
                    glob.glob(os.path.join(png_spath, f"r{run:03d}*.png"))
                )
                for fn_png in png_files:
                    ftar.add(fn_png)
                # print(files)
                # Listing the files in tar
                for fn_png in ftar.getnames():
                    print("added %s" % fn_png)
                ftar.close()

        if test_mode and i_run >= 0:
            break
