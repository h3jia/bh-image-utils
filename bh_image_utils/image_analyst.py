# Python standard modules
import os
import struct

# Plotting modules
import matplotlib
matplotlib.use('agg')
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Other modules
import numpy as np
from scipy.ndimage import gaussian_filter
import multiprocess as mp
from collections import OrderedDict

__all__ = ['ImageAnalyst']


keys_0 = ['mass_msun', 'width', 'frequency', 'adaptive_num_levels']
keys_1 = ['I_nu', 'Q_nu', 'U_nu', 'V_nu', 'time', 'length', 'lambda', 'emission', 'tau']
keys_2 = ['lambda_ave_rho', 'lambda_ave_n_e', 'lambda_ave_p_gas', 'lambda_ave_Theta_e',
          'lambda_ave_B', 'lambda_ave_sigma', 'lambda_ave_beta_inverse', 'emission_ave_rho',
          'emission_ave_n_e', 'emission_ave_p_gas', 'emission_ave_Theta_e', 'emission_ave_B',
          'emission_ave_sigma', 'emission_ave_beta_inverse', 'tau_int_rho', 'tau_int_n_e',
          'tau_int_p_gas', 'tau_int_Theta_e', 'tau_int_B', 'tau_int_sigma', 'tau_int_beta_inverse']
keys_3 = ['distance_pc', 'x_rg', 'x_cm', 'x_rad', 'dx_rad', 'x_muas', 'dx_muas', 'f_nu_factor',
          'f_nu_jy']
keys_4 = ['f_nu_jy_mean', 'correction']
keys_5 = ['I_nu_blur', 'Q_nu_blur', 'U_nu_blur', 'V_nu_blur', 'x_muas_blur']

keys_iquv = ['I_nu', 'Q_nu', 'U_nu', 'V_nu', 'I_nu_blur', 'Q_nu_blur', 'U_nu_blur', 'V_nu_blur']

c_cms = 2.99792458e10
gg_msun = 1.32712440018e26
pc_cm = 9.69394202136e18 / np.pi
jy_factor = 1.0e-23

x_label = r'$x$ ($\mathrm{\mu as}$)'
y_label = r'$y$ ($\mathrm{\mu as}$)'

inu_scale = 1.0e-4
inu_label_t = [r'${}_\nu$', r' ($10^{-4}\ \mathrm{erg/s/cm^2/sr/Hz}$)']
inu_cmap = plt.get_cmap('inferno')
inu_cmap.set_bad('gray')


class ImageAnalyst:

    def __init__(self, distance_pc=1.68e7, f_nu_target=0.66):
        self._dict = None
        self._distance_pc = float(distance_pc)
        self._f_nu_target = float(f_nu_target)
        self.set_inu_range()

    @property
    def dict(self):
        return self._dict

    def set_inu_range(self, inu_vmin=0., inu_vmax=None):
        self.inu_vmin = inu_vmin
        self.inu_vmax = inu_vmax

    def get_images(self, dir_pattern, n, n_worker=10, full_data=False):
        self._full_data = full_data
        if isinstance(n, int):
            n = range(n)
        elif hasattr(n, '__iter__') or n is None:
            pass
        else:
            raise ValueError

        if n is None:
            self._npz = [np.load(dir_pattern)]
        else:
            self._npz = [np.load(dir_pattern.format(i)) for i in n]
        self._dict = OrderedDict()
        keys_all = (keys_0 + keys_1 + keys_2) if full_data else (keys_0 + keys_1)
        for k in keys_all:
            self._dict[k] = np.array([n[k] for n in self._npz])
        del self._npz

        self.dict['distance_pc'] = self._distance_pc
        nx = self.dict['I_nu'].shape[-1]
        width = self.dict['width'].flatten()[0]
        mass_msun = self.dict['mass_msun'].flatten()[0]
        self.dict['x_rg'] = np.linspace(-width / 2, width / 2, nx + 1, endpoint=True)
        rg_cm = mass_msun * gg_msun / c_cms**2
        self.dict['x_cm'] = self.dict['x_rg'] * rg_cm
        distance_cm = self.dict['distance_pc'] * pc_cm
        self.dict['x_rad'] = np.arctan(self.dict['x_cm'] / distance_cm)
        self.dict['dx_rad'] = self.dict['x_rad'][1] - self.dict['x_rad'][0]
        self.dict['x_muas'] = self.dict['x_rad'] * 180.0 / np.pi * 3600.0 * 1.0e6
        self.dict['dx_muas'] = self.dict['x_muas'][1] - self.dict['x_muas'][0]
        self.dict['f_nu_factor'] = self.dict['dx_rad'] * self.dict['dx_rad'] / jy_factor
        self.dict['f_nu_jy'] = np.nansum(
            self.dict['I_nu'] * self.dict['f_nu_factor'][..., np.newaxis], axis=(-1, -2))
        if self.dict['f_nu_jy'].ndim == 1:
            self.dict['f_nu_jy_mean'] = np.mean(self.dict['f_nu_jy'])
            self.dict['correction'] = self._f_nu_target / self.dict['f_nu_jy_mean']

    @staticmethod
    def blur_image(image, muas_per_pix, fwhm=20.):
        return gaussian_filter(
            np.nan_to_num(image), fwhm / 2 / np.sqrt(2 * np.log(2)) / muas_per_pix, mode='constant')

    @staticmethod
    def pad_image(image, n_pad):
        image = np.asarray(image)
        if not image.ndim >= 2:
            raise ValueError
        i_shape = np.array(image.shape)
        i_shape[-1] += 2 * n_pad
        i_shape[-2] += 2 * n_pad
        foo = np.zeros(i_shape)
        foo[..., n_pad:-n_pad, n_pad:-n_pad] = image
        return foo

    @staticmethod
    def pad_muas(muas, n_pad):
        muas = np.asarray(muas)
        assert muas.ndim == 1 and muas.size >= 2 and n_pad > 0
        foo = np.empty(muas.size + n_pad * 2)
        foo[:n_pad] = np.linspace(muas[0] - n_pad * (muas[1] - muas[0]), muas[0], n_pad,
                                  endpoint=False)
        foo[n_pad:-n_pad] = muas
        foo[-n_pad:] = np.linspace(muas[-1] + muas[-1] - muas[-2], muas[-1] + (n_pad + 1) *
                                   (muas[-1] - muas[-2]), n_pad, endpoint=False)
        return foo

    def blur_images(self, fwhm=20., n_worker=10, polar=True, f_pad=1):
        muas_per_pix = float(self.dict['dx_muas'])
        n_pad = int(np.rint(f_pad * fwhm / self.dict['dx_muas']))
        worker = lambda x: ImageAnalyst.blur_image(x, muas_per_pix, fwhm)
        with mp.Pool(n_worker) as pool:
            self.dict['I_nu_blur'] = np.array(
                pool.map(worker, self.pad_image(self.dict['I_nu'], n_pad), chunksize=25))
            if polar:
                self.dict['Q_nu_blur'] = np.array(
                    pool.map(worker, self.pad_image(self.dict['Q_nu'], n_pad), chunksize=25))
                self.dict['U_nu_blur'] = np.array(
                    pool.map(worker, self.pad_image(self.dict['U_nu'], n_pad), chunksize=25))
                self.dict['V_nu_blur'] = np.array(
                    pool.map(worker, self.pad_image(self.dict['V_nu'], n_pad), chunksize=25))
        self.dict['x_muas_blur'] = self.pad_muas(self.dict['x_muas'], n_pad)

    def plot_image(self, i, target='I_nu', correction=False, save=False, save_name='0.png',
                   subplot=False, dpi=100, vmin=None, vmax=None, scale=None, cblabel=None,
                   cmap=None, fontsize_label=15, fontsize_cblabel=15, fontsize_ticks=13):
        if not subplot:
            plt.figure(figsize=(6, 4), dpi=dpi)
        if cblabel is None and scale is None and target in keys_iquv:
            cblabel = inu_label_t[0].format(target[0]) + inu_label_t[1]
        if scale is None:
            scale = inu_scale if target in keys_iquv else 1.
        if cmap is None:
            cmap = inu_cmap

        if np.asarray(i).ndim == 2:
            image = np.asarray(i)
        else:
            image = self.dict[target][i]
        if correction:
            image = image * self.dict['correction']
        vmin = self.inu_vmin if vmin is None else vmin
        vmax = self.inu_vmax if vmax is None else vmax
        if image.shape[-1] == self.dict['x_muas'].size - 1:
            x_muas = self.dict['x_muas']
        elif 'x_muas_blur' in self.dict and image.shape[-1] == self.dict['x_muas_blur'].size - 1:
            x_muas = self.dict['x_muas_blur']
        else:
            raise ValueError
        im = plt.pcolormesh(x_muas, x_muas, image / scale, cmap=cmap, vmin=vmin, vmax=vmax,
                            rasterized=True)

        # Make colorbar
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel(cblabel, fontsize=fontsize_cblabel)
        cbar.ax.tick_params(labelsize=fontsize_ticks)

        # Adjust axes
        plt.xlim((x_muas[0], x_muas[-1]))
        plt.xlabel(x_label, fontsize=fontsize_label)
        plt.ylim((x_muas[0], x_muas[-1]))
        plt.ylabel(y_label, fontsize=fontsize_label)
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.gca().set_aspect(1.0)

        # Adjust layout
        if not subplot:
            plt.tight_layout()

        if save:
            plt.savefig(save_name, dpi=dpi)
            plt.close()
        elif not subplot:
            plt.show()

    def quadsolve(self, x, y, c=None):
        c = self._f_nu_target if c is None else float(c)
        a, b, c = np.polyfit(x, y, 2) - np.array([0, 0, c])
        return (-b + np.sqrt(b**2 - 4 * a * c)) / 2 / a, (-b - np.sqrt(b**2 - 4 * a * c)) / 2 / a
