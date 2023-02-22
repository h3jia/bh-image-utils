import numpy as np
from scipy.interpolate import RectBivariateSpline
from ._ring_analyst import _ri_theta, _ri_io
from scipy.optimize import minimize
from collections import namedtuple
import multiprocess as mp
from scipy.stats import circmean, circstd
import warnings

__all__ = ['RingAnalyst']


RingReport = namedtuple('RingReport', 'x_center, n_bad, r_theta, r_mean, r_std, '
                        'r_in_theta, r_in_mean, r_in_std, r_out_theta, r_out_mean, '
                        'r_out_std, d_theta, d_mean, d_std, w_theta, '
                        'w_mean, w_std, eta_all, eta_mean, eta_std, '
                        'A_all, A_mean, A_std, fc',
                        defaults=(None, np.nan, np.nan,
                                  None, np.nan, np.nan, None, np.nan,
                                  np.nan, None, np.nan, np.nan, None,
                                  np.nan, np.nan, None, np.nan, np.nan,
                                  None, np.nan, np.nan, np.nan))


PolarRingReport = namedtuple('PolarRingReport', 'x_center, n_bad, r_theta, r_mean, r_std, '
                             'r_in_theta, r_in_mean, r_in_std, r_out_theta, r_out_mean, '
                             'r_out_std, d_theta, d_mean, d_std, w_theta, '
                             'w_mean, w_std, eta_all, eta_mean, eta_std, '
                             'A_all, A_mean, A_std, fc, beta_2',
                             defaults=(None, np.nan, np.nan,
                                       None, np.nan, np.nan, None, np.nan,
                                       np.nan, None, np.nan, np.nan, None,
                                       np.nan, np.nan, None, np.nan, np.nan,
                                       None, np.nan, np.nan, np.nan, np.nan))


PolarBinRingReport = namedtuple('PolarBinRingReport', 'x_center, n_bad, r_theta, r_mean, r_std, '
                                'r_in_theta, r_in_mean, r_in_std, r_out_theta, r_out_mean, '
                                'r_out_std, d_theta, d_mean, d_std, w_theta, '
                                'w_mean, w_std, eta_all, eta_mean, eta_std, '
                                'A_all, A_mean, A_std, fc, beta_2, beta_2_bin',
                                defaults=(None, np.nan, np.nan,
                                          None, np.nan, np.nan, None, np.nan,
                                          np.nan, None, np.nan, np.nan, None,
                                          np.nan, np.nan, None, np.nan, np.nan,
                                          None, np.nan, np.nan, np.nan, np.nan, []))


def circ_w(samples, weights=None, high=2*np.pi, low=0):
    samples = np.asarray(samples)
    diff = high - low
    if diff != 2 * np.pi:
        raise NotImplementedError
    if weights is None:
        weights = np.ones_like(samples)
    else:
        weights = np.asarray(weights)
        assert weights.shape == samples.shape
    if not (np.all(weights >= 0) and np.sum(weights) > 0):
        warnings.warn('please check the weights', RuntimeWarning)
    sin_m = np.sum(weights * np.sin(samples)) / np.sum(weights)
    cos_m = np.sum(weights * np.cos(samples)) / np.sum(weights)
    circ_m = np.arctan2(sin_m, cos_m)
    circ_m = (circ_m % (2. * np.pi)) + low
    R = np.minimum(1, np.hypot(sin_m, cos_m))
    circ_s = ((high - low) / 2. / np.pi) * np.sqrt(-2. * np.log(R))
    return circ_m, circ_s


class RingAnalyst:

    def __init__(self, image_analyst, n_theta=100, r_max=60, dr=None, center_region=10,
                 center_out_loss=1e6, bad_dir_loss=5, alpha=0.5, beta=0.85, gamma=0.5,
                 optimize_options=None, fc_region=5, blur=True, eta_A_w=True):
        self.image_analyst = image_analyst
        x_muas = image_analyst.dict['x_muas_blur' if blur else 'x_muas']
        self.dx = x_muas[1] - x_muas[0]
        self.x_interp = x_muas[:-1] + self.dx / 2
        self.n_theta = n_theta
        self.r_max = r_max
        self.dr = self.dx if dr is None else dr
        self.center_region = center_region
        self.center_out_loss = center_out_loss
        self.bad_dir_loss = bad_dir_loss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.optimize_options = ({'initial_simplex': [[5, 0], [-3, 4], [-3, -4]], 'fatol': 1e-6}
                                 if optimize_options is None else optimize_options)
        self.fc_region = fc_region
        self.eta_A_w = eta_A_w

    def set_i(self, i=None, blur=None):
        if i is not None:
            self.i = i
        if blur is not None:
            self.blur = blur
        if i is not None or blur is not None:
            self.image = self.image_analyst.dict['I_nu_blur' if self.blur else 'I_nu'][i].T
            self.f_interp = RectBivariateSpline(self.x_interp, self.x_interp, self.image, kx=1,
                                                ky=1)

    def disk_f(self, x):
        x = np.asarray(x)
        if x.shape != (2,):
            raise NotImplementedError
        disk_cs = np.stack((np.cos(np.linspace(0, 2 * np.pi, self.n_theta, False)),
                            np.sin(np.linspace(0, 2 * np.pi, self.n_theta, False))))
        disk_r = np.linspace(0, self.r_max, int(np.ceil(self.r_max / self.dr)))
        disk_xy = np.einsum('ij,k->ijk', disk_cs, disk_r)
        disk_f = self.f_interp.ev(x[0] + disk_xy[0], x[1] + disk_xy[1])
        return disk_f

    def _r_theta(self, disk_f):
        n_theta, n_r = disk_f.shape
        ri_theta = np.zeros(n_theta, dtype=np.int_)
        is_bad = np.zeros(n_theta, dtype=np.int_)
        _ri_theta(disk_f, ri_theta, is_bad, n_theta, n_r, self.alpha, self.beta)
        fm_theta = disk_f[np.arange(n_theta), ri_theta]
        r_theta = ri_theta * self.dx
        return r_theta, is_bad, ri_theta, fm_theta

    def r_theta(self, x):
        return self._r_theta(self.disk_f(x))

    def _r_mean_std(self, r_theta, is_bad):
        is_bad_bool = is_bad.astype(bool)
        r_theta_good = r_theta[~is_bad_bool]
        assert np.all(r_theta_good >= 0)
        # r_mean, r_std, n_bad
        return np.mean(r_theta_good), np.std(r_theta_good), np.sum(is_bad_bool)

    def _loss(self, disk_f):
        n_theta, n_r = disk_f.shape
        r_theta, is_bad, ri_theta, fm_theta = self._r_theta(disk_f)
        r_mean, r_std, n_bad = self._r_mean_std(r_theta, is_bad)
        loss_0 = r_std / r_mean
        loss_1 = self.bad_dir_loss * n_bad / n_theta
        return loss_0 + loss_1

    def loss(self, x):
        loss_2 = self.center_out_loss if np.any(np.abs(x) > self.center_region) else 0 
        return self._loss(self.disk_f(x)) + loss_2

    def optimize(self):
        foo = minimize(self.loss, [0, 0], method='Nelder-Mead', options=self.optimize_options)
        return foo.x, foo.success

    def eta_A_single(self, x_center, r):
        ring_cs = np.stack((np.cos(np.linspace(0, 2 * np.pi, self.n_theta, False)),
                            np.sin(np.linspace(0, 2 * np.pi, self.n_theta, False))))
        ring_xy = r * ring_cs
        ring_f = self.f_interp.ev(x_center[0] + ring_xy[0], x_center[1] + ring_xy[1])
        itg = (np.mean(ring_f * np.exp(1j * np.linspace(0, 2 * np.pi, self.n_theta, False))) * 2 *
               np.pi)
        return np.angle(itg), np.absolute(itg) / (np.mean(ring_f) * 2 * np.pi)

    def eta_A(self, x_center, r_in, r_out):
        n_r = int(np.ceil((r_out - r_in) / self.dr))
        r_all = np.linspace(r_in, r_out, n_r)
        eta_A_all = [self.eta_A_single(x_center, r_i) for r_i in r_all]
        eta_all = np.asarray([foo[0] for foo in eta_A_all])
        A_all = np.asarray([foo[1] for foo in eta_A_all])
        # eta_mean = circmean(eta_all)
        # eta_std = circstd(eta_all)
        # A_mean = np.mean(A_all)
        # A_std = np.std(A_all)
        eta_mean, eta_std = circ_w(eta_all, r_all if self.eta_A_w else None)
        A_mean = np.average(A_all, weights=r_all if self.eta_A_w else None)
        A_std = np.sqrt(np.average((A_all - A_mean)**2, weights=r_all if self.eta_A_w else None))
        return eta_all, eta_mean, eta_std, A_all, A_mean, A_std

    def _f_mean_single(self, x_center, r):
        ring_cs = np.stack((np.cos(np.linspace(0, 2 * np.pi, self.n_theta, False)),
                            np.sin(np.linspace(0, 2 * np.pi, self.n_theta, False))))
        ring_xy = r * ring_cs
        ring_f = self.f_interp.ev(x_center[0] + ring_xy[0], x_center[1] + ring_xy[1])
        itg = np.mean(ring_f)
        return itg

    def fc(self, x_center, r_mean):
        f_r_mean = self._f_mean_single(x_center, r_mean)
        n_r = int(np.ceil(self.fc_region / self.dr))
        # dr = self.fc_region / n_r
        r_all = np.linspace(0, self.fc_region, n_r)
        f_in_all = np.array([self._f_mean_single(x_center, r_i) for r_i in r_all])
        # f_in = np.sum(f_in_all * r_all * dr) / np.sum(r_all * dr)
        f_in = np.sum(f_in_all * r_all) / np.sum(r_all)
        return f_in / f_r_mean

    def beta_m(self, i, x_center, m=2, bin_beta=False):
        y_grid, x_grid = np.meshgrid(self.x_interp, self.x_interp)
        phi_grid = ((np.arctan2(y_grid - x_center[1], x_grid - x_center[0]) - np.pi / 2) %
                    (2 * np.pi))
        eimp_grid = np.exp(-1j * m * phi_grid)
        inus = self.image_analyst.dict['I_nu_blur' if self.blur else 'I_nu'][i].T
        qnus = self.image_analyst.dict['Q_nu_blur' if self.blur else 'Q_nu'][i].T
        unus = self.image_analyst.dict['U_nu_blur' if self.blur else 'U_nu'][i].T
        beta_2 = np.sum((qnus + 1j * unus) * eimp_grid) / np.sum(inus)
        beta_2_bin = []
        if bin_beta is False:
            return beta_2
        else:
            r_grid = np.sqrt((x_grid - x_center[0])**2 + (y_grid - x_center[1])**2)
            for k in range(len(bin_beta) - 1):
                mask = np.where((bin_beta[k] < r_grid) * (r_grid < bin_beta[k + 1]))
                beta_2_bin.append(np.sum(((qnus + 1j * unus) * eimp_grid)[mask]) /
                                  np.sum(inus[mask]))
            return beta_2, beta_2_bin

    def run_single(self, i, blur=True, polar=False, bin_beta=False):
        self.set_i(i, blur)
        x_center, success = self.optimize()
        if success:
            disk_f = self.disk_f(x_center)
            n_theta, n_r = disk_f.shape
            r_theta, is_bad, ri_theta, fm_theta = self._r_theta(disk_f)
            ri_in = np.zeros(n_theta, dtype=np.int_)
            ri_out = np.zeros(n_theta, dtype=np.int_)
            _ri_io(disk_f, ri_theta, is_bad, ri_in, ri_out, n_theta, n_r, fm_theta * self.gamma)
            r_in_theta = ri_in * self.dx
            r_out_theta = ri_out * self.dx
            r_mean, r_std, n_bad = self._r_mean_std(r_theta, is_bad)
            r_in_mean, r_in_std, n_bad = self._r_mean_std(r_in_theta, is_bad)
            r_out_mean, r_out_std, n_bad = self._r_mean_std(r_out_theta, is_bad)
            w_theta = r_out_theta - r_in_theta
            w_mean, w_std, n_bad = self._r_mean_std(w_theta, is_bad)
            eta_all, eta_mean, eta_std, A_all, A_mean, A_std = self.eta_A(x_center, r_in_mean,
                                                                          r_out_mean)
            fc = self.fc(x_center, r_mean)

            if polar:
                if bin_beta is not False:
                    beta_2, beta_2_bin = self.beta_m(i, x_center, 2, bin_beta)
                    return PolarBinRingReport(x_center, n_bad, r_theta, r_mean, r_std,
                                              r_in_theta, r_in_mean, r_in_std,
                                              r_out_theta, r_out_mean, r_out_std,
                                              2 * r_theta, 2 * r_mean, 2 * r_std,
                                              w_theta, w_mean, w_std,
                                              eta_all, eta_mean, eta_std,
                                              A_all, A_mean, A_std, fc, beta_2, beta_2_bin)
                else:
                    beta_2 = self.beta_m(i, x_center, 2, bin_beta)
                    return PolarRingReport(x_center, n_bad, r_theta, r_mean, r_std,
                                           r_in_theta, r_in_mean, r_in_std,
                                           r_out_theta, r_out_mean, r_out_std,
                                           2 * r_theta, 2 * r_mean, 2 * r_std,
                                           w_theta, w_mean, w_std,
                                           eta_all, eta_mean, eta_std,
                                           A_all, A_mean, A_std, fc, beta_2)

            else:
                return RingReport(x_center, n_bad, r_theta, r_mean, r_std,
                                  r_in_theta, r_in_mean, r_in_std,
                                  r_out_theta, r_out_mean, r_out_std,
                                  2 * r_theta, 2 * r_mean, 2 * r_std,
                                  w_theta, w_mean, w_std,
                                  eta_all, eta_mean, eta_std,
                                  A_all, A_mean, A_std, fc)
        else:
            if polar:
                return PolarRingReport(x_center, -1)
            else:
                return RingReport(x_center, -1)

    def run(self, i_min=0, i_max=None, blur=True, polar=False, bin_beta=False, n_worker=10):
        if i_max is None:
            i_max = len(self.image_analyst.dict['I_nu'])
        with mp.Pool(n_worker) as pool:
            return pool.starmap(self.run_single, zip(range(i_min, i_max), [blur] * (i_max - i_min),
                                [polar] * (i_max - i_min), [bin_beta] * (i_max - i_min)))
