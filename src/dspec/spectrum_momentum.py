# coding: utf-8
"""
fork from C-WAVE algorithm to gete statistical momentum from an arbitrary decomposition of a spectrum described in theta and r (directions,wavenumbers)

C-WAVE params consist in 20 values, that are integrals of convolution of the spectrum with different kernels weightening some region of the spectrum
"""
import logging
import numpy as np
import xarray as xr
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon

KMAX = 2 * np.pi / 60
KMIN = 2 * np.pi / 625


def get_polygon_wavenumber(wavenumber_val):
    """

    :param wavenumber_val: float [meter]
    :return:
    """

    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    N_pts = 100
    phi = np.linspace(0, 2 * np.pi, N_pts)
    radius = np.ones(N_pts) * 2 * np.pi / wavenumber_val
    x_pos_vect = []
    y_pos_vect = []
    for uu in range(N_pts):
        x_pos, y_pos = pol2cart(radius[uu], phi[uu])
        x_pos_vect.append(x_pos)
        y_pos_vect.append(y_pos)
    x_pos_vect = np.array(x_pos_vect)
    y_pos_vect = np.array(y_pos_vect)
    coords_cart = np.stack([x_pos_vect, y_pos_vect]).T

    return Polygon(coords_cart)


def transform_k_into_index(inter, k_rg, k_az):
    """

    :param inter: shapely MultiPoint intersection composed of shapely Point()
    :param k_rg: np.ndarray wavenumber in range azix
    :param k_az: np.ndarray wavenumber in azimuth axis
    :return:
    """
    range_indexes = []
    az_indexes = []
    for uu in inter.geoms:
        range_index = np.where(k_rg == uu.x)[0][0]
        azimuth_index = np.where(k_az == uu.y)[0][0]
        range_indexes.append(range_index)
        az_indexes.append(azimuth_index)
    range_indexes = np.array(range_indexes)
    az_indexes = np.array(az_indexes)
    return range_indexes, az_indexes


def filter_cartesian_with_wavelength_ring(lower_wl, higher_wl, one_spec_re):
    """

    :param lower_wl: float
    :param higher_wl: float
    :param one_spec_re: xarray.DataArray
    :return:
    """
    pol_lower = get_polygon_wavenumber(wavenumber_val=lower_wl)
    pol_higher = get_polygon_wavenumber(wavenumber_val=higher_wl)
    KX, KY = np.meshgrid(one_spec_re.k_rg, one_spec_re.k_az)

    pp = np.stack([KX.ravel(), KY.ravel()]).T
    multipp = MultiPoint(pp)
    ring = pol_higher.symmetric_difference(pol_lower)
    inter = ring.intersection(multipp)
    range_indexes, az_indexes = transform_k_into_index(inter, k_rg=one_spec_re.k_rg, k_az=one_spec_re.k_az)
    all_zeros = np.zeros(one_spec_re.shape)
    coords = (az_indexes, range_indexes)
    range_indexes_ds = xr.DataArray(range_indexes, dims=['pts'])
    az_indexes_ds = xr.DataArray(az_indexes, dims=['pts'])
    subspec = one_spec_re.isel(k_rg=range_indexes_ds, k_az=az_indexes_ds).values
    np.add.at(all_zeros, coords, subspec)
    final_spec = xr.DataArray(all_zeros, coords=one_spec_re.coords, attrs=one_spec_re.attrs, dims=one_spec_re.dims)
    return final_spec


def getWeightingMatrix(kmin, kmax, cspc):
    """

    :param kmin: float
    :param kmax: float
    :param cspc: xarray.DataArray with k_rg and k_az coordinates
    :return:
    """
    gamma = 2
    a1 = (gamma ** 2 - np.power(gamma, 4)) / (gamma ** 2 * kmin ** 2 - kmax ** 2)
    a2 = (kmax ** 2 - np.power(gamma, 4) * kmin ** 2) / (kmax ** 2 - gamma ** 2 * kmin ** 2)
    # % Ellipse
    KX, KY = np.meshgrid(cspc.k_rg, cspc.k_az)
    logging.debug('KX: %s', KX.shape)
    tmp = a1 * np.power(KX, 4) + a2 * KX ** 2 + KY ** 2
    logging.debug('tmp ellipse : %s', tmp.shape)
    # eta
    eta = np.sqrt((2. * tmp) / ((KX ** 2 + KY ** 2) * tmp * np.log10(kmax / kmin)))
    logging.debug('eta = %s', eta.shape)
    alphak = 2. * ((np.log10(np.sqrt(tmp)) - np.log10(kmin)) / np.log10(kmax / kmin)) - 1
    alphak[(alphak ** 2) > 1] = 1.
    alphat = np.arctan2(KY, KX)
    logging.debug('alphat = %s', alphat.shape)

    # Gegenbauer polynomials
    tmp = abs(np.sqrt(1 - alphak ** 2))  # % imaginary???
    g1 = 1 / 2. * np.sqrt(3) * tmp
    g2 = 1 / 2. * np.sqrt(15) * alphak * tmp
    g3 = np.dot((1 / 4.) * np.sqrt(7. / 6.), (15. * np.power(alphak, 2) - 3.)) * tmp  #
    g4 = (1 / 4.) * np.sqrt(9. / 10) * (35. * np.power(alphak, 3) - 15. * alphak ** 2) * tmp
    logging.debug('g1 = %s', g1.shape)

    # Harmonic functions
    f1 = np.sqrt(1 / np.pi) * np.cos(0. * alphat)
    f2 = np.sqrt(2 / np.pi) * np.sin(2. * alphat)
    f3 = np.sqrt(2 / np.pi) * np.cos(2. * alphat)
    f4 = np.sqrt(2 / np.pi) * np.sin(4. * alphat)
    f5 = np.sqrt(2 / np.pi) * np.cos(4. * alphat)

    # Weighting functions
    logging.debug('KX shape = %s', KX.shape)
    h = np.ones((KX.shape[0], KX.shape[1], 20))
    logging.debug('h computation g1 = %s f1 = %s eta= %s', g1.shape, f1.shape, eta.shape)
    h[:, :, 0] = g1 * f1 * eta
    h[:, :, 1] = g1 * f2 * eta
    h[:, :, 2] = g1 * f3 * eta
    h[:, :, 3] = g1 * f4 * eta
    h[:, :, 4] = g1 * f5 * eta
    h[:, :, 5] = g2 * f1 * eta
    h[:, :, 6] = g2 * f2 * eta
    h[:, :, 7] = g2 * f3 * eta
    h[:, :, 8] = g2 * f4 * eta
    h[:, :, 9] = g2 * f5 * eta
    h[:, :, 10] = g3 * f1 * eta
    h[:, :, 11] = g3 * f2 * eta
    h[:, :, 12] = g3 * f3 * eta
    h[:, :, 13] = g3 * f4 * eta
    h[:, :, 14] = g3 * f5 * eta
    h[:, :, 15] = g4 * f1 * eta
    h[:, :, 16] = g4 * f2 * eta
    h[:, :, 17] = g4 * f3 * eta
    h[:, :, 18] = g4 * f4 * eta
    h[:, :, 19] = g4 * f5 * eta
    return h


def orthogonalDecompSpec(kmin, kmax, cspc):
    """

    :param kmin: float
    :param kmax: float
    :param cspc: xarray.DataArray with k_rg and k_az coordinates
    :return:
        S: np.ndarray orthogonal moments of the input spectra
    """
    # Compute Orthogonal Moments
    ns = 20  # % number of variables in orthogonal decomposition
    S = np.ones((ns, 1)) * np.nan
    unique_kx = cspc.k_rg
    unique_ky = cspc.k_az
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    # logging.debug('unique rg : %s %s', len(unique_kx), is_sorted(unique_kx))
    # logging.debug('unique az : %s %s', len(unique_ky), is_sorted(unique_ky))
    dkx = np.diff(unique_kx)[0] * np.ones(unique_kx.shape)
    dky = np.diff(unique_ky)[0] * np.ones(unique_ky.shape)
    logging.info('dkx: %s dky: %s', dkx.shape, dky.shape)
    DKX, DKY = np.meshgrid(dkx, dky)
    h = getWeightingMatrix(kmin, kmax, cspc)
    P = cspc / (np.nansum(np.nansum(cspc * DKX * DKY)))
    logging.debug('P shape = %s', P.shape)
    logging.debug('h shape = %s', h.shape)
    for jj in range(S.size):
        small_h = h[:, :, jj].squeeze().T
        S[jj] = np.nansum(np.nansum(small_h * P.T * DKX.T * DKY.T))
    logging.debug('S = %s', len(S))
    return S


def prepareSpectra(real_part_spectrum, imaginary_part_spectrum, kmax=KMAX, kmin=KMIN):
    """
    apply a k_min, k_max filter on the cartesian spectrum
    :args:
        real_part_spectrum : (xarray.DataArray): described in coordinates (kx,ky)
        imaginary_part_spectrum : (xarray.DataArray): described in coordinates (kx,ky)
    :returns:
        CWAVE: (np.ndarray) 20 floats computed on X spectra
    """
    lower_wl = 2 * np.pi / kmax
    higher_lambda = 2 * np.pi / kmin
    sub_re = filter_cartesian_with_wavelength_ring(lower_wl, higher_lambda, real_part_spectrum)
    sub_im = filter_cartesian_with_wavelength_ring(lower_wl, higher_lambda, imaginary_part_spectrum)
    cspc = np.sqrt(sub_re ** 2 + sub_im ** 2)
    return cspc
