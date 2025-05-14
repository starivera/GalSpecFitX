#!/usr/bin/env python
"""
This script runs the spectral fitting algorithm pPXF.

Author: Isabel Rivera
"""
import os
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from lmfit import Model
import plotly.tools as tls
import plotly.io as pio
import plotly.graph_objects as go

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

# Constants
SPEED_OF_LIGHT = 299792.458  # Speed of light in km/s

def reddy_attenuation(lam, a_v, delta=None, f_nodust=None, uv_bump=None):
    """
    Combines the attenuation curves from
    `Kriek & Conroy (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...775L..16K>`_
    hereafter KC13,
    `Calzetti et al. (2000) <http://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C>`_
    hereafter C+00,
    `Reddy et al. (2016): <https://doi.org/10.3847/0004-637X/828/2/107>`_,
    `Noll et al. (2009) <https://ui.adsabs.harvard.edu/abs/2009A%26A...499...69N>`_,
    and `Lower et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022ApJ...931...14L>`_.

    When ``delta = uv_bump = f_nodust = None`` this function returns the Reddy+16 and C+00
    reddening curve. When ``uv_bump = f_nodust = None`` this function uses the
    ``delta - uv_bump`` relation by KC13. The parametrization of the UV bump
    comes from Noll+09. The modelling of the attenuated fraction follows Lower+22.

    Input Parameters
    ----------------

    lam: array_like with shape (n_pixels,)
        Restframe wavelength in Angstroms of each pixel in the galaxy spectrum.
    a_v: float
        Assumed attenuation of the spectrum, in mag, at 5500 Å (V-band).
    delta: float, optional
        UV slope of the spectrum.
    f_nodust: float, optional
        Fraction of stellar light that is not attenuated.
    uv_bump: float, optional
        Amplitude of the UV bump. If ``uv_bump=None`` uses the relation of
        KC13 to predict ``uv_bump`` from ``delta``.

    Output Parameters
    -----------------

    frac: array_like with shape (n_pixels,)
        Fraction by which the spectrum flux at each wavelength has to be
        multiplied, to model the attenuation effect.

    """
    lam = lam/1e4   # Angstroms --> micron
    r_v = 3.1
    e_bv = a_v/r_v

    # print('lam is', lam)

    # Reddy+16 equation (3) for lam > 0.095, C+00 otherwise
    k1 = np.where(lam >= 0.095, 2.191 + 0.974/lam,
                r_v + ((0.029249/lam - 0.526482)/lam + 4.01243)/lam - 5.7328)

    if (delta is None) and (uv_bump is None):
        a_lam = e_bv*k1
    else:
        if uv_bump is None:
            uv_bump = 0.85 - 1.9*delta  # eq.(3) KC13
        lam_0 = 0.2175                  # Peak wavelength of UV bump in micron
        delta_lam = 0.035               # Width of UV bump in micron
        d_lam = uv_bump*(lam*delta_lam)**2/((lam**2 - lam_0**2)**2 + (lam*delta_lam)**2)    # eq.(2) KC13
        lam_v = 0.55                    # Effective V-band wavelength in micron
        a_lam = e_bv*(k1 + d_lam)*(lam/lam_v)**delta                                        # eq.(1) KC13

    frac = 10**(-0.4*a_lam.clip(0))     # C+00 equation (2) with opposite sign

    if f_nodust is not None:
        frac = f_nodust + (1 - f_nodust)*frac

    return frac     # The model spectrum has to be multiplied by this vector


class InstrumentInfo:
    """
    Class to handle instrument-related information and calculations.
    """

    def __init__(self, R: float, instr_lam_min: float, instr_lam_max: float):
        """
        Initialize with instrument resolving power and wavelength range.

        :param R: The instrument's resolving power (dimensionless).
        :param instr_lam_min: Minimum wavelength of the instrument filter for the data (in Angstroms, Å).
        :param instr_lam_max: Maximum wavelength of the instrument filter for the data (in Angstroms, Å).
        """
        self.R = R
        self.instr_lam_min = instr_lam_min
        self.instr_lam_max = instr_lam_max

    def calc_FWHM_gal(self) -> float:
        """
        Calculate the spectral resolution FWHM of the galaxy spectrum.

        The FWHM is calculated using the instrument's resolving power `R` and the geometric mean
        of the minimum and maximum wavelengths of the instrument filter.

        :return: FWHM of the galaxy spectrum in Angstroms (Å).
        """
        FWHM_gal = np.sqrt(self.instr_lam_min * self.instr_lam_max) / self.R
        logging.info(f"FWHM_gal: {FWHM_gal:.5f} Å")

        return FWHM_gal

class LibraryHandler(ABC):
    """
    Abstract base class for template library handlers. This class defines the
    interface for retrieving stellar population templates from different libraries.
    """

    @abstractmethod
    def retrieve_templates(self, velscale: float, age_range: List[float], metal_range: List[float], norm_range: List[float], FWHM_gal: float) -> any:
        """
        Retrieve templates from the specific template library.

        :param velscale: Velocity scale per pixel in km/s.
        :param age_range: List of two floats representing the age range in Gyr for the templates to be retrieved `[age_min, age_max]`.
        :param metal_range: List of two floats representing the metallicity range for the templates to be retrieved `[metal_min, metal_max]` (e.g., 0.020 = Z_solar).
        :param norm_range: List of two floats representing the wavelength range in Angstroms within which to compute the templates' normalization `[norm_min, norm_max]`.
        :param FWHM_gal: Full Width at Half Maximum (FWHM) of the galaxy's spectral line spread, in km/s.

        :return: The retrieved templates. The specific type depends on the implementation.
        """
        pass

class StarburstLibraryHandler(LibraryHandler):
    """
    Handler for Starburst99 stellar population models. This handler is used to retrieve
    templates from the Starburst99 library based on the specified parameters.
    """

    def __init__(self, IMF_slope: str, star_form: str, star_evol: str, lib_path: str, evol_track: str):
        """
        Initialize the handler with the necessary parameters for the Starburst99 library.

        :param IMF_slope: Initial Mass Function (IMF) slope for the Starburst99 templates.
        :param star_form: Star formation scenario (e.g., instantaneous, continuous).
        :param star_evol: Star evolution scenario (e.g., single, binary).
        :param lib_path: Path to the base directory of the Starburst99 library.
        :param evol_track: Evolutionary track to be used for the Starburst99 models.
        """
        import GalSpecFitX.starburst99_util as lib  # Import the library only if using this handler
        self.lib = lib
        self.IMF_slope = IMF_slope
        self.star_form = star_form
        self.star_evol = star_evol
        self.lib_path = lib_path
        self.evol_track = evol_track


    def retrieve_templates(self, velscale: float, age_range: List[float], metal_range: List[float], norm_range:List[float], FWHM_gal: float) -> any:
        """
        Retrieve the Starburst99 templates based on the specified parameters.

        :param velscale: Velocity scale per pixel in km/s.
        :param age_range: List of two floats representing the age range in Gyr for the templates to be retrieved `[age_min, age_max]`.
        :param metal_range: List of two floats representing the metallicity range for the templates to be retrieved `[metal_min, metal_max]` (e.g., 0.020 = Z_solar).
        :param norm_range: List of two floats representing the wavelength range in Angstroms within which to compute the templates' normalization `[norm_min, norm_max]`.
        :param FWHM_gal: Full Width at Half Maximum (FWHM) of the galaxy's spectral line spread, in km/s.

        :return: The retrieved Starburst99 templates.
        """
        pathname = os.path.join(self.lib_path, 'STARBURST99', self.evol_track, self.star_form, self.star_evol, self.IMF_slope, '*.fits')
        starburst99_lib = self.lib.starburst(pathname, velscale, self.lib_path, self.evol_track, age_range=age_range, metal_range=metal_range, norm_range=norm_range, FWHM_gal=FWHM_gal)

        return starburst99_lib

class BPASSLibraryHandler(LibraryHandler):
    """
    Handler for BPASS (Binary Population and Spectral Synthesis) stellar population models.
    This handler retrieves templates from the BPASS library based on the specified parameters.
    """

    def __init__(self, IMF_slope: str, star_form: str, star_evol: str, lib_path: str):
        """
        Initialize the handler with the necessary parameters for the BPASS library.

        :param IMF_slope: Initial Mass Function (IMF) slope for the BPASS templates.
        :param star_form: Star formation scenario (e.g., single, binary).
        :param star_evol: Star evolution scenario (e.g., single, binary).
        :param lib_path: Path to the base directory of the BPASS library.
        """
        import GalSpecFitX.bpass_util as lib  # Import only if using this handler
        self.lib = lib
        self.IMF_slope = IMF_slope
        self.star_form = star_form
        self.star_evol = star_evol
        self.lib_path = lib_path

    def retrieve_templates(self, velscale: float, age_range: List[float], metal_range: List[float], norm_range: List[float], FWHM_gal: float) -> any:
        """
        Retrieve the BPASS templates based on the specified parameters.

        :param velscale: Velocity scale per pixel in km/s.
        :param age_range: List of two floats representing the age range in Gyr for the templates to be retrieved `[age_min, age_max]`.
        :param metal_range: List of two floats representing the metallicity range for the templates to be retrieved `[metal_min, metal_max]` (e.g., 0.020 = Z_solar).
        :param norm_range: List of two floats representing the wavelength range in Angstroms within which to compute the templates' normalization `[norm_min, norm_max]`.
        :param FWHM_gal: Full Width at Half Maximum (FWHM) of the galaxy's spectral line spread, in km/s.

        :return: The retrieved BPASS templates.
        """
        pathname = os.path.join(self.lib_path, 'BPASS', self.star_form, self.star_evol, self.IMF_slope, '*.fits')
        bpass_lib = self.lib.bpass(pathname, velscale, self.lib_path, age_range=age_range, metal_range=metal_range, norm_range=norm_range, FWHM_gal=FWHM_gal)

        return bpass_lib


class TemplateRetrieval:
    """
    Class to handle the retrieval of spectral templates, and to run pPXF software.

    """

    def __init__(self, gal_spectrum: Tuple[np.ndarray, np.ndarray, np.ndarray], library_handler: LibraryHandler):
        """
        Initialize with galaxy spectrum and a specific library handler.

        :param gal_spectrum: Tuple containing arrays of wavelengths, fluxes, and errors.
        :param library_handler: Instance of a library handler responsible for retrieving spectral templates.
        """
        self.gal_spectrum = gal_spectrum
        self.library_handler = library_handler

    def retrieve_spectral_templates(self, age_range: List[float], metal_range: List[float], norm_range: List[float], default_noise: float, FWHM_gal: float) -> Tuple[any, float]:
        """
        Retrieve spectral templates from the library.

        This function uses the provided age range, the spectral resolution (FWHM), and galaxy spectrum to retrieve the corresponding stellar templates from a given library.

        :param age_range: List specifying the age range for the templates.
        :param metal_range: List specifying the metalicity range for the templates.
        :param norm_range: List specifying the normalization range for the templates.
        :param default_noise: Value used to replace zero error values in the galaxy spectrum.
        :param FWHM_gal: Spectral resolution (FWHM) of the galaxy spectrum.

        :return: A tuple containing the spectral templates (LibraryInstance) and the velocity scale in km/s.
        """
        gal_lam, gal_flux, gal_err = self.gal_spectrum

        # Ensure the error array is not zero to avoid division errors
        gal_err[gal_err == 0] = default_noise

        velscale = SPEED_OF_LIGHT * np.diff(np.log(gal_lam[[0, -1]])) / (gal_lam.size - 1)  # eq.(8) of Cappellari (2017)
        velscale = velscale[0]
        logging.info(f"Velocity scale per pixel: {velscale:.5f} km/s")

        library = self.library_handler.retrieve_templates(velscale, age_range, metal_range, norm_range, FWHM_gal)

        return library, velscale


    def fit_spectrum(self, library: object, templates: np.ndarray, velscale: float, start: List[List[float]], dust: dict, moments: List[int], lam_temp: np.ndarray,
                     reg_dim: Tuple[int, ...], output_path: str, n_iterations: int,
                     **kwargs) -> None:
        """
        Fit the combined spectrum using pPXF.

        This function uses the pPXF software to fit the galaxy spectrum to the stellar templates, optionally with multiple iterations to compute uncertainties.

        :param library: Template library.
        :param templates: Stellar templates.
        :param velscale: Velocity scale per pixel.
        :param start: Initial guess for the fit.
        :param dust: Dictionary with dust extinction information.
        :param moments: List of moments for each component.
        :param lam_temp: Template wavelengths.
        :param reg_dim: Regularization dimensions.
        :param output_path: Path to save the fitted spectrum.
        :param n_iterations: Number of iterations to run for uncertainty calculation.
        :param kwargs: Additional keyword arguments.
        """
        gal_lam, gal_flux, gal_err = self.gal_spectrum

        mask = kwargs.get('mask', None)

        if mask is not None:
            mask_lam = np.ones(gal_lam.shape, dtype=bool)  # Initialize mask as boolean array

            # Function to apply a range mask
            def apply_range_mask(mask_lam, gal_lam, mask_range):
                return np.logical_and(mask_lam, ~((gal_lam > mask_range[0]) & (gal_lam < mask_range[1])))

            try:
                # If mask is a single range (expected to be a list or tuple of length 2)
                if len(mask) == 2:
                    mask_lam = apply_range_mask(mask_lam, gal_lam, mask)
                else:
                    # Otherwise, assume it's a list of ranges
                    for mask_range in mask:
                        mask_lam = apply_range_mask(mask_lam, gal_lam, mask_range)

            except (TypeError, IndexError) as e:
                raise ValueError("Mask should be a 2-element range or a list of ranges.") from e

            kwargs['mask'] = mask_lam


        pp = ppxf(templates, gal_flux, gal_err, velscale, start, dust=dust, moments=moments, lam=gal_lam, lam_temp=lam_temp,
                  reg_dim=reg_dim,
                  **kwargs)

        if pp.dust is not None:
            a_v = pp.dust[0]['sol'][0]
        else:
            a_v = 0.0


        # Assuming pp.bestfit is a 1D array-like object
        bestfit_table = Table([pp.bestfit], names=('flux',))

        # Convert the table to a FITS Binary Table HDU
        bestfit_hdu = fits.BinTableHDU(bestfit_table, name='BESTFIT')

        # Open bestfit.fits which should already exist in the output path and append the new table
        with fits.open(os.path.join(output_path,'bestfit.fits'), mode='append') as hdul:
            hdul.append(bestfit_hdu)

        # Create the Matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 4))
        pp.plot()
        plt.grid(alpha=0.5)
        lines = plt.gca().lines
        lines[0].set_linewidth(0.5)
        lines[0].set_linewidth(1.0)

        # NV doublet (1238.82 and 1242.80 Å)
        plt.axvline(x=.123882, color='purple', linestyle='--')
        plt.axvline(x=.124280, color='purple', linestyle='--')
        plt.text(.123882 - 0.0002, 1.4, 'N V', color='purple', va='bottom', ha='right')

        # OV line (1371 Å)
        plt.axvline(x=.137130, color='orange', linestyle='--')
        plt.text(.137130 - 0.0002, 1.4, 'O V', color='orange', va='bottom', ha='right')

        plt.xlim(.1220, .1270)
        plt.ylim(0.0, 1.75)
        plt.savefig(os.path.join(output_path,'fitted_spectrum_static.png'), dpi=150)

        # Convert Matplotlib figure to Plotly
        plotly_fig = tls.mpl_to_plotly(fig)

        # Save the interactive plot as an HTML file
        pio.write_html(plotly_fig, file=os.path.join(output_path,'interactive_fitted_spectrum.html'), auto_open=False)

        # Create light weights plot
        light_weights = pp.weights
        light_weights = light_weights.reshape(reg_dim)  # Reshape to (n_ages, n_metal)
        light_weights /= light_weights.sum()            # Normalize to light fractions

        # Compute uncertainties if n_iterations is specified
        std_ages, std_metallicities, std_A_v = None, None, None
        if n_iterations is not None:
            std_ages, std_metallicities, std_A_v = self._compute_uncertainties(
                n_iterations, library, templates, gal_flux, gal_err, velscale, start, dust, moments, gal_lam, lam_temp, reg_dim, **kwargs
            )

        library.plot(light_weights, output_path, std_ages, std_metallicities, a_v, std_A_v)

    def _compute_uncertainties(self, n_iterations: int, library, templates, gal_flux, gal_err, velscale, start, dust, moments, gal_lam, lam_temp, reg_dim, **kwargs):
        """
        Compute uncertainties via Monte Carlo iterations.
        """
        ages, metallicities, A_v = [], [], []

        for i in range(n_iterations):
            simulated_flux = gal_flux + np.random.normal(0, gal_err)

            try:
                pp_sim = ppxf(templates, simulated_flux, gal_err, velscale, start, dust=dust, moments=moments,
                              lam=gal_lam, lam_temp=lam_temp, reg_dim=reg_dim, **kwargs)

                light_weights_sim = pp_sim.weights.reshape(reg_dim)
                light_weights_sim /= light_weights_sim.sum()

                mean_age, mean_z = library.plot(light_weights_sim, plot=False)
                ages.append(mean_age)
                metallicities.append(mean_z)

                if pp_sim.dust is not None:
                    A_v.append(pp_sim.dust[0]['sol'][0])

            except ValueError:
                continue

        return np.std(ages), np.std(metallicities), np.std(A_v) if A_v else None


class SpectrumProcessor:
    """
    Class to process, and fit a spectrum.

    This class handles spectral data processing, including masking spectral lines, retrieving templates, and fitting the spectrum with pPXF.
    """

    def __init__(self, config: dict, gal_spectrum: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """
        Initialize with configuration settings.

        :param config: Configuration dictionary with keys controlling processing options, such as:
                       - 'absorp_lam': List of absorption line wavelengths.
                       - 'output_path': Path to save output files.
        :param gal_spectrum: Tuple which contains three arrays (wavelength, flux, error).
        """
        self.config = config
        self.gal_spectrum = gal_spectrum

    def mask_spectral_lines(self, R: float) -> List[Tuple[float, float]]:
        """
        Mask Milky Way absorption lines by fitting Gaussian models and estimating FWHM.

        :param R: Instrumental resolution as a float.
        :return: List of wavelength range tuples `(float, float)` masked during processing.
        """
        logging.info(f'Beginning spectral line fitting')

        # Extract spectrum data
        lam_gal, flux_gal, err_gal = self.gal_spectrum

        updated_mask = updated_mask = self.config.get('mask', [])

        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        absorp_config = self.config.get('absorp_lam', {})

        if isinstance(absorp_config, list):
            absorp_dict = {lam: (lam / R) * 5 for lam in absorp_config}
        elif isinstance(absorp_config, dict):
            absorp_dict = {float(k): float(v) for k, v in absorp_config.items()}
        else:
            logging.warning("absorp_lam must be a list or dictionary. Skipping masking.")
            return updated_mask

        for absorp_lam, window in absorp_dict.items():
            mask = (lam_gal > absorp_lam - window) & (lam_gal < absorp_lam + window)

            if not np.any(mask):
                logging.warning(f"No data points found for absorption line at {absorp_lam}. Skipping.")
                continue

            x = lam_gal[mask]
            y = flux_gal[mask]
            y = y / np.max(y)  # Normalize

            # Estimate error; fall back to constant error if none
            if err_gal is not None:
                yerr = err_gal[mask]
                weights = 1 / np.maximum(yerr, 1e-3)
            else:
                weights = np.ones_like(x)

            # Initial parameter guesses: amp, mu, sigma
            amp_guess = 1 - np.min(y)
            mu_guess = absorp_lam
            sigma_guess = (absorp_lam / R) / 2.355

            try:
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(
                    gaussian, x, 1 - y,
                    p0=[amp_guess, mu_guess, sigma_guess],
                    sigma=1 / weights,
                    absolute_sigma=True
                )
                amp, mu, sigma = popt
                fwhm = 2.355 * abs(sigma)
                masked_region = (mu - fwhm, mu + fwhm)
                updated_mask.append(masked_region)

                logging.info(f"Fitted absorption line at {mu:.2f} Å with FWHM = {fwhm:.2f} Å")
            except RuntimeError as e:
                logging.warning(f"Gaussian fit failed for line at {absorp_lam} Å: {e}")
                continue

        self.config['mask'] = updated_mask
        logging.info(f'Updated mask: {updated_mask}')
        logging.info(f'Ending spectral line fitting')

        return updated_mask



    def process_spectrum(self, library_handler: LibraryHandler) -> None:
        """
        Process the spectral data through steps like retrieving templates, setting parameters, and fitting the spectrum with pPXF.

        :param library_handler: Instance of `LibraryHandler` for spectral template retrieval.
        """

        # Step 1: Retrieve spectral templates
        template_retrieval = TemplateRetrieval(self.gal_spectrum, library_handler)
        library, velscale = template_retrieval.retrieve_spectral_templates(age_range = self.config['age_range'], metal_range = self.config['metal_range'], norm_range = self.config['norm_range'], default_noise = self.config['default_noise'], FWHM_gal = self.config['FWHM_gal'])

        # Step 2: Set fit parameters
        reg_dim = library.templates.shape[1:]
        stars_templates = library.templates.reshape(library.templates.shape[0], -1)
        lam_temp = library.lam_temp
        n_stars = stars_templates.shape[1]
        component = [0] * n_stars

        start_stars = self.config['start']

        if start_stars is None:
            start_stars = [0.0, 3 * velscale]

        moments = len(start_stars)

        dust_stars = self.config['dust']

        if dust_stars is not None:
            dust_stars["component"] = np.array(component) == 0
            dust_stars["func"] = reddy_attenuation
            dust_stars = [dust_stars]


        # Step 3: Fit the spectrum using pPXF
        fit_kwargs = {
            key: value for key, value in self.config.items()
            if key not in ['age_range', 'metal_range', 'norm_range', 'FWHM_gal', 'output_path', 'absorp_lam', 'segment', 'default_noise', 'n_iterations', 'start', 'dust',
            ]
        }

        template_retrieval.fit_spectrum(
            library,
            stars_templates,
            velscale,
            start_stars,
            dust_stars,
            moments,
            lam_temp,
            reg_dim,
            output_path = self.config['output_path'],
            n_iterations = self.config['n_iterations'],

            **fit_kwargs
        )
