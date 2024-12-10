#!/usr/bin/env python
"""
This script runs the spectral fitting algorithm pPXF.

Author: Isabel Rivera
"""
import os
from astropy.table import Table
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import logging
from abc import ABC, abstractmethod
from lmfit.models import GaussianModel
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
        Assumed attenuation of the spectrum, in mag, at 5500 A (V-band).
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


class SegmentCombiner:
    """
    Class to combine multiple spectral segments into a single spectrum.
    """

    def __init__(self, segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """
        Initialize with segments to be combined.

        :param segments: List of tuples, each containing arrays of wavelengths, fluxes, and errors.
        """
        self.segments = segments

    def combine_segments(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Combine processed segments into a single spectrum.

        :return: Tuple containing arrays of combined wavelengths, fluxes, and errors.
        """
        combined_waves = np.concatenate([seg[0] for seg in self.segments])
        combined_fluxes = np.concatenate([seg[1] for seg in self.segments])
        combined_errors = np.concatenate([seg[2] for seg in self.segments])

        # Sort by wavelength
        sorted_indices = np.argsort(combined_waves)
        combined_waves = combined_waves[sorted_indices]
        combined_fluxes = combined_fluxes[sorted_indices]
        combined_errors = combined_errors[sorted_indices]

        return combined_waves, combined_fluxes, combined_errors

class InstrumentInfo:
    """
    Class to handle instrument-related information and calculations.
    """

    def __init__(self, R: float, instr_lam_min: float, instr_lam_max: float):
        """
        Initialize with instrument resolving power and wavelength range.

        :param R: The instrument's resolving power.
        :param instr_lam_min: Minimum wavelength of the instrument filter for the data (in μm).
        :param instr_lam_max: Maximum wavelength of the instrument filter for the data (in μm).
        """
        self.R = R
        self.instr_lam_min = instr_lam_min
        self.instr_lam_max = instr_lam_max

    def calc_FWHM_gal(self) -> float:
        """
        Calculate the spectral resolution FWHM of the galaxy spectrum.

        :return: FWHM of the galaxy spectrum in Angstroms.
        """
        FWHM_gal = 1e4 * np.sqrt(self.instr_lam_min * self.instr_lam_max) / self.R
        logging.info(f"FWHM_gal: {FWHM_gal:.1f} Å")
        return FWHM_gal

class LibraryHandler(ABC):
    """
    Abstract base class for template library handlers.
    """

    @abstractmethod
    def retrieve_templates(self, velscale: float, age_range: List[float]) -> Tuple[np.ndarray, Tuple[int, ...], np.ndarray, np.ndarray]:
        """
        Retrieve templates from the library.

        :param velscale: Velocity scale per pixel.
        :param age_range: List specifying the age range for the templates.
        :return: Tuple containing the stellar templates, regularization dimensions, template wavelengths,
                 and logarithm of template wavelengths.
        """
        pass

class StarburstLibraryHandler(LibraryHandler):

    def __init__(self, IMF_slope: str, star_form: str, lib_path: str, evol_track: str):
        import GalSpecFitX.starburst99_util as lib  # Import the library only if using this handler
        self.lib = lib
        self.IMF_slope = IMF_slope
        self.star_form = star_form
        self.lib_path = lib_path
        self.evol_track = evol_track


    def retrieve_templates(self, velscale: float, age_range: List[float]) -> Tuple[np.ndarray, Tuple[int, ...], np.ndarray, np.ndarray]:

        pathname = os.path.join(self.lib_path, 'STARBURST99', self.evol_track, self.star_form, self.IMF_slope, '*.fits')
        starburst99_lib = self.lib.starburst(pathname, velscale, self.lib_path, self.evol_track, age_range=age_range)

        return starburst99_lib

class BPASSLibraryHandler(LibraryHandler):
    """
    Handler for BPASS (Binary Population and Spectral Synthesis) stellar population models.
    """

    def __init__(self, IMF_slope: str, star_form: str, lib_path: str):
        """
        Initialize with specific BPASS parameters.

        :param IMF_slope: Initial Mass Function (IMF) slope for BPASS templates.
        """
        import GalSpecFitX.bpass_util as lib  # Import only if using this handler
        self.lib = lib
        self.IMF_slope = IMF_slope
        self.star_form = star_form
        self.lib_path = lib_path

    def retrieve_templates(self, velscale: float, age_range: List[float]) -> Tuple[np.ndarray, Tuple[int, ...], np.ndarray, np.ndarray]:
        """
        Retrieve BPASS templates from the library.

        :param velscale: Velocity scale per pixel.
        :param age_range: List specifying the age range for the templates.
        :return: Tuple containing the stellar templates, regularization dimensions, template wavelengths,
                 and logarithm of template wavelengths.
        """
        # ppxf_dir = os.path.dirname(os.path.realpath(self.lib.__file__))
        pathname = os.path.join(self.lib_path, 'BPASS', self.star_form, self.IMF_slope, '*.fits')

        bpass_lib = self.lib.bpass(pathname, velscale, self.lib_path, age_range=age_range, norm_range="continuum")

        # reg_dim = bpass_lib.templates.shape[1:]
        # stars_templates = bpass_lib.templates.reshape(bpass_lib.templates.shape[0], -1)
        # lam_temp = bpass_lib.lam_temp
        # ln_lam_temp = bpass_lib.ln_lam_temp

        return bpass_lib


class TemplateRetrieval:
    """
    Class to handle the retrieval of spectral templates, and to run pPXF software.
    """

    def __init__(self, gal_spectrum: Tuple[np.ndarray, np.ndarray, np.ndarray], library_handler: LibraryHandler):
        """
        Initialize with galaxy spectrum and a specific library handler.

        :param gal_spectrum: Tuple containing arrays of wavelengths, fluxes, and errors.
        :param library_handler: Instance of a library handler.
        """
        self.gal_spectrum = gal_spectrum
        self.library_handler = library_handler

    def retrieve_spectral_templates(self, age_range: List[float], default_noise: float) -> Tuple[np.ndarray, int, float, Tuple[int, ...], np.ndarray, np.ndarray]:
        """
        Retrieve spectral templates from the library.

        :param age_range: List specifying the age range for the templates.
        :return: Tuple containing the stellar templates, number of stars, velocity scale, regularization dimensions,
                 template wavelengths, and logarithm of template wavelengths.
        """
        gal_lam, gal_flux, gal_err = self.gal_spectrum

        # Ensure the error array is not zero to avoid division errors
        gal_err[gal_err == 0] = default_noise

        velscale = SPEED_OF_LIGHT * np.diff(np.log(gal_lam[[0, -1]])) / (gal_lam.size - 1)  # eq.(8) of Cappellari (2017)
        velscale = velscale[0]
        logging.info(f"Velocity scale per pixel: {velscale:.2f} km/s")

        library = self.library_handler.retrieve_templates(velscale, age_range)

        return library, velscale

    def gaussian_emission_lines(self, FWHM_gal: float, ln_lam_temp: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray, int]:
        """
        Generate Gaussian emission lines templates.

        :param FWHM_gal: spectral resolution FWHM of the galaxy spectrum.
        :param ln_lam_temp: Logarithm of template wavelengths.
        :return: Tuple containing gas templates, gas names, line wavelengths, and number of gas components.
        """
        gal_lam, gal_flux, gal_err = self.gal_spectrum

        lam_range_gal = [np.min(gal_lam), np.max(gal_lam)]
        gas_templates, gas_names, line_wave = util.emission_lines(ln_lam_temp, lam_range_gal, FWHM_gal)
        n_gas = len(gas_names)

        return gas_templates, gas_names, line_wave, n_gas

    def stack_stars_gas(self, stars_templates: np.ndarray, gas_templates: np.ndarray, n_stars: int, n_gas: int,
                        start_stars: List[float], start_gas: List[float]) -> Tuple[np.ndarray, List[int], Optional[np.ndarray], List[int], List[List[float]]]:
        """
        Stack stellar and gas templates.

        :param stars_templates: Stellar templates.
        :param gas_templates: Gas templates.
        :param n_stars: Number of stellar components.
        :param n_gas: Number of gas components.
        :param start_stars: Initial guess for stellar components.
        :param start_gas: Initial guess for gas components.
        :return: Tuple containing the combined templates, component array, gas component array, moments array, and start array.
        """
        templates = np.column_stack([stars_templates, gas_templates])

        if not gas_templates.size:
            component = [0] * n_stars
            gas_component = None
            moments = len(start_stars)
            start = start_stars
        else:
            component = [0] * n_stars + [1] * n_gas
            gas_component = np.array(component) > 0
            moments = [len(start_stars), len(start_gas)]
            start = [start_stars, start_gas]

        return templates, np.array(component), gas_component, moments, start

    def fit_spectrum(self, library, templates: np.ndarray, velscale: float, start: List[List[float]], dust, bounds, fixed, moments: List[int], lam_temp: np.ndarray,
                     reg_dim: Tuple[int, ...], component: List[int], gas_component: Optional[np.ndarray], gas_names: List[str], output_path: str,
                     **kwargs) -> None:
        """
        Fit the combined spectrum using pPXF.

        :param templates: Stars and gas templates.
        :param velscale: Velocity scale per pixel.
        :param start: Initial guess for the fit.
        :param moments: List of moments for each component.
        :param lam_temp: Template wavelengths.
        :param reg_dim: Regularization dimensions.
        :param component: List indicating which templates belong to which component.
        :param gas_component: Array indicating gas components.
        :param gas_names: Names of the gas components.
        :param mask: List of tuples containing wavelength ranges to exclude from the fit.
        :param output_path: Path to save the fitted spectrum.
        """
        gal_lam, gal_flux, gal_err = self.gal_spectrum

        mask = kwargs['mask']

        if mask is not None:

            mask_lam = np.ones(gal_lam.shape)

            try:
                mask_lam = np.logical_and(mask_lam, ~((gal_lam > mask[0]) & (gal_lam < mask[1])))

            except:
                for range in mask:
                    mask_lam = np.logical_and(mask_lam, ~((gal_lam > range[0]) & (gal_lam < range[1])))


            kwargs['mask'] = mask_lam

            # print('Dust is', dust)
            # print("Reddening is", kwargs['reddening'])


        pp = ppxf(templates, gal_flux, gal_err, velscale, start, dust=dust, bounds=bounds, fixed=fixed, moments=moments, lam=gal_lam, lam_temp=lam_temp,
                  reg_dim=reg_dim, component=component, gas_component=gas_component, gas_names=gas_names,
                  **kwargs)


        # Assuming pp.bestfit is a 1D array-like object
        table = Table([pp.bestfit], names=('flux',))

        # Convert the table to a FITS Binary Table HDU
        hdu1 = fits.BinTableHDU(table, name='BESTFIT')

        # Create a Primary HDU with no data
        primary_hdu = fits.PrimaryHDU()

        # Create an HDU list with the primary HDU and the table HDU
        hdul = fits.HDUList([primary_hdu, hdu1])

        # Write the HDU list to a new FITS file
        hdul.writeto('bestfit.fits', overwrite=True)


        # Create the Matplotlib figure
        fig, ax = plt.subplots(figsize=(11, 5))
        pp.plot()
        plt.grid(alpha=0.5)
        plt.savefig(os.path.join(output_path,'fitted_spectrum_static.png'))

        # Convert Matplotlib figure to Plotly
        plotly_fig = tls.mpl_to_plotly(fig)

        # Save the interactive plot as an HTML file
        pio.write_html(plotly_fig, file=os.path.join(output_path,'interactive_fitted_spectrum.html'), auto_open=False)

        # Create light weights plot
        light_weights = pp.weights
        light_weights = light_weights.reshape(reg_dim)  # Reshape to (n_ages, n_metal)
        light_weights /= light_weights.sum()            # Normalize to light fractions

        # Save light weights fractions
        fig, ax = plt.subplots(figsize=(11, 5))
        library.plot(light_weights)
        plt.title("Light Weights Fractions");
        plt.savefig(os.path.join(output_path,'light_weights.png'))


class SpectrumProcessor:
    """
    Class to process, combine, and fit spectra.
    """

    def __init__(self, config: dict, segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """
        Initialize with configuration settings.

        :param config: Configuration dictionary.
        :param segments: List of tuples, each containing arrays of wavelengths, fluxes, and errors.
        """
        self.config = config
        self.segments = segments

    def mask_spectral_lines(self, R: float, n_pix: Tuple[int, int]) -> List[Tuple[float, float]]:
        """
        Mask Milky Way absorption lines and fit Gaussian models to correct spectra.

        :param R: Instrumental resolution.
        :param n_pix: Tuple of two integers defining pixel range around each rest wavelength.
        :return: List of tuples defining wavelength ranges masked by Milky Way lines.
        """

        for i, segment in enumerate(self.config['segments']):

            lam_gal_log_rebin, norm_flux_gal_log_rebin, norm_err_gal_log_rebin = self.segments[i]

            logging.info('mask before adding milky way lines is: \n {}'.format(self.config['mask']))

            for absorp_lam in self.config['absorp_lam']:

                mask = (lam_gal_log_rebin > (absorp_lam-n_pix[0])) & (lam_gal_log_rebin < (absorp_lam+n_pix[1]))
                # print('mask',mask)
                # print('wavelength array', lam_gal_log_rebin)
                wavelength = lam_gal_log_rebin[mask]

                logging.info(f'wavelength range for milky way line at {absorp_lam}: {wavelength}')

                if wavelength.size > 0:

                    spectrum = norm_flux_gal_log_rebin[mask]
                    baseline = np.mean(spectrum)
                    spectrum -= baseline

                    sigma_inst = SPEED_OF_LIGHT/(R*2.355)

                    velocity = (wavelength - absorp_lam) / absorp_lam * SPEED_OF_LIGHT
                    # print('velocity', velocity)

                    # Step 4: Create lmfit model and set initial parameters
                    model = GaussianModel()
                    params = model.make_params(amplitude=-1, center=0, sigma=sigma_inst)  # sigma set to instrumental resolution

                    # Step 5: Fit the model to the spectrum
                    result = model.fit(spectrum, params, x=velocity)

                    # Step 6: Subtract the best-fit Gaussian from the spectrum
                    spectrum_corrected = spectrum - result.best_fit

                    if result.params['sigma'].value > 10*sigma_inst:

                        spectrum = spectrum_corrected

                        model = GaussianModel()
                        params = model.make_params(amplitude=-0.5, center=0, sigma=sigma_inst)  # sigma set to instrumental resolution

                        # Step 5: Fit the model to the spectrum
                        result = model.fit(spectrum, params, x=velocity)

                        # Step 6: Subtract the best-fit Gaussian from the spectrum
                        spectrum_corrected = spectrum - result.best_fit

                    fwhm_velocity = result.params['fwhm'].value  # FWHM in km/s

                    fwhm_wavelength = fwhm_velocity * absorp_lam / SPEED_OF_LIGHT

                    if self.config['mask'] == None:
                        self.config['mask'] = [(absorp_lam-fwhm_wavelength, absorp_lam+fwhm_wavelength)]

                    else:
                        try:
                            self.config['mask'].append((absorp_lam-fwhm_wavelength, absorp_lam+fwhm_wavelength))
                        except AttributeError:
                            self.config['mask'] = [self.config['mask']]
                            self.config['mask'].append((absorp_lam-fwhm_wavelength, absorp_lam+fwhm_wavelength))


                    # Print the fitting result
                    logging.info(result.fit_report())

                    # Create a Plotly figure
                    fig = go.Figure()

                    # Add traces for the plots
                    fig.add_trace(go.Scatter(x=wavelength, y=spectrum, mode='lines', name='Original Spectrum'))
                    fig.add_trace(go.Scatter(x=wavelength, y=result.best_fit, mode='lines', name='Fitted Gaussian', line=dict(dash='dash')))
                    fig.add_trace(go.Scatter(x=wavelength, y=spectrum_corrected, mode='lines', name='Corrected Spectrum'))
                    fig.add_trace(go.Scatter(x=[absorp_lam, absorp_lam], y=[np.min(spectrum_corrected), np.max(spectrum_corrected)],
                                             mode='lines', name=f'{absorp_lam}', line=dict(color='red', dash='dot')))

                    # Update layout
                    fig.update_layout(
                        title='Spectral Data',
                        xaxis_title='Wavelength (Angstroms)',
                        yaxis_title='Intensity',
                        legend_title='Legend',
                        template='plotly_white'
                    )

                    # Save the interactive plot as an HTML file
                    fig.write_html(os.path.join(self.config['output_path'], f'{segment}_{absorp_lam}_masked.html'), auto_open=False)


                logging.info('mask is now \n {}'.format(self.config['mask']))

        return self.config['mask']

    def process_combined_spectrum(self, library_handler: LibraryHandler) -> None:
        """
        Process the spectral data through various steps including combining spectra,
        retrieving templates, generating emission lines, stacking templates, and fitting the spectrum.

        :param library_handler: Handler for spectral library.
        """
        # Step 1: Combine the segments
        combiner = SegmentCombiner(self.segments)
        combined_spectrum = combiner.combine_segments()

        # Step 2: Initialize TemplateRetrieval with the combined spectrum
        template_retrieval = TemplateRetrieval(combined_spectrum, library_handler)

        # Step 3: Retrieve spectral templates
        age_range = self.config['age_range']
        library, velscale = template_retrieval.retrieve_spectral_templates(age_range, self.config['default_noise'])

        reg_dim = library.templates.shape[1:]
        stars_templates = library.templates.reshape(library.templates.shape[0], -1)
        lam_temp = library.lam_temp
        ln_lam_temp = library.ln_lam_temp
        n_stars = stars_templates.shape[1]

        # Step 4: Generate Gaussian emission lines templates
        FWHM_gal = self.config['FWHM_gal']
        gas_templates, gas_names, line_wave, n_gas = template_retrieval.gaussian_emission_lines(FWHM_gal, ln_lam_temp)

        # Step 5: Stack the stellar and gas templates
        start_stars = self.config['start_stars']
        start_gas = self.config['start_gas']

        if start_stars is None:
            start_stars = [0.0, 3 * velscale]
        if start_gas is None:
            start_gas = [0.0, 3 * velscale]

        templates, component, gas_component, moments, start = template_retrieval.stack_stars_gas(
            stars_templates, gas_templates, n_stars, n_gas, start_stars, start_gas
        )

        dust_stars = self.config['dust_stars']
        dust_gas = self.config['dust_stars']

        if dust_stars is not None:
            dust_stars["component"] = np.array(component) == 0
            dust_stars["func"] = reddy_attenuation
        elif dust_gas is not None:
            dust_gas["component"] = np.array(component) == 1
            dust_gas["func"] = reddy_attenuation

        bounds_stars = self.config['bounds_stars']
        bounds_gas = self.config['bounds_gas']
        fixed_stars = self.config['fixed_stars']
        fixed_gas = self.config['fixed_gas']

        if gas_component is None:
            dust = None if dust_stars is None else [dust_stars]
            bounds = bounds_stars
            fixed = fixed_stars
        else:
            dust = [dust_stars, dust_gas]
            bounds = [bounds_stars, bounds_gas]
            fixed = [fixed_stars, fixed_gas]

        # Step 6: Fit the spectrum using pPXF
        fit_kwargs = {
            key: value for key, value in self.config.items()
            if key not in ['age_range', 'start_stars', 'start_gas', 'FWHM_gal', 'output_path', 'absorp_lam', 'segments', 'default_noise', 'dust_stars', 'dust_gas', 'bounds_stars', 'bounds_gas', 'fixed_stars', 'fixed_gas'
            ]
        }

        light_weights = template_retrieval.fit_spectrum(
            library,
            templates,
            velscale,
            start,
            dust,
            bounds,
            fixed,
            moments,
            lam_temp,
            reg_dim,
            component,
            gas_component,
            gas_names,
            output_path = self.config['output_path'],

            **fit_kwargs
        )
