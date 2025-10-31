###############################################################################
# This module provides utilities and classes to handle stellar
# population synthesis (SPS) template grids. The code supports:
#
#   - Loading grids of Single Stellar Population (SSP) spectral templates
#   - Extracting age and metallicity from filenames
#   - Logarithmic rebinning and convolving templates for use with spectral fitting codes (e.g., pPXF)
#   - Trimming, normalizing, and visualizing template grids
#
# Filename Convention
# ------------------
# Template filenames must encode age and metallicity using a consistent pattern:
#     '...Zsol<M>.T<A>...'
# where:
#     M = metallicity in fractional solar units (Z/Z☉)
#     A = age in Gyr
# This convention is compatible with any SPS library following the same pattern.
#
# Grids
# -----
# The code assumes templates form a complete Cartesian grid in age–metallicity space.
#
# Classes
# -------
# SPSLibrary
#     Manages a library of SSP templates, handles convolution, rebinning, normalization,
#     subsetting, and plotting of template weights.
#
# Functions
# ---------
# age_metal(filename)
#     Extracts age and metallicity from a template filename.
#
# Usage Example
# -------------
# >>> from sps_library import SPSLibrary, age_metal
# >>> # Extract age and metallicity from a filename
# >>> age, Z = age_metal('SSP_Zsol0.001T0.00001.fits')
# >>> # Load and process a grid of templates
# >>> sps = SPSLibrary(
# >>>     pathname='templates/*.fits',
# >>>     lam='template_wavelength.fits',
# >>>     velscale=50,  # km/s per pixel
# >>>     FWHM_gal=2.5,  # Å
# >>>     FWHM_tem=1.0,  # Å
# >>>     age_range=(0.001, 13.0),  # Gyr
# >>>     metal_range=(0.0001, 0.03)
# >>> )
# >>> # Plot template light fractions with a weight matrix
# >>> mean_age, mean_Z = sps.plot(weights, output_path='plots/')
#
# Originally adapted from `miles_util.py` and `sps_util.py` in the pPXF
# package (v8.2.1 and v9.1.1).
#
# Developed by Isabel Rivera, STScI, 16 June 2024
###############################################################################

import os
import glob, re

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import ppxf.ppxf_util as util

z_sol = 0.020

def age_metal(filename):
    """
    Extract stellar population age and metallicity from a template filename.

    This function assumes that template filenames encode age and metallicity
    in a consistent pattern:
        'Zsol<M>.T<A>'
    where:
        M = metallicity in fractional solar units (Z/Z☉)
        A = age in Gyr
    Both values are returned as floats.

    This approach is compatible with any SPS library that follows this
    naming convention, regardless of the specific model source.

    Parameters
    ----------
    filename : str
        Full or relative path to a template FITS file.

    Returns
    -------
    age : float
        Stellar population age in Gyr.
    metal : float
        Metallicity in fractional solar units (e.g., 0.020 = Z☉).

    Raises
    ------
    ValueError
        If the filename does not contain the expected 'Zsol...T...' pattern.

    Example
    -------
    >>> age_metal('SSP_Zsol0.001T0.00001.fits')
    (1e-05, 0.001)
    """

    match = re.search(r'Zsol([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)T([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', filename)

    if match:
        metal = float(match.group(1))
        age = float(match.group(2))
        return age, metal
    else:
        raise ValueError(f"Filename {filename} does not contain 'Zsol...T...' pattern.")

###############################################################################

class SPSLibrary:
    """
    Class for managing a library of Single Stellar Population (SSP) spectral templates.

    This class supports:
    - Reading grids of SSP templates from FITS files
    - Extracting age and metallicity from filenames
    - Logarithmic rebinning and Gaussian convolution to match the resolution of observed spectra
    - Subsetting by age, metallicity, and wavelength range
    - Normalizing templates for spectral fitting
    - Visualizing template weights as light fractions

    Templates are expected to follow a consistent filename convention to encode
    age and metallicity:
        'Zsol<M>.T<A>'
    where:
        M = metallicity in fractional solar units (Z/Z☉)
        A = age in Gyr

    Attributes
    ----------
    templates_full : np.ndarray
        Full 3D array of SSP templates (n_wavelengths, n_ages, n_metallicities).
    ln_lam_temp_full : np.ndarray
        Logarithmically rebinned wavelength array corresponding to `templates_full`.
    lam_temp_full : np.ndarray
        Linear wavelength array corresponding to `templates_full`.
    templates : np.ndarray
        Possibly subsetted templates according to `age_range`, `metal_range`, and `wave_range`.
    ln_lam_temp : np.ndarray
        Logarithmically rebinned wavelength array for `templates`.
    lam_temp : np.ndarray
        Linear wavelength array for `templates`.
    age_grid : np.ndarray
        2D array of SSP ages corresponding to `templates`.
    metal_grid : np.ndarray
        2D array of SSP metallicities corresponding to `templates`.
    n_ages : int
        Number of ages in the (possibly subsetted) template library.
    n_metal : int
        Number of metallicities in the (possibly subsetted) template library.
    flux : np.ndarray or float
        Median flux normalization values for each template, or a single scalar if `norm_range` is None.

    Methods
    -------
    plot(weights, config_filename='config', output_path=None, std_ages=None,
         std_metallicities=None, a_v=0.0, std_A_v=None, plot=True)
        Plot the light-weight fractions of SSP templates given a weight matrix.
    """

    def __init__(self, pathname, lam, velscale, FWHM_gal=None, FWHM_tem=None,
                 age_range=None, metal_range=None, norm_range=None, wave_range=None):

        """
        Initialize the SPSLibrary by reading and processing SSP templates.

        Templates are read from FITS files matching `pathname`. Each template is
        optionally convolved to match the resolution of the observed spectrum
        and logarithmically rebinned to the desired velocity scale.

        Parameters
        ----------
        pathname : str
            File path (can include wildcards) to the SSP FITS templates.
        lam : array_like
            Wavelength array corresponding to the SSP templates in Ångströms.
        velscale : float
            Velocity scale for logarithmic rebinning of the templates.
        FWHM_gal : float or None, optional
            Full-width at half-maximum of the galaxy spectrum in Ångströms.
            Convolution of templates to match the galaxy resolution will only be applied
            if both `FWHM_gal` and `FWHM_tem` are provided. If either is None, no convolution is performed.
        FWHM_tem : float or None, optional
            Instrumental resolution of the SSP templates in Ångströms.
            Used in combination with `FWHM_gal` to compute the Gaussian kernel for convolution.
            If either `FWHM_gal` or `FWHM_tem` is None, convolution is skipped.
        age_range : tuple of float, optional
            Minimum and maximum ages to include in the library (same units as SSP ages in the FITS files).
        metal_range : tuple of float, optional
            Minimum and maximum metallicities to include in the library.
        norm_range : tuple of float, optional
            Wavelength range to normalize each template.
        wave_range : tuple of float, optional
            Wavelength range to subset the final templates.

        Raises
        ------
        AssertionError
            If no files match the pathname, or if the age-metallicity grid is incomplete.

        Notes
        -----
        - Templates are expected to form a complete Cartesian grid in age–metallicity space.
        - Normalization is applied per template if `norm_range` is given, otherwise a scalar normalization is applied.
        - Gaussian convolution to match the galaxy resolution is only performed if:
            1. Both `FWHM_gal` and `FWHM_tem` are provided, and
            2. The galaxy spectrum has a lower resolution than the templates (i.e., FWHM_gal > FWHM_tem).
          If either condition is not met, the original template resolution is preserved.
        """

        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname
        files.sort()

        all = [age_metal(f) for f in files]
        all_ages, all_metals = np.array(all).T
        ages, metals = np.unique(all_ages), np.unique(all_metals)
        n_ages, n_metal = len(ages), len(metals)

        assert set(all) == set([(a, b) for a in ages for b in metals]), \
            'Ages and Metals do not form a Cartesian grid'

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the galaxy spectrum, to determine the
        # size needed for the array which will contain the template spectra.
        hdu = fits.open(files[0])
        ssp = hdu[0].data

        lam_range_temp = fits.getdata(lam)

        cenwave_range_temp = np.zeros(len(lam_range_temp))
        cenwave_range_temp[:-1] = (lam_range_temp[1:] + lam_range_temp[:-1]) / 2
        cenwave_range_temp[-1] = lam_range_temp[-1] + ((lam_range_temp[-1] - lam_range_temp[-2]) / 2)

        ssp_new, ln_lam_temp = util.log_rebin(cenwave_range_temp, ssp, velscale=velscale)[:2]

        lam_temp = np.exp(ln_lam_temp)

        if norm_range is not None:
            band = (norm_range[0] <= lam_temp) & (lam_temp <= norm_range[1])

        templates = np.empty((ssp_new.size, n_ages, n_metal))
        age_grid, metal_grid, flux = np.empty((3, n_ages, n_metal))

        # Convolve the chosen library of spectral templates
        # with the quadratic difference between the galaxy and the
        # template resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> galaxy
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.

        if FWHM_gal is not None and FWHM_tem is not None:
            FWHM_tem = np.ones(len(cenwave_range_temp))* FWHM_tem
            FWHM_diff = (FWHM_gal**2 - FWHM_tem**2).clip(0)
            if np.any(FWHM_diff <= 0):
                print("WARNING: the template's resolution dlam is larger than the galaxy's")
            sigma = np.sqrt(FWHM_diff)/np.sqrt(4*np.log(4))


        # Here we make sure the spectra are sorted in both [M/H] and Age
        # along the two axes of the rectangular grid of templates.
        for j, age in enumerate(ages):
            for k, met in enumerate(metals):
                p = all.index((age, met))
                hdu = fits.open(files[p])
                ssp = hdu[0].data
                if FWHM_gal is not None:
                    ssp = util.varsmooth(cenwave_range_temp, ssp, sigma)

                ssp_new = util.log_rebin(cenwave_range_temp, ssp, velscale=velscale)[0]

                if norm_range is not None:
                    flux[j, k] = np.median(ssp_new[band])
                    ssp_new /= flux[j, k]   # Normalize every spectrum

                templates[:, j, k] = ssp_new
                age_grid[j, k] = age
                metal_grid[j, k] = met

        if age_range is not None:
            w = (age_range[0] <= ages) & (ages <= age_range[1])
            templates = templates[:, w, :]
            age_grid = age_grid[w, :]
            metal_grid = metal_grid[w, :]
            flux = flux[w, :]

        if metal_range is not None:
            w = (metal_range[0] <= metals) & (metals <= metal_range[1])
            templates = templates[:, :, w]
            age_grid = age_grid[:, w]
            metal_grid = metal_grid[:, w]
            flux = flux[:, w]

        if norm_range is None:
            flux = np.median(templates[templates > 0])
            templates /= flux  # Normalize by a scalar

        self.templates_full = templates
        self.ln_lam_temp_full = ln_lam_temp
        self.lam_temp_full = lam_temp
        if wave_range is not None:
            w = (wave_range[0] <= lam_temp) & (lam_temp <= wave_range[1])
            ln_lam_temp = ln_lam_temp[w]
            templates = templates[w, :, :]

        self.templates = templates
        self.ln_lam_temp = ln_lam_temp
        self.lam_temp = lam_temp
        self.age_grid = age_grid
        self.metal_grid = metal_grid
        self.n_ages, self.n_metal = age_grid.shape
        self.flux = flux


    def plot(self, weights, config_filename='config', output_path=None, std_ages=None, std_metallicities=None,
             a_v=0.0, std_A_v=None, plot=True):
        """
        Plot the light-weight fractions of SSP templates based on a weight matrix.

        Parameters
        ----------
        weights : np.ndarray
            2D array of template weights matching `age_grid` and `metal_grid`.
        config_filename : str, optional
            Filename to use for saving the plot (default 'config').
        output_path : str or None, optional
            Directory to save the plot. If None, the plot is not saved.
        std_ages : float or None, optional
            Standard deviation of ages for display.
        std_metallicities : float or None, optional
            Standard deviation of metallicities for display.
        a_v : float, optional
            Extinction value for display (default 0.0).
        std_A_v : float or None, optional
            Standard deviation of A_v for display.
        plot : bool, optional
            If True, generates the plot. If False, only computes mean age and metallicity.

        Returns
        -------
        mean_age : float
            Weighted mean age of the stellar population (in Myr).
        mean_metallicity : float
            Weighted mean metallicity in units of solar metallicity (Z☉).

        Raises
        ------
        AssertionError
            If `weights` is not 2D, does not match the grid shape, or all weights are zero.

        Notes
        -----
        - The function supports visualization of SSP weights as stacked bar charts.
        - Mean age and metallicity are weighted by the light fraction of each template.
        """

        assert weights.ndim == 2, "`weights` must be 2-dim"
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        assert np.any(weights > 0), "All weights are zero or empty."

        # Convert age grid to Myr
        xgrid = self.age_grid * 1e3
        ygrid = self.metal_grid

        # Get unique ages and metallicities
        unique_ages = np.unique(xgrid)
        unique_metals = np.unique(ygrid)

        # Calculate mean age
        mean_age = np.average(unique_ages, weights=weights.sum(axis=1))

        # Calculate mean metallicity
        mean_z = np.average(unique_metals, weights=weights.sum(axis=0))

        if plot:

            # Creating the bar chart
            fig, ax = plt.subplots()

            # Number of unique metallicities
            n_metallicities = len(unique_metals)

            # Dynamically generate colors
            colors = plt.cm.viridis(np.linspace(0, 1, n_metallicities))  # Use colormap for enough colors

            # Initialize bottom positions for the stacked bar chart
            bottoms = np.zeros(len(unique_ages))

            # Plot for each metallicity
            for i in range(n_metallicities):
                metal_mask = ygrid == unique_metals[i]  # Mask for the current metallicity
                age_for_metal = xgrid[metal_mask]  # Ages corresponding to the current metallicity
                weight_for_metal = weights[metal_mask]  # Weights for the current metallicity

                # Plot bars
                ax.bar(
                    age_for_metal,
                    weight_for_metal,
                    width=(np.max(unique_ages) - np.min(unique_ages)) / len(unique_ages) * 0.8,
                    color=colors[i],
                    bottom=bottoms,
                    label=f"{unique_metals[i]/z_sol:.2f} * Z☉"
                )
                bottoms += weight_for_metal

            # Plot mean age line
            ax.axvline(mean_age, color='k', linestyle='--', linewidth=1.5)

            # Display mean age
            if std_ages is not None:
                std_age_str = f"{std_ages:.2f}" if int(round(std_ages * 10)) % 10 == 1 else f"{std_ages:.1f}"
                mean_age_str = f"{mean_age:.2f}" if int(round(std_ages * 10)) % 10 == 1 else f"{mean_age:.1f}"
                age_text = f"<Age> = {mean_age_str} \u00B1 {std_age_str}"
            else:
                age_text = f"<Age> = {mean_age:.1f}"
            ax.text(
                mean_age+1, 0.9, age_text,
                verticalalignment='center', horizontalalignment='left', fontsize=10
            )

            # Display mean metallicity
            if std_metallicities is not None:
                std_z_str = f"{std_metallicities:.2f}" if int(round(std_metallicities * 10)) % 10 == 1 else f"{std_metallicities:.1f}"
                mean_z_str = f"{(mean_z / z_sol):.2f}" if int(round(std_metallicities * 10)) % 10 == 1 else f"{(mean_z / z_sol):.1f}"
                metallicity_text = f"<Z> = ({mean_z_str} \u00B1 {std_z_str}) * Z☉"
            else:
                metallicity_text = f"<Z> = {(mean_z / z_sol):.1f} * Z☉"
            ax.text(
                mean_age+1, 0.5, metallicity_text,
                verticalalignment='center', horizontalalignment='left', fontsize=10
            )

            # Display dust component
            if std_A_v is not None:
                std_str = f"{std_A_v:.2f}" if int(round(std_A_v * 10)) % 10 == 1 else f"{std_A_v:.1f}"
                a_v_str = f"{a_v:.2f}" if int(round(std_A_v * 10)) % 10 == 1 else f"{a_v:.1f}"
                dust_text = f"A_v = {a_v_str} \u00B1 {std_str}"
            else:
                dust_text = f"A_v = {a_v:.1f}"

            ax.text(
                mean_age+1, 0.7, dust_text,
                verticalalignment='center', horizontalalignment='left', fontsize=10
            )


            # Set title, axis labels and legend
            plt.title("Light Weights Fractions")
            ax.set_xlabel('Stellar population age (Myr)')
            ax.set_ylabel('Light fraction')
            ax.set_xlim(np.min(unique_ages) - 1, np.max(unique_ages) + 1)
            ax.set_ylim(0, 1)
            ax.legend(loc='best')
            plt.grid(alpha=0.5)


            if output_path != None:
                plt.savefig(os.path.join(output_path,f'light_weights_{config_filename}.png'), dpi=600)

        return mean_age, mean_z / z_sol
