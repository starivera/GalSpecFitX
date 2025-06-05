###############################################################################
# This module defines the `bpass` class and supporting functions to:
#   - Load a grid of BPASS SSP spectral templates
#   - Extract age and metallicity from filenames
#   - Process and rebin spectra for use with pPXF
#   - Visualize template light-weight solutions
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
    Extract age and metallicity from a BPASS template filename.

    Assumes the filename includes a substring of the form:
    'Zp[M].T[A]', where M is the metallicity (Z) and A is the age (T),
    both in float format (Gyr for age).

    Example:
        '...Zp0.001T0.00001_inst.fits'

    Parameters
    ----------
    filename : str
        Full or relative path to a BPASS template FITS file.

    Returns
    -------
    age : float
        Stellar population age in Gyr.
    metal : float
        Metallicity in fractional solar units (e.g., 0.020 = Solar).
    """

    match = re.search(r'Zp([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)T([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', filename)

    if match:
        metal = float(match.group(1))
        age = float(match.group(2))
        return age, metal
    else:
        raise ValueError(f"Filename {filename} does not contain 'Zp...T...' pattern.")

###############################################################################

class bpass:
    """
    Constructs a grid of logarithmically rebinned SSP templates from the
    BPASS (Binary Population and Spectral Synthesis) spectral library,
    suitable for stellar population fitting with pPXF.

    Templates are read from FITS files with filenames encoding age and metallicity,
    and are assumed to span a full Cartesian grid across those parameters.

    Features:
      - Reads and rebins BPASS spectra to a given velocity scale
      - Optionally smooths templates to match galaxy resolution
      - Normalizes templates within a specified wavelength range
      - Extracts template age and metallicity from filenames
      - Provides tools to visualize light-weighted template solutions

    Parameters
    ----------
    pathname : str
        Glob path to BPASS FITS template files
        (e.g., "BPASS/instantaneous/single/imf135all_100/*Zp*T*.fits").
    velscale : float
        Desired velocity scale in km/s for logarithmic rebinning.
    lib_path : str
        Path to the wavelength array file (e.g., "BPASS/*lam.fits").
    FWHM_gal : float or ndarray, optional
        Galaxy instrumental resolution (in Angstroms). If provided,
        templates will be convolved to match it.
    FWHM_tem : float, default=1.0
        Intrinsic resolution of the BPASS templates in Angstroms.
    age_range : array_like, shape (2,), optional
        Minimum and maximum ages (in Gyr) to include in the grid.
    metal_range : array_like, shape (2,), optional
        Minimum and maximum metallicities (in Z/Z☉) to include in the grid.
    norm_range : array_like, shape (2,), optional
        Wavelength range (in Angstroms) to use for flux normalization.
    wave_range : array_like, shape (2,), optional
        Wavelength range (in Angstroms) to trim the templates.

    Attributes
    ----------
    templates : ndarray
        Reprocessed templates as a 2D array (n_pix, n_templates).
    ln_lam_temp : ndarray
        Natural log of wavelength values (Angstroms).
    lam_temp : ndarray
        Linear wavelength array in Angstroms.
    age_grid : ndarray
        Age values in Gyr, reshaped to match the template grid.
    metal_grid : ndarray
        Metallicity values (Z) in the template grid.
    flux : ndarray
        Median flux in the normalization range for each template.
    n_ages : int
        Number of unique age bins in the template grid.
    n_metal : int
        Number of unique metallicity bins in the template grid.

    References
    ----------
    - Eldridge, J.J., Stanway, E.R., Xiao, L., et al. (2017), "BPASS: Binary Population and Spectral Synthesis", PASA, 34, e058.
    - Stanway, E.R., & Eldridge, J.J. (2018), "Reevaluating old stellar populations", MNRAS, 479, 75.
    - BPASS project website: https://bpass.auckland.ac.nz
    """

    def __init__(self, pathname, velscale, lib_path, FWHM_gal=None, FWHM_tem=1.0,
                 age_range=None, metal_range=None, norm_range=None, wave_range=None):

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

        lam = glob.glob(os.path.join(lib_path, "BPASS", "*lam.fits"))[0]
        lam_range_temp = fits.getdata(lam)

        cenwave_range_temp = np.zeros(len(lam_range_temp))
        cenwave_range_temp[:-1] = (lam_range_temp[1:] + lam_range_temp[:-1]) / 2
        cenwave_range_temp[-1] = lam_range_temp[-1] + ((lam_range_temp[-1] - lam_range_temp[-2]) / 2)

        FWHM_tem = np.ones(len(cenwave_range_temp))* FWHM_tem

        ssp_new, ln_lam_temp = util.log_rebin(cenwave_range_temp, ssp, velscale=velscale)[:2]

        lam_temp = np.exp(ln_lam_temp)

        if norm_range is not None:
            band = (norm_range[0] <= lam_temp) & (lam_temp <= norm_range[1])

        templates = np.empty((ssp_new.size, n_ages, n_metal))
        age_grid, metal_grid, flux = np.empty((3, n_ages, n_metal))

        # Convolve the chosen BPASS library of spectral templates
        # with the quadratic difference between the galaxy and
        # BPASS resolution (~1 Å). Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> galaxy
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        if FWHM_gal is not None:
            FWHM_diff = (FWHM_gal**2 - FWHM_tem**2).clip(0)
            if np.any(FWHM_diff == 0):
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


    def plot(self, weights, output_path=None, std_ages=None, std_metallicities=None,
             a_v=0.0, std_A_v=None, plot=True):
        """
        Visualize light-fraction weights across the BPASS template grid.

        Creates a stacked bar plot of light fractions grouped by metallicity,
        and optionally displays best-fit mean age, metallicity, and extinction.

        Parameters
        ----------
        weights : ndarray, shape (n_ages, n_metallicities)
            Light-fraction weights for each template in the grid.
        output_path : str, optional
            Directory path to save the plot as 'light_weights.png'.
        std_ages : float, optional
            Standard deviation of the mean age (in Myr).
        std_metallicities : float, optional
            Standard deviation of the mean metallicity (in Z/Z☉).
        a_v : float, default=0.0
            V-band extinction (A_V).
        std_A_v : float, optional
            Standard deviation of A_V.
        plot : bool, default=True
            If False, disables plotting but still returns weighted means.

        Returns
        -------
        mean_age : float
            Light-weighted mean stellar age (in Myr).
        mean_z : float
            Light-weighted mean stellar metallicity in solar units (Z/Z☉).
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

        if plot==True:

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
                    label=f"{unique_metals[i]/z_sol:.2f} * Zsol"
                )
                bottoms += weight_for_metal

            # Plot mean age line
            ax.axvline(mean_age, color='k', linestyle='--', linewidth=1.5)

            # Display mean age
            if std_ages is not None:
                std_age_str = f"{std_ages:.2f}" if int(round(std_ages * 10)) % 10 == 1 else f"{std_ages:.1f}"
                age_text = f"<Age> = {mean_age:.1f} \u00B1 {std_age_str}"
            else:
                age_text = f"<Age> = {mean_age:.1f}"
            ax.text(
                mean_age+1, 0.9, age_text,
                verticalalignment='center', horizontalalignment='left', fontsize=10
            )

            # Display mean metallicity
            if std_metallicities is not None:
                std_z_str = f"{std_metallicities:.2f}" if int(round(std_metallicities * 10)) % 10 == 1 else f"{std_metallicities:.1f}"
                metallicity_text = f"<Z> = ({(mean_z / z_sol):.1f} \u00B1 {std_z_str}) * Zsol"
            else:
                metallicity_text = f"<Z> = {(mean_z / z_sol):.1f} * Zsol"
            ax.text(
                mean_age+1, 0.5, metallicity_text,
                verticalalignment='center', horizontalalignment='left', fontsize=10
            )

            # Display dust component
            if std_A_v is not None:
                std_str = f"{std_A_v:.2f}" if int(round(std_A_v * 10)) % 10 == 1 else f"{std_A_v:.1f}"
                dust_text = f"A_v = {a_v:.1f} \u00B1 {std_str}"
            else:
                dust_text = f"A_v = {a_v:.1f}"

            ax.text(
                mean_age+1, 0.7, dust_text,
                verticalalignment='center', horizontalalignment='left', fontsize=10
            )


            # Set title, axis labels and legend
            plt.title("Light Weights Fractions");
            ax.set_xlabel('Stellar population Age (Myr)')
            ax.set_ylabel('Light fraction')
            ax.set_xlim(np.min(unique_ages) - 1, np.max(unique_ages) + 1)
            ax.set_ylim(0, 1)
            ax.legend(loc='best')
            plt.grid(alpha=0.5)


            if output_path != None:
                plt.savefig(os.path.join(output_path,'light_weights.png'), dpi=150)

        return mean_age, mean_z / z_sol
