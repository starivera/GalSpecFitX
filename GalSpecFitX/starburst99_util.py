###############################################################################
# This file contains the 'starburst' class with functions to construct
# a library of STARBURST99 templates and interpret and display the output
# of pPXF when using those templates as input.

import os
import glob, re

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from astropy.io import fits

import ppxf.ppxf_util as util

z_sol = 0.020

def age_metal(filename):
    """
    Extract the age and metallicity from the name of a file of
    the STARBURST99 library of Single Stellar Population models as
    downloaded from https://www.stsci.edu/science/starburst99/docs/default.htm as of 2024

    This function relies on the template file containing a substring of the
    precise form like Zp0.001T0.00001, specifying the metallicity and age.

    :param filename: string possibly including full path
        (e.g. 'starburst_lib/stellar_templates/GENEVA_high_1.30_2.30.Zp0.001T0.00001_inst.fits')
    :return: age (Gyr), [M/H]

    """
    match = re.search(r'Zp([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)T([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', filename)
    if match:
        metal = float(match.group(1))
        age = float(match.group(2))

    # print(f'age, metal: {age}, {metal}')

    return age, metal


###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Adapted from miles_util.py and sps_util.py provided in the pPXF
#       package version 8.2.1 and 9.1.1 respectively.
#     - Written by Isabel Rivera, STScI, 16 June 2024 for specific use of the
#       STARBURST99 templates.

class starburst:
    """
    This code produces an array of logarithmically-binned templates by reading
    the spectra from the Single Stellar Population (SSP) STARBURST99
    library by Claus Leitherer, Daniel Schaerer, Jeff Goldader, Rosa Gonzalez-Delgado,
    Carmelle Robert, Denis Foo Kune, Duilia de Mello, Daniel Devost, Timothy M. Heckman,
    Alessandra Aloisi, Lucimara Martins, and Gerardo Vazquez.
    A description of the input physics is in Leitherer et al. (1999; ApJS, 123, 3),
    Vazquez & Leitherer (2005; ApJ, 621, 695) and Leitherer et al. (2010; ApJS, 189,309)
    and Leitherer et al. (2014).

    The code checks that the model spectra form a rectangular grid
    in age and metallicity and properly sorts them in both parameters.
    The code also returns the age and metallicity of each template
    by reading these parameters directly from the file names.
    The templates are broadened by a Gaussian with dispersion
    ``sigma_diff = np.sqrt(sigma_gal**2 - sigma_tem**2)``.

    This script is designed to use the files naming convention adopted by
    the MILES library, where SSP spectra file names have the form like below::

        ...Z[Metallicity]T[Age]...fits
        e.g. Eun1.30Zm0.40T00.0631_iPp0.00_baseFe_linear_FWHM_variable.fits

    Input Parameters
    ----------------

    pathname:
        path with wildcards returning the list of files to use
        (e.g. ``sample_libraries/STARBURST99/geneva_high/instantaneous/salpeter/*.fits``).
        The files must form a Cartesian grid in age and metallicity and the procedure returns an error if
        they do not.
    velscale:
        desired velocity scale for the output templates library in km/s
        (e.g. 60). This is generally the same or an integer fraction of the
        ``velscale`` of the galaxy spectrum used as input to ``ppxf``.
    FWHM_gal:
        scalar with the FWHM of the instrumental resolution of the
        galaxy spectrum in Angstrom.

        - If ``FWHM_gal=None`` (default), no convolution is performed.

    Optional Keywords
    -----------------

    age_range: array_like with shape (2,)
        ``[age_min, age_max]`` optional age range (inclusive) in Gyr for the
        STARBURST99 models. This can be useful e.g. to limit the age of the templates
        to be younger than the age of the Universe at a given redshift.
    metal_range: array_like with shape (2,)
        ``[metal_min, metal_max]`` optional metallicity [M/H] range (inclusive)
        for the STARBURST99 models (e.g.`` metal_range = [0.0, 0.020]`` to select only
        the spectra with up to Solar metallicity).
    norm_range: array_like with shape (2,)
        A two-elements vector specifying the wavelength range in Angstrom
        within which to compute the templates normalization
        (e.g. ``norm_range=[5070, 5950]`` for the FWHM of the V-band).
    wave_range: array_like with shape (2,)
        A two-elements vector specifying the wavelength range in Angstrom for
        which to extract the stellar templates. Restricting the wavelength
        range of the templates to the range of the galaxy data is useful to
        save some computational time. By default ``wave_range=[3541, 1e4]``

    Output Parameters
    -----------------

    Stored as attributes of the ``starburst`` class:

    .age_grid: array_like with shape (n_ages, n_metals)
        Age in Gyr of every template.
    .flux: array_like with shape (n_ages, n_metals)
        ``.flux`` contains the mean flux in each template spectrum.

        The weights returned by ``ppxf`` represent light contributed by each SSP population template.
        One can then use this ``.flux`` attribute to convert the light weights
        into fractional masses as follows::

            pp = ppxf(...)                                  # Perform the ppxf fit
            light_weights = pp.weights[~gas_component]      # Exclude gas templates weights
            light_weights = light_weights.reshape(reg_dim)  # Reshape to a 2D matrix
            mass_weights = light_weights/starburst.flux         # Divide by this attribute
            mass_weights /= mass_weights.sum()              # Normalize to sum=1

    .ln_lam_temp: array_like with shape (npixels,)
        Natural logarithm of the wavelength in Angstrom of every pixel.
    .metal_grid: array_like with shape (n_ages, n_metals)
        Metallicity [M/H] of every template.
    .n_ages:
        Number of different ages.
    .n_metal:
        Number of different metallicities.
    .templates: array_like with shape (npixels, n_ages, n_metals)
        Array with the spectral templates.

    """

    def __init__(self, pathname, velscale, lib_path, evol_track, FWHM_gal=None, FWHM_tem=0.4,
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

        lam = glob.glob(os.path.join(lib_path, "STARBURST99", evol_track, "*lam.fits"))[0]
        lam_range_temp = fits.getdata(lam)

        cenwave_range_temp = np.zeros(len(lam_range_temp))
        cenwave_range_temp[:-1] = (lam_range_temp[1:] + lam_range_temp[:-1]) / 2
        cenwave_range_temp[-1] = lam_range_temp[-1] + ((lam_range_temp[-1] - lam_range_temp[-2]) / 2)

        # cenwave_range_temp = cenwave_range_temp[[0, -1]]
        FWHM_tem = np.ones(len(cenwave_range_temp))* FWHM_tem

        ssp_new, ln_lam_temp = util.log_rebin(cenwave_range_temp, ssp, velscale=velscale)[:2]

        lam_temp = np.exp(ln_lam_temp)

        if norm_range is not None:
            band = (norm_range[0] <= lam_temp) & (lam_temp <= norm_range[1])

        templates = np.empty((ssp_new.size, n_ages, n_metal))
        age_grid, metal_grid, flux = np.empty((3, n_ages, n_metal))

        # Convolve the chosen STARBURST99 library of spectral templates
        # with the quadratic difference between the galaxy and the
        # Leitherer et al. (2010) resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> galaxy
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.

        if FWHM_gal is not None:
            FWHM_diff = (FWHM_gal**2 - FWHM_tem**2).clip(0)
            if np.any(FWHM_diff == 0):
                print("WARNING: the template's resolution dlam is larger than the galaxy's")
            sigma = np.sqrt(FWHM_diff)/np.sqrt(4*np.log(4))

            # FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
            # sigma = FWHM_dif/2.355/h2['CDELT1']   # Sigma difference in pixels


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


###############################################################################

    def plot(self, weights, output_path=None, std_ages=None, std_metallicities=None, a_v=0.0, std_A_v=None, plot=True):

        assert weights.ndim == 2, "`weights` must be 2-dim"
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        assert np.any(weights > 0), "All weights are zero or empty."

        # Convert age grid to Myr
        xgrid = self.age_grid * 1e3
        ygrid = self.metal_grid
        # print(xgrid, ygrid, weights)

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
