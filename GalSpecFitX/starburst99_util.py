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
    s = re.findall(r'Z[m|p][0-9]\.[0-9]{3}T[0-9]\.[0-9]{5}', filename)[0]
    metal = s[:7]
    # print(metal)
    age = float(s[8:])
    if "Zm" in metal:
        metal = -float(metal[2:])
    elif "Zp" in metal:
        metal = float(metal[2:])

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
        (e.g. ``starburst_lib/stellar_templates/GENEVA_high_1.30_2.30.Zp0.001T0.00001_inst.fits``).
        The files must form a Cartesian grid in age and metallicity and the procedure returns an error if
        they do not.
    velscale:
        desired velocity scale for the output templates library in km/s
        (e.g. 60). This is generally the same or an integer fraction of the
        ``velscale`` of the galaxy spectrum used as input to ``ppxf``.
    FWHM_gal:
        scalar or vector with the FWHM of the instrumental resolution of the
        galaxy spectrum in Angstrom at every pixel of the stellar templates.

        - If ``FWHM_gal=None`` (default), no convolution is performed.

    Optional Keywords
    -----------------

    age_range: array_like with shape (2,)
        ``[age_min, age_max]`` optional age range (inclusive) in Gyr for the
        STARBURST99 models. This can be useful e.g. to limit the age of the templates
        to be younger than the age of the Universe at a given redshift.
    metal_range: array_like with shape (2,)
        ``[metal_min, metal_max]`` optional metallicity [M/H] range (inclusive)
        for the STARBURST99 models (e.g.`` metal_range = [0, np.inf]`` to select only
        the spectra with Solar metallicity and above).
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

    def __init__(self, pathname, velscale, FWHM_gal=None, FWHM_tem=0.4,
                 age_range=None, metal_range=None, wave_range=None):

        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname

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

        # Get the directory where main.py is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lam_range_temp = fits.getdata(os.path.join(script_dir, "starburst_lib/stellar_templates/GENEVA_high_lam.fits"))

        cenwave_range_temp = np.zeros(len(lam_range_temp))

        cenwave_range_temp[:-1] = (lam_range_temp[1:] + lam_range_temp[:-1]) / 2

        cenwave_range_temp[-1] = lam_range_temp[-1] + ((lam_range_temp[-1] - lam_range_temp[-2]) / 2)
        print('original wavelength array', lam_range_temp)
        print('cenwaves', cenwave_range_temp)

        ssp_new, ln_lam_temp = util.log_rebin(cenwave_range_temp, ssp, velscale=velscale)[:2]

        lam_temp = np.exp(ln_lam_temp)

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
            fwhm_diff = (fwhm_gal**2 - fwhm_tem**2).clip(0)
            if np.any(fwhm_diff == 0):
                logging.info("WARNING: the template's resolution dlam is larger than the galaxy's")
            sigma = np.sqrt(fwhm_diff)/np.sqrt(4*np.log(4))

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
                    if np.isscalar(FWHM_gal):
                        if sigma > 0.1:   # Skip convolution for nearly zero sigma
                            ssp = ndimage.gaussian_filter1d(ssp, sigma)
                    else:
                        ssp = util.gaussian_filter1d(ssp, sigma)  # convolution with variable sigma
                # print(files[p], cenwave_range_temp.shape, ssp.shape)
                ssp_new = util.log_rebin(cenwave_range_temp, ssp, velscale=velscale)[0]

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

    def plot(self, weights, nodots=False, colorbar=True, **kwargs):
        assert weights.ndim == 2, "`weights` must be 2-dim"
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        # Convert age grid to Myr
        xgrid = self.age_grid * 1e3
        ygrid = self.metal_grid
        print(xgrid, ygrid, weights)

        # Creating the bar chart
        fig, ax = plt.subplots()

        # Get unique ages and metallicities
        unique_ages = np.unique(xgrid)
        unique_metals = np.unique(ygrid)

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

            print(metal_mask, age_for_metal, weight_for_metal)

            ax.bar(age_for_metal, weight_for_metal, width=1.0, color=colors[i], bottom=bottoms, label=f"{unique_metals[i]/z_sol} * Zsol")
            bottoms += weight_for_metal

        # Calculate and plot mean age line
        print("Shapes", unique_ages.shape, weights.sum(axis=1))
        mean_age = np.average(unique_ages, weights=weights.sum(axis=1))
        ax.axvline(mean_age, color='k', linestyle='--', linewidth=1.5)
        ax.text(mean_age + 1, 0.9, f'<Age> = {mean_age:.2f}', verticalalignment='center', horizontalalignment='right', fontsize=10)

        # Calculate mean metallicity and reddening text
        mean_z = np.average(unique_metals, weights=weights.sum(axis=0))
        ax.text(4, 0.9, f'<Z> = {(mean_z/z_sol):.2f} * Zsol', verticalalignment='center', horizontalalignment='left', fontsize=10)

        # Set axis labels and legend
        ax.set_xlabel('Stellar population Age (Myr)')
        ax.set_ylabel('Light fraction')
        ax.set_xlim(0, np.max(xgrid) + 5)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')

        # Show plot
        plt.show()
