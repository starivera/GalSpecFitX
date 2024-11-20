# GalSpecFitX
Full Spectrum Galaxy Fitting Software Utilizing [STARBURST99](https://www.stsci.edu/science/starburst99/docs/default.htm) and [BPASS](https://bpass.auckland.ac.nz/) stellar population models.

This software applies the Penalized PiXel-Fitting method (pPXF) created and distributed by [Cappellari (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C), which extracts the stellar or gas kinematics and stellar population from galaxy spectra via full spectrum fitting using a maximum penalized likelihood approach. The GalSpecFitX software provides several routines for preparing galaxy data to ensure the highest efficiency with the fitting routine. It also provides additional enhancements such as Milky Way absorption line masking. A full suite of STARBURST99 and BPASS libraries have been included for compatible use with the software.

## HOW TO RUN
1. Clone the GalSpecFitX repository.
2. Create a new Conda environment: ```conda env create -n galspecfitx```
3. Open the GalSpecFitX root directory in your terminal and run ```pip install .``` You should now be able to run the software by calling ```galspecfitx``` from the command line.
4. Create a directory for your galaxy data and copy over the 'config.ini' file from the GalSpecFitX directory.
5. You can use this configuration file as a template and adjust the parameters accordingly (see the Configuration File Parameters and Spectral Fitting Parameters - Recommended sections below).
6. Command line options:
   ```
   galspecfitx [--options]
      galspecfitx --input_path --config_file --output_path

   options
      --input_path : Input path containing galaxy data. If not provided it is assumed to be located in the current directory. Please provide galaxy filename in your configuration file.
      --config_file : Configuration filename (default: config.ini). If it is not located in input_path please include the whole path to the file. E.g. /path/to/config.ini
      --output_path : Output path for results. If not provided results will be generated in input_path.
   ```

## DATA PREPARATION
This software prepares a raw galaxy spectrum for spectral fitting by performing de-redshifting, binning, log re-binning and median normalization routines. This ensures the best compatibility with the fitting algorithm. Multiple spectra can be combined by creating an Astropy FITS table for each spectrum with columns 'wavelength', 'flux', and 'error'. Each table should then be stored in a separate hdu extension under one FITS file. Please provide a list of the extension numbers to the 'segments' parameter of your configuration file (e.g. 1,2,3,...). If multiple 'segments' are listed, the code will automatically combine the data into one combined spectrum ordered by smallest to largest wavelength before performing the fit.

Please see the example below for instructions on how to save your spectrum to an Astropy table and save it as a FITS file in Python.

```
from astropy.table import Table
from astropy.io import fits
import numpy as np

# Creating sample spectra

def create_sample_spectrum(lam_min, lam_max):
    wavelength = np.linspace(lam_min, lam_max, 1000)
    flux = np.random.random(1000)
    noise = np.full_like(flux, 0.05)
    return wavelength, flux, error

# Creating multiple spectra
spectrum1 = create_sample_data(4000, 5000)
spectrum2 = create_sample_data(5000, 6000)
spectrum3 = create_sample_data(6000, 7000)

# Create Astropy Tables
table1 = Table(spectrum1, names=('wavelength', 'flux', 'error'))
table2 = Table(spectrum2, names=('wavelength', 'flux', 'error'))
table3 = Table(spectrum3, names=('wavelength', 'flux', 'error'))

# Convert the tables to FITS HDUs and give them specific names
hdu1 = fits.BinTableHDU(table1, name='SPECTRUM1')
hdu2 = fits.BinTableHDU(table2, name='SPECTRUM2')
hdu3 = fits.BinTableHDU(table3, name='SPECTRUM3')

# Create a Primary HDU
primary_hdu = fits.PrimaryHDU()

# Create an HDU list with the primary HDU and the table HDUs:
hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])

# Write the HDU list to a new FITS file
hdul.writeto('example_multiple_spectra.fits', overwrite=True)
```

NOTE: For HST/COS data the 'segments' parameter should be a list of the detector segments contained in an x1dsum.fits file (e.g. FUVA,FUVB).

## Optional: Milky Way Absorption line masking

GalSpecFitX allows the user to mask parts of a spectrum containing milky way absorption lines and exclude them from the fit. This is NOT line subtraction. The original flux density of the spectrum remains preserved. Masking is done by fitting a single gaussian to a line at a given wavelength and finding its approximate width. The pixels within this range will simply not be included in the fit.

## Configuration File Parameters

The software requires a configuration `.ini` file to run. The configuration file is divided into several sections, each of which contains specific parameters. Below is a breakdown of the required parameters for each section.

### 1. Settings Section
This section contains general settings related to the galaxy data processing.

| Parameter         | Type   | Description                                                                 |
|-------------------|--------|-----------------------------------------------------------------------------|
| `galaxy_filename` | string | Name of the galaxy spectrum file.                                           |
| `use_hst_cos`     | bool   | Whether to use HST/COS data (`True` or `False`).                            |
| `segments`        | string | Comma-separated list of segments to process.                                |
| `bin_width`       | int    | Width for binning the galaxy spectrum.                                      |
| `default_noise`   | float  | Default noise value for the galaxy spectrum.                                |
| `z_guess`         | float  | Initial guess for the redshift of the galaxy.                               |

### 2. Instrument Section
This section contains information about the instrument used for the observations.

| Parameter         | Type   | Description                                                                 |
|-------------------|--------|-----------------------------------------------------------------------------|
| `FWHM_gal`        | float  | Full width at half maximum (FWHM) for the galaxy spectrum (in Angstroms).   |
| *or the following three parameters if `FWHM_gal` is not provided:*                                       |
| `instr_lam_min`   | float  | Minimum wavelength of the instrument (in microns).                          |
| `instr_lam_max`   | float  | Maximum wavelength of the instrument (in microns).                           |
| `R`               | float  | Resolving power of the instrument.                                          |

### 3. Library Section
This section defines the stellar population models used for fitting.

| Parameter   | Type   | Description                                                                       |
|-------------|--------|-----------------------------------------------------------------------------------|
| `lib_path`  | str/None | Path to library (BPASS or STARBURST99). If None or not provided only sample libraries are used.  |
| `Library`   | string | Name of the library for stellar population templates (`STARBURST99` or `BPASS`).  |
| `IMF`       | string | Initial mass function (IMF) used in the library (See Choosing an IMF section).    |
| `star_form` | string | Star formation model (See Choosing Star Formation section).                       |
| `age_min`   | float  | (Optional) Minimum stellar population age for fitting (in Gyr).                   |
| `age_max`   | float  | (Optional) Maximum stellar population age for fitting (in Gyr).                   |

### 4. Fit Section
This section contains additional parameters for customizing the fitting process. These parameters are optional and can vary depending on the user's needs. All available fitting parameters and default values are also listed in the provided `config.ini` template.

| **Parameter**         | **Type**        | **Description**                                                                                               |
|-----------------------|-----------------|---------------------------------------------------------------------------------------------------------------|
| `start_stars`         | None/list       | Initial kinematic parameters (velocity and sigma required) for stars; defaults are used if set to None.       |
| `start_gas`           | None/list       | Initial kinematic parameters (velocity and sigma required) for gas; defaults are used if set to None.         |
| `bias`                | float           | Optional bias term to control fit sensitivity; default is None.*                                              |
| `bounds_stars`        | None/list       | Parameter bounds (e.g., min and max values) for fitting constraints in start_stars; default is None.          |
| `bounds_gas`          | None/list       | Parameter bounds (e.g., min and max values) for fitting constraints in start_gas; default is None.            |
| `clean`               | bool            | Enables outlier removal if True; default is False.                                                            |
| `constr_templ`        | dict            | Constraints applied to templates during fitting; default is None.*                                             |
| `constr_kinem`        | dict            | Constraints on kinematic parameters (e.g., velocity); default is None.*                                        |
| `degree`              | int             | Degree of additive polynomial for continuum fitting; default is 4. Set ``degree=-1`` to not include any additive polynomial.|
| `dust_stars`          | None/dict       | Dust attenuation parameters for stars; default is None. {"start":..., "bounds":..., "fixed":...}              |
| `dust_gas`            | None/dict       | Dust attenuation parameters for gas; default is None. {"start":..., "bounds":..., "fixed":...}                |
| `fixed_stars`         | None/list       | Boolean vector set to ``True`` where a given kinematic parameter has to be held fixed with the value given in ``start_stars``. This is a list with the same dimensions as ``start_stars``. |
| `fixed_gas`           | None/list       | Boolean vector set to ``True`` where a given kinematic parameter has to be held fixed with the value given in ``start_gas``. This is a list with the same dimensions as ``start_gas``. |
| `fraction`            | float           | Ratio between stars and gas component.*                                                                        |
| `ftol`                | float           | Tolerance level for fit convergence; default is 1e-4.                                                         |
| `global_search`**     | bool or dict    | Enables global optimization of the nonlinear parameters (kinematics) before starting the usual local optimizer.if True; default is False. |
| `linear`              | bool            | Uses linear fitting only if True; default is False.                                                           |
| `linear_method`       | str             | Method for linear fitting (options vary based on pPXF settings); default is `lsq_box`.                        |
| `mask`                | None/list       | List of wavelength ranges to exclude from fit; default is None (e.g. [[lam_i1, lam_f1], [lami2, lamf2], ...]                                              |
| `method`              | str             | Algorithm to perform the non-linear minimization step (options vary based on pPXF settings); default is `capfit`. |
| `mdegree`             | int             | Degree of multiplicative polynomial for continuum fitting; default is 0.                                      |
| `quiet`               | bool            | Suppresses verbose output of the best fitting parameters at the end of the fit if True; default is False.     |
| `rest_wavelengths`    | None/list       | Absorption line central wavelengths for milky way line masking; default is None.                              |
| `sigma_diff`          | float           | Quadratic difference in km/s defined as: ```sigma_diff**2 = sigma_inst**2 - sigma_temp**2``` between the instrumental dispersion of the galaxy spectrum and the instrumental dispersion of the template spectra.                                                |
| `trig`                | bool            | Enables trigonometric series as an alternative to Legendre polynomials, for both the additive and multiplicative polynomials if True; default is False. |
| `vsyst`               | float           | Reference velocity in ``km/s``; default is 0.0. Output will be w.r.t. this velocity.                                                             |
| `x0`                  | None            | Initialization for linear solution; default is None.                                                                 |


---
*See pPXF documentation for further explanation on these parameters.<br>
**Computationally intensive and generally unnecessary.

## Spectral Fitting Parameters - Recommended

These are the parameters I recommend focusing on as they tend to have the greatest influence on the quality of the fit:

- **start_stars**: Initial guess for the parameters (V, sigma) for the stars component.  
- **start_gas**: Initial guess for the parameters (V, sigma) for the gas component.  
- **Degree**: Degree of the additive Legendre polynomial used to correct the template continuum shape during the fit.  
- **Linear**: Keeps all nonlinear parameters fixed and only performs a linear least-squares routine for the templates and additive polynomial weights.

## How to Access Full Suite of Libraries (STARBURST99 AND BPASS) using GIT LFS
After following the installation instructions you will have automatic access to the sample libraries located in the GalSpecFitX subfolder ``sample_libraries``. The sample libraries provided are the Starburst99 Geneva High evolutionary track, Salpeter IMF for instantaneous star formation models (see ), and BPASS Salpeter IMF single star formation models. The provided config.ini file uses these libraries by default by setting lib_path to None.

### <u>All Available libraries</u>

A full suite of STARBURST99 and BPASS are currently available, and are provided in the root directory of the repository. The table below identifies the full criteria and keywords for selecting a library a set of models in your configuration file based on parameters such as evolutionary track, IMF slopes, and type of star formation.

For Starburst99:

| **Evolutionary Track**                               | **Star Formation**       | **Initial Mass Function (IMF) Slopes** |
|------------------------------------------------------|--------------------------|----------------------------------------|
| `geneva_high`<br> Geneva tracks with high mass loss  | `inst`<br> Instantaneous | `salpeter`<br> `kroupa`<br>            |


