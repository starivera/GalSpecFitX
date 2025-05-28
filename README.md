# GalSpecFitX
Full Spectrum Galaxy Fitting Software Utilizing [STARBURST99](https://www.stsci.edu/science/starburst99/docs/default.htm) and [BPASS](https://bpass.auckland.ac.nz/) stellar population models.

This software applies the Penalized PiXel-Fitting method (pPXF) created and distributed by [Cappellari (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C), which extracts the stellar or gas kinematics and stellar population from galaxy spectra via full spectrum fitting using a maximum penalized likelihood approach. The GalSpecFitX software provides several routines for preparing galaxy data to ensure the highest efficiency with the fitting routine. It also provides additional enhancements such as Milky Way absorption line masking. A full suite of STARBURST99 and BPASS libraries have been included for compatible use with the software.

## HOW TO RUN
1. Clone the GalSpecFitX repository.
2. In the GalSpecFitX root directory create a new Conda environment: ```conda env create -n galspecfitx -f environment.yml```
3. After creating and activating the environment run ```pip install .``` You should now be able to run the software by calling ```galspecfitx``` from the command line.
4. Create a directory for your galaxy data and copy over the 'config.ini' file from the GalSpecFitX root directory.
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
This software prepares a raw galaxy spectrum for spectral fitting by performing de-reddening, de-redshifting, binning, log re-binning and median normalization routines. This ensures the best compatibility with the fitting algorithm. GalSpecFitX requires a FITS file containing an Astropy table with ‘wavelength,’ ‘flux,’ and ‘error,’ columns. The spectrum must be evenly sampled; otherwise, continuum fitting may become misaligned. 

Please see the example below for instructions on how to use Python to merge multiple spectra and the spectres package to achieve even sampling. The final data are stored in an Astropy table within a single HDU extension of a FITS file. 

```
from astropy.table import Table
from astropy.io import fits
import numpy as np
from spectres import spectres

# Function to create sample spectrum
def create_sample_spectrum(lam_min, lam_max):
    wavelength = np.linspace(lam_min, lam_max, 1000)
    flux = np.random.random(1000)
    noise = np.full_like(flux, 0.05)
    return wavelength, flux, noise  

# Creating multiple spectra
spectrum1 = create_sample_spectrum(4000, 5000) 
spectrum2 = create_sample_spectrum(5000, 6000)
spectrum3 = create_sample_spectrum(6000, 7000)

# Combine multiple spectra
combined_lam = np.concatenate([spectrum1[0], spectrum2[0], spectrum3[0]])
combined_flux = np.concatenate([spectrum1[1], spectrum2[1], spectrum3[1]])
combined_noise = np.concatenate([spectrum1[2], spectrum2[2], spectrum3[2]])

# Create evenly sampled wavelength array
combined_lam_resamp = np.arange(combined_lam[0], combined_lam[-1], combined_lam[1] - combined_lam[0]) 

# Use spectres to resample the combined spectrum
combined_flux_resamp, combined_noise_resamp = spectres(combined_lam_resamp, combined_lam, combined_flux, combined_noise, fill=0)

spectrum_resamp = [combined_lam_resamp, combined_flux_resamp, combined_noise_resamp]

# Create Astropy Table
table = Table(spectrum_resamp, names=('wavelength', 'flux', 'error'))

# Convert the table to a FITS HDU 
hdu = fits.BinTableHDU(table, name='FULL_SPECTRUM')

# Create a Primary HDU
primary_hdu = fits.PrimaryHDU()

# Create an HDU list with the primary HDU and the table HDU:
hdul = fits.HDUList([primary_hdu, hdu])

# Write the HDU list to a new FITS file
hdul.writeto('resampled_spectrum.fits', overwrite=True)
```

## Configuration File Parameters

The software requires a configuration `.ini` file to run. The configuration file is divided into several sections, each of which contains specific parameters. Below is a breakdown of the required parameters for each section.

### 1. Settings Section
This section contains general settings related to the galaxy data processing.

| Parameter         | Type   | Description                                                                 |
|-------------------|--------|-----------------------------------------------------------------------------|
| `galaxy_filename` | string | Name of the galaxy spectrum FITS file.                                           |
| `segment`         | string | FITS HDU extension number corresponding to the data you want to process.    |
| `bin_width`       | int    | Width for binning the galaxy spectrum. 1 performs no binning.     |
| `default_noise`   | float  | Default noise value for the galaxy spectrum. Default is 1.                  |
| `z_guess`         | float  | Initial guess for the redshift of the galaxy. 0 performs no redshifting.                               |

### 2. Instrument Section
This section contains information about the instrument used for the observations. Used for convolution of the spectral templates and the absorption masking feature.

| Parameter         | Type   | Description                                                                 |
|-------------------|--------|-----------------------------------------------------------------------------|
| `FWHM_gal`        | float  | Spectral resolution full width at half maximum (FWHM) in Angstroms.         |
| *or the following three parameters if `FWHM_gal` is not provided:*                                       |
| `instr_lam_min`   | float  | Minimum wavelength of the instrument in microns.                            |
| `instr_lam_max`   | float  | Maximum wavelength of the instrument in microns.                            |
| `R`               | float  | Resolving power of the instrument.                                          |

### 3. Dereddening Section
This section contains parameters for removing Milky Way foreground from a galaxy spectrum.

| Parameter   | Type   | Description                                                                       |
|-------------|--------|-----------------------------------------------------------------------------------|
| `ebv`       | float  | E(B-V) reddening value to use in the extinction correction. 0 performs no reddening. |
| `model_name`| string | Name of the extinction model. Options are CCM89, 094, F99, F04, VCG04, GCC09, M14, G16, F19, D22, G23.  |
| `Rv`        | float  | Total-to-selective extinction ratio Rv (usually 3.1 for Milky Way). |


### 4. Library Section
This section defines the stellar population models used for fitting.

| Parameter   | Type   | Description                                                                       |
|-------------|--------|-----------------------------------------------------------------------------------|
| `lib_path`  | str/None | Path to library (BPASS or STARBURST99). If None or not provided only sample libraries are used.  |
| `Library`   | string | Name of the library for stellar population templates (`STARBURST99` or `BPASS`).  |
| `evol_track`| string | Evolutionary track. Only applies to Starburst99 libraries. Default is `geneva_high`. |
| `IMF`       | string | Initial mass function (IMF) used in the library (See Accessing Libraries section).    |
| `star_form` | string | Star formation model (Instantaneous or Continuous).                       |
| `star_pop`  | string | Type of stellar population (Single or Binary).                       |
| `age_range` | list of float | Age range for stellar templates (in Gyr) (e.g.[0.0, 1.0]).                    |
| `metal_range`| list of float | Metallicity range for stellar templates (e.g. [0.0, 0.020], Z_solar = 0.020).                   |
| `norm_range` | list of float | Wavelength range to be used to normalize the stellar templates and galaxy spectrum (in Å). If None provided median normalization of the entire spectrum is performed. |

### 5. Fit Section
This section contains additional parameters for customizing the fitting process. These parameters are optional and can vary depending on the user's needs. All available fitting parameters and default values are also listed in the provided `config.ini` template.

| **Parameter**         | **Type**        | **Description**                                                                                               |
|-----------------------|-----------------|---------------------------------------------------------------------------------------------------------------|
| `start`               | None/list       | Initial kinematic parameters (velocity and sigma required in km/s) for stars. Setting this to None will set V, sigma = [0.0, 3*velocity scale per pixel].|
| `absorp_lam`          | None/list/dict  | The wavelengths of known Milky Way absorption lines to be masked during spectral fitting (see Milky Way Absorption Line Masking section). |
| `bias`                | float           | Optional bias term to control fit sensitivity; default is None.*                                              |
| `bounds`              | None/list       | Parameter bounds (e.g., min and max values) for fitting constraints in start; default is None.          |
| `clean`               | bool            | Enables outlier removal if True; default is False.                                                            |
| `constr_templ`        | dict            | Constraints applied to templates during fitting; default is None.*                                             |
| `constr_kinem`        | dict            | Constraints on kinematic parameters (e.g., velocity); default is None.*                                        |
| `degree`              | int             | Degree of additive polynomial for continuum fitting; default is 4. Set ``degree=-1`` to not include any additive polynomial.|
| `dust`                | None/dict       | Dust attenuation parameters for stars; default is None. {"start":..., "bounds":..., "fixed":...}              |
| `fixed`               | None/list       | Boolean vector set to ``True`` where a given kinematic parameter has to be held fixed with the value given in ``start``. This is a list with the same dimensions as ``start``. |
| `fraction`            | float           | Ratio between stars and gas component.*                                                                        |
| `ftol`                | float           | Tolerance level for fit convergence; default is 1e-4.                                                         |
| `global_search`**     | bool or dict    | Enables global optimization of the nonlinear parameters (kinematics) before starting the usual local optimizer.if True; default is False. |
| `linear`              | bool            | Only performs linear fitting if set to True; default is False.                                                           |
| `linear_method`       | str             | Method for linear fitting (options vary based on pPXF settings); default is `lsq_box`.                        |
| `mask`                | None/list       | List of wavelength ranges to exclude from fit; default is None (e.g. [[lam_i1, lam_f1], [lami2, lamf2], ...]                                              |
| `method`              | str             | Algorithm to perform the non-linear minimization step (options vary based on pPXF settings); default is `capfit`. |
| `mdegree`             | int             | Degree of multiplicative polynomial for continuum fitting; default is 0.                                      |
| `n_iterations`        | int             | Number of iterations of the fit to perform. Calculates uncertaintaties using Monte Carlo simulations (see Rivera et al. 2025 for more detail).      |
| `quiet`               | bool            | Suppresses verbose output of the best fitting parameters at the end of the fit if True; default is False.     |
| `absorp_lam`          | None/list       | Absorption line central wavelengths for milky way line masking; default is None.                              |
| `sigma_diff`          | float           | Quadratic difference in km/s defined as: ```sigma_diff**2 = sigma_inst**2 - sigma_temp**2``` between the instrumental dispersion of the galaxy spectrum and the instrumental dispersion of the template spectra.                                                |
| `trig`                | bool            | Enables trigonometric series as an alternative to Legendre polynomials, for both the additive and multiplicative polynomials if True; default is False. |
| `vsyst`               | float           | Reference velocity in ``km/s``; default is 0.0. Output will be w.r.t. this velocity.                                                             |
| `x0`                  | None            | Initialization for linear solution; default is None.                                                                 |


---
*See pPXF documentation for further explanation on these parameters.<br>
**Computationally intensive and generally unnecessary.

## Optional: Milky Way Absorption Line Masking

GalSpecFitX allows the user to mask parts of a spectrum containing milky way absorption lines and exclude them from the fit. This is NOT line subtraction. Masking is done by fitting a single gaussian to a line at a given wavelength and approximate width. The pixels within this range will simply not be included during the fit. 
You can provide this as either:
- A list of wavelengths: default masking window of 5×(wavelength / R) will be used for each line. Example: absorp_lam = [5175.0, 5890.0, 3933.7]
- A dictionary mapping each wavelength to a custom window (in Å). Example: absorp_lam = {"5175.0": 10.0, "5890.0": 15.0}
If not provided or set to None, no absorption line masking will be applied.

## Spectral Fitting Parameters - Recommended

These are the parameters I recommend focusing on as they tend to have the greatest influence on the quality of the fit:

- **start**: Initial guess for the LOSVD parameters (V, sigma, ...) for the stars component.  
- **Degree**: Degree of the additive Legendre polynomial used to correct the template continuum shape during the fit. Set ``degree=-1`` to not include any additive polynomial. 
- **Linear**: If set to True only performs a linear least-squares routine for the templates and additive polynomial weights. Setting this to true may provide a better fit to the kinematic components (V, sigma, ...), but note that dust attenuation will not be fit when this is done. A workaround would be to set the **bounds** parameter based on the fit this provides, and then set Linear back to False.

## Accessing the Starburst99 and BPASS Libraries
After following the installation instructions you will have automatic access to the sample libraries located in the GalSpecFitX subfolder ``sample_libraries``. The sample libraries provided are the Starburst99 Geneva High evolutionary track, Salpeter IMF for instantaneous star formation models, and BPASS Salpeter IMF (`imf135all_100`) single star formation models. The provided config.ini file uses these libraries by default by setting the `lib_path` parameter to None.

#### How to access the full suite of libraries using GIT LFS

The full suite of libraries although present in the root folder of the repository under `full_suite` will not be useable when you've first cloned the repository. Git Large File Storage (LFS) must be installed to fetch and download them. The first thing you will need to do is install Git LFS. If you have Homebrew you can do this by running:

```brew install git-lfs``` or by way of one of the methods listed in the Git LFS [docs](https://git-lfs.com/).

Next, run ```git lfs install``` to initialize it.

Now you can download the models by running ```git lfs pull``` in the directory containing full_suite. **WARNING**: These libraries are LARGE, so if you would like to store the libraries somewhere with more storage space feel free to move the `full_suite` folder after downloading them.

Finally, all that needs to be done to start using the full suite is to direct the code to the directory via the `lib_path` parameter of your configuration file (e.g. lib_path=/path/to/full_suite). Now, you can explore all the models by choosing parameters based on the **All Available Libraries** section directly below.

### <u>All Available libraries</u>

A full suite of STARBURST99 and BPASS are currently available, and are provided in the root directory of the repository. The table below identifies the full criteria and keywords for selecting a set of models in your configuration file based on parameters such as evolutionary track, IMF slopes, and type of star formation.

For Starburst99:

| **Evolutionary Track (`evol_track`)**                | **Star Formation (`star_form`)** | **Initial Mass Function (`IMF`)** |
|------------------------------------------------------|----------------------------------|-----------------------------------|
| `padova` -> selection of the 1992 - 1994 Padova tracks.         | `inst` -> Instantaneous          | `salpeter`<br> `kroupa`<br>       |
| `padova_agb` -> selection of the 1992 - 1994 Padova tracks with thermally pulsing AGB stars added.          | `inst` -> Instantaneous          | `salpeter`<br> `kroupa`<br>   |
| `geneva_std` -> selection of the 1994 Geneva tracks with "standard" mass-loss rates.   | `inst` -> Instantaneous          | `salpeter`<br> `kroupa`<br>       |
| `geneva_high` -> Geneva tracks with high mass-loss rates.  | `inst` -> Instantaneous          | `salpeter`<br> `kroupa`<br>       |

For further explanation of these choices see: https://massivestars.stsci.edu/starburst99/docs/run.html#IZ

For BPASS:

| **Star Formation (`star_form`)** | **Initial Mass Function (`IMF`)** |
|----------------------------------|-----------------------------------|
| `single`<br> `binary`            | `imf_chab100`<br> `imf_chab300`<br> `imf100_100`<br> `imf100_300`<br> `imf135_100`<br>  `imf135_300`<br> `imf135all_100`<br> `imf170_100`<br> `imf170_300`<br> |

For further explanation of these choices see the BPASS [manual](https://livewarwickac.sharepoint.com/sites/Physics-BinaryPopulationandSpectralSynthesisBPASS/Shared%20Documents/Forms/AllItems.aspx?ga=1&id=%2Fsites%2FPhysics%2DBinaryPopulationandSpectralSynthesisBPASS%2FShared%20Documents%2FBPASS%5Fv2%2E2%5Frelease%2FBPASS%20v2%2E2%2E1%20full%20release%2FBPASSv2%2E2%2E1%5FManual%2Epdf&viewid=141639b8%2D0962%2D4a5a%2Db1e4%2D8977a94c88eb&parent=%2Fsites%2FPhysics%2DBinaryPopulationandSpectralSynthesisBPASS%2FShared%20Documents%2FBPASS%5Fv2%2E2%5Frelease%2FBPASS%20v2%2E2%2E1%20full%20release).

## <u>Output</u>

Following a run of the GalSpecFitX software, the following outputs are produced:

| **Filename**                         | **Format**          | **Description**                                                                                               |
|--------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------|
| `bestfit.fits`                       | FITS format         | Contains two extensions, one with the processed spectrum `PROCESSED_DATA_<segment>`, including fluxes and associated errors, and one containing the best-fit continuum `BESTFIT`.      |
| `fitted_spectrum_static`             | PNG                 | Static plot of the best fitting solution and residuals.         |
| `interactive_fitted_spectrum`        | HTML                | Interactive plot of the best fitting solution.                                            |
| `light_weights`                      | PNG                 | The fraction of each model in the best-fit continuum, where model ages are plotted on the x-axis and model metallicities are provided in different colors. The best-fit age and metallicity are given in this plot.          |
| `normalized_log_rebinned_spectrum_<segment>` | PNG | Plot of the galaxy spectrum after de-reddening, de-redshifting, binning, log-rebinning, and median normalization.             |
| `spectral_fitting.log`               | Log File            | Log containing the input configuration file parameters and best fit parameters.         |
