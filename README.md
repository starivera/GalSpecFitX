# GalSpecFitX
Full Spectrum Galaxy Fitting Software Utilizing [Starburst99](https://www.stsci.edu/science/starburst99/docs/default.htm) and [BPASS](https://bpass.auckland.ac.nz/) stellar population models.

This software applies the Penalized Pixel-Fitting method (pPXF) developed by [Cappellari (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C), which derives stellar population properties from galaxy spectra through full-spectrum fitting using a maximum penalized likelihood approach. GalSpecFitX performs a suite of preprocessing routines—such as Galactic dereddening, deredshifting, binning, and normalization—incorporates enhanced masking capabilities, and enables seamless integration with established stellar population synthesis libraries.

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
This software prepares a raw galaxy spectrum for spectral fitting by performing optional dereddening, deredshifting, and binning, as well as log-rebinning and median normalization—both required for use with GalSpecFitX—to ensure optimal compatibility with the fitting routine. GalSpecFitX requires a FITS file containing an Astropy table with ‘wavelength,’ ‘flux,’ and ‘error,’ columns. The galaxy spectrum must be evenly sampled; otherwise, the continuum fit may misalign with the observed spectrum.

Please see the example below for instructions on how to use Python to merge multiple spectra and the `spectres` package to achieve even sampling. The final data are stored in an Astropy table within a single HDU extension of a FITS file.

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
spectrum1 = create_sample_spectrum(1000, 2000)
spectrum2 = create_sample_spectrum(2100, 3200)
spectrum3 = create_sample_spectrum(3500, 4000)

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

The software requires a configuration `.ini` file to run. The configuration file is divided into several sections, each of which contains specific parameters. Below is a breakdown of the required parameters for each section. The repository includes a template `config.ini` file with all parameters set to their default values.

### 1. Settings Section
This section contains general settings related to the galaxy data processing.

| Parameter         | Type   | Description                                                                 |
|-------------------|--------|-----------------------------------------------------------------------------|
| `galaxy_filename` | string | Name of the galaxy spectrum FITS file (e.g. galaxy_spectrum.fits)                                       |
| `segment`         | string | FITS HDU extension name corresponding to the data you want to process (e.g 'FULL_SPECTRUM').    |
| `bin_width`       | int    | Width for binning the galaxy spectrum. 1 performs no binning.     |
| `default_noise`   | float  | Default noise value for the galaxy spectrum. Default is 1.                  |
| `z_guess`         | float  | Initial guess for the redshift of the galaxy. 0 performs no redshifting.                               |

### 2. Instrument Section
This section contains information about the instrument used for the observations. Used for convolution of the spectral templates and the absorption masking feature.

| Parameter         | Type   | Description                                                                 |
|-------------------|--------|-----------------------------------------------------------------------------|
| `FWHM_gal`        | float  | Spectral resolution full width at half maximum (FWHM) in Angstroms.         |
| *or the following three parameters if `FWHM_gal` is not provided:*                                       |
| `instr_lam_min`   | float  | Minimum wavelength of the instrument in Angstroms.                            |
| `instr_lam_max`   | float  | Maximum wavelength of the instrument in Angstroms.                            |
| `R`               | float  | Resolving power of the instrument.                                           |

### 3. Dereddening Section
This section contains parameters for removing Milky Way foreground. Dereddening is performed using Python's [dust_extinction](https://dust-extinction.readthedocs.io/en/latest/) package.

| Parameter   | Type   | Description                                                                       |
|-------------|--------|-----------------------------------------------------------------------------------|
| `ebv`       | float  | E(B-V) reddening value to use in the extinction correction. 0 performs no reddening. |
| `model_name`| string | Name of the extinction model. Options are CCM89, 094, F99, F04, VCG04, GCC09, M14, G16, F19, D22, G23.  |
| `Rv`        | float  | Total-to-selective extinction ratio Rv (usually 3.1 for Milky Way). |


### 4. Library Section
This section allows the user to select and refine the stellar population models used for fitting.

| Parameter   | Type   | Description                                                                       |
|-------------|--------|-----------------------------------------------------------------------------------|
| `lib_path`  | str/None | Path to library (e.g. `lib_path=/path/to/full_suite`). `None` defaults to sample_libraries automatically provided.  |
| `Library`   | string | Name of the library for stellar population templates (`STARBURST99` or `BPASS`).  |
| `evol_track`| string | Evolutionary track. Only applies to Starburst99 libraries. Default is `geneva_high`. |
| `IMF`       | string | Initial mass function (IMF) (See **All Available libraries**).    |
| `upper_mass`| int | Upper limit on the stellar mass distribution (100, or 300).    |
| `star_form` | string | Star formation model (instantaneous or continuous). Only instantaneous models are available at this time.                       |
| `star_pop`  | string | Type of stellar population (single or binary).                       |
| `age_range` | list of float | Age range for stellar templates (in Gyr) (e.g.[0.0, 1.0]).                    |
| `metal_range`| list of float | Metallicity range for stellar templates (e.g. [0.0, 0.020], Z☉ = 0.020).                   |
| `norm_range` | list of float | Wavelength range to be used to normalize the stellar templates and galaxy spectrum (in Å). If None provided median normalization of the entire spectrum is performed. |

### 5. Fit Section
This section lists the fit parameters that must be specified and may vary depending on the user's needs. All available fitting parameters and their default values are also provided in the `config.ini` template.

| **Parameter**         | **Type**        | **Description**                                                                                               |
|-----------------------|-----------------|---------------------------------------------------------------------------------------------------------------|
| `start`               | None/list       | LOSVD kinematic components (velocity and sigma required in km/s) for stars. Setting this to None will set V, sigma = [0.0, 3*velocity scale per pixel].|
| `absorp_lam`          | None/list/dict  | The wavelengths of known Milky Way absorption lines to be masked during spectral fitting (see Milky Way Absorption Line Masking section). |
| `bias`                | float           | Optional bias term to control fit sensitivity; default is None.*                                              |
| `bounds`              | None/list       | Parameter bounds (e.g., min and max values) for start; default is None. E.g. [[0.0, 100.], [0.0, 40.]] sets min and max bounds for V[0.0, 100.] and sigma[0.0, 40.] during fit. |
| `clean`               | bool            | Enables outlier removal if True; default is False.                                                            |
| `constr_kinem`        | dict            | Linear constraints on the kinematic parameters; default is None.*                                        |
| `constr_templ`        | dict            | Linear constraints on the template weights; default is None.*                                             |
| `degree`              | int             | Degree of additive polynomial used to correct the template continuum shape during the fit; default is 4. Set ``degree=-1`` to not include any additive polynomial.|
| `dust`                | None/dict       | Parameters for the attenuation curve to be applied for stars; default is None. {"start":..., "bounds":..., "fixed":...}              |
| `fixed`               | None/list       | Boolean vector set to ``True`` where a given kinematic parameter has to be held fixed with the value given in ``start``. This is a list with the same dimensions as ``start``. |
| `fraction`            | float           | Ratio between stars and gas component.*                                                                        |
| `ftol`                | float           | Tolerance level for fit convergence; default is 1e-4.                                                         |
| `global_search`**     | bool or dict    | Enables global optimization of the nonlinear parameters (kinematics) before starting the usual local optimizer if True; default is False. |
| `linear`              | bool            | Only performs a linear fit for the templates and additive polynomials weights if set to True; default is False.                                           |
| `linear_method`       | str             | Method for linear fitting. Options are `nnls`, `lsq_box`, `lsq_lin`, `cvxopt`; default is `lsq_box`.                        |
| `mask`                | None/list       | List of wavelength ranges to exclude from fit; default is None (e.g. [[lam_i1, lam_f1], [lami2, lamf2], ...]                                              |
| `method`              | str             | Algorithm to perform the non-linear minimization step. Options are `capfit`, `trf`, `dogbox`, `lm`; default is `capfit`. |
| `mdegree`             | int             | Degree of multiplicative polynomial for continuum fitting; default is 0.                                      |
| `n_iterations`        | int             | Number of iterations of the fit to perform. Calculates uncertaintaties through Monte Carlo perturbations of the observed spectrum.      |
| `quiet`               | bool            | Suppresses verbose output of the best fitting parameters at the end of the fit if True; default is False.     |
| `absorp_lam`          | None/list       | Absorption line central wavelengths for milky way line masking; default is None. Resolving power `R` must be provided in the `Instrument` section.        |
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

These are the parameters I recommend focusing on as they tend to have the greatest influence on the the fit:

- **start**: Initial guess for the LOSVD parameters (V, sigma, ...) for the stars component.  
- **Degree**: Degree of the additive Legendre polynomial used to correct the template continuum shape during the fit. Set ``degree=-1`` to not include any additive polynomial.
- **Linear**: If set to True only performs a linear least-squares routine for the templates and additive polynomial weights. Setting this to true may provide a better fit to the kinematic components (V, sigma, ...), but note that dust attenuation will not be fit when this is done. A workaround would be to set the **bounds** parameter based on the fit this provides, and then set Linear back to False.

## Accessing the Starburst99 and BPASS Libraries

**Basic access:**

After following the installation instructions, you will automatically have access to the sample libraries included in the repository under the folder ``GalSpecFitX/sample_libraries``. These sample libraries include:

- **Starburst99**: Geneva High evolutionary track, Salpeter and Kroupa IMFs for instantaneous, single-star formation models

The provided `config.ini` file uses these sample libraries by default by setting the ``lib_path`` parameter to ``None``.

#### Accessing the Full Suite of Libraries (Using Git LFS)

The full suite of libraries for Starburst99 and BPASS is stored in the ``full_suite`` folder at the root of the repository. Because these files are large (~10 GB is the current total size of the full suite library), they are managed with Git Large File Storage (Git LFS). If you have Git LFS installed prior to cloning the repository and want to avoid automatically downloading the data during cloning, you can use the command `GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/starivera/GalSpecFitX.git`. If you don't have Git LFS installed beforehand or if you use `GIT_LFS_SKIP_SMUDGE=1`, the data in the `full_suite/` directory within your local clone will be replaced with small pointer files until you explicitly fetch the full data using Git LFS.

**To download and use the full suite after cloning the repo follow these steps:**

**1. Install Git LFS**

- If you have [Homebrew](https://brew.sh/), run:
```
brew install git-lfs
```
- Alternatively, follow the installation instructions for your platform on the [Git LFS website](https://git-lfs.com/).

**2. Initialize Git LFS**

Run this command once to set up Git LFS on your machine:
```
git lfs install
```

**3. Download the full libraries**

Navigate to the root of your cloned repository (where the ``full_suite`` folder is located) and run:
 ```
 git lfs pull
```
This will download all the large model files. You can also choose specific files and folders to download by adding the `--include` or `--exclude` tag.

***Note:** The full suite of libraries is large (~10 GB). Make sure you have enough free disk space where you have your local clone.

**4. (Optional) Move the ``full_suite`` folder**

If you want to store the libraries in a different location with more disk space, you can move the entire ``full_suite`` folder. You must have run `git lfs pull` prior to doing this, so you must still ensure you have enough disk space where the clone is originally located. Just remember to update your ``config.ini`` accordingly (see step 5).

**5. Update your configuration to use the full suite**

Open your ``config.ini`` file and set the ``lib_path`` parameter to the path where your ``full_suite`` folder is located. For example:
```
lib_path=/path/to/full_suite
```
This tells the program to load models from the full library instead of the default sample set.

##### Summary

- Use the sample libraries out of the box without extra setup.

- To use the full library set, install Git LFS, download the files with ``git lfs pull``, and update ``lib_path`` in your config.

- The full library files are large (~10 GB), so plan your disk space accordingly.

### <u>All Available libraries</u>

A full suite of Starburst99 and BPASS are currently available, and are provided in the root directory of the repository. The table below identifies the full criteria and keywords for selecting a set of models in your configuration file based on parameters such as evolutionary track, IMF slopes, upper mass cut-off, and type of star formation.

For Starburst99:

| **Evolutionary Track (`evol_track`)**                | **Star Formation (`star_form`)** | **Initial Mass Function (`IMF`)** | **Upper Mass Limit (`upper_mass`)** |
|------------------------------------------------------|----------------------------------|-----------------------------------|-----------------------------------|
| `padova_std` -> selection of the 1992 - 1994 Padova tracks.         | `inst` -> Instantaneous          | `salpeter`<br> `kroupa`<br>       | 100 |
| `padova_agb` -> selection of the 1992 - 1994 Padova tracks with thermally pulsing AGB stars added.          | `inst` -> Instantaneous          | `salpeter`<br> `kroupa`<br>   | 100 |
| `geneva_std` -> selection of the 1994 Geneva tracks with "standard" mass-loss rates.   | `inst` -> Instantaneous          | `salpeter`<br> `kroupa`<br>       | 100 |
| `geneva_high` -> Geneva tracks with high mass-loss rates.  | `inst` -> Instantaneous          | `salpeter`<br> `kroupa`<br>       | 100 |
For further explanation of these choices see: https://massivestars.stsci.edu/starburst99/docs/run.html#IZ

For BPASS:

| **Star Formation (`star_form`)** | **Initial Mass Function (`IMF`)** | **Upper Mass Limit (`upper_mass`)** |
|----------------------------------|-----------------------------------|-----------------------------------|
| `single`<br> `binary`            | `100`<br> `135`<br> `135all`<br> `170`<br> `chab`<br>  | 100<br> 300 <br> |
For further explanation of these models see page 7 of the BPASSv2.2.1 [manual](https://warwick.ac.uk/fac/sci/physics/research/astro/research/catalogues/bpass/v2p2/bpassv2.2_manual-arial.pdf).

## <u>Output</u>

Following a run of the GalSpecFitX software, the following outputs are produced:

| **Filename**                         | **Format**          | **Description**                                                                                               |
|--------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------|
| `bestfit_<config_filename>.fits`                       | FITS format         | Contains two extensions, one with the preprocessed spectrum `PREPROCESSED_SPECTRUM`, containing wavelengths, fluxes, and errors, and one containing the best-fit continuum `BESTFIT`.      |
| `bestfit_<config_filename>_static.png`             | PNG                 | Static plot of the best fitting solution and residuals.         |
| `bestfit_<config_filename>_interactive.html`        | HTML                | Interactive plot of the best fitting solution.                                            |
| `light_weights_<config_filename>.png`                      | PNG                 | The fraction of each model in the best-fit continuum, where model ages are plotted on the x-axis and model metallicities are provided in different colors. The best-fit age and metallicity are given in this plot.          |
| `<config_filename>_preprocessed.png` | PNG | Plot of the galaxy spectrum after de-reddening, de-redshifting, binning, log-rebinning, and median normalization.             |
| `spectral_fitting_<config_filename>.log`               | Log File            | Log containing the input configuration file parameters and best fit parameters.         |

## License
This software is licensed under the BSD 3-Clause License (see the LICENSE file for details).
