---
title: "GalSpecFitX: A Python package for full-spectrum fitting of galaxy spectra"
tags:
  - Python
  - astronomy
  - spectroscopy
  - galaxy evolution
  - stellar populations
authors:
  - name: Isabel M. Rivera
    affiliation: 1
    email: irivera@stsci.edu
    orcid: 0009-0001-6380-1476
    corresponding: true
  - name: Nimisha Kumari
    affiliation: 2
  - name: Claus Leitherer
    affiliation: 1
  - name: Jaden Sicotte
    affiliation: 3
  - name: Renske Smit
    affiliation: 4
  - name: Alessandra Aloisi
    affiliation: 1
affiliations:
  - name: Space Telescope Science Institute, Baltimore, MD, USA
    index: 1
  - name: AURA/ESA Office at STScI, Baltimore, MD, USA
    index: 2
  - name: George Washington University, Washington, DC, USA
    index: 3
  - name: Astrophysics Research Institute, Liverpool John Moores University, UK
    index: 4
date: 2025-09-12
bibliography: paper.bib
---

# Summary
`GalSpecFitX` is an open-source Python package for the full-spectrum fitting of galaxy spectra using the widely adopted Penalized Pixel-Fitting method (`pPXF`; @Cappellari2004; @Cappellari2017). The software automates the common preprocessing steps required for reliable fitting—including Galactic extinction correction, redshift correction, binning, normalization, and masking—and provides straightforward access to stellar population synthesis models such as `Starburst99` [@Leitherer1999; @Leitherer2014] and `BPASS` [@Eldridge2017].

By combining a reproducible configuration system with modular design, `GalSpecFitX` lowers the barrier to entry for researchers new to spectral fitting, while providing flexibility for advanced users. The package enables astronomers to derive key physical properties of galaxies—such as stellar ages, metallicities, dust attenuation, and star formation histories—without requiring extensive expertise in data preparation or model handling.

While originally motivated by ultraviolet spectroscopic data from the Hubble Space Telescope, the software is broadly applicable to optical and UV galaxy spectra obtained with a variety of telescopes and instruments. Although `GalSpecFitX` provides native support for `Starburst99` and `BPASS` in the UV–optical range, its modular framework enables seamless integration of other stellar population models. `GalSpecFitX` is freely available at https://github.com/starivera/GalSpecFitX under the BSD 3-Clause license.

# Statement of need
Full-spectrum fitting is a cornerstone of extragalactic astronomy. By comparing observed galaxy spectra with synthetic stellar population models, researchers can extract information about stellar content, kinematics, and chemical enrichment. Among available approaches, the `pPXF` algorithm has become widely adopted due to its robustness, flexibility, and efficiency. However, applying `pPXF` effectively requires careful preparation of input data, management of model templates, and handling of wavelength-dependent effects, all of which can present a steep learning curve for new users.

Existing tools address related needs. Packages such as `STARLIGHT` [@Cid-Fernandes2005], `FIREFLY` [@Wilkinson2017], and `Prospector` [@Johnson2021] provide powerful modeling capabilities, but are often tailored to specific wavelength ranges, datasets, or parameter inference frameworks. In contrast, `pPXF` itself is highly general but leaves preprocessing and workflow design to the user.

`GalSpecFitX` bridges this gap by providing:

- Automated preprocessing routines (extinction correction, redshift correction, binning, normalization, masking).

- Built-in support for widely used stellar population synthesis models (`Starburst99` and `BPASS`).

- A configuration-driven system for reproducible and customizable analyses.

- A modular design that allows users to extend or swap individual components.

This combination makes `GalSpecFitX` especially useful for researchers who want to leverage `pPXF’s` capabilities without needing to implement the surrounding infrastructure from scratch.

# Software Overview

`GalSpecFitX` is organized into three primary scripts and two supporting utilities for handling stellar population synthesis (SPS) libraries:

- `main.py` — Central driver script that reads the configuration file, loads the input galaxy data, and orchestrates the preprocessing and fitting steps.

- `galaxy_preprocess.py` — Handles all preprocessing operations (e.g., dereddening, deredshifting, binning, normalization, and masking).

- `galaxy_fit.py` — Retrieves the appropriate SPS templates and executes the pPXF algorithm using user-defined parameters.

- Utility scripts:
  - `starburst99_util.py` — Interfaces with the `Starburst99` library.

  - `bpass_util.py` — Interfaces with the `BPASS` library.

Figure 1 outlines the key responsibilities of each script, along with associated configuration options.

![Structure of the `GalSpecFitX` software. `main.py` initializes parameters from the configuration file before executing `galaxy_preprocess.py` and `galaxy_fit.py`. The latter handles the spectral fitting workflow using `pPXF`.](figures/structure_of_galspecfitx.png)

## Template Retrieval

`GalSpecFitX` includes access to sample spectral templates for both `Starburst99` and `BPASS`, with full libraries obtainable via Git Large File Storage (LFS). Templates are converted to FITS format with filenames encoding the IMF, metallicity, and age of each model (see the README for details).

Example filename conventions:

- `Starburst99`:

```
<evol_track>_<star_form>_<star_pop>_<IMF_slope>_<M_max>.Zsol<metal>T<age>.fits
```

- `BPASS`:

```
BPASS_2.2.1_<star_form>_<star_pop>_<IMF_slope>_<M_max>.Zsol<metal>T<age>.fits
```

where `evol_track` = evolutionary track, `star_form` = star formation law (instantaneous or continuous), `star_pop` = stellar population type (single or binary), `IMF_slope` = follows the source file conventions (e.g., for @Kroupa2001 IMF: `1.30_2.30` for `Starburst99`, `imf135_100` for `BPASS`), `M_max` = upper mass cutoff, `metal` = solar metallicity, and `age` = age in Gyr.

A flexible, modular system retrieves the appropriate templates during fitting. Dedicated handler classes—`Starburst99LibraryHandler` and `BPASSLibraryHandler`—interface with their respective utilities and are integrated into the `SpectrumProcessor` within `galaxy_fit.py`. Users can filter templates by age, metallicity, and normalization range directly via the configuration file. Both the galaxy spectrum and templates are median-normalized before fitting.

## Output and Results

`GalSpecFitX` produces a comprehensive set of outputs, including processed data, best-fit continua, diagnostic plots, and log files. Results are stored in a user-defined output directory, e.g.:

```
output/
├── bestfit.fits
├── fitted_spectrum_static.png
├── interactive_fitted_spectrum.html
├── light_weights.png
├── normalized_log_rebinned_spectrum_<hdu_ext>.html
└── spectral_fitting.log
```

Key outputs include:

- `bestfit.fits` — Contains preprocessed data and the best-fit continuum.

- `spectral_fitting.log` — Summarizes configuration settings and derived parameters (e.g., velocity dispersion, attenuation, fit quality).

- Static and interactive visualizations — `Plotly`-based figures allow for zooming, panning, and exporting, while static PNGs summarize final fits and light-weighted stellar population contributions.

# Example Usage
To demonstrate `GalSpecFitX`, we fit the ultraviolet HST/COS G130M+G160M spectrum of the starburst dwarf galaxy Pox 186. Before running the code, the raw COS segments were resampled onto a common wavelength grid (0.01 Å pixel size). Subsequent preprocessing steps were handled automatically by `GalSpecFitX` via configuration file parameters:

- **Extinction correction:** Applied Milky Way curve from [@Cardelli1989] with E(B–V) = 0.0385 [@Schlafly2011].  
- **Redshift correction:** Adopted z = 0.0040705 [@Kumari2024].  
- **Normalization:** Scaled to the median flux between feature-less spectral region 1350–1360 Å.  
- **Binning:** Applied 6-pixel binning for smoother continuum fits.  
- **Masking:** Excluded spectral regions containing strong ISM/nebular features to prevent bias in the continuum fit.  

All example scripts and configuration files used in this analysis are included in the repository under `pox186_demo/`.

The software supports fitting with multiple stellar population models. Figures 2 and 3 show continuum fits to the Pox 186 spectrum using `Starburst99` and `BPASS` models, respectively. In both cases, `GalSpecFitX` recovered young, metal-poor stellar populations consistent with previous studies [e.g., @Kumari2024; @Rogers2023].

![`GalSpecFitX` fit to the COS spectrum of Pox 186 using `Starburst99` models. The black line shows the dereddened, deredshifted spectrum; the red line shows the best-fit model. Masked regions are greyed out. Residuals (green dots) are shown below. The inset zooms into the P Cygni N V λ1240 line.](figures/pox186_starburst99.png)

![Same as above, but using `BPASS` models.](figures/pox186_bpass.png)

`GalSpecFitX` can also generate light-weighted population diagnostics, which quantify the contributions of stellar populations of different ages and metallicities to the best-fit population synthesis models (Figure 4).

![Light-weighted stellar population contributions inferred from `Starburst99` (left) and `BPASS` (right) fits to Pox 186. The x-axis shows stellar age, while colors represent metallicity. The vertical dashed line marks the average stellar age.](figures/pox186_lightweights_combined.png)

Parameter estimates from both models are in good agreement with published results [@Rogers2023]. For example, `GalSpecFitX` recovered average stellar ages of ~2–3 Myr, subsolar metallicities (Z ~0.2–0.3 Z⊙), and attenuation values of AV ~0.6–0.7 mag. The software also allows users to estimate uncertainties through an optional Monte Carlo module, which perturbs the input spectrum and re-fits multiple realizations. The number of iterations is controlled by the `n_iterations` parameter in the configuration file. These results highlight the software’s ability to extract physically meaningful parameters from ultraviolet spectra in a fully reproducible workflow.

# References
