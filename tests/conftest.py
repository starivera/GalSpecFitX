import os
import sys
import json
import shutil
import textwrap
import numpy as np
import pytest
from pathlib import Path

from astropy.table import Table
from astropy.io import fits

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture
def tmp_run(tmp_path: Path) -> Path:
    # per-test working directory
    return tmp_path

@pytest.fixture
def example_spectrum_fits(tmp_run: Path) -> Path:
    """
    Create an evenly-sampled spectrum (fast!) in a FITS file with
    one table HDU named 'FULL_SPECTRUM' and columns wavelength/flux/error.
    """
    lam = np.linspace(1000.0, 2000.0, 1000)
    flux = np.random.random(1000)
    noise = np.full_like(flux, 0.05)

    table = Table([lam, flux, noise], names=("wavelength", "flux", "error"))
    hdu = fits.BinTableHDU(table, name="FULL_SPECTRUM")
    primary = fits.PrimaryHDU()
    hdul = fits.HDUList([primary, hdu])

    out = tmp_run / "example_spectrum.fits"
    hdul.writeto(out, overwrite=True)
    return out

@pytest.fixture
def minimal_config(tmp_run: Path, example_spectrum_fits: Path) -> Path:
    """
    Create a minimal config.ini that:
      * uses the sample libraries (lib_path=None),
      * keeps things fast (small polynomials, linear-only option), and
      * targets the 'FULL_SPECTRUM' HDU.
    """
    config_text = textwrap.dedent(f"""
    [Settings]
    galaxy_filename = {example_spectrum_fits.name}
    segment = FULL_SPECTRUM
    bin_width = 1
    default_noise = 0.05
    z_guess = 0.0

    [Instrument]
    # Keep it simple: give an FWHM (Å) directly
    FWHM_gal = 1.0

    [Dereddening]
    ebv = 0.0
    model_name = CCM89
    Rv = 3.1

    [Library]
    lib_path = None
    Library = STARBURST99
    evol_track = geneva_high
    IMF = salpeter
    star_form = instantaneous
    star_pop = single
    age_range = [0.0, 0.02]
    metal_range = [0.0, 0.02]
    # use entire range for normalization (small file anyway)
    norm_range = None

    [Fit]
    # keep it FAST and deterministic-ish for CI:
    start = None
    degree = 4
    dust = None
    mdegree = 0
    linear = True
    n_iterations = 1
    quiet = True
    """).strip()

    cfg = tmp_run / "config.ini"
    cfg.write_text(config_text)
    return cfg

@pytest.fixture
def ensure_cli_in_path(repo_root: Path, monkeypatch):
    """
    Ensure 'galspecfitx' resolves (either from an editable install or by running
    the package as a module if you expose a __main__ / CLI in setup.py).
    Suggestion: in CI, run 'pip install -e .' first.
    """
    # Nothing to do here other than ensure cwd is repo root for relative imports
    monkeypatch.chdir(repo_root)
