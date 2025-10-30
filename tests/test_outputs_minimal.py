import subprocess
import sys
from pathlib import Path

import pytest
from astropy.io import fits

config_filename = 'config'

REQUIRED_OUTPUTS = [
    f"{config_filename}_preprocessed.html",
    f"bestfit_{config_filename}.fits",
    f"bestfit_{config_filename}_static.png",
    f"bestfit_{config_filename}_interactive.html",
    f"light_weights_{config_filename}.png",
    f"spectral_fitting_{config_filename}.log",
]

@pytest.mark.integration
@pytest.mark.usefixtures("ensure_cli_in_path")
def test_minimal_run_creates_outputs(tmp_run, example_spectrum_fits, minimal_config):
    # Arrange
    input_path = tmp_run
    output_path = tmp_run / "out"
    output_path.mkdir(parents=True, exist_ok=True)

    # Act: run the CLI end-to-end
    cmd = [
        "galspecfitx",
        "--input_path", str(input_path),
        "--config_file", str(minimal_config),
        "--output_path", str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # If it fails, show stdout/stderr in pytest output
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    # Assert: required files exist
    for fname in REQUIRED_OUTPUTS:
        assert (output_path / fname).exists(), f"Missing: {fname}"

    # Assert: FITS has the expected HDUs (per README)
    bestfit = output_path / f"bestfit_{config_filename}.fits"
    with fits.open(bestfit) as hdul:
        names = [h.name for h in hdul]

    assert any(n.startswith("PREPROCESSED_SPECTRUM") for n in names), names
    assert "BESTFIT" in names, names
