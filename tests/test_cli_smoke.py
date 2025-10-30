import subprocess
import sys
import shutil

import pytest

def _has_cli() -> bool:
    return shutil.which("galspecfitx") is not None

@pytest.mark.skipif(not _has_cli(), reason="CLI 'galspecfitx' not on PATH. Did you pip install . ?")
def test_cli_help_runs():
    # `galspecfitx -h` (or --help) should exit 0
    proc = subprocess.run(["galspecfitx", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    # a couple of expected strings per README
    assert "input_path" in proc.stdout.lower()
    assert "config_file" in proc.stdout.lower()
    assert "output_path" in proc.stdout.lower()
