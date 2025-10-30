#!/usr/bin/env python

import ast
import sys
import argparse
import logging
import configparser
from pathlib import Path
import contextlib
import matplotlib.pyplot as plt
import plotly.io as pio
import numpy as np

from GalSpecFitX.galaxy_preprocess import (
    GalaxySpectrum,
    DeReddeningOperation,
    DeRedshiftOperation,
    BinningOperation,
    LogRebinningOperation,
    NormalizationOperation,
    ProcessedGalaxySpectrum,
)
from GalSpecFitX.combine_and_fit import (
    SpectrumProcessor,
    InstrumentInfo,
    Starburst99LibraryHandler,
    BPASSLibraryHandler,
)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def read_config(filename: str, input_path: Path) -> configparser.ConfigParser:
    """Read and validate configuration file."""
    config_file = input_path / filename
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file {config_file} not found.")

    config = configparser.ConfigParser()
    config.read(config_file)

    required_sections = ["Settings", "Instrument", "Library", "Fit"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise KeyError(f"Missing required sections: {', '.join(missing)}")

    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Galaxy spectral fitting pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_path", "-i", type=str, default=".",
        help="Input directory containing galaxy data and config file."
    )
    parser.add_argument(
        "--config_file", "-c", type=str, default="config.ini",
        help="Configuration filename (expected in input_path)."
    )
    parser.add_argument(
        "--output_path", "-o", type=str, default=None,
        help="Output directory for results."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging."
    )

    return parser.parse_args()


def setup_logging(log_path: Path, verbose: bool = False) -> None:
    """Set up logging to both console and file."""
    log_level = logging.DEBUG if verbose else logging.INFO
    handlers = [
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )
    logging.info(f"Logging initialized → {log_path}")

class Logger:
    def __init__(self, filename: Path):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate writing to the log file

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def __enter__(self):
        # Context management setup
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Automatically close the log file when exiting the context
        self.log_file.close()

def get_optional(section: configparser.SectionProxy, key: str, default=None, convert=None):
    """Retrieve an optional config value with type conversion."""
    value = section.get(key, fallback=None)
    if value is None or value.lower() == "none":
        return default
    try:
        return convert(value) if convert else value
    except Exception as e:
        raise ValueError(f"Failed to convert {key}='{value}': {e}")


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path or input_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config = read_config(args.config_file, input_path)
    config_filename = Path(args.config_file).stem

    log_filename = output_path / f"spectral_fitting_{config_filename}.log"
    setup_logging(log_filename, verbose=args.verbose)

    settings = config["Settings"]
    instrument = config["Instrument"]
    dered = config["Dereddening"]
    library = config["Library"]
    fit = config["Fit"]

    # --- Settings ---
    gal_filename = settings["galaxy_filename"]
    segment = settings["segment"]
    bin_width = settings.getint("bin_width")
    default_noise = settings.getfloat("default_noise")
    z_guess = settings.getfloat("z_guess")

    # --- Instrument ---
    FWHM_gal = get_optional(instrument, "FWHM_gal", convert=float)
    if FWHM_gal is None:
        lam_min = instrument.getfloat("instr_lam_min")
        lam_max = instrument.getfloat("instr_lam_max")
        R = instrument.getfloat("R")
        FWHM_gal = InstrumentInfo(R, lam_min, lam_max).calc_FWHM_gal()

    # --- Dereddening ---
    ebv = float(dered.get("ebv", 0.0))
    Rv = float(dered.get("Rv", 3.1))
    ext_model = dered.get("model_name", "CCM89").upper()

    # --- Library ---
    lib_path = get_optional(library, "lib_path", default=None, convert=str)
    lib_path = Path(lib_path or Path(__file__).parent / "sample_libraries")

    library_name = library["Library"].upper()
    if library_name not in {"STARBURST99", "BPASS"}:
        raise ValueError(f"Unsupported library: {library_name}")

    evol_track = get_optional(library, "evol_track", default=None, convert=str)
    if evol_track is None and library_name == "STARBURST99":
        evol_track = "geneva_high"

    imf = library["IMF"].lower()
    star_form = library["star_form"].lower()
    star_pop = library["star_pop"].lower()
    age_range = ast.literal_eval(library["age_range"])
    metal_range = ast.literal_eval(library["metal_range"])
    norm_range = ast.literal_eval(library["norm_range"])

    # --- Fit parameters ---
    processor_config = {
        "age_range": age_range,
        "metal_range": metal_range,
        "norm_range": norm_range,
        "FWHM_gal": FWHM_gal,
        "output_path": str(output_path),
        "default_noise": default_noise,
        "config_filename": config_filename,
    }

    for key, value in fit.items():
        try:
            if value.lower() == "none":
                parsed_value = None
            elif value.lower() in ("true", "false"):
                parsed_value = value.lower() == "true"
            else:
                parsed_value = ast.literal_eval(value)
        except Exception:
            parsed_value = value

        if key == "absorp_lam":
            if isinstance(parsed_value, dict):
                parsed_value = {float(k): float(v) for k, v in parsed_value.items()}
        processor_config[key] = parsed_value

    # Log all configuration parameters
    logging.info("Configuration Parameters:")
    for section in config.sections():
        logging.info(f"[{section}]")
        for key, value in config[section].items():
            logging.info(f"{key}: {value}")

    # --- Spectrum Processing ---
    base = GalaxySpectrum(str(input_path / gal_filename), segment)
    operations = [
        DeReddeningOperation(ebv, ext_model, Rv),
        DeRedshiftOperation(z_guess),
        BinningOperation(bin_width),
        LogRebinningOperation(),
        NormalizationOperation(norm_range),
    ]
    processed = ProcessedGalaxySpectrum(base, operations).apply_operations()

    try:
        base.create_new_table(processed, output_path, config_filename)
        base.create_plots(processed, output_path, config_filename)
    except Exception as e:
        logging.error(f"Error saving or plotting: {e}")

    # --- Fitting ---
    processor = SpectrumProcessor(processor_config, processed)
    if processor_config.get("absorp_lam"):
        R = instrument.getfloat("R", fallback=None)
        if R is None:
            raise ValueError("Must provide resolving power 'R' to use spectral line masking.")
        processor_config["mask"] = processor.mask_spectral_lines(R)

    if library_name == "STARBURST99":
        lib_handler = Starburst99LibraryHandler(imf, star_form, star_pop, str(lib_path), evol_track)
    else:
        lib_handler = BPASSLibraryHandler(imf, star_form, star_pop, str(lib_path))

    # --- Execute fitting with redirected stdout and stderr ---
    with Logger(log_filename) as logger:
        sys.stdout = logger
        sys.stderr = logger
        try:
            processor.process_spectrum(lib_handler)
        finally:
            # Restore original stdout/stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
