#!/usr/bin/env python

import os
import ast
import sys
import argparse
import logging
import configparser
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.io as pio
import numpy as np
from GalSpecFitX.galaxy_prep import GalaxySpectrum, DeReddeningOperation, DeRedshiftOperation, BinningOperation, LogRebinningOperation, NormalizationOperation, ProcessedGalaxySpectrum
from GalSpecFitX.combine_and_fit import SpectrumProcessor, InstrumentInfo, StarburstLibraryHandler, BPASSLibraryHandler

def read_config(filename: str, input_path: str) -> configparser.ConfigParser:
    """Read configuration file.

    Args:
    ----
    filename : str
        Name of the configuration file.
    input_path : str
        Directory path where the configuration file is located.

    Returns:
    -------
    config : configparser.ConfigParser
        Configuration parser object containing parsed data.

    Raises:
    -------
    FileNotFoundError
        If the specified configuration file does not exist.
    """

    config_file = os.path.join(input_path, filename) if input_path else filename

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} does not exist.")

    config = configparser.ConfigParser()

    try:
        config.read(config_file)
    except configparser.Error as e:
        raise ValueError(f"Error parsing the configuration file: {e}")

    required_sections = ['Settings', 'instrument', 'library', 'fit']
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise KeyError(f"Missing required sections in the configuration file: {', '.join(missing_sections)}")

    return config

def parse_args() -> argparse.Namespace:
    """Parses command line arguments.

    Returns:
    --------
        args : argparse.Namespace object
            An argparse object containing all of the added arguments.

    """

    #Create help strings:
    input_path_help = "Input path containing galaxy data. If not provided assumes current directory. Please provide galaxy filename in your configuration file."
    config_file_help = "Configuration filename (default: config.ini) which is expected to be in input_path."
    output_path_help = "Output path for results. If not provided results will be generated in input_path."


    # Add arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-input_path', dest = 'input_path', action = 'store',
                        type = str, required = False, help = input_path_help)
    parser.add_argument('--config_file', '-config_file', dest = 'config_file', action = 'store',
                        type = str, required = False, default='config.ini', help = config_file_help)
    parser.add_argument('--output_path', '-output_path', dest = 'output_path', action = 'store',
                        type = str, required = False, help = output_path_help)


    # Parse args:
    args = parser.parse_args()

    return args

def configure_logging(filename: str) -> None:
    """Configures logging to a specified file.

    Args:
    ----
    filename : str
        Name of the log file to configure.
    """

    try:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename=filename, filemode='w')
        logging.info('Logging configured successfully')
    except Exception as e:
        logging.exception(f"Failed to configure logging: {e}")

class Logger:
    def __init__(self, filename="logfile.log"):
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

def get_optional_config(section, key, default=None, convert_to=None):
    value = section.get(key, default)

    if value is not None:
        if value.lower() == "none":
            value=None

        else:
            try:
                value = convert_to(value)
            except Exception as e:
                raise ValueError(f"Failed to convert value '{value}' for key '{key}' in section '{section.name}': {e}")

    return value


def main() -> None:

    """Main function."""

    # user input arguments
    args           = parse_args()
    input_path     = args.input_path
    config_file    = args.config_file
    output_path    = args.output_path
    input_path = input_path if input_path else os.getcwd()
    output_path = output_path if output_path else input_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    config = read_config(config_file, input_path)
    settings = config['Settings']
    instrument = config['instrument']
    dereddening = config['dereddening']
    library = config['library']
    fit = config['fit']

    # Settings parameters
    gal_filename = settings['galaxy_filename']
    segment = settings['segment']
    bin_width = int(settings['bin_width'])
    default_noise = float(settings['default_noise'])
    z_guess = float(settings['z_guess'])


    log_file = 'spectral_fitting.log'
    log_filename = os.path.join(output_path, log_file)
    configure_logging(log_filename)

    # Instrument parameters
    FWHM_gal = get_optional_config(instrument, 'FWHM_gal', default=None, convert_to=float)

    if FWHM_gal is None:
        instr_lam_min = instrument.getfloat('instr_lam_min')
        instr_lam_max = instrument.getfloat('instr_lam_max')
        R = instrument.getfloat('R')
        FWHM_gal = InstrumentInfo(R, instr_lam_min, instr_lam_max).calc_FWHM_gal()

    # Dereddening Parameters
    ebv = float(dereddening.get('ebv', 0.0))
    Rv = float(dereddening.get('Rv', 3.1))
    ext_model = dereddening.get('model_name', 'CCM89').upper()

    # Library parameters
    lib_path = get_optional_config(library, 'lib_path', default=None, convert_to=str)

    if lib_path is None:
        lib_path = os.path.dirname(os.path.abspath(__file__)) + '/sample_libraries'

    library_name = library['Library'].upper()

    if library_name not in ['STARBURST99', 'BPASS']:
        logging.error(f"Unknown library: {library_name}")
        raise ValueError(f"Unsupported library: {library_name}")

    evol_track = get_optional_config(library, 'evol_track', default=None, convert_to=str)

    if evol_track is None and library_name == 'STARBURST99':
        evol_track = 'geneva_high'
    elif evol_track is not None and library_name != 'STARBURST99':
        logging.info("evol_track parameter can only be used with Starburst99 and will therefore be ignored.")

    imf = library['IMF'].lower()

    star_form = library['star_form'].lower()
    star_evol = library['star_evol'].lower()

    age_range_str = library['age_range']  # e.g., "[0.0, 0.4]"
    age_range = ast.literal_eval(age_range_str)  # Converts the string to a list

    metal_range_str = library['metal_range']
    metal_range = ast.literal_eval(metal_range_str)

    norm_range_str = library['norm_range']
    norm_range = ast.literal_eval(norm_range_str)

    # Fit parameters
    processor_config = {
        'age_range': age_range,
        'metal_range': metal_range,
        'norm_range': norm_range,
        'FWHM_gal': FWHM_gal,
        'output_path': output_path,
        'segment': segment,
        'default_noise': default_noise,
    }

    for key, value in fit.items():
        if value.lower() == "none":
            value = None
        elif value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        else:
            try:
                # Use ast.literal_eval() to safely evaluate the value
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError) as e:
                logging.error(f"Error processing {key}: {e}")
                continue

        processor_config[key] = value

    if key == 'absorp_lam':
        if isinstance(value, list):
            # Leave as list for default-window logic
            processor_config[key] = value
        elif isinstance(value, dict):
            # Ensure keys are float if provided as strings
            processor_config[key] = {float(k): float(v) for k, v in value.items()}
        else:
            raise ValueError("Invalid format for 'absorp_lam'; must be a list or a dict.")
    else:
        processor_config[key] = value

    # Log all configuration parameters
    logging.info("Configuration Parameters:")
    for section in config.sections():
        logging.info(f"[{section}]")
        for key, value in config[section].items():
            logging.info(f"{key}: {value}")

    base_spectrum = GalaxySpectrum(input_path+'/'+gal_filename, segment)
    operations = [
        DeReddeningOperation(ebv, ext_model, Rv),
        DeRedshiftOperation(z_guess),
        BinningOperation(bin_width),
        LogRebinningOperation(),
        NormalizationOperation(norm_range)
    ]
    processed_spectrum = ProcessedGalaxySpectrum(base_spectrum, operations)
    processed_data = processed_spectrum.apply_operations()

    # Save processed spectrum to a FITS file named bestfit.fits
    base_spectrum.create_new_table(processed_data, output_path)

    lam_gal_log_rebin, norm_flux_gal_log_rebin, norm_err_gal_log_rebin = processed_data

    # Create the Matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 5))
    plt.plot(lam_gal_log_rebin, norm_flux_gal_log_rebin, label=f'{segment} spectrum')
    plt.plot(lam_gal_log_rebin, norm_err_gal_log_rebin, linestyle='None', marker='.', markersize=1, label='Error')
    plt.title(f'{gal_filename} - Deredshifted, SpectRes Binned, Log-Rebinned, Median Normalized Spectrum')
    plt.xlabel('Rest Wavelength (Å)')
    plt.ylabel('Median Normalized Flux')
    plt.legend()
    plt.grid(alpha=0.5)

    # Convert Matplotlib figure to Plotly
    plotly_fig = tls.mpl_to_plotly(fig)

    # Save the interactive plot as an HTML file
    pio.write_html(plotly_fig, file=os.path.join(output_path,f'normalized_log_rebinned_spectrum_{segment}.html'), auto_open=False)


    # Initialize SpectrumProcessor for combining and fitting
    processor = SpectrumProcessor(processor_config, processed_data)

    # Call mask_spectral_lines if aborption wavelengths are provided
    if processor_config['absorp_lam'] is not None:

        assert instrument['R'].lower() != "none", "Must provide resolving power to use spectral line masking."

        R = float(instrument['R'])
        processor_config['mask'] = processor.mask_spectral_lines(R)

    if library_name == 'STARBURST99':
        library_handler = StarburstLibraryHandler(imf, star_form, star_evol, lib_path, evol_track)
    elif library_name == 'BPASS':
        library_handler = BPASSLibraryHandler(imf, star_form, star_evol, lib_path)

    with Logger(log_filename) as logger:
        sys.stdout = logger
        processor.process_spectrum(library_handler)
        sys.stdout = sys.stdout.terminal  # Restore the original stdout

if __name__ == '__main__':
    main()
