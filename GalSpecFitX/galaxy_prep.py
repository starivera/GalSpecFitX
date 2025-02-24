#!/usr/bin/env python
"""
This script prepares a galaxy spectrum for spectral fitting by performing operations like
de-redshifting, binning, and normalization.

Author: Isabel Rivera
"""
import os
from typing import Protocol, Tuple, List

import configparser
import numpy as np
from astropy.io import fits
from astropy.table import Table
from spectres import spectres

from ppxf.ppxf_util import log_rebin

class SpectrumOperation(Protocol):
    """Protocol for spectrum operation classes."""

    def apply(self, spectrum: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the spectrum operation."""

        pass

class DeRedshiftOperation:
    """Class for de-redshifting the spectrum."""

    def __init__(self, z_guess: float):
        """
        Initialize the de-redshifting operation.

        Parameters:
        z_guess (float): Estimated redshift of the galaxy.
        """

        self.z_guess = z_guess

    def apply(self, spectrum: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the de-redshifting operation.

        Parameters:
        spectrum (Tuple[np.ndarray, np.ndarray, np.ndarray]): The input spectrum as (wavelengths, fluxes, errors).

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The de-redshifted spectrum.
        """

        waves, fluxes, errors = spectrum
        waves /= (1 + self.z_guess)

        return waves, fluxes, errors

class BinningOperation:
    """Class for binning the spectrum."""

    def __init__(self, bin_width: int, fill_value: float = 0.0):
        """
        Initialize the binning operation.

        Parameters:
        bin_width (int): Width of the bins.
        fill_value (float): Value to fill pixels where the new wavelength array is out of range of the original wavelength array. Default: 0.
        """

        self.bin_width = bin_width
        self.fill_value = fill_value

    def apply(self, spectrum: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the binning operation.

        Parameters:
        spectrum (Tuple[np.ndarray, np.ndarray, np.ndarray]): The input spectrum as (wavelengths, fluxes, errors).

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The binned spectrum.
        """

        waves, fluxes, errors = spectrum

        non_zero_i, non_zero_f = np.where(errors != 0)[0][0], np.where(errors != 0)[0][-1] + 1
        waves = waves[non_zero_i:non_zero_f]
        fluxes = fluxes[non_zero_i:non_zero_f]
        errors = errors[non_zero_i:non_zero_f]

        binned_waves = waves[:(waves.size // self.bin_width) * self.bin_width].reshape(-1, self.bin_width).mean(axis=1)
        binned_fluxes, binned_errors = spectres(binned_waves, waves, fluxes, errors, fill=self.fill_value)

        return binned_waves, binned_fluxes, binned_errors

class LogRebinningOperation:
    """Class for logarithmic rebinning of the spectrum."""

    def apply(self, spectrum: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the log rebinning operation.

        Parameters:
        spectrum (Tuple[np.ndarray, np.ndarray, np.ndarray]): The input spectrum as (wavelengths, fluxes, errors).

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The log-rebinned spectrum.
        """

        waves, fluxes, errors = spectrum
        non_zero_i, non_zero_f = np.where(errors != 0)[0][0], np.where(errors != 0)[0][-1] + 1
        waves = waves[non_zero_i:non_zero_f]
        fluxes = fluxes[non_zero_i:non_zero_f]
        errors = errors[non_zero_i:non_zero_f]

        cenwaves = np.zeros(len(waves))
        cenwaves[:-1] = (waves[1:] + waves[:-1]) / 2
        cenwaves[-1] = waves[-1] + ((waves[-1] - waves[-2]) / 2)

        log_rebin_fluxes, log_rebin_waves, velscale = log_rebin(cenwaves, fluxes)

        errors_squared = errors**2
        log_rebin_errs, _, _ = log_rebin(cenwaves, errors_squared)
        log_rebin_errs = np.sqrt(log_rebin_errs)

        return np.exp(log_rebin_waves), log_rebin_fluxes, log_rebin_errs

class NormalizationOperation:
    """Class for median normalizing the spectrum."""

    def __init__(self, norm_range: List[float]):
        """
        Initialize the de-redshifting operation.

        Parameters:
        norm_range (List[float]): List of two floats representing the wavelength range in Angstroms within which to compute the galaxy spectrum's normalization `[norm_min, norm_max]`.
        """

        self.norm_range = norm_range

    def apply(self, spectrum: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the normalization operation.

        Parameters:
        spectrum (Tuple[np.ndarray, np.ndarray, np.ndarray]): The input spectrum as (wavelengths, fluxes, errors).

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The median normalized spectrum.
        """

        waves, fluxes, errors = spectrum

        if self.norm_range is None:
            median_flux = np.median(fluxes)
            median_error = np.median(errors)

        else:
            median_flux = np.median(fluxes[(waves >= self.norm_range[0]) & (waves <= self.norm_range[1])])
            median_error =  np.median(errors[(waves >= self.norm_range[0]) & (waves <= self.norm_range[1])])

        med_norm_fluxes = fluxes / median_flux
        med_norm_errors = errors / median_error

        return waves, med_norm_fluxes, med_norm_errors

class GalaxySpectrum:
    """Class for managing a galaxy spectrum."""

    def __init__(self, filename: str, segment: str):
        """Initialize the galaxy spectrum.

        Parameters:
        filename (str): Name of the FITS file containing the spectrum.
        segment (str): Segment of the spectrum to use.
        """

        self.filename = filename
        self.segment = segment

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve the spectrum data from the FITS file.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The spectrum data as (wavelengths, fluxes, errors).
        """

        segment = int(self.segment)
        with fits.open(self.filename) as hdul:
            table_data = hdul[segment].data

        # Extract the columns
        wavelengths = table_data['wavelength']
        fluxes = table_data['flux']
        errors = table_data['error']

        return wavelengths, fluxes, errors

    def create_new_table(self, spectrum: Tuple[np.ndarray, np.ndarray, np.ndarray], output_path: str) -> None:
        """Create and save a new FITS table containing spectrum data.

        Parameters:
        spectrum (Tuple[np.ndarray, np.ndarray, np.ndarray]): The spectrum data as (wavelengths, fluxes, errors).
        output_path (str): The directory where the FITS file will be saved.
        """

        # Unpack spectrum data
        wavelengths, fluxes, errors = spectrum

        # Construct the table
        new_table = Table(
            [[self.segment], [wavelengths], [fluxes], [errors]],
            names=('SEGMENT', 'WAVELENGTH', 'FLUX', 'ERROR')
        )

        # Create a FITS binary table HDU
        new_hdu = fits.BinTableHDU(new_table, name=f'PROCESSED_DATA_{self.segment}')

        # Define output file path
        file_name = os.path.join(output_path, 'bestfit.fits')

        # Write FITS file with a primary HDU and the new table
        fits.HDUList([fits.PrimaryHDU(), new_hdu]).writeto(file_name, overwrite=True)


class ProcessedGalaxySpectrum:
    """Class for applying operations to a galaxy spectrum."""

    def __init__(self, base_spectrum: GalaxySpectrum, operations: List[SpectrumOperation]):
        """Initialize the processed galaxy spectrum.

        Parameters:
        base_spectrum (GalaxySpectrum): The base galaxy spectrum.
        operations (List[SpectrumOperation]): List of operations to apply to the spectrum.
        """

        self.base_spectrum = base_spectrum
        self.operations = operations

    def apply_operations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the operations to the base spectrum.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The processed spectrum as (wavelengths, fluxes, errors).
        """

        spectrum_data = self.base_spectrum.get_data()

        for operation in self.operations:
            spectrum_data = operation.apply(spectrum_data)

        return spectrum_data
