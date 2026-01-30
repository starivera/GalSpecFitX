from typing import List
import os
import glob

from GalSpecFitX.library_registry import register_library
from GalSpecFitX.galaxy_fit import LibraryHandler, LibraryFactoryMixin
import GalSpecFitX.sps_util as lib


@register_library
class BPASSLibraryHandler(LibraryHandler, LibraryFactoryMixin):
    """
    Handler for BPASS (Binary Population and Spectral Synthesis) stellar population models.
    This handler retrieves templates from the BPASS library based on the specified parameters.

    References
    ----------
    - Eldridge, J.J., Stanway, E.R., Xiao, L., et al. (2017), "BPASS: Binary Population and Spectral Synthesis", PASA, 34, e058.
    - Stanway, E.R., & Eldridge, J.J. (2018), "Reevaluating old stellar populations", MNRAS, 479, 75.
    - BPASS project website: https://bpass.auckland.ac.nz
    """
    name = "BPASS"

    def __init__(
        self,
        IMF_slope: str,
        upper_mass: str,
        star_form: str,
        star_pop: str,
        lib_path: str,
    ):
        """
        Initialize the handler with the necessary parameters for the BPASS library.

        :param IMF_slope: Initial Mass Function (IMF) slope for the BPASS templates.
        :param upper_mass: Upper solar mass cutoff limit for the BPASS templates.
        :param star_form: Star formation scenario (e.g., single, binary).
        :param star_pop: Star population scenario (e.g., single, binary).
        :param lib_path: Path to the base directory of the BPASS library.
        """
        self.IMF_slope = IMF_slope
        self.upper_mass = upper_mass
        self.star_form = star_form
        self.star_pop = star_pop
        self.lib_path = lib_path

    def retrieve_templates(
        self,
        velscale: float,
        age_range: List[float],
        metal_range: List[float],
        norm_range: List[float],
        FWHM_gal: float,
    ) -> lib.SPSLibrary:
        """
        Retrieve the BPASS templates based on the specified parameters.

        :param velscale: Velocity scale per pixel in km/s.
        :param age_range: List of two floats representing the age range in Gyr for the templates to be retrieved `[age_min, age_max]`.
        :param metal_range: List of two floats representing the metallicity range for the templates to be retrieved `[metal_min, metal_max]` (e.g., 0.020 = Z☉).
        :param norm_range: List of two floats representing the wavelength range in Angstroms within which to compute the templates' normalization `[norm_min, norm_max]`.
        :param FWHM_gal: Full Width at Half Maximum (FWHM) of the galaxy's spectral line spread, in km/s.

        :return: The retrieved BPASS templates.
        """
        pathname = os.path.join(
            self.lib_path,
            "BPASS",
            self.star_form,
            self.star_pop,
            "imf_"+self.IMF_slope,
            self.upper_mass,
            "*.fits",
        )
        lam = glob.glob(os.path.join(self.lib_path, "BPASS", "*lam.fits"))[0]

        bpass_lib = lib.SPSLibrary(
            pathname,
            lam,
            velscale,
            FWHM_gal=FWHM_gal,
            FWHM_tem=1.0,
            age_range=age_range,
            metal_range=metal_range,
            norm_range=norm_range,
        )

        return bpass_lib

    @classmethod
    def validate_config(cls, cfg):
        if cfg["IMF"].lower() not in {"100", "135", "135all", "170", "chab"}:
            raise ValueError("Invalid IMF for BPASS")
        if cfg.getint("upper_mass") not in {100, 300}:
            raise ValueError("Invalid upper mass limit for BPASS")
        if cfg["star_form"].lower() not in {"instantaneous"}:
            raise ValueError("Invalid star formation mode")
        if cfg["star_pop"].lower() not in {"single", "binary"}:
            raise ValueError("Invalid star population")

    @classmethod
    def from_config(cls, cfg, lib_path):
        return cls(
            IMF_slope=cfg["IMF"].lower(),
            upper_mass=str(cfg.getint("upper_mass")),
            star_form=cfg["star_form"].lower(),
            star_pop=cfg["star_pop"].lower(),
            lib_path=str(lib_path),
        )
