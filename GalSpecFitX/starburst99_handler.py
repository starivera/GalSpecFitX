from typing import List
import os
import glob

from GalSpecFitX.library_registry import register_library
from GalSpecFitX.galaxy_fit import LibraryHandler, LibraryFactoryMixin
import GalSpecFitX.sps_util as lib


@register_library
class Starburst99LibraryHandler(LibraryHandler, LibraryFactoryMixin):
    """
    Handler for Starburst99 stellar population models. This handler is used to retrieve
    templates from the Starburst99 library based on the specified parameters.

    References
    ----------
    Leitherer et al. (1999), ApJS, 123, 3
    Vázquez & Leitherer (2005), ApJ, 621, 695
    Leitherer et al. (2010), ApJS, 189, 309
    Leitherer et al. (2014)
    """
    name = "STARBURST99"

    def __init__(
        self,
        IMF_slope: str,
        upper_mass: str,
        star_form: str,
        star_pop: str,
        lib_path: str,
        evol_track: str,
    ):
        """
        Initialize the handler with the necessary parameters for the Starburst99 library.

        :param IMF_slope: Initial Mass Function (IMF) slope for the Starburst99 templates.
        :param upper_mass: Upper solar mass cutoff limit for the Starburst99 templates.
        :param star_form: Star formation scenario (e.g., instantaneous, continuous).
        :param star_pop: Star population scenario (e.g., single, binary).
        :param lib_path: Path to the base directory of the Starburst99 library.
        :param evol_track: Evolutionary track to be used for the Starburst99 models.
        """
        self.IMF_slope = IMF_slope
        self.upper_mass = upper_mass
        self.star_form = star_form
        self.star_pop = star_pop
        self.lib_path = lib_path
        self.evol_track = evol_track

    def retrieve_templates(
        self,
        velscale: float,
        age_range: List[float],
        metal_range: List[float],
        norm_range: List[float],
        FWHM_gal: float,
    ) -> lib.SPSLibrary:
        """
        Retrieve the Starburst99 templates based on the specified parameters.

        :param velscale: Velocity scale per pixel in km/s.
        :param age_range: List of two floats representing the age range in Gyr for the templates to be retrieved `[age_min, age_max]`.
        :param metal_range: List of two floats representing the metallicity range for the templates to be retrieved `[metal_min, metal_max]` (e.g., 0.020 = Z☉).
        :param norm_range: List of two floats representing the wavelength range in Angstroms within which to compute the templates' normalization `[norm_min, norm_max]`.
        :param FWHM_gal: Full Width at Half Maximum (FWHM) of the galaxy's spectral line spread, in km/s.

        :return: The retrieved Starburst99 templates.
        """
        pathname = os.path.join(
            self.lib_path,
            "STARBURST99",
            self.evol_track,
            self.star_form,
            self.star_pop,
            self.IMF_slope,
            self.upper_mass,
            "*.fits",
        )
        lam = glob.glob(
            os.path.join(self.lib_path, "STARBURST99", self.evol_track, "*lam.fits")
        )[0]

        starburst99_lib = lib.SPSLibrary(
            pathname,
            lam,
            velscale,
            FWHM_gal=FWHM_gal,
            FWHM_tem=0.4,
            age_range=age_range,
            metal_range=metal_range,
            norm_range=norm_range,
        )
        return starburst99_lib

    @classmethod
    def validate_config(cls, cfg):
        if cfg["IMF"].lower() not in {"salpeter", "kroupa"}:
            raise ValueError("Invalid IMF for Starburst99")
        if cfg.getint("upper_mass") not in {100}:
            raise ValueError("Invalid upper mass limit for Starburst99")
        if cfg["star_form"].lower() not in {"instantaneous"}:
            raise ValueError("Invalid star formation mode")
        if cfg["star_pop"].lower() not in {"single"}:
            raise ValueError("Invalid star population")

    @classmethod
    def from_config(cls, cfg, lib_path):
        return cls(
            IMF_slope=cfg["IMF"].lower(),
            upper_mass=str(cfg.getint("upper_mass")),
            star_form=cfg["star_form"].lower(),
            star_pop=cfg["star_pop"].lower(),
            lib_path=str(lib_path),
            evol_track=cfg.get("evol_track", "geneva_high"),
        )
