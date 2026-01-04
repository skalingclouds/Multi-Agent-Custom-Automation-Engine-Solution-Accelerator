"""Scrapers package for RFP Radar.

This package provides an extensible framework for scraping RFP data
from various government portals and aggregators.

The SCRAPERS registry maps RFPSource values to their corresponding
scraper classes. Scrapers are registered when their modules are imported.
"""

from typing import Dict, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rfp_radar.scrapers.base import BaseScraper

# Registry mapping source names to scraper classes
# This will be populated by individual scraper modules when imported
SCRAPERS: Dict[str, Type["BaseScraper"]] = {}


def register_scraper(source_name: str):
    """Decorator to register a scraper class in the SCRAPERS registry.

    Args:
        source_name: The source identifier (e.g., "govtribe", "opengov", "bidnet")

    Returns:
        Decorator function that registers the class

    Example:
        @register_scraper("govtribe")
        class GovTribeScraper(BaseScraper):
            ...
    """
    def decorator(cls: Type["BaseScraper"]) -> Type["BaseScraper"]:
        SCRAPERS[source_name.lower()] = cls
        return cls
    return decorator


def get_scraper(source_name: str) -> Optional[Type["BaseScraper"]]:
    """Get a scraper class by source name.

    Args:
        source_name: The source identifier (e.g., "govtribe", "opengov", "bidnet")

    Returns:
        The scraper class if found, None otherwise
    """
    return SCRAPERS.get(source_name.lower())


def get_available_sources() -> list:
    """Get a list of all available scraper source names.

    Returns:
        List of registered source names
    """
    return list(SCRAPERS.keys())


# Import scraper implementations to trigger registration
# These imports will fail gracefully if the modules don't exist yet
try:
    from rfp_radar.scrapers.govtribe import GovTribeScraper  # noqa: F401
except ImportError:
    pass

try:
    from rfp_radar.scrapers.opengov import OpenGovScraper  # noqa: F401
except ImportError:
    pass

try:
    from rfp_radar.scrapers.bidnet import BidNetScraper  # noqa: F401
except ImportError:
    pass


__all__ = [
    "SCRAPERS",
    "register_scraper",
    "get_scraper",
    "get_available_sources",
]
