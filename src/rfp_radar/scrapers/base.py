"""Base scraper abstract class for RFP Radar.

This module provides the abstract base class that all RFP scrapers must inherit from.
It defines the common interface and provides shared functionality for:
- HTTP requests with retry logic and exponential backoff
- RFP filtering by age and geography
- Structured logging
- Error handling
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import RequestException

from rfp_radar.models import RFP, RFPSource, ScraperResult


class BaseScraper(ABC):
    """Abstract base class for RFP scrapers.

    All concrete scraper implementations must inherit from this class and
    implement the required abstract methods. The base class provides:
    - HTTP request handling with retry logic
    - Common filtering by age and geography
    - Structured logging
    - Error handling patterns

    Attributes:
        source: The RFPSource enum value for this scraper
        base_url: Base URL for the scraper's target portal
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1.0)
        request_timeout: Request timeout in seconds (default: 60)
        session: Requests session for connection pooling

    Example:
        @register_scraper("govtribe")
        class GovTribeScraper(BaseScraper):
            source = RFPSource.GOVTRIBE
            base_url = "https://api.govtribe.com"

            def fetch_listings(self, **kwargs):
                # Implementation
                pass

            def parse_listing(self, raw_data):
                # Implementation
                pass
    """

    # Class-level attributes to be overridden by subclasses
    source: RFPSource = RFPSource.MANUAL
    base_url: str = ""

    # Default configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: int = 60

    def __init__(
        self,
        max_age_days: int = 3,
        us_only: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the base scraper.

        Args:
            max_age_days: Maximum age in days for RFPs to include (default: 3)
            us_only: If True, filter to US-based RFPs only (default: True)
            logger: Optional logger instance. If None, creates one.
        """
        self.max_age_days = max_age_days
        self.us_only = us_only
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.session = requests.Session()
        self._configure_session()

    def _configure_session(self) -> None:
        """Configure the requests session with default headers."""
        self.session.headers.update({
            "User-Agent": "NAITIVE-RFP-Radar/1.0",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def close(self) -> None:
        """Close the requests session and release resources."""
        if self.session:
            self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes session."""
        self.close()
        return False

    @abstractmethod
    def fetch_listings(self, **kwargs) -> List[Dict[str, Any]]:
        """Fetch raw RFP listings from the source portal.

        This method must be implemented by all concrete scraper classes.
        It should return a list of raw data dictionaries that can be
        parsed into RFP objects.

        Args:
            **kwargs: Source-specific parameters (e.g., date range, category)

        Returns:
            List of raw RFP data dictionaries

        Raises:
            RequestException: If the request fails after all retries
        """
        pass

    @abstractmethod
    def parse_listing(self, raw_data: Dict[str, Any]) -> Optional[RFP]:
        """Parse a raw listing dictionary into an RFP object.

        This method must be implemented by all concrete scraper classes.
        It should handle the source-specific data format and create
        a normalized RFP object.

        Args:
            raw_data: Raw listing data from the source

        Returns:
            RFP object if parsing succeeds, None if the data is invalid
        """
        pass

    def scrape(self, **kwargs) -> ScraperResult:
        """Execute the full scraping pipeline.

        This method orchestrates the complete scraping process:
        1. Fetch raw listings from the source
        2. Parse each listing into RFP objects
        3. Filter by age and geography
        4. Return the results

        Args:
            **kwargs: Source-specific parameters passed to fetch_listings

        Returns:
            ScraperResult containing the scraped and filtered RFPs
        """
        start_time = time.time()
        self.logger.info(
            "Starting scrape",
            extra={
                "source": self.source.value,
                "max_age_days": self.max_age_days,
                "us_only": self.us_only,
            },
        )

        try:
            # Fetch raw listings
            raw_listings = self.fetch_listings(**kwargs)
            total_found = len(raw_listings)
            self.logger.info(
                "Fetched listings",
                extra={"source": self.source.value, "count": total_found},
            )

            # Parse and filter
            rfps = []
            for raw_data in raw_listings:
                try:
                    rfp = self.parse_listing(raw_data)
                    if rfp is not None:
                        # Apply filters
                        if self._passes_filters(rfp):
                            rfps.append(rfp)
                except Exception as e:
                    self.logger.warning(
                        "Failed to parse listing",
                        extra={
                            "source": self.source.value,
                            "error": str(e),
                            "raw_data_keys": list(raw_data.keys()) if raw_data else [],
                        },
                    )
                    continue

            duration = time.time() - start_time
            self.logger.info(
                "Scrape completed",
                extra={
                    "source": self.source.value,
                    "total_found": total_found,
                    "after_filter": len(rfps),
                    "duration_seconds": round(duration, 2),
                },
            )

            return ScraperResult(
                source=self.source,
                scraped_at=datetime.utcnow(),
                success=True,
                rfps=rfps,
                total_found=total_found,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "Scrape failed",
                extra={
                    "source": self.source.value,
                    "error": str(e),
                    "duration_seconds": round(duration, 2),
                },
            )
            return ScraperResult(
                source=self.source,
                scraped_at=datetime.utcnow(),
                success=False,
                error_message=str(e),
                rfps=[],
                total_found=0,
                duration_seconds=duration,
            )

    def _passes_filters(self, rfp: RFP) -> bool:
        """Check if an RFP passes all configured filters.

        Args:
            rfp: The RFP to check

        Returns:
            True if the RFP passes all filters, False otherwise
        """
        # Check age filter
        if not rfp.is_within_age_limit(self.max_age_days):
            self.logger.debug(
                "RFP filtered by age",
                extra={
                    "rfp_id": rfp.id,
                    "age_days": rfp.age_in_days(),
                    "max_age_days": self.max_age_days,
                },
            )
            return False

        # Check geography filter
        if self.us_only and not rfp.is_us_based():
            self.logger.debug(
                "RFP filtered by geography",
                extra={"rfp_id": rfp.id, "country": rfp.country},
            )
            return False

        return True

    def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> requests.Response:
        """Make an HTTP request with exponential backoff retry.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments passed to requests

        Returns:
            Response object

        Raises:
            RequestException: If all retry attempts fail
        """
        kwargs.setdefault("timeout", self.request_timeout)
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except RequestException as e:
                last_exception = e
                wait_time = self.retry_delay * (2 ** attempt)

                if attempt < self.max_retries - 1:
                    self.logger.warning(
                        "Request failed, retrying",
                        extra={
                            "url": url,
                            "attempt": attempt + 1,
                            "max_retries": self.max_retries,
                            "wait_seconds": wait_time,
                            "error": str(e),
                        },
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        "Request failed after all retries",
                        extra={
                            "url": url,
                            "attempts": self.max_retries,
                            "error": str(e),
                        },
                    )

        raise last_exception

    def get(self, url: str, **kwargs) -> requests.Response:
        """Make a GET request with retry logic.

        Args:
            url: Request URL
            **kwargs: Additional arguments passed to requests

        Returns:
            Response object
        """
        return self._request_with_retry("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        """Make a POST request with retry logic.

        Args:
            url: Request URL
            **kwargs: Additional arguments passed to requests

        Returns:
            Response object
        """
        return self._request_with_retry("POST", url, **kwargs)

    def build_url(self, path: str) -> str:
        """Build a full URL from the base URL and path.

        Args:
            path: URL path to append to base URL

        Returns:
            Full URL string
        """
        base = self.base_url.rstrip("/")
        path = path.lstrip("/")
        return f"{base}/{path}" if path else base

    @classmethod
    def get_source_name(cls) -> str:
        """Get the source name for this scraper.

        Returns:
            Lowercase source name string
        """
        return cls.source.value.lower()

    def __repr__(self) -> str:
        """Return a string representation of the scraper."""
        return (
            f"{self.__class__.__name__}("
            f"source={self.source.value}, "
            f"max_age_days={self.max_age_days}, "
            f"us_only={self.us_only})"
        )
