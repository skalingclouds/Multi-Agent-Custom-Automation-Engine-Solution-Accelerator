"""OpenGov portal scraper for RFP Radar.

This module provides a scraper for OpenGov, a platform used by many
state and local governments for procurement and bidding opportunities.

OpenGov provides a unified interface for accessing government
procurement opportunities at the state and local level.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from rfp_radar.models import RFP, RFPSource
from rfp_radar.scrapers import register_scraper
from rfp_radar.scrapers.base import BaseScraper


@register_scraper("opengov")
class OpenGovScraper(BaseScraper):
    """Scraper for OpenGov procurement portal.

    OpenGov provides access to state and local government procurement
    opportunities. This scraper fetches opportunities and normalizes
    them into the RFP format.

    Attributes:
        source: RFPSource.OPENGOV
        base_url: OpenGov API base URL

    Example:
        with OpenGovScraper(max_age_days=3) as scraper:
            result = scraper.scrape()
            for rfp in result.rfps:
                print(f"Found: {rfp.title}")
    """

    source: RFPSource = RFPSource.OPENGOV
    base_url: str = "https://api.opengov.com/v1"

    # OpenGov-specific configuration
    opportunities_endpoint: str = "/bids"
    default_page_size: int = 50
    max_pages: int = 10

    def _configure_session(self) -> None:
        """Configure the requests session with OpenGov-specific headers."""
        super()._configure_session()
        # OpenGov API may require additional headers
        self.session.headers.update({
            "X-Client-Name": "NAITIVE-RFP-Radar",
            "X-Client-Version": "1.0",
        })

    def fetch_listings(
        self,
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
        category: Optional[str] = None,
        state: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Fetch raw RFP listings from OpenGov API.

        Fetches bid opportunities from OpenGov, handling
        pagination automatically.

        Args:
            page_size: Number of results per page (default: 50)
            max_pages: Maximum pages to fetch (default: 10)
            category: Optional category filter
            state: Optional state filter (e.g., "CA", "NY")
            **kwargs: Additional query parameters

        Returns:
            List of raw bid dictionaries from the API

        Raises:
            RequestException: If the API request fails after retries
        """
        page_size = page_size or self.default_page_size
        max_pages = max_pages or self.max_pages
        all_listings = []

        self.logger.info(
            "Fetching OpenGov bids",
            extra={
                "page_size": page_size,
                "max_pages": max_pages,
                "category": category,
                "state": state,
            },
        )

        for page in range(1, max_pages + 1):
            params = self._build_query_params(
                page=page,
                page_size=page_size,
                category=category,
                state=state,
                **kwargs,
            )

            try:
                response = self.get(
                    self.build_url(self.opportunities_endpoint),
                    params=params,
                )
                data = response.json()

                # Handle different response formats
                listings = self._extract_listings(data)
                if not listings:
                    self.logger.info(
                        "No more listings found",
                        extra={"page": page, "total_fetched": len(all_listings)},
                    )
                    break

                all_listings.extend(listings)

                # Check if we've reached the last page
                if len(listings) < page_size:
                    break

            except Exception as e:
                self.logger.warning(
                    "Failed to fetch page",
                    extra={
                        "page": page,
                        "error": str(e),
                        "total_fetched": len(all_listings),
                    },
                )
                # Continue with what we have rather than failing completely
                break

        self.logger.info(
            "Completed fetching OpenGov bids",
            extra={"total_listings": len(all_listings)},
        )

        return all_listings

    def _build_query_params(
        self,
        page: int,
        page_size: int,
        category: Optional[str] = None,
        state: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build query parameters for the OpenGov API.

        Args:
            page: Page number (1-indexed)
            page_size: Number of results per page
            category: Optional category filter
            state: Optional state filter
            **kwargs: Additional query parameters

        Returns:
            Dictionary of query parameters
        """
        params = {
            "page": page,
            "per_page": page_size,
            "sort_by": "posted_date",
            "sort_order": "desc",
            "status": "open",  # Only open bids
        }

        if category:
            params["category"] = category

        if state:
            params["state"] = state

        # Add any additional parameters
        params.update(kwargs)

        return params

    def _extract_listings(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract listings from the API response.

        Handles different response formats that the API might return.

        Args:
            data: Raw API response data

        Returns:
            List of bid dictionaries
        """
        # Handle different possible response structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common response wrapper keys
            for key in ["bids", "opportunities", "results", "data", "items"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If the dict looks like a single bid
            if "id" in data or "title" in data or "bid_number" in data:
                return [data]
        return []

    def parse_listing(self, raw_data: Dict[str, Any]) -> Optional[RFP]:
        """Parse a raw OpenGov listing into an RFP object.

        Normalizes the OpenGov-specific data format into the
        standard RFP model.

        Args:
            raw_data: Raw bid data from OpenGov API

        Returns:
            RFP object if parsing succeeds, None if data is invalid
        """
        if not raw_data:
            return None

        try:
            # Extract required fields with fallbacks
            title = self._extract_field(raw_data, ["title", "name", "bid_title", "subject"])
            if not title:
                self.logger.debug(
                    "Skipping listing without title",
                    extra={"raw_keys": list(raw_data.keys())},
                )
                return None

            # Parse dates
            posted_date = self._parse_date(
                self._extract_field(
                    raw_data,
                    ["posted_date", "publish_date", "created_at", "open_date"],
                )
            )
            due_date = self._parse_date(
                self._extract_field(
                    raw_data,
                    ["due_date", "close_date", "deadline", "response_deadline", "bid_due_date"],
                )
            )

            # Extract location info
            location, state, country = self._parse_location(raw_data)

            # Build RFP object
            rfp = RFP(
                id=self._generate_id(raw_data),
                title=title,
                description=self._extract_field(
                    raw_data,
                    ["description", "summary", "synopsis", "details", "scope"],
                    default="",
                ),
                agency=self._extract_field(
                    raw_data,
                    ["agency", "organization", "department", "issuer", "buyer", "government_entity"],
                    default="",
                ),
                source=self.source,
                source_url=self._extract_field(
                    raw_data,
                    ["url", "source_url", "link", "bid_url", "portal_url"],
                    default="",
                ),
                posted_date=posted_date,
                due_date=due_date,
                location=location,
                state=state,
                country=country,
                pdf_url=self._extract_field(
                    raw_data,
                    ["pdf_url", "document_url", "attachment_url", "bid_document"],
                    default=None,
                ),
                attachments=self._extract_attachments(raw_data),
                naics_codes=self._extract_naics_codes(raw_data),
                set_aside=self._extract_field(
                    raw_data,
                    ["set_aside", "set_aside_type", "preference"],
                    default="",
                ),
                estimated_value=self._parse_value(
                    self._extract_field(
                        raw_data,
                        ["estimated_value", "value", "contract_value", "budget"],
                    )
                ),
                contract_type=self._extract_field(
                    raw_data,
                    ["contract_type", "type", "procurement_type", "bid_type"],
                    default="",
                ),
                raw_data=raw_data,
            )

            return rfp

        except Exception as e:
            self.logger.warning(
                "Failed to parse OpenGov listing",
                extra={
                    "error": str(e),
                    "raw_keys": list(raw_data.keys()) if raw_data else [],
                },
            )
            return None

    def _extract_field(
        self,
        data: Dict[str, Any],
        keys: List[str],
        default: Any = None,
    ) -> Any:
        """Extract a field value trying multiple possible keys.

        Args:
            data: Dictionary to extract from
            keys: List of keys to try in order
            default: Default value if no key is found

        Returns:
            First found value or default
        """
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return default

    def _generate_id(self, raw_data: Dict[str, Any]) -> str:
        """Generate a unique ID for the RFP.

        Args:
            raw_data: Raw listing data

        Returns:
            Unique identifier string
        """
        # Try to use existing ID from the source
        for key in ["id", "_id", "bid_id", "bid_number", "solicitation_number", "reference_number"]:
            if key in raw_data and raw_data[key]:
                return f"opengov-{raw_data[key]}"

        # Fallback: generate from title hash
        title = self._extract_field(raw_data, ["title", "name", "bid_title"], "unknown")
        import hashlib
        hash_value = hashlib.sha256(str(title).encode()).hexdigest()[:12]
        return f"opengov-{hash_value}"

    def _parse_date(self, date_value: Any) -> Optional[datetime]:
        """Parse a date value from various formats.

        Args:
            date_value: Raw date value (string, datetime, or None)

        Returns:
            Parsed datetime or None
        """
        if date_value is None:
            return None

        if isinstance(date_value, datetime):
            return date_value

        if isinstance(date_value, str):
            date_formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%m/%d/%Y %H:%M:%S",
                "%d-%m-%Y",
                "%B %d, %Y",  # e.g., "January 15, 2024"
            ]
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue

        return None

    def _parse_location(
        self,
        raw_data: Dict[str, Any],
    ) -> tuple:
        """Parse location information from raw data.

        Args:
            raw_data: Raw listing data

        Returns:
            Tuple of (location, state, country)
        """
        location = self._extract_field(
            raw_data,
            ["place_of_performance", "location", "address", "city", "jurisdiction"],
            default="",
        )

        state = self._extract_field(
            raw_data,
            ["state", "state_code", "region", "state_province"],
            default="",
        )

        # OpenGov primarily serves US state and local governments
        country = self._extract_field(
            raw_data,
            ["country", "country_code"],
            default="US",
        )

        # Handle nested location objects
        if isinstance(location, dict):
            state = location.get("state", location.get("state_code", state))
            country = location.get("country", location.get("country_code", country))
            location = location.get("city", location.get("address", ""))

        return str(location), str(state), str(country)

    def _extract_attachments(self, raw_data: Dict[str, Any]) -> List[str]:
        """Extract attachment URLs from raw data.

        Args:
            raw_data: Raw listing data

        Returns:
            List of attachment URLs
        """
        attachments = []

        for key in ["attachments", "documents", "files", "bid_documents"]:
            if key in raw_data:
                items = raw_data[key]
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str):
                            attachments.append(item)
                        elif isinstance(item, dict):
                            url = item.get("url") or item.get("href") or item.get("link") or item.get("file_url")
                            if url:
                                attachments.append(url)

        return attachments

    def _extract_naics_codes(self, raw_data: Dict[str, Any]) -> List[str]:
        """Extract NAICS codes from raw data.

        Args:
            raw_data: Raw listing data

        Returns:
            List of NAICS code strings
        """
        naics_codes = []

        for key in ["naics_codes", "naics", "naics_code", "commodity_codes"]:
            if key in raw_data:
                value = raw_data[key]
                if isinstance(value, list):
                    naics_codes.extend([str(code) for code in value if code])
                elif isinstance(value, (str, int)):
                    naics_codes.append(str(value))

        return naics_codes

    def _parse_value(self, value: Any) -> Optional[float]:
        """Parse a contract value to float.

        Args:
            value: Raw value (string, number, or None)

        Returns:
            Float value or None
        """
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Remove common formatting
            cleaned = value.replace("$", "").replace(",", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return None

        return None
