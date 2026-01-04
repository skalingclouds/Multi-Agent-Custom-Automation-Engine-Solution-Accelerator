# src/rfp_radar/tests/test_scrapers.py
"""
Unit tests for RFP Radar scraper framework.

Tests cover:
- Scrapers registry (SCRAPERS, register_scraper, get_scraper, get_available_sources)
- BaseScraper abstract class interface enforcement
- BaseScraper initialization with default and custom values
- Session configuration and context manager support
- HTTP request handling with retry logic
- RFP filtering by age and geography
- Full scrape pipeline orchestration
- Concrete scraper implementations (GovTribe, OpenGov, BidNet)
- Data parsing and field extraction
- Date parsing from various formats
- Location parsing including nested objects
- Attachment and NAICS code extraction
- ID generation with fallback to hash
- Error handling and edge cases
"""
import os
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch, PropertyMock
import pytest
import requests
from requests.exceptions import RequestException, HTTPError

from rfp_radar.models import (
    RFP,
    RFPSource,
    RFPStatus,
    ScraperResult,
)


# Mock environment variables required for config
MOCK_ENV_VARS = {
    "APP_ENV": "dev",
    "AZURE_STORAGE_ACCOUNT_URL": "https://mockstorageaccount.blob.core.windows.net",
    "AZURE_SEARCH_ENDPOINT": "https://mock-search.search.windows.net",
    "AZURE_OPENAI_ENDPOINT": "https://mock-openai.openai.azure.com",
    "SLACK_BOT_TOKEN": "xoxb-mock-token-12345",
    "RFP_RELEVANCE_THRESHOLD": "0.55",
}


# ==============================================================================
# Scrapers Registry Tests
# ==============================================================================


class TestScrapersRegistry:
    """Tests for the scrapers registry module."""

    @pytest.mark.unit
    def test_scrapers_registry_is_dict(self):
        """Test SCRAPERS is a dictionary."""
        from rfp_radar.scrapers import SCRAPERS

        assert isinstance(SCRAPERS, dict)

    @pytest.mark.unit
    def test_scrapers_registry_has_govtribe(self):
        """Test GovTribe scraper is registered."""
        from rfp_radar.scrapers import SCRAPERS

        assert "govtribe" in SCRAPERS

    @pytest.mark.unit
    def test_scrapers_registry_has_opengov(self):
        """Test OpenGov scraper is registered."""
        from rfp_radar.scrapers import SCRAPERS

        assert "opengov" in SCRAPERS

    @pytest.mark.unit
    def test_scrapers_registry_has_bidnet(self):
        """Test BidNet scraper is registered."""
        from rfp_radar.scrapers import SCRAPERS

        assert "bidnet" in SCRAPERS

    @pytest.mark.unit
    def test_get_scraper_returns_class(self):
        """Test get_scraper returns the correct scraper class."""
        from rfp_radar.scrapers import get_scraper
        from rfp_radar.scrapers.govtribe import GovTribeScraper

        result = get_scraper("govtribe")
        assert result is GovTribeScraper

    @pytest.mark.unit
    def test_get_scraper_case_insensitive(self):
        """Test get_scraper is case insensitive."""
        from rfp_radar.scrapers import get_scraper

        assert get_scraper("GOVTRIBE") is not None
        assert get_scraper("GovTribe") is not None
        assert get_scraper("govtribe") is not None

    @pytest.mark.unit
    def test_get_scraper_returns_none_for_unknown(self):
        """Test get_scraper returns None for unknown sources."""
        from rfp_radar.scrapers import get_scraper

        result = get_scraper("unknown_scraper")
        assert result is None

    @pytest.mark.unit
    def test_get_available_sources_returns_list(self):
        """Test get_available_sources returns a list."""
        from rfp_radar.scrapers import get_available_sources

        result = get_available_sources()
        assert isinstance(result, list)

    @pytest.mark.unit
    def test_get_available_sources_contains_registered(self):
        """Test get_available_sources contains all registered scrapers."""
        from rfp_radar.scrapers import get_available_sources

        sources = get_available_sources()
        assert "govtribe" in sources
        assert "opengov" in sources
        assert "bidnet" in sources

    @pytest.mark.unit
    def test_register_scraper_decorator(self):
        """Test register_scraper decorator registers a class."""
        from rfp_radar.scrapers import register_scraper, SCRAPERS
        from rfp_radar.scrapers.base import BaseScraper

        @register_scraper("test_scraper")
        class TestScraper(BaseScraper):
            source = RFPSource.MANUAL

            def fetch_listings(self, **kwargs):
                return []

            def parse_listing(self, raw_data):
                return None

        assert "test_scraper" in SCRAPERS
        assert SCRAPERS["test_scraper"] is TestScraper

        # Clean up
        del SCRAPERS["test_scraper"]


# ==============================================================================
# BaseScraper Abstract Class Tests
# ==============================================================================


class TestBaseScraperAbstractInterface:
    """Tests for BaseScraper abstract interface enforcement."""

    @pytest.mark.unit
    def test_cannot_instantiate_base_scraper_directly(self):
        """Test BaseScraper cannot be instantiated without implementing abstract methods."""
        from rfp_radar.scrapers.base import BaseScraper

        with pytest.raises(TypeError) as exc_info:
            BaseScraper()

        assert "abstract" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_subclass_must_implement_fetch_listings(self):
        """Test subclass must implement fetch_listings method."""
        from rfp_radar.scrapers.base import BaseScraper

        class IncompleteScraper(BaseScraper):
            def parse_listing(self, raw_data):
                return None

        with pytest.raises(TypeError):
            IncompleteScraper()

    @pytest.mark.unit
    def test_subclass_must_implement_parse_listing(self):
        """Test subclass must implement parse_listing method."""
        from rfp_radar.scrapers.base import BaseScraper

        class IncompleteScraper(BaseScraper):
            def fetch_listings(self, **kwargs):
                return []

        with pytest.raises(TypeError):
            IncompleteScraper()


class ConcreteTestScraper:
    """Helper class to create a concrete test scraper for testing BaseScraper."""

    @staticmethod
    def create():
        """Create a concrete test scraper class."""
        from rfp_radar.scrapers.base import BaseScraper

        class TestScraper(BaseScraper):
            source = RFPSource.MANUAL
            base_url = "https://test.example.com"

            def fetch_listings(self, **kwargs):
                return []

            def parse_listing(self, raw_data):
                if not raw_data:
                    return None
                return RFP(
                    id=raw_data.get("id", "test-id"),
                    title=raw_data.get("title", "Test Title"),
                    country=raw_data.get("country", "US"),
                    posted_date=raw_data.get("posted_date"),
                )

        return TestScraper


class TestBaseScraperInitialization:
    """Tests for BaseScraper initialization."""

    @pytest.mark.unit
    def test_init_with_default_values(self):
        """Test scraper initialization with default values."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        assert scraper.max_age_days == 3
        assert scraper.us_only is True
        assert scraper.session is not None
        assert scraper.logger is not None

    @pytest.mark.unit
    def test_init_with_custom_max_age_days(self):
        """Test scraper initialization with custom max_age_days."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper(max_age_days=7)

        assert scraper.max_age_days == 7

    @pytest.mark.unit
    def test_init_with_custom_us_only(self):
        """Test scraper initialization with custom us_only flag."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper(us_only=False)

        assert scraper.us_only is False

    @pytest.mark.unit
    def test_init_with_custom_logger(self):
        """Test scraper initialization with custom logger."""
        import logging

        TestScraper = ConcreteTestScraper.create()
        custom_logger = logging.getLogger("custom_test_logger")
        scraper = TestScraper(logger=custom_logger)

        assert scraper.logger is custom_logger

    @pytest.mark.unit
    def test_session_has_default_headers(self):
        """Test session is configured with default headers."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        assert "User-Agent" in scraper.session.headers
        assert "NAITIVE-RFP-Radar" in scraper.session.headers["User-Agent"]
        assert scraper.session.headers["Accept"] == "application/json"


class TestBaseScraperContextManager:
    """Tests for BaseScraper context manager support."""

    @pytest.mark.unit
    def test_context_manager_enter_returns_self(self):
        """Test context manager __enter__ returns scraper instance."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        with scraper as ctx:
            assert ctx is scraper

    @pytest.mark.unit
    def test_context_manager_closes_session(self):
        """Test context manager closes session on exit."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()
        scraper.session = MagicMock()

        with scraper:
            pass

        scraper.session.close.assert_called_once()

    @pytest.mark.unit
    def test_close_closes_session(self):
        """Test close() method closes the session."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()
        scraper.session = MagicMock()

        scraper.close()

        scraper.session.close.assert_called_once()


class TestBaseScraperBuildUrl:
    """Tests for BaseScraper.build_url() method."""

    @pytest.mark.unit
    def test_build_url_basic(self):
        """Test build_url constructs URL correctly."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        result = scraper.build_url("/api/v1/opportunities")
        assert result == "https://test.example.com/api/v1/opportunities"

    @pytest.mark.unit
    def test_build_url_strips_slashes(self):
        """Test build_url handles extra slashes."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()
        scraper.base_url = "https://test.example.com/"

        result = scraper.build_url("/api/v1/opportunities")
        assert result == "https://test.example.com/api/v1/opportunities"

    @pytest.mark.unit
    def test_build_url_empty_path(self):
        """Test build_url with empty path."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        result = scraper.build_url("")
        assert result == "https://test.example.com"


class TestBaseScraperGetSourceName:
    """Tests for BaseScraper.get_source_name() class method."""

    @pytest.mark.unit
    def test_get_source_name_returns_lowercase(self):
        """Test get_source_name returns lowercase source name."""
        TestScraper = ConcreteTestScraper.create()

        result = TestScraper.get_source_name()
        assert result == "manual"

    @pytest.mark.unit
    def test_get_source_name_govtribe(self):
        """Test get_source_name for GovTribeScraper."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper

        result = GovTribeScraper.get_source_name()
        assert result == "govtribe"


class TestBaseScraperFilters:
    """Tests for BaseScraper._passes_filters() method."""

    @pytest.mark.unit
    def test_passes_filters_us_rfp_recent(self):
        """Test US-based recent RFP passes filters."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper(max_age_days=3, us_only=True)

        rfp = RFP(
            id="test-1",
            title="Test RFP",
            country="US",
            posted_date=datetime.utcnow() - timedelta(days=1),
        )

        assert scraper._passes_filters(rfp) is True

    @pytest.mark.unit
    def test_passes_filters_non_us_rfp_filtered(self):
        """Test non-US RFP is filtered when us_only=True."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper(max_age_days=3, us_only=True)

        rfp = RFP(
            id="test-1",
            title="Test RFP",
            country="CA",
            posted_date=datetime.utcnow() - timedelta(days=1),
        )

        assert scraper._passes_filters(rfp) is False

    @pytest.mark.unit
    def test_passes_filters_non_us_allowed_when_disabled(self):
        """Test non-US RFP passes when us_only=False."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper(max_age_days=3, us_only=False)

        rfp = RFP(
            id="test-1",
            title="Test RFP",
            country="CA",
            posted_date=datetime.utcnow() - timedelta(days=1),
        )

        assert scraper._passes_filters(rfp) is True

    @pytest.mark.unit
    def test_passes_filters_old_rfp_filtered(self):
        """Test old RFP is filtered by age."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper(max_age_days=3, us_only=True)

        rfp = RFP(
            id="test-1",
            title="Test RFP",
            country="US",
            posted_date=datetime.utcnow() - timedelta(days=5),
        )

        assert scraper._passes_filters(rfp) is False

    @pytest.mark.unit
    def test_passes_filters_no_posted_date_passes(self):
        """Test RFP without posted_date passes age filter."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper(max_age_days=3, us_only=True)

        rfp = RFP(
            id="test-1",
            title="Test RFP",
            country="US",
            posted_date=None,
        )

        assert scraper._passes_filters(rfp) is True


class TestBaseScraperRequestWithRetry:
    """Tests for BaseScraper._request_with_retry() method."""

    @pytest.mark.unit
    def test_request_with_retry_success_first_attempt(self):
        """Test successful request on first attempt."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        scraper.session.request = MagicMock(return_value=mock_response)

        result = scraper._request_with_retry("GET", "https://test.com/api")

        assert result is mock_response
        scraper.session.request.assert_called_once()

    @pytest.mark.unit
    def test_request_with_retry_retries_on_failure(self):
        """Test request retries on failure."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()
        scraper.retry_delay = 0.01  # Fast retry for testing

        mock_success = MagicMock()
        mock_success.raise_for_status = MagicMock()
        scraper.session.request = MagicMock(
            side_effect=[RequestException("Error"), mock_success]
        )

        result = scraper._request_with_retry("GET", "https://test.com/api")

        assert result is mock_success
        assert scraper.session.request.call_count == 2

    @pytest.mark.unit
    def test_request_with_retry_raises_after_max_retries(self):
        """Test request raises exception after max retries."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()
        scraper.max_retries = 2
        scraper.retry_delay = 0.01

        scraper.session.request = MagicMock(
            side_effect=RequestException("Connection error")
        )

        with pytest.raises(RequestException):
            scraper._request_with_retry("GET", "https://test.com/api")

        assert scraper.session.request.call_count == 2

    @pytest.mark.unit
    def test_request_with_retry_uses_timeout(self):
        """Test request uses default timeout."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        scraper.session.request = MagicMock(return_value=mock_response)

        scraper._request_with_retry("GET", "https://test.com/api")

        call_kwargs = scraper.session.request.call_args[1]
        assert call_kwargs["timeout"] == 60

    @pytest.mark.unit
    def test_get_method_uses_request_with_retry(self):
        """Test get() method uses _request_with_retry."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        with patch.object(scraper, "_request_with_retry") as mock_request:
            mock_request.return_value = MagicMock()
            scraper.get("https://test.com/api")

            mock_request.assert_called_once_with("GET", "https://test.com/api")

    @pytest.mark.unit
    def test_post_method_uses_request_with_retry(self):
        """Test post() method uses _request_with_retry."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        with patch.object(scraper, "_request_with_retry") as mock_request:
            mock_request.return_value = MagicMock()
            scraper.post("https://test.com/api", json={"key": "value"})

            mock_request.assert_called_once_with(
                "POST", "https://test.com/api", json={"key": "value"}
            )


class TestBaseScraperScrape:
    """Tests for BaseScraper.scrape() method."""

    @pytest.mark.unit
    def test_scrape_returns_scraper_result(self):
        """Test scrape() returns ScraperResult."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        result = scraper.scrape()

        assert isinstance(result, ScraperResult)
        assert result.source == RFPSource.MANUAL
        assert result.success is True

    @pytest.mark.unit
    def test_scrape_filters_rfps(self):
        """Test scrape() filters RFPs correctly."""
        from rfp_radar.scrapers.base import BaseScraper

        class FilterTestScraper(BaseScraper):
            source = RFPSource.MANUAL

            def fetch_listings(self, **kwargs):
                return [
                    {"id": "1", "title": "US Recent", "country": "US",
                     "posted_date": datetime.utcnow()},
                    {"id": "2", "title": "US Old", "country": "US",
                     "posted_date": datetime.utcnow() - timedelta(days=10)},
                    {"id": "3", "title": "Canada Recent", "country": "CA",
                     "posted_date": datetime.utcnow()},
                ]

            def parse_listing(self, raw_data):
                return RFP(
                    id=raw_data["id"],
                    title=raw_data["title"],
                    country=raw_data["country"],
                    posted_date=raw_data["posted_date"],
                )

        scraper = FilterTestScraper(max_age_days=3, us_only=True)
        result = scraper.scrape()

        assert result.total_found == 3
        assert len(result.rfps) == 1  # Only US Recent passes
        assert result.rfps[0].id == "1"

    @pytest.mark.unit
    def test_scrape_handles_parse_errors(self):
        """Test scrape() handles parse errors gracefully."""
        from rfp_radar.scrapers.base import BaseScraper

        class ErrorScraper(BaseScraper):
            source = RFPSource.MANUAL

            def fetch_listings(self, **kwargs):
                return [
                    {"id": "1", "title": "Good"},
                    {"id": "2"},  # Missing title
                    {"id": "3", "title": "Also Good"},
                ]

            def parse_listing(self, raw_data):
                if "title" not in raw_data:
                    raise ValueError("Missing title")
                return RFP(
                    id=raw_data["id"],
                    title=raw_data["title"],
                    country="US",
                    posted_date=datetime.utcnow(),
                )

        scraper = ErrorScraper()
        result = scraper.scrape()

        assert result.success is True
        assert len(result.rfps) == 2

    @pytest.mark.unit
    def test_scrape_handles_fetch_failure(self):
        """Test scrape() handles fetch failure."""
        from rfp_radar.scrapers.base import BaseScraper

        class FailingScraper(BaseScraper):
            source = RFPSource.MANUAL

            def fetch_listings(self, **kwargs):
                raise RequestException("API Error")

            def parse_listing(self, raw_data):
                return None

        scraper = FailingScraper()
        result = scraper.scrape()

        assert result.success is False
        assert "API Error" in result.error_message
        assert result.rfps == []

    @pytest.mark.unit
    def test_scrape_records_duration(self):
        """Test scrape() records duration."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        result = scraper.scrape()

        assert result.duration_seconds >= 0

    @pytest.mark.unit
    def test_scrape_sets_scraped_at_timestamp(self):
        """Test scrape() sets scraped_at timestamp."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        before = datetime.utcnow()
        result = scraper.scrape()
        after = datetime.utcnow()

        assert before <= result.scraped_at <= after


class TestBaseScraperRepr:
    """Tests for BaseScraper.__repr__() method."""

    @pytest.mark.unit
    def test_repr_contains_class_name(self):
        """Test __repr__ contains class name."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        repr_str = repr(scraper)

        assert "TestScraper" in repr_str

    @pytest.mark.unit
    def test_repr_contains_source(self):
        """Test __repr__ contains source."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper()

        repr_str = repr(scraper)

        assert "manual" in repr_str

    @pytest.mark.unit
    def test_repr_contains_config(self):
        """Test __repr__ contains configuration."""
        TestScraper = ConcreteTestScraper.create()
        scraper = TestScraper(max_age_days=7, us_only=False)

        repr_str = repr(scraper)

        assert "max_age_days=7" in repr_str
        assert "us_only=False" in repr_str


# ==============================================================================
# GovTribeScraper Tests
# ==============================================================================


class TestGovTribeScraperAttributes:
    """Tests for GovTribeScraper class attributes."""

    @pytest.mark.unit
    def test_source_is_govtribe(self):
        """Test GovTribeScraper source is GOVTRIBE."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper

        assert GovTribeScraper.source == RFPSource.GOVTRIBE

    @pytest.mark.unit
    def test_base_url_is_set(self):
        """Test GovTribeScraper has base_url set."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper

        assert GovTribeScraper.base_url == "https://api.govtribe.com/v1"

    @pytest.mark.unit
    def test_has_opportunities_endpoint(self):
        """Test GovTribeScraper has opportunities endpoint."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper

        assert GovTribeScraper.opportunities_endpoint == "/opportunities"


class TestGovTribeScraperParseListing:
    """Tests for GovTribeScraper.parse_listing() method."""

    @pytest.fixture
    def govtribe_scraper(self):
        """Create a GovTribeScraper instance."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper
        return GovTribeScraper()

    @pytest.mark.unit
    def test_parse_listing_basic(self, govtribe_scraper):
        """Test parsing a basic listing."""
        raw_data = {
            "id": "123",
            "title": "Test Opportunity",
            "description": "This is a test.",
            "agency": "Test Agency",
            "posted_date": "2024-01-15",
            "country": "US",
        }

        rfp = govtribe_scraper.parse_listing(raw_data)

        assert rfp is not None
        assert rfp.id == "govtribe-123"
        assert rfp.title == "Test Opportunity"
        assert rfp.description == "This is a test."
        assert rfp.agency == "Test Agency"
        assert rfp.source == RFPSource.GOVTRIBE

    @pytest.mark.unit
    def test_parse_listing_returns_none_for_empty(self, govtribe_scraper):
        """Test parse_listing returns None for empty data."""
        result = govtribe_scraper.parse_listing({})
        # Empty dict may return None due to missing title
        assert result is None

    @pytest.mark.unit
    def test_parse_listing_returns_none_for_none(self, govtribe_scraper):
        """Test parse_listing returns None for None input."""
        result = govtribe_scraper.parse_listing(None)
        assert result is None

    @pytest.mark.unit
    def test_parse_listing_returns_none_without_title(self, govtribe_scraper):
        """Test parse_listing returns None when title is missing."""
        raw_data = {
            "id": "123",
            "description": "No title here",
        }

        result = govtribe_scraper.parse_listing(raw_data)
        assert result is None

    @pytest.mark.unit
    def test_parse_listing_handles_alternate_field_names(self, govtribe_scraper):
        """Test parse_listing handles alternate field names."""
        raw_data = {
            "opportunity_id": "456",
            "name": "Alternate Title Field",
            "synopsis": "Alternate description field",
            "organization": "Alternate agency field",
        }

        rfp = govtribe_scraper.parse_listing(raw_data)

        assert rfp is not None
        assert rfp.title == "Alternate Title Field"
        assert rfp.description == "Alternate description field"
        assert rfp.agency == "Alternate agency field"


class TestGovTribeScraperDateParsing:
    """Tests for GovTribeScraper date parsing."""

    @pytest.fixture
    def govtribe_scraper(self):
        """Create a GovTribeScraper instance."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper
        return GovTribeScraper()

    @pytest.mark.unit
    def test_parse_date_iso_format(self, govtribe_scraper):
        """Test parsing ISO format date."""
        result = govtribe_scraper._parse_date("2024-01-15T10:30:00Z")

        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    @pytest.mark.unit
    def test_parse_date_simple_format(self, govtribe_scraper):
        """Test parsing simple date format."""
        result = govtribe_scraper._parse_date("2024-01-15")

        assert result is not None
        assert result.year == 2024

    @pytest.mark.unit
    def test_parse_date_us_format(self, govtribe_scraper):
        """Test parsing US date format."""
        result = govtribe_scraper._parse_date("01/15/2024")

        assert result is not None
        assert result.month == 1
        assert result.day == 15

    @pytest.mark.unit
    def test_parse_date_returns_none_for_none(self, govtribe_scraper):
        """Test parsing None returns None."""
        result = govtribe_scraper._parse_date(None)
        assert result is None

    @pytest.mark.unit
    def test_parse_date_passes_through_datetime(self, govtribe_scraper):
        """Test parsing datetime passes through."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = govtribe_scraper._parse_date(dt)

        assert result is dt


class TestGovTribeScraperLocationParsing:
    """Tests for GovTribeScraper location parsing."""

    @pytest.fixture
    def govtribe_scraper(self):
        """Create a GovTribeScraper instance."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper
        return GovTribeScraper()

    @pytest.mark.unit
    def test_parse_location_basic(self, govtribe_scraper):
        """Test parsing basic location."""
        raw_data = {
            "location": "Washington",
            "state": "DC",
            "country": "US",
        }

        location, state, country = govtribe_scraper._parse_location(raw_data)

        assert location == "Washington"
        assert state == "DC"
        assert country == "US"

    @pytest.mark.unit
    def test_parse_location_nested_object(self, govtribe_scraper):
        """Test parsing nested location object."""
        raw_data = {
            "location": {
                "city": "New York",
                "state": "NY",
                "country": "US",
            }
        }

        location, state, country = govtribe_scraper._parse_location(raw_data)

        assert location == "New York"
        assert state == "NY"
        assert country == "US"

    @pytest.mark.unit
    def test_parse_location_defaults_to_us(self, govtribe_scraper):
        """Test location defaults to US country."""
        raw_data = {}

        location, state, country = govtribe_scraper._parse_location(raw_data)

        assert country == "US"


class TestGovTribeScraperExtractListings:
    """Tests for GovTribeScraper._extract_listings() method."""

    @pytest.fixture
    def govtribe_scraper(self):
        """Create a GovTribeScraper instance."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper
        return GovTribeScraper()

    @pytest.mark.unit
    def test_extract_listings_from_list(self, govtribe_scraper):
        """Test extracting listings from list response."""
        data = [{"id": "1"}, {"id": "2"}]

        result = govtribe_scraper._extract_listings(data)

        assert result == data

    @pytest.mark.unit
    def test_extract_listings_from_dict_wrapper(self, govtribe_scraper):
        """Test extracting listings from dict wrapper."""
        data = {
            "opportunities": [{"id": "1"}, {"id": "2"}],
            "meta": {"total": 2}
        }

        result = govtribe_scraper._extract_listings(data)

        assert result == [{"id": "1"}, {"id": "2"}]

    @pytest.mark.unit
    def test_extract_listings_from_results_key(self, govtribe_scraper):
        """Test extracting listings from 'results' key."""
        data = {
            "results": [{"id": "1"}],
        }

        result = govtribe_scraper._extract_listings(data)

        assert result == [{"id": "1"}]

    @pytest.mark.unit
    def test_extract_listings_single_opportunity(self, govtribe_scraper):
        """Test extracting single opportunity."""
        data = {"id": "1", "title": "Single Opportunity"}

        result = govtribe_scraper._extract_listings(data)

        assert result == [data]

    @pytest.mark.unit
    def test_extract_listings_empty_response(self, govtribe_scraper):
        """Test extracting from empty response."""
        data = {}

        result = govtribe_scraper._extract_listings(data)

        assert result == []


class TestGovTribeScraperExtractField:
    """Tests for GovTribeScraper._extract_field() method."""

    @pytest.fixture
    def govtribe_scraper(self):
        """Create a GovTribeScraper instance."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper
        return GovTribeScraper()

    @pytest.mark.unit
    def test_extract_field_first_key(self, govtribe_scraper):
        """Test extracting with first key."""
        data = {"title": "Test Title"}

        result = govtribe_scraper._extract_field(data, ["title", "name"])

        assert result == "Test Title"

    @pytest.mark.unit
    def test_extract_field_second_key(self, govtribe_scraper):
        """Test extracting with second key."""
        data = {"name": "Test Name"}

        result = govtribe_scraper._extract_field(data, ["title", "name"])

        assert result == "Test Name"

    @pytest.mark.unit
    def test_extract_field_default(self, govtribe_scraper):
        """Test extracting returns default."""
        data = {}

        result = govtribe_scraper._extract_field(
            data, ["title", "name"], default="Unknown"
        )

        assert result == "Unknown"

    @pytest.mark.unit
    def test_extract_field_skips_none_values(self, govtribe_scraper):
        """Test extracting skips None values."""
        data = {"title": None, "name": "Actual Name"}

        result = govtribe_scraper._extract_field(data, ["title", "name"])

        assert result == "Actual Name"


class TestGovTribeScraperAttachments:
    """Tests for GovTribeScraper attachment extraction."""

    @pytest.fixture
    def govtribe_scraper(self):
        """Create a GovTribeScraper instance."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper
        return GovTribeScraper()

    @pytest.mark.unit
    def test_extract_attachments_string_list(self, govtribe_scraper):
        """Test extracting attachments from string list."""
        raw_data = {
            "attachments": ["https://example.com/doc1.pdf", "https://example.com/doc2.pdf"]
        }

        result = govtribe_scraper._extract_attachments(raw_data)

        assert len(result) == 2
        assert "https://example.com/doc1.pdf" in result

    @pytest.mark.unit
    def test_extract_attachments_dict_list(self, govtribe_scraper):
        """Test extracting attachments from dict list."""
        raw_data = {
            "documents": [
                {"url": "https://example.com/doc1.pdf"},
                {"href": "https://example.com/doc2.pdf"},
            ]
        }

        result = govtribe_scraper._extract_attachments(raw_data)

        assert len(result) == 2

    @pytest.mark.unit
    def test_extract_attachments_empty(self, govtribe_scraper):
        """Test extracting from empty data."""
        result = govtribe_scraper._extract_attachments({})

        assert result == []


class TestGovTribeScraperNaicsCodes:
    """Tests for GovTribeScraper NAICS code extraction."""

    @pytest.fixture
    def govtribe_scraper(self):
        """Create a GovTribeScraper instance."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper
        return GovTribeScraper()

    @pytest.mark.unit
    def test_extract_naics_list(self, govtribe_scraper):
        """Test extracting NAICS from list."""
        raw_data = {"naics_codes": ["541512", "541519"]}

        result = govtribe_scraper._extract_naics_codes(raw_data)

        assert result == ["541512", "541519"]

    @pytest.mark.unit
    def test_extract_naics_single_value(self, govtribe_scraper):
        """Test extracting single NAICS value."""
        raw_data = {"naics": "541512"}

        result = govtribe_scraper._extract_naics_codes(raw_data)

        assert result == ["541512"]

    @pytest.mark.unit
    def test_extract_naics_integer(self, govtribe_scraper):
        """Test extracting NAICS from integer."""
        raw_data = {"naics": 541512}

        result = govtribe_scraper._extract_naics_codes(raw_data)

        assert result == ["541512"]


class TestGovTribeScraperValueParsing:
    """Tests for GovTribeScraper value parsing."""

    @pytest.fixture
    def govtribe_scraper(self):
        """Create a GovTribeScraper instance."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper
        return GovTribeScraper()

    @pytest.mark.unit
    def test_parse_value_float(self, govtribe_scraper):
        """Test parsing float value."""
        result = govtribe_scraper._parse_value(1500000.50)

        assert result == 1500000.50

    @pytest.mark.unit
    def test_parse_value_int(self, govtribe_scraper):
        """Test parsing integer value."""
        result = govtribe_scraper._parse_value(1500000)

        assert result == 1500000.0

    @pytest.mark.unit
    def test_parse_value_string_with_formatting(self, govtribe_scraper):
        """Test parsing formatted string value."""
        result = govtribe_scraper._parse_value("$1,500,000.00")

        assert result == 1500000.00

    @pytest.mark.unit
    def test_parse_value_none(self, govtribe_scraper):
        """Test parsing None returns None."""
        result = govtribe_scraper._parse_value(None)

        assert result is None

    @pytest.mark.unit
    def test_parse_value_invalid_string(self, govtribe_scraper):
        """Test parsing invalid string returns None."""
        result = govtribe_scraper._parse_value("not a number")

        assert result is None


class TestGovTribeScraperIdGeneration:
    """Tests for GovTribeScraper ID generation."""

    @pytest.fixture
    def govtribe_scraper(self):
        """Create a GovTribeScraper instance."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper
        return GovTribeScraper()

    @pytest.mark.unit
    def test_generate_id_from_id_field(self, govtribe_scraper):
        """Test generating ID from 'id' field."""
        raw_data = {"id": "12345"}

        result = govtribe_scraper._generate_id(raw_data)

        assert result == "govtribe-12345"

    @pytest.mark.unit
    def test_generate_id_from_solicitation_number(self, govtribe_scraper):
        """Test generating ID from solicitation_number."""
        raw_data = {"solicitation_number": "ABC-2024-001"}

        result = govtribe_scraper._generate_id(raw_data)

        assert result == "govtribe-ABC-2024-001"

    @pytest.mark.unit
    def test_generate_id_fallback_to_hash(self, govtribe_scraper):
        """Test generating ID falls back to hash."""
        raw_data = {"title": "Unique Title"}

        result = govtribe_scraper._generate_id(raw_data)

        assert result.startswith("govtribe-")
        assert len(result) > len("govtribe-")


# ==============================================================================
# OpenGovScraper Tests
# ==============================================================================


class TestOpenGovScraperAttributes:
    """Tests for OpenGovScraper class attributes."""

    @pytest.mark.unit
    def test_source_is_opengov(self):
        """Test OpenGovScraper source is OPENGOV."""
        from rfp_radar.scrapers.opengov import OpenGovScraper

        assert OpenGovScraper.source == RFPSource.OPENGOV

    @pytest.mark.unit
    def test_base_url_is_set(self):
        """Test OpenGovScraper has base_url set."""
        from rfp_radar.scrapers.opengov import OpenGovScraper

        assert OpenGovScraper.base_url == "https://api.opengov.com/v1"

    @pytest.mark.unit
    def test_has_bids_endpoint(self):
        """Test OpenGovScraper has bids endpoint."""
        from rfp_radar.scrapers.opengov import OpenGovScraper

        assert OpenGovScraper.opportunities_endpoint == "/bids"


class TestOpenGovScraperParseListing:
    """Tests for OpenGovScraper.parse_listing() method."""

    @pytest.fixture
    def opengov_scraper(self):
        """Create an OpenGovScraper instance."""
        from rfp_radar.scrapers.opengov import OpenGovScraper
        return OpenGovScraper()

    @pytest.mark.unit
    def test_parse_listing_basic(self, opengov_scraper):
        """Test parsing a basic listing."""
        raw_data = {
            "bid_id": "456",
            "bid_title": "Test Bid",
            "summary": "This is a test bid.",
            "buyer": "Test City",
        }

        rfp = opengov_scraper.parse_listing(raw_data)

        assert rfp is not None
        assert rfp.id == "opengov-456"
        assert rfp.title == "Test Bid"
        assert rfp.description == "This is a test bid."
        assert rfp.source == RFPSource.OPENGOV

    @pytest.mark.unit
    def test_parse_listing_handles_opengov_date_format(self, opengov_scraper):
        """Test parsing OpenGov-specific date format."""
        raw_data = {
            "title": "Test",
            "posted_date": "January 15, 2024",
        }

        rfp = opengov_scraper.parse_listing(raw_data)

        assert rfp is not None
        assert rfp.posted_date is not None
        assert rfp.posted_date.month == 1
        assert rfp.posted_date.day == 15


class TestOpenGovScraperExtractListings:
    """Tests for OpenGovScraper._extract_listings() method."""

    @pytest.fixture
    def opengov_scraper(self):
        """Create an OpenGovScraper instance."""
        from rfp_radar.scrapers.opengov import OpenGovScraper
        return OpenGovScraper()

    @pytest.mark.unit
    def test_extract_listings_from_bids_key(self, opengov_scraper):
        """Test extracting listings from 'bids' key."""
        data = {"bids": [{"id": "1"}, {"id": "2"}]}

        result = opengov_scraper._extract_listings(data)

        assert result == [{"id": "1"}, {"id": "2"}]

    @pytest.mark.unit
    def test_extract_listings_single_bid(self, opengov_scraper):
        """Test extracting single bid."""
        data = {"bid_number": "2024-001", "title": "Single Bid"}

        result = opengov_scraper._extract_listings(data)

        assert result == [data]


# ==============================================================================
# BidNetScraper Tests
# ==============================================================================


class TestBidNetScraperAttributes:
    """Tests for BidNetScraper class attributes."""

    @pytest.mark.unit
    def test_source_is_bidnet(self):
        """Test BidNetScraper source is BIDNET."""
        from rfp_radar.scrapers.bidnet import BidNetScraper

        assert BidNetScraper.source == RFPSource.BIDNET

    @pytest.mark.unit
    def test_base_url_is_set(self):
        """Test BidNetScraper has base_url set."""
        from rfp_radar.scrapers.bidnet import BidNetScraper

        assert BidNetScraper.base_url == "https://api.bidnet.com/v1"

    @pytest.mark.unit
    def test_has_solicitations_endpoint(self):
        """Test BidNetScraper has solicitations endpoint."""
        from rfp_radar.scrapers.bidnet import BidNetScraper

        assert BidNetScraper.opportunities_endpoint == "/solicitations"


class TestBidNetScraperParseListing:
    """Tests for BidNetScraper.parse_listing() method."""

    @pytest.fixture
    def bidnet_scraper(self):
        """Create a BidNetScraper instance."""
        from rfp_radar.scrapers.bidnet import BidNetScraper
        return BidNetScraper()

    @pytest.mark.unit
    def test_parse_listing_basic(self, bidnet_scraper):
        """Test parsing a basic listing."""
        raw_data = {
            "solicitation_id": "789",
            "solicitation_title": "Test Solicitation",
            "overview": "This is a test solicitation.",
            "purchasing_agency": "Test County",
        }

        rfp = bidnet_scraper.parse_listing(raw_data)

        assert rfp is not None
        assert rfp.id == "bidnet-789"
        assert rfp.title == "Test Solicitation"
        assert rfp.description == "This is a test solicitation."
        assert rfp.source == RFPSource.BIDNET

    @pytest.mark.unit
    def test_parse_listing_handles_bidnet_date_format(self, bidnet_scraper):
        """Test parsing BidNet-specific date formats."""
        raw_data = {
            "title": "Test",
            "posted_date": "01/15/2024 02:30 PM",
        }

        rfp = bidnet_scraper.parse_listing(raw_data)

        assert rfp is not None
        assert rfp.posted_date is not None
        assert rfp.posted_date.month == 1

    @pytest.mark.unit
    def test_parse_listing_abbreviated_month_format(self, bidnet_scraper):
        """Test parsing abbreviated month format."""
        raw_data = {
            "title": "Test",
            "due_date": "Jan 15, 2024",
        }

        rfp = bidnet_scraper.parse_listing(raw_data)

        assert rfp is not None
        assert rfp.due_date is not None


class TestBidNetScraperExtractListings:
    """Tests for BidNetScraper._extract_listings() method."""

    @pytest.fixture
    def bidnet_scraper(self):
        """Create a BidNetScraper instance."""
        from rfp_radar.scrapers.bidnet import BidNetScraper
        return BidNetScraper()

    @pytest.mark.unit
    def test_extract_listings_from_solicitations_key(self, bidnet_scraper):
        """Test extracting listings from 'solicitations' key."""
        data = {"solicitations": [{"id": "1"}, {"id": "2"}]}

        result = bidnet_scraper._extract_listings(data)

        assert result == [{"id": "1"}, {"id": "2"}]

    @pytest.mark.unit
    def test_extract_listings_single_solicitation(self, bidnet_scraper):
        """Test extracting single solicitation."""
        data = {"solicitation_number": "2024-001", "title": "Single"}

        result = bidnet_scraper._extract_listings(data)

        assert result == [data]


class TestBidNetScraperAttachments:
    """Tests for BidNetScraper attachment extraction."""

    @pytest.fixture
    def bidnet_scraper(self):
        """Create a BidNetScraper instance."""
        from rfp_radar.scrapers.bidnet import BidNetScraper
        return BidNetScraper()

    @pytest.mark.unit
    def test_extract_attachments_with_download_url(self, bidnet_scraper):
        """Test extracting attachments with download_url field."""
        raw_data = {
            "solicitation_documents": [
                {"download_url": "https://example.com/doc1.pdf"},
            ]
        }

        result = bidnet_scraper._extract_attachments(raw_data)

        assert len(result) == 1
        assert result[0] == "https://example.com/doc1.pdf"


class TestBidNetScraperQueryParams:
    """Tests for BidNetScraper query parameter building."""

    @pytest.fixture
    def bidnet_scraper(self):
        """Create a BidNetScraper instance."""
        from rfp_radar.scrapers.bidnet import BidNetScraper
        return BidNetScraper()

    @pytest.mark.unit
    def test_build_query_params_basic(self, bidnet_scraper):
        """Test building basic query parameters."""
        params = bidnet_scraper._build_query_params(page=1, page_size=50)

        assert params["page"] == 1
        assert params["limit"] == 50
        assert params["sort"] == "-publish_date"
        assert params["status"] == "active"

    @pytest.mark.unit
    def test_build_query_params_with_agency_type(self, bidnet_scraper):
        """Test building query parameters with agency_type."""
        params = bidnet_scraper._build_query_params(
            page=1, page_size=50, agency_type="federal"
        )

        assert params["agency_type"] == "federal"

    @pytest.mark.unit
    def test_build_query_params_with_state(self, bidnet_scraper):
        """Test building query parameters with state."""
        params = bidnet_scraper._build_query_params(
            page=1, page_size=50, state="CA"
        )

        assert params["state"] == "CA"


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestScraperIntegration:
    """Integration tests for the scraper framework."""

    @pytest.mark.unit
    def test_all_registered_scrapers_are_instantiable(self):
        """Test all registered scrapers can be instantiated."""
        from rfp_radar.scrapers import SCRAPERS

        for name, scraper_class in SCRAPERS.items():
            scraper = scraper_class()
            assert scraper is not None
            assert hasattr(scraper, "scrape")
            scraper.close()

    @pytest.mark.unit
    def test_all_scrapers_have_correct_source(self):
        """Test all scrapers have correct source attribute."""
        from rfp_radar.scrapers import SCRAPERS

        expected_sources = {
            "govtribe": RFPSource.GOVTRIBE,
            "opengov": RFPSource.OPENGOV,
            "bidnet": RFPSource.BIDNET,
        }

        for name, scraper_class in SCRAPERS.items():
            if name in expected_sources:
                assert scraper_class.source == expected_sources[name]

    @pytest.mark.unit
    def test_all_scrapers_inherit_from_base(self):
        """Test all scrapers inherit from BaseScraper."""
        from rfp_radar.scrapers import SCRAPERS
        from rfp_radar.scrapers.base import BaseScraper

        for name, scraper_class in SCRAPERS.items():
            assert issubclass(scraper_class, BaseScraper)

    @pytest.mark.unit
    def test_scraper_context_manager_workflow(self):
        """Test typical scraper workflow with context manager."""
        from rfp_radar.scrapers.govtribe import GovTribeScraper

        with GovTribeScraper(max_age_days=7, us_only=False) as scraper:
            assert scraper.max_age_days == 7
            assert scraper.us_only is False
            assert scraper.session is not None

    @pytest.mark.unit
    def test_mock_scrape_workflow(self):
        """Test complete mock scrape workflow."""
        from rfp_radar.scrapers.base import BaseScraper

        class MockScraper(BaseScraper):
            source = RFPSource.MANUAL

            def fetch_listings(self, **kwargs):
                return [
                    {"id": "1", "title": "AI Project", "country": "US",
                     "posted_date": datetime.utcnow().isoformat()},
                    {"id": "2", "title": "Cloud Migration", "country": "US",
                     "posted_date": datetime.utcnow().isoformat()},
                ]

            def parse_listing(self, raw_data):
                posted_date = raw_data.get("posted_date")
                if isinstance(posted_date, str):
                    posted_date = datetime.fromisoformat(posted_date)
                return RFP(
                    id=raw_data["id"],
                    title=raw_data["title"],
                    country=raw_data.get("country", "US"),
                    posted_date=posted_date,
                )

        with MockScraper(max_age_days=3, us_only=True) as scraper:
            result = scraper.scrape()

            assert result.success is True
            assert result.total_found == 2
            assert len(result.rfps) == 2
            assert result.source == RFPSource.MANUAL
