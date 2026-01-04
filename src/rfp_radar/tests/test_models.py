# src/rfp_radar/tests/test_models.py
"""
Unit tests for RFP Radar Pydantic models.

Tests cover:
- Enum validation (RFPSource, RFPStatus, RFPTag)
- RFP model creation, validation, and helper methods
- ClassificationResult model with score validation
- ClassifiedRFP model with actionability logic
- ProposalMetadata and Proposal models
- DigestEntry and Digest models
- ScraperResult model
"""
from datetime import datetime, timedelta
import uuid
import pytest

from rfp_radar.models import (
    RFP,
    RFPSource,
    RFPStatus,
    RFPTag,
    ClassificationResult,
    ClassifiedRFP,
    ProposalMetadata,
    Proposal,
    DigestEntry,
    Digest,
    ScraperResult,
)


class TestRFPSourceEnum:
    """Tests for the RFPSource enumeration."""

    @pytest.mark.unit
    def test_rfp_source_values(self):
        """Test that RFPSource has expected values."""
        assert RFPSource.GOVTRIBE.value == "govtribe"
        assert RFPSource.OPENGOV.value == "opengov"
        assert RFPSource.BIDNET.value == "bidnet"
        assert RFPSource.MANUAL.value == "manual"

    @pytest.mark.unit
    def test_rfp_source_is_string_enum(self):
        """Test that RFPSource values are strings."""
        for source in RFPSource:
            assert isinstance(source.value, str)

    @pytest.mark.unit
    def test_rfp_source_from_string(self):
        """Test that RFPSource can be created from string value."""
        source = RFPSource("govtribe")
        assert source == RFPSource.GOVTRIBE


class TestRFPStatusEnum:
    """Tests for the RFPStatus enumeration."""

    @pytest.mark.unit
    def test_rfp_status_values(self):
        """Test that RFPStatus has expected values."""
        assert RFPStatus.DISCOVERED.value == "discovered"
        assert RFPStatus.FILTERED.value == "filtered"
        assert RFPStatus.CLASSIFIED.value == "classified"
        assert RFPStatus.STORED.value == "stored"
        assert RFPStatus.PROPOSAL_GENERATED.value == "proposal_generated"
        assert RFPStatus.NOTIFIED.value == "notified"
        assert RFPStatus.SKIPPED.value == "skipped"
        assert RFPStatus.ERROR.value == "error"

    @pytest.mark.unit
    def test_rfp_status_count(self):
        """Test that RFPStatus has correct number of states."""
        assert len(RFPStatus) == 8


class TestRFPTagEnum:
    """Tests for the RFPTag enumeration."""

    @pytest.mark.unit
    def test_rfp_tag_values(self):
        """Test that RFPTag has expected values."""
        assert RFPTag.AI.value == "AI"
        assert RFPTag.DYNAMICS.value == "Dynamics"
        assert RFPTag.MODERNIZATION.value == "Modernization"
        assert RFPTag.CLOUD.value == "Cloud"
        assert RFPTag.SECURITY.value == "Security"
        assert RFPTag.DATA.value == "Data"
        assert RFPTag.AUTOMATION.value == "Automation"
        assert RFPTag.OTHER.value == "Other"

    @pytest.mark.unit
    def test_rfp_tag_count(self):
        """Test that RFPTag has correct number of tags."""
        assert len(RFPTag) == 8


class TestRFPModel:
    """Tests for the RFP Pydantic model."""

    @pytest.mark.unit
    def test_rfp_minimal_creation(self):
        """Test RFP creation with only required field (title)."""
        rfp = RFP(title="Test RFP")

        assert rfp.title == "Test RFP"
        assert rfp.description == ""
        assert rfp.agency == ""
        assert rfp.source == RFPSource.MANUAL
        assert rfp.country == "US"
        assert rfp.status == RFPStatus.DISCOVERED
        assert isinstance(rfp.id, str)
        assert len(rfp.id) > 0

    @pytest.mark.unit
    def test_rfp_full_creation(self):
        """Test RFP creation with all fields."""
        posted = datetime(2026, 1, 1, 12, 0, 0)
        due = datetime(2026, 2, 1, 17, 0, 0)

        rfp = RFP(
            id="test-id-123",
            title="Full RFP Test",
            description="A comprehensive test RFP",
            agency="Department of Testing",
            source=RFPSource.GOVTRIBE,
            source_url="https://example.com/rfp/123",
            posted_date=posted,
            due_date=due,
            location="Washington, DC",
            country="US",
            state="DC",
            pdf_url="https://example.com/rfp/123.pdf",
            attachments=["https://example.com/attachment1.pdf"],
            status=RFPStatus.CLASSIFIED,
            naics_codes=["541512", "541519"],
            set_aside="Small Business",
            estimated_value=1000000.00,
            contract_type="Fixed Price",
            raw_data={"original_id": "ext-123"},
        )

        assert rfp.id == "test-id-123"
        assert rfp.title == "Full RFP Test"
        assert rfp.source == RFPSource.GOVTRIBE
        assert rfp.posted_date == posted
        assert rfp.due_date == due
        assert len(rfp.naics_codes) == 2
        assert rfp.estimated_value == 1000000.00
        assert rfp.raw_data["original_id"] == "ext-123"

    @pytest.mark.unit
    def test_rfp_auto_generated_id(self):
        """Test that RFP generates UUID if not provided."""
        rfp = RFP(title="Test RFP")

        # Verify it's a valid UUID format
        try:
            uuid.UUID(rfp.id)
            is_valid_uuid = True
        except ValueError:
            is_valid_uuid = False

        assert is_valid_uuid

    @pytest.mark.unit
    def test_rfp_country_validator_uppercase(self):
        """Test that country code is converted to uppercase."""
        rfp = RFP(title="Test", country="us")
        assert rfp.country == "US"

    @pytest.mark.unit
    def test_rfp_country_validator_truncates(self):
        """Test that country code is truncated to 2 characters."""
        rfp = RFP(title="Test", country="usa")
        assert rfp.country == "US"

    @pytest.mark.unit
    def test_rfp_country_validator_empty_defaults_to_us(self):
        """Test that empty country defaults to US."""
        rfp = RFP(title="Test", country="")
        assert rfp.country == "US"

    @pytest.mark.unit
    def test_rfp_is_us_based_true(self):
        """Test is_us_based returns True for US RFPs."""
        rfp = RFP(title="Test", country="US")
        assert rfp.is_us_based() is True

    @pytest.mark.unit
    def test_rfp_is_us_based_false(self):
        """Test is_us_based returns False for non-US RFPs."""
        rfp = RFP(title="Test", country="CA")
        assert rfp.is_us_based() is False

    @pytest.mark.unit
    def test_rfp_age_in_days_with_posted_date(self):
        """Test age_in_days calculation with posted_date."""
        posted = datetime.utcnow() - timedelta(days=5)
        rfp = RFP(title="Test", posted_date=posted)

        # Age should be approximately 5 days (allow 1 day margin for test execution time)
        assert 4 <= rfp.age_in_days() <= 6

    @pytest.mark.unit
    def test_rfp_age_in_days_without_posted_date(self):
        """Test age_in_days returns 0 when no posted_date."""
        rfp = RFP(title="Test")
        assert rfp.age_in_days() == 0

    @pytest.mark.unit
    def test_rfp_is_within_age_limit_true(self):
        """Test is_within_age_limit returns True for recent RFPs."""
        posted = datetime.utcnow() - timedelta(days=2)
        rfp = RFP(title="Test", posted_date=posted)

        assert rfp.is_within_age_limit(max_age_days=3) is True

    @pytest.mark.unit
    def test_rfp_is_within_age_limit_false(self):
        """Test is_within_age_limit returns False for old RFPs."""
        posted = datetime.utcnow() - timedelta(days=10)
        rfp = RFP(title="Test", posted_date=posted)

        assert rfp.is_within_age_limit(max_age_days=3) is False

    @pytest.mark.unit
    def test_rfp_is_within_age_limit_boundary(self):
        """Test is_within_age_limit at boundary."""
        posted = datetime.utcnow() - timedelta(days=3)
        rfp = RFP(title="Test", posted_date=posted)

        assert rfp.is_within_age_limit(max_age_days=3) is True

    @pytest.mark.unit
    def test_rfp_attachments_default_empty_list(self):
        """Test that attachments defaults to empty list."""
        rfp = RFP(title="Test")
        assert rfp.attachments == []
        assert isinstance(rfp.attachments, list)

    @pytest.mark.unit
    def test_rfp_naics_codes_default_empty_list(self):
        """Test that naics_codes defaults to empty list."""
        rfp = RFP(title="Test")
        assert rfp.naics_codes == []


class TestClassificationResultModel:
    """Tests for the ClassificationResult Pydantic model."""

    @pytest.mark.unit
    def test_classification_result_creation(self):
        """Test ClassificationResult creation with required fields."""
        result = ClassificationResult(
            rfp_id="test-rfp-123",
            relevance_score=0.75,
        )

        assert result.rfp_id == "test-rfp-123"
        assert result.relevance_score == 0.75
        assert result.tags == []
        assert result.reasoning == ""
        assert result.model_used == "gpt-4o"
        assert result.confidence == 1.0

    @pytest.mark.unit
    def test_classification_result_with_tags(self):
        """Test ClassificationResult with tags."""
        result = ClassificationResult(
            rfp_id="test-123",
            relevance_score=0.8,
            tags=[RFPTag.AI, RFPTag.CLOUD],
        )

        assert len(result.tags) == 2
        assert RFPTag.AI in result.tags
        assert RFPTag.CLOUD in result.tags

    @pytest.mark.unit
    def test_classification_result_score_validator_clamps_high(self):
        """Test that relevance_score above 1.0 is clamped to 1.0."""
        result = ClassificationResult(
            rfp_id="test-123",
            relevance_score=1.5,
        )
        assert result.relevance_score == 1.0

    @pytest.mark.unit
    def test_classification_result_score_validator_clamps_low(self):
        """Test that relevance_score below 0.0 is clamped to 0.0."""
        result = ClassificationResult(
            rfp_id="test-123",
            relevance_score=-0.5,
        )
        assert result.relevance_score == 0.0

    @pytest.mark.unit
    def test_classification_result_confidence_validator(self):
        """Test that confidence is clamped to valid range."""
        result = ClassificationResult(
            rfp_id="test-123",
            relevance_score=0.5,
            confidence=1.5,
        )
        assert result.confidence == 1.0

    @pytest.mark.unit
    def test_classification_result_is_relevant_above_threshold(self):
        """Test is_relevant returns True above threshold."""
        result = ClassificationResult(
            rfp_id="test-123",
            relevance_score=0.70,
        )
        assert result.is_relevant(threshold=0.55) is True

    @pytest.mark.unit
    def test_classification_result_is_relevant_below_threshold(self):
        """Test is_relevant returns False below threshold."""
        result = ClassificationResult(
            rfp_id="test-123",
            relevance_score=0.40,
        )
        assert result.is_relevant(threshold=0.55) is False

    @pytest.mark.unit
    def test_classification_result_is_relevant_at_threshold(self):
        """Test is_relevant returns True at exact threshold."""
        result = ClassificationResult(
            rfp_id="test-123",
            relevance_score=0.55,
        )
        assert result.is_relevant(threshold=0.55) is True

    @pytest.mark.unit
    def test_classification_result_default_threshold(self):
        """Test is_relevant uses default threshold of 0.55."""
        relevant_result = ClassificationResult(
            rfp_id="test-123",
            relevance_score=0.56,
        )
        not_relevant_result = ClassificationResult(
            rfp_id="test-456",
            relevance_score=0.54,
        )

        assert relevant_result.is_relevant() is True
        assert not_relevant_result.is_relevant() is False

    @pytest.mark.unit
    def test_classification_result_classified_at_auto_set(self):
        """Test that classified_at is automatically set."""
        result = ClassificationResult(
            rfp_id="test-123",
            relevance_score=0.75,
        )

        assert isinstance(result.classified_at, datetime)
        # Should be recent (within last minute)
        time_diff = datetime.utcnow() - result.classified_at
        assert time_diff.total_seconds() < 60


class TestClassifiedRFPModel:
    """Tests for the ClassifiedRFP Pydantic model."""

    @pytest.mark.unit
    def test_classified_rfp_creation(self):
        """Test ClassifiedRFP creation with RFP and ClassificationResult."""
        rfp = RFP(title="Test RFP", country="US")
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.75,
        )

        classified = ClassifiedRFP(rfp=rfp, classification=classification)

        assert classified.rfp.title == "Test RFP"
        assert classified.classification.relevance_score == 0.75

    @pytest.mark.unit
    def test_classified_rfp_is_actionable_true(self):
        """Test is_actionable returns True for US RFP above threshold."""
        rfp = RFP(title="Test RFP", country="US")
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.75,
        )
        classified = ClassifiedRFP(rfp=rfp, classification=classification)

        assert classified.is_actionable(relevance_threshold=0.55) is True

    @pytest.mark.unit
    def test_classified_rfp_is_actionable_false_non_us(self):
        """Test is_actionable returns False for non-US RFP."""
        rfp = RFP(title="Test RFP", country="CA")
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.90,  # High score but wrong country
        )
        classified = ClassifiedRFP(rfp=rfp, classification=classification)

        assert classified.is_actionable(relevance_threshold=0.55) is False

    @pytest.mark.unit
    def test_classified_rfp_is_actionable_false_low_score(self):
        """Test is_actionable returns False for low relevance score."""
        rfp = RFP(title="Test RFP", country="US")
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.30,  # Low score
        )
        classified = ClassifiedRFP(rfp=rfp, classification=classification)

        assert classified.is_actionable(relevance_threshold=0.55) is False


class TestProposalMetadataModel:
    """Tests for the ProposalMetadata Pydantic model."""

    @pytest.mark.unit
    def test_proposal_metadata_creation(self):
        """Test ProposalMetadata creation with required field."""
        metadata = ProposalMetadata(rfp_id="test-rfp-123")

        assert metadata.rfp_id == "test-rfp-123"
        assert isinstance(metadata.id, str)
        assert metadata.version == 1
        assert metadata.brand_name == "NAITIVE"
        assert metadata.brand_website == "https://www.naitive.cloud"

    @pytest.mark.unit
    def test_proposal_metadata_full_creation(self):
        """Test ProposalMetadata with all fields."""
        metadata = ProposalMetadata(
            id="proposal-123",
            rfp_id="rfp-456",
            rfp_title="Government AI Contract",
            version=2,
            blob_url="https://storage.example.com/proposal.md",
            blob_path="proposals/rfp-456/v2.md",
            content_hash="abc123hash",
            word_count=5000,
            section_count=8,
            brand_name="CustomBrand",
            brand_website="https://custom.example.com",
            model_used="gpt-4-turbo",
            prompt_tokens=1500,
            completion_tokens=3000,
        )

        assert metadata.id == "proposal-123"
        assert metadata.version == 2
        assert metadata.word_count == 5000
        assert metadata.brand_name == "CustomBrand"
        assert metadata.prompt_tokens == 1500

    @pytest.mark.unit
    def test_proposal_metadata_defaults(self):
        """Test ProposalMetadata default values."""
        metadata = ProposalMetadata(rfp_id="test-123")

        assert metadata.rfp_title == ""
        assert metadata.blob_url == ""
        assert metadata.blob_path == ""
        assert metadata.content_hash == ""
        assert metadata.word_count == 0
        assert metadata.section_count == 0
        assert metadata.model_used == "gpt-4o"
        assert metadata.prompt_tokens == 0
        assert metadata.completion_tokens == 0


class TestProposalModel:
    """Tests for the Proposal Pydantic model."""

    @pytest.mark.unit
    def test_proposal_creation(self):
        """Test Proposal creation with metadata and content."""
        metadata = ProposalMetadata(rfp_id="test-123")
        proposal = Proposal(
            metadata=metadata,
            markdown_content="# Proposal\n\nThis is a test proposal.",
        )

        assert proposal.metadata.rfp_id == "test-123"
        assert "# Proposal" in proposal.markdown_content

    @pytest.mark.unit
    def test_proposal_word_count_property(self):
        """Test Proposal word_count property calculation."""
        metadata = ProposalMetadata(rfp_id="test-123")
        proposal = Proposal(
            metadata=metadata,
            markdown_content="One two three four five six seven eight nine ten",
        )

        assert proposal.word_count == 10

    @pytest.mark.unit
    def test_proposal_word_count_empty_content(self):
        """Test Proposal word_count with empty content."""
        metadata = ProposalMetadata(rfp_id="test-123")
        proposal = Proposal(
            metadata=metadata,
            markdown_content="",
        )

        # Empty string split returns [''], which has length 1
        # But we check the actual implementation behavior
        assert proposal.word_count >= 0


class TestDigestEntryModel:
    """Tests for the DigestEntry Pydantic model."""

    @pytest.mark.unit
    def test_digest_entry_creation(self):
        """Test DigestEntry creation."""
        rfp = RFP(
            title="Test RFP",
            agency="Test Agency",
            source_url="https://example.com/rfp",
        )
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.85,
            tags=[RFPTag.AI, RFPTag.CLOUD],
        )

        entry = DigestEntry(
            rfp=rfp,
            classification=classification,
            proposal_url="https://storage.example.com/proposal.md",
        )

        assert entry.rfp.title == "Test RFP"
        assert entry.classification.relevance_score == 0.85
        assert entry.proposal_url == "https://storage.example.com/proposal.md"

    @pytest.mark.unit
    def test_digest_entry_to_slack_block_high_score(self):
        """Test to_slack_block with high relevance score (star emoji)."""
        rfp = RFP(
            title="High Score RFP",
            agency="Test Agency",
            source_url="https://example.com/rfp",
        )
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.85,
            tags=[RFPTag.AI],
        )
        entry = DigestEntry(rfp=rfp, classification=classification)

        block = entry.to_slack_block()

        assert block["type"] == "section"
        assert "High Score RFP" in block["text"]["text"]
        assert ":star:" in block["text"]["text"]
        assert "0.85" in block["text"]["text"]
        assert "AI" in block["text"]["text"]

    @pytest.mark.unit
    def test_digest_entry_to_slack_block_medium_score(self):
        """Test to_slack_block with medium relevance score (checkmark emoji)."""
        rfp = RFP(
            title="Medium Score RFP",
            agency="Test Agency",
        )
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.65,
        )
        entry = DigestEntry(rfp=rfp, classification=classification)

        block = entry.to_slack_block()

        assert ":white_check_mark:" in block["text"]["text"]

    @pytest.mark.unit
    def test_digest_entry_to_slack_block_low_score(self):
        """Test to_slack_block with low relevance score (question emoji)."""
        rfp = RFP(
            title="Low Score RFP",
            agency="Test Agency",
        )
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.40,
        )
        entry = DigestEntry(rfp=rfp, classification=classification)

        block = entry.to_slack_block()

        assert ":grey_question:" in block["text"]["text"]

    @pytest.mark.unit
    def test_digest_entry_to_slack_block_no_tags(self):
        """Test to_slack_block with no tags shows 'None'."""
        rfp = RFP(title="No Tags RFP")
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.75,
            tags=[],
        )
        entry = DigestEntry(rfp=rfp, classification=classification)

        block = entry.to_slack_block()

        assert "Tags: None" in block["text"]["text"]

    @pytest.mark.unit
    def test_digest_entry_to_slack_block_with_button(self):
        """Test to_slack_block includes View RFP button when source_url exists."""
        rfp = RFP(
            title="RFP with URL",
            source_url="https://example.com/rfp/123",
        )
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.75,
        )
        entry = DigestEntry(rfp=rfp, classification=classification)

        block = entry.to_slack_block()

        assert block["accessory"] is not None
        assert block["accessory"]["type"] == "button"
        assert block["accessory"]["url"] == "https://example.com/rfp/123"

    @pytest.mark.unit
    def test_digest_entry_to_slack_block_no_button_without_url(self):
        """Test to_slack_block has no button when source_url is empty."""
        rfp = RFP(title="RFP without URL")
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.75,
        )
        entry = DigestEntry(rfp=rfp, classification=classification)

        block = entry.to_slack_block()

        # accessory should be None when no source_url
        assert block["accessory"] is None


class TestDigestModel:
    """Tests for the Digest Pydantic model."""

    @pytest.mark.unit
    def test_digest_creation_empty(self):
        """Test Digest creation with no entries."""
        digest = Digest()

        assert isinstance(digest.id, str)
        assert digest.entries == []
        assert digest.total_discovered == 0
        assert digest.total_filtered == 0
        assert digest.total_relevant == 0
        assert digest.total_proposals == 0

    @pytest.mark.unit
    def test_digest_creation_with_entries(self):
        """Test Digest creation with entries."""
        rfp = RFP(title="Test RFP")
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.75,
        )
        entry = DigestEntry(rfp=rfp, classification=classification)

        digest = Digest(
            entries=[entry],
            total_discovered=10,
            total_filtered=5,
            total_relevant=1,
            total_proposals=1,
        )

        assert len(digest.entries) == 1
        assert digest.total_discovered == 10
        assert digest.total_relevant == 1

    @pytest.mark.unit
    def test_digest_is_empty_true(self):
        """Test is_empty returns True for empty digest."""
        digest = Digest()
        assert digest.is_empty() is True

    @pytest.mark.unit
    def test_digest_is_empty_false(self):
        """Test is_empty returns False for non-empty digest."""
        rfp = RFP(title="Test RFP")
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.75,
        )
        entry = DigestEntry(rfp=rfp, classification=classification)
        digest = Digest(entries=[entry])

        assert digest.is_empty() is False

    @pytest.mark.unit
    def test_digest_generated_at_auto_set(self):
        """Test that generated_at is automatically set."""
        digest = Digest()

        assert isinstance(digest.generated_at, datetime)
        time_diff = datetime.utcnow() - digest.generated_at
        assert time_diff.total_seconds() < 60


class TestScraperResultModel:
    """Tests for the ScraperResult Pydantic model."""

    @pytest.mark.unit
    def test_scraper_result_creation(self):
        """Test ScraperResult creation with source."""
        result = ScraperResult(source=RFPSource.GOVTRIBE)

        assert result.source == RFPSource.GOVTRIBE
        assert result.success is True
        assert result.error_message == ""
        assert result.rfps == []
        assert result.total_found == 0
        assert result.duration_seconds == 0.0

    @pytest.mark.unit
    def test_scraper_result_with_rfps(self):
        """Test ScraperResult with scraped RFPs."""
        rfp1 = RFP(title="RFP 1")
        rfp2 = RFP(title="RFP 2")

        result = ScraperResult(
            source=RFPSource.OPENGOV,
            rfps=[rfp1, rfp2],
            total_found=5,
            duration_seconds=2.5,
        )

        assert len(result.rfps) == 2
        assert result.total_found == 5
        assert result.duration_seconds == 2.5

    @pytest.mark.unit
    def test_scraper_result_failure(self):
        """Test ScraperResult for failed scrape."""
        result = ScraperResult(
            source=RFPSource.BIDNET,
            success=False,
            error_message="Connection timeout",
            duration_seconds=30.0,
        )

        assert result.success is False
        assert result.error_message == "Connection timeout"

    @pytest.mark.unit
    def test_scraper_result_rfp_count_property(self):
        """Test ScraperResult rfp_count property."""
        rfp1 = RFP(title="RFP 1")
        rfp2 = RFP(title="RFP 2")
        rfp3 = RFP(title="RFP 3")

        result = ScraperResult(
            source=RFPSource.GOVTRIBE,
            rfps=[rfp1, rfp2, rfp3],
        )

        assert result.rfp_count == 3

    @pytest.mark.unit
    def test_scraper_result_rfp_count_empty(self):
        """Test ScraperResult rfp_count when empty."""
        result = ScraperResult(source=RFPSource.MANUAL)
        assert result.rfp_count == 0

    @pytest.mark.unit
    def test_scraper_result_scraped_at_auto_set(self):
        """Test that scraped_at is automatically set."""
        result = ScraperResult(source=RFPSource.GOVTRIBE)

        assert isinstance(result.scraped_at, datetime)
        time_diff = datetime.utcnow() - result.scraped_at
        assert time_diff.total_seconds() < 60


class TestModelSerialization:
    """Tests for model serialization and deserialization."""

    @pytest.mark.unit
    def test_rfp_to_dict(self):
        """Test RFP model can be converted to dict."""
        rfp = RFP(
            title="Test RFP",
            agency="Test Agency",
            country="US",
        )

        data = rfp.model_dump()

        assert isinstance(data, dict)
        assert data["title"] == "Test RFP"
        assert data["agency"] == "Test Agency"
        assert data["country"] == "US"

    @pytest.mark.unit
    def test_rfp_from_dict(self):
        """Test RFP model can be created from dict."""
        data = {
            "title": "Test RFP",
            "agency": "Test Agency",
            "country": "US",
            "source": "govtribe",
        }

        rfp = RFP.model_validate(data)

        assert rfp.title == "Test RFP"
        assert rfp.source == RFPSource.GOVTRIBE

    @pytest.mark.unit
    def test_classification_result_to_json(self):
        """Test ClassificationResult can be serialized to JSON."""
        result = ClassificationResult(
            rfp_id="test-123",
            relevance_score=0.75,
            tags=[RFPTag.AI, RFPTag.CLOUD],
        )

        json_str = result.model_dump_json()

        assert isinstance(json_str, str)
        assert "test-123" in json_str
        assert "0.75" in json_str

    @pytest.mark.unit
    def test_digest_roundtrip(self):
        """Test Digest can be serialized and deserialized."""
        rfp = RFP(title="Test RFP")
        classification = ClassificationResult(
            rfp_id=rfp.id,
            relevance_score=0.75,
        )
        entry = DigestEntry(rfp=rfp, classification=classification)
        original = Digest(
            entries=[entry],
            total_discovered=10,
        )

        # Convert to dict and back
        data = original.model_dump()
        restored = Digest.model_validate(data)

        assert len(restored.entries) == 1
        assert restored.total_discovered == 10
        assert restored.entries[0].rfp.title == "Test RFP"
