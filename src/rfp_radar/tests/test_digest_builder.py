# src/rfp_radar/tests/test_digest_builder.py
"""
Unit tests for RFP Radar digest builder module.

Tests cover:
- DigestBuilder initialization with custom and default values
- Digest building from classified RFPs and proposals via build_digest()
- Convenience method build_digest_from_results() for ProposalGenerator output
- Empty digest building via build_empty_digest()
- Slack Block Kit formatting via format_slack_blocks()
- Plain text fallback formatting via format_fallback_text()
- Single entry formatting via format_entry_text()
- Score-based emoji selection via get_score_emoji()
- All private block building methods
- Error notification formatting
- Statistics and configuration retrieval
- Context manager support
- Edge cases and boundary conditions
"""
import os
from datetime import datetime, timedelta
from unittest.mock import patch
import pytest

from rfp_radar.models import (
    RFP,
    RFPSource,
    RFPTag,
    ClassificationResult,
    ClassifiedRFP,
    Proposal,
    ProposalMetadata,
    Digest,
    DigestEntry,
)


# Mock environment variables required for config
MOCK_ENV_VARS = {
    "APP_ENV": "dev",
    "AZURE_STORAGE_ACCOUNT_URL": "https://mockstorageaccount.blob.core.windows.net",
    "AZURE_SEARCH_ENDPOINT": "https://mock-search.search.windows.net",
    "AZURE_OPENAI_ENDPOINT": "https://mock-openai.openai.azure.com",
    "SLACK_BOT_TOKEN": "xoxb-mock-token-12345",
    "NAITIVE_BRAND_NAME": "NAITIVE",
    "NAITIVE_WEBSITE": "https://www.naitive.cloud",
}


@pytest.fixture
def sample_rfp():
    """Create a sample RFP for testing."""
    return RFP(
        id="test-rfp-123",
        title="AI-Powered Government Analytics Platform",
        description="Seeking a contractor to develop an AI-powered analytics platform.",
        agency="Department of Testing",
        source=RFPSource.GOVTRIBE,
        source_url="https://govtribe.com/rfp/test-rfp-123",
        country="US",
        naics_codes=["541512", "541519"],
        estimated_value=1500000.00,
        contract_type="Fixed Price",
        set_aside="Small Business",
        location="Washington, DC",
        due_date=datetime.utcnow() + timedelta(days=30),
    )


@pytest.fixture
def sample_rfp_minimal():
    """Create a minimal RFP for testing."""
    return RFP(
        id="test-rfp-minimal",
        title="Basic RFP",
        description="A simple RFP description.",
    )


@pytest.fixture
def sample_classification():
    """Create a sample classification result for testing."""
    return ClassificationResult(
        rfp_id="test-rfp-123",
        relevance_score=0.85,
        tags=[RFPTag.AI, RFPTag.CLOUD, RFPTag.DATA],
        reasoning="This RFP aligns well with AI and cloud capabilities.",
        confidence=0.9,
        model_used="gpt-4o",
    )


@pytest.fixture
def sample_classification_high():
    """Create a high-score classification for testing."""
    return ClassificationResult(
        rfp_id="test-rfp-high",
        relevance_score=0.92,
        tags=[RFPTag.AI, RFPTag.AUTOMATION],
        reasoning="Excellent fit.",
    )


@pytest.fixture
def sample_classification_medium():
    """Create a medium-score classification for testing."""
    return ClassificationResult(
        rfp_id="test-rfp-medium",
        relevance_score=0.65,
        tags=[RFPTag.CLOUD],
        reasoning="Moderate fit.",
    )


@pytest.fixture
def sample_classification_low():
    """Create a low-score classification for testing."""
    return ClassificationResult(
        rfp_id="test-rfp-low",
        relevance_score=0.45,
        tags=[],
        reasoning="Low relevance.",
    )


@pytest.fixture
def sample_classified_rfp(sample_rfp, sample_classification):
    """Create a sample ClassifiedRFP for testing."""
    return ClassifiedRFP(
        rfp=sample_rfp,
        classification=sample_classification,
    )


@pytest.fixture
def sample_proposal():
    """Create a sample proposal for testing."""
    return Proposal(
        metadata=ProposalMetadata(
            rfp_id="test-rfp-123",
            rfp_title="AI-Powered Government Analytics Platform",
            blob_url="https://mockstorageaccount.blob.core.windows.net/rfp-radar/proposals/test-rfp-123.md",
            brand_name="NAITIVE",
            brand_website="https://www.naitive.cloud",
        ),
        markdown_content="# Proposal\n\n## Executive Summary\n\nContent here.",
    )


class TestDigestBuilderInitialization:
    """Tests for DigestBuilder initialization."""

    @pytest.mark.unit
    def test_init_with_default_values(self):
        """Test builder initialization with default values."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()

            assert builder.brand_name == "NAITIVE"
            assert builder.brand_website == "https://www.naitive.cloud"
            assert builder.max_entries_per_digest == 10

    @pytest.mark.unit
    def test_init_with_custom_branding(self):
        """Test builder initialization with custom branding."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder(
                brand_name="CustomBrand",
                brand_website="https://custom.example.com",
            )

            assert builder.brand_name == "CustomBrand"
            assert builder.brand_website == "https://custom.example.com"

    @pytest.mark.unit
    def test_init_with_custom_max_entries(self):
        """Test builder initialization with custom max entries."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder(max_entries=5)

            assert builder.max_entries_per_digest == 5

    @pytest.mark.unit
    def test_init_with_zero_max_entries(self):
        """Test builder initialization with zero max entries."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder(max_entries=0)

            assert builder.max_entries_per_digest == 0


class TestDigestBuilderBuildDigest:
    """Tests for DigestBuilder.build_digest() method."""

    @pytest.mark.unit
    def test_build_digest_basic(self, sample_classified_rfp):
        """Test building a basic digest."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_digest(
                classified_rfps=[sample_classified_rfp],
                total_discovered=10,
                total_filtered=5,
            )

            assert isinstance(digest, Digest)
            assert len(digest.entries) == 1
            assert digest.total_discovered == 10
            assert digest.total_filtered == 5
            assert digest.total_relevant == 1
            assert digest.total_proposals == 0

    @pytest.mark.unit
    def test_build_digest_with_proposals(self, sample_classified_rfp, sample_proposal):
        """Test building digest with proposals."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            proposals = {sample_classified_rfp.rfp.id: sample_proposal}

            digest = builder.build_digest(
                classified_rfps=[sample_classified_rfp],
                proposals=proposals,
            )

            assert digest.total_proposals == 1
            assert digest.entries[0].proposal_url is not None
            assert "mockstorageaccount" in digest.entries[0].proposal_url

    @pytest.mark.unit
    def test_build_digest_empty_list(self):
        """Test building digest with empty RFP list."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_digest(classified_rfps=[])

            assert digest.is_empty()
            assert len(digest.entries) == 0
            assert digest.total_relevant == 0

    @pytest.mark.unit
    def test_build_digest_sorts_by_score(self):
        """Test digest entries are sorted by relevance score descending."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            classified_rfps = [
                ClassifiedRFP(
                    rfp=RFP(id=f"rfp-{i}", title=f"RFP {i}"),
                    classification=ClassificationResult(
                        rfp_id=f"rfp-{i}",
                        relevance_score=score,
                    ),
                )
                for i, score in enumerate([0.5, 0.9, 0.7])
            ]

            builder = DigestBuilder()
            digest = builder.build_digest(classified_rfps=classified_rfps)

            scores = [e.classification.relevance_score for e in digest.entries]
            assert scores == [0.9, 0.7, 0.5]

    @pytest.mark.unit
    def test_build_digest_without_proposals(self, sample_classified_rfp):
        """Test building digest without proposals returns None proposal_url."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_digest(
                classified_rfps=[sample_classified_rfp],
                proposals=None,
            )

            assert digest.entries[0].proposal_url is None


class TestDigestBuilderBuildDigestFromResults:
    """Tests for DigestBuilder.build_digest_from_results() method."""

    @pytest.mark.unit
    def test_build_digest_from_results_basic(
        self, sample_classified_rfp, sample_proposal
    ):
        """Test building digest from proposal generation results."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            results = [(sample_classified_rfp, sample_proposal)]

            builder = DigestBuilder()
            digest = builder.build_digest_from_results(
                results=results,
                total_discovered=20,
                total_filtered=10,
            )

            assert len(digest.entries) == 1
            assert digest.total_discovered == 20
            assert digest.total_filtered == 10
            assert digest.total_proposals == 1
            assert digest.entries[0].proposal_url is not None

    @pytest.mark.unit
    def test_build_digest_from_results_empty(self):
        """Test building digest from empty results."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_digest_from_results(results=[])

            assert digest.is_empty()
            assert digest.total_proposals == 0

    @pytest.mark.unit
    def test_build_digest_from_results_multiple(self):
        """Test building digest from multiple results."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            results = [
                (
                    ClassifiedRFP(
                        rfp=RFP(id=f"rfp-{i}", title=f"RFP {i}"),
                        classification=ClassificationResult(
                            rfp_id=f"rfp-{i}",
                            relevance_score=0.8,
                        ),
                    ),
                    Proposal(
                        metadata=ProposalMetadata(
                            rfp_id=f"rfp-{i}",
                            rfp_title=f"RFP {i}",
                            blob_url=f"https://storage.blob.core.windows.net/proposals/rfp-{i}.md",
                        ),
                        markdown_content="# Proposal",
                    ),
                )
                for i in range(5)
            ]

            builder = DigestBuilder()
            digest = builder.build_digest_from_results(results=results)

            assert len(digest.entries) == 5
            assert digest.total_proposals == 5


class TestDigestBuilderBuildEmptyDigest:
    """Tests for DigestBuilder.build_empty_digest() method."""

    @pytest.mark.unit
    def test_build_empty_digest(self):
        """Test building an empty digest."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_empty_digest(
                total_discovered=50,
                total_filtered=25,
            )

            assert digest.is_empty()
            assert digest.total_discovered == 50
            assert digest.total_filtered == 25
            assert digest.total_relevant == 0
            assert digest.total_proposals == 0

    @pytest.mark.unit
    def test_build_empty_digest_defaults(self):
        """Test building empty digest with default values."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_empty_digest()

            assert digest.is_empty()
            assert digest.total_discovered == 0
            assert digest.total_filtered == 0


class TestDigestBuilderFormatSlackBlocks:
    """Tests for DigestBuilder.format_slack_blocks() method."""

    @pytest.mark.unit
    def test_format_slack_blocks_non_empty_digest(self, sample_classified_rfp):
        """Test formatting non-empty digest as Slack blocks."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_digest(classified_rfps=[sample_classified_rfp])
            blocks = builder.format_slack_blocks(digest)

            assert isinstance(blocks, list)
            assert len(blocks) > 0

            # Verify header block
            header_block = blocks[0]
            assert header_block["type"] == "header"
            assert "RFP Radar" in header_block["text"]["text"]

    @pytest.mark.unit
    def test_format_slack_blocks_empty_digest(self):
        """Test formatting empty digest as Slack blocks."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_empty_digest()
            blocks = builder.format_slack_blocks(digest)

            assert isinstance(blocks, list)
            assert len(blocks) > 0

            # Should have header, empty summary, divider, footer
            block_types = [b["type"] for b in blocks]
            assert "header" in block_types
            assert "section" in block_types
            assert "context" in block_types

    @pytest.mark.unit
    def test_format_slack_blocks_includes_dividers(self, sample_classified_rfp):
        """Test Slack blocks include dividers."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_digest(classified_rfps=[sample_classified_rfp])
            blocks = builder.format_slack_blocks(digest)

            divider_count = sum(1 for b in blocks if b.get("type") == "divider")
            assert divider_count >= 1

    @pytest.mark.unit
    def test_format_slack_blocks_truncates_entries(self):
        """Test Slack blocks truncate entries beyond max."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            # Create 15 classified RFPs
            classified_rfps = [
                ClassifiedRFP(
                    rfp=RFP(id=f"rfp-{i}", title=f"Test RFP {i}"),
                    classification=ClassificationResult(
                        rfp_id=f"rfp-{i}",
                        relevance_score=0.8,
                    ),
                )
                for i in range(15)
            ]

            builder = DigestBuilder(max_entries=5)
            digest = builder.build_digest(classified_rfps=classified_rfps)
            blocks = builder.format_slack_blocks(digest)

            # Should include truncation notice
            block_texts = []
            for block in blocks:
                if block.get("type") == "context":
                    elements = block.get("elements", [])
                    for elem in elements:
                        block_texts.append(elem.get("text", ""))

            assert any("more RFPs" in text for text in block_texts)

    @pytest.mark.unit
    def test_format_slack_blocks_includes_proposal_links(
        self, sample_classified_rfp, sample_proposal
    ):
        """Test Slack blocks include proposal links when available."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            proposals = {sample_classified_rfp.rfp.id: sample_proposal}

            builder = DigestBuilder()
            digest = builder.build_digest(
                classified_rfps=[sample_classified_rfp],
                proposals=proposals,
            )
            blocks = builder.format_slack_blocks(digest)

            # Look for proposal link in context blocks
            proposal_link_found = False
            for block in blocks:
                if block.get("type") == "context":
                    elements = block.get("elements", [])
                    for elem in elements:
                        text = elem.get("text", "")
                        if "View Proposal" in text:
                            proposal_link_found = True

            assert proposal_link_found


class TestDigestBuilderFormatFallbackText:
    """Tests for DigestBuilder.format_fallback_text() method."""

    @pytest.mark.unit
    def test_format_fallback_text_non_empty(self, sample_classified_rfp):
        """Test fallback text for non-empty digest."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_digest(
                classified_rfps=[sample_classified_rfp],
                total_discovered=10,
            )
            text = builder.format_fallback_text(digest)

            assert "NAITIVE" in text
            assert "RFP Radar" in text
            assert "1" in text  # 1 relevant RFP

    @pytest.mark.unit
    def test_format_fallback_text_empty(self):
        """Test fallback text for empty digest."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_empty_digest(total_discovered=50, total_filtered=20)
            text = builder.format_fallback_text(digest)

            assert "NAITIVE" in text
            assert "No new relevant RFPs" in text
            assert "50" in text
            assert "20" in text

    @pytest.mark.unit
    def test_format_fallback_text_with_proposals(
        self, sample_classified_rfp, sample_proposal
    ):
        """Test fallback text includes proposal count."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            proposals = {sample_classified_rfp.rfp.id: sample_proposal}

            builder = DigestBuilder()
            digest = builder.build_digest(
                classified_rfps=[sample_classified_rfp],
                proposals=proposals,
            )
            text = builder.format_fallback_text(digest)

            assert "1 proposals" in text or "1 relevant" in text


class TestDigestBuilderFormatEntryText:
    """Tests for DigestBuilder.format_entry_text() method."""

    @pytest.mark.unit
    def test_format_entry_text_full(self, sample_rfp, sample_classification):
        """Test formatting entry with all fields."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            entry = DigestEntry(
                rfp=sample_rfp,
                classification=sample_classification,
                proposal_url="https://storage.blob.core.windows.net/proposals/test.md",
            )

            builder = DigestBuilder()
            text = builder.format_entry_text(entry)

            assert sample_rfp.title in text
            assert "85%" in text  # Score formatted as percentage
            assert "AI" in text
            assert sample_rfp.agency in text
            assert "Proposal:" in text

    @pytest.mark.unit
    def test_format_entry_text_minimal(self, sample_rfp_minimal):
        """Test formatting entry with minimal fields."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            classification = ClassificationResult(
                rfp_id=sample_rfp_minimal.id,
                relevance_score=0.6,
                tags=[],
            )
            entry = DigestEntry(
                rfp=sample_rfp_minimal,
                classification=classification,
            )

            builder = DigestBuilder()
            text = builder.format_entry_text(entry)

            assert sample_rfp_minimal.title in text
            assert "Tags: None" in text
            assert "Proposal:" not in text

    @pytest.mark.unit
    def test_format_entry_text_with_due_date(self, sample_rfp, sample_classification):
        """Test formatting entry includes due date."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            entry = DigestEntry(
                rfp=sample_rfp,
                classification=sample_classification,
            )

            builder = DigestBuilder()
            text = builder.format_entry_text(entry)

            assert "Due:" in text


class TestDigestBuilderGetScoreEmoji:
    """Tests for DigestBuilder.get_score_emoji() method."""

    @pytest.mark.unit
    def test_get_score_emoji_high(self):
        """Test high score returns star emoji."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()

            assert builder.get_score_emoji(0.95) == ":star:"
            assert builder.get_score_emoji(0.8) == ":star:"

    @pytest.mark.unit
    def test_get_score_emoji_medium(self):
        """Test medium score returns check emoji."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()

            assert builder.get_score_emoji(0.79) == ":white_check_mark:"
            assert builder.get_score_emoji(0.55) == ":white_check_mark:"

    @pytest.mark.unit
    def test_get_score_emoji_low(self):
        """Test low score returns question emoji."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()

            assert builder.get_score_emoji(0.54) == ":grey_question:"
            assert builder.get_score_emoji(0.0) == ":grey_question:"

    @pytest.mark.unit
    def test_get_score_emoji_boundary_values(self):
        """Test emoji selection at exact boundaries."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()

            # At high threshold (0.8)
            assert builder.get_score_emoji(0.8) == ":star:"
            assert builder.get_score_emoji(0.799) == ":white_check_mark:"

            # At medium threshold (0.55)
            assert builder.get_score_emoji(0.55) == ":white_check_mark:"
            assert builder.get_score_emoji(0.549) == ":grey_question:"


class TestDigestBuilderPrivateMethods:
    """Tests for DigestBuilder private helper methods."""

    @pytest.mark.unit
    def test_build_header_block(self):
        """Test _build_header_block creates valid header."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            block = builder._build_header_block()

            assert block["type"] == "header"
            assert "NAITIVE" in block["text"]["text"]
            assert "RFP Radar" in block["text"]["text"]
            assert block["text"]["emoji"] is True

    @pytest.mark.unit
    def test_build_summary_block(self, sample_classified_rfp):
        """Test _build_summary_block for non-empty digest."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_digest(
                classified_rfps=[sample_classified_rfp],
                total_discovered=100,
                total_filtered=50,
            )
            block = builder._build_summary_block(digest)

            assert block["type"] == "section"
            text = block["text"]["text"]
            assert "Found" in text
            assert "100" in text
            assert "50" in text

    @pytest.mark.unit
    def test_build_empty_summary_block(self):
        """Test _build_empty_summary_block for empty digest."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            digest = builder.build_empty_digest(total_discovered=100, total_filtered=50)
            block = builder._build_empty_summary_block(digest)

            assert block["type"] == "section"
            text = block["text"]["text"]
            assert "No new relevant RFPs" in text
            assert "100" in text

    @pytest.mark.unit
    def test_build_entry_block_with_source_url(self, sample_rfp, sample_classification):
        """Test _build_entry_block includes View RFP button when source_url exists."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            entry = DigestEntry(rfp=sample_rfp, classification=sample_classification)

            builder = DigestBuilder()
            block = builder._build_entry_block(entry)

            assert block["type"] == "section"
            assert "accessory" in block
            assert block["accessory"]["type"] == "button"
            assert block["accessory"]["url"] == sample_rfp.source_url

    @pytest.mark.unit
    def test_build_entry_block_without_source_url(self, sample_classification):
        """Test _build_entry_block without button when no source_url."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            rfp = RFP(id="no-url", title="RFP without URL")
            entry = DigestEntry(rfp=rfp, classification=sample_classification)

            builder = DigestBuilder()
            block = builder._build_entry_block(entry)

            assert "accessory" not in block

    @pytest.mark.unit
    def test_build_entry_block_truncates_long_title(self, sample_classification):
        """Test _build_entry_block truncates long titles."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            long_title = "A" * 150
            rfp = RFP(id="long-title", title=long_title)
            entry = DigestEntry(rfp=rfp, classification=sample_classification)

            builder = DigestBuilder()
            block = builder._build_entry_block(entry)

            text = block["text"]["text"]
            # Title should be truncated to 100 chars + "..."
            assert "..." in text
            assert len(text.split("\n")[0]) <= 110

    @pytest.mark.unit
    def test_build_proposal_link_block(self, sample_rfp, sample_classification):
        """Test _build_proposal_link_block creates valid context block."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            entry = DigestEntry(
                rfp=sample_rfp,
                classification=sample_classification,
                proposal_url="https://storage.blob.core.windows.net/proposals/test.md",
            )

            builder = DigestBuilder()
            block = builder._build_proposal_link_block(entry)

            assert block["type"] == "context"
            assert len(block["elements"]) > 0
            assert "View Proposal" in block["elements"][0]["text"]

    @pytest.mark.unit
    def test_build_truncation_notice(self):
        """Test _build_truncation_notice shows remaining count."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            block = builder._build_truncation_notice(5)

            assert block["type"] == "context"
            assert "5 more" in block["elements"][0]["text"]

    @pytest.mark.unit
    def test_build_footer_block(self):
        """Test _build_footer_block includes branding."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            block = builder._build_footer_block()

            assert block["type"] == "context"
            text = block["elements"][0]["text"]
            assert "NAITIVE" in text
            assert "naitive.cloud" in text


class TestDigestBuilderFormatErrorBlocks:
    """Tests for DigestBuilder.format_error_blocks() method."""

    @pytest.mark.unit
    def test_format_error_blocks_basic(self):
        """Test formatting basic error blocks."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            blocks = builder.format_error_blocks("Connection timeout")

            assert isinstance(blocks, list)
            assert len(blocks) == 3

            # Check header
            assert blocks[0]["type"] == "header"
            assert "Error" in blocks[0]["text"]["text"]
            assert "NAITIVE" in blocks[0]["text"]["text"]

            # Check error message in code block
            assert "Connection timeout" in blocks[1]["text"]["text"]

    @pytest.mark.unit
    def test_format_error_blocks_custom_type(self):
        """Test formatting error blocks with custom type."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            blocks = builder.format_error_blocks(
                "Failed to scrape portal",
                error_type="Scraping Error",
            )

            assert "Scraping Error" in blocks[0]["text"]["text"]

    @pytest.mark.unit
    def test_format_error_blocks_truncates_long_message(self):
        """Test error blocks truncate long messages."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            long_message = "X" * 5000
            builder = DigestBuilder()
            blocks = builder.format_error_blocks(long_message)

            # Message should be truncated to 2500 chars
            message_text = blocks[1]["text"]["text"]
            assert len(message_text) <= 2510  # Account for code block markers


class TestDigestBuilderFormatErrorFallback:
    """Tests for DigestBuilder.format_error_fallback() method."""

    @pytest.mark.unit
    def test_format_error_fallback_basic(self):
        """Test formatting error fallback text."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            text = builder.format_error_fallback("Connection failed")

            assert "NAITIVE" in text
            assert "RFP Radar" in text
            assert "Connection failed" in text

    @pytest.mark.unit
    def test_format_error_fallback_custom_type(self):
        """Test formatting error fallback with custom type."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()
            text = builder.format_error_fallback(
                "API rate limited",
                error_type="Rate Limit Error",
            )

            assert "Rate Limit Error" in text

    @pytest.mark.unit
    def test_format_error_fallback_truncates_long_message(self):
        """Test error fallback truncates long messages."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            long_message = "Y" * 500
            builder = DigestBuilder()
            text = builder.format_error_fallback(long_message)

            # Message should be truncated to 200 chars
            assert len(text) < 300


class TestDigestBuilderGetStats:
    """Tests for DigestBuilder.get_stats() method."""

    @pytest.mark.unit
    def test_get_stats_returns_config(self):
        """Test get_stats returns builder configuration."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder(
                brand_name="TestBrand",
                brand_website="https://test.example.com",
                max_entries=7,
            )
            stats = builder.get_stats()

            assert stats["brand_name"] == "TestBrand"
            assert stats["brand_website"] == "https://test.example.com"
            assert stats["max_entries_per_digest"] == 7
            assert "score_thresholds" in stats
            assert stats["score_thresholds"]["high"] == 0.8
            assert stats["score_thresholds"]["medium"] == 0.55


class TestDigestBuilderContextManager:
    """Tests for DigestBuilder context manager support."""

    @pytest.mark.unit
    def test_context_manager_enter_returns_self(self):
        """Test context manager __enter__ returns builder instance."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            builder = DigestBuilder()

            with builder as ctx:
                assert ctx is builder

    @pytest.mark.unit
    def test_context_manager_works_correctly(self, sample_classified_rfp):
        """Test context manager can be used for building digests."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            with DigestBuilder() as builder:
                digest = builder.build_digest(
                    classified_rfps=[sample_classified_rfp]
                )
                assert len(digest.entries) == 1


class TestDigestBuilderClassAttributes:
    """Tests for DigestBuilder class attributes."""

    @pytest.mark.unit
    def test_default_max_entries(self):
        """Test DEFAULT_MAX_ENTRIES constant."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            assert DigestBuilder.DEFAULT_MAX_ENTRIES == 10

    @pytest.mark.unit
    def test_score_thresholds(self):
        """Test score threshold constants."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            assert DigestBuilder.SCORE_HIGH_THRESHOLD == 0.8
            assert DigestBuilder.SCORE_MEDIUM_THRESHOLD == 0.55

    @pytest.mark.unit
    def test_emoji_constants(self):
        """Test emoji constants."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            assert DigestBuilder.EMOJI_HIGH == ":star:"
            assert DigestBuilder.EMOJI_MEDIUM == ":white_check_mark:"
            assert DigestBuilder.EMOJI_LOW == ":grey_question:"
            assert DigestBuilder.EMOJI_RADAR == ":radar:"


class TestDigestBuilderIntegration:
    """Integration-style tests for DigestBuilder."""

    @pytest.mark.unit
    def test_full_digest_workflow(self):
        """Test complete digest building workflow."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            # Create test data
            classified_rfps = [
                ClassifiedRFP(
                    rfp=RFP(
                        id=f"rfp-{i}",
                        title=f"Test RFP {i}",
                        agency=f"Agency {i}",
                        source_url=f"https://example.com/rfp/{i}",
                    ),
                    classification=ClassificationResult(
                        rfp_id=f"rfp-{i}",
                        relevance_score=0.9 - (i * 0.1),
                        tags=[RFPTag.AI],
                    ),
                )
                for i in range(3)
            ]

            proposals = {
                "rfp-0": Proposal(
                    metadata=ProposalMetadata(
                        rfp_id="rfp-0",
                        blob_url="https://storage.blob.core.windows.net/rfp-0.md",
                    ),
                    markdown_content="# Proposal 0",
                ),
            }

            with DigestBuilder() as builder:
                # Build digest
                digest = builder.build_digest(
                    classified_rfps=classified_rfps,
                    proposals=proposals,
                    total_discovered=100,
                    total_filtered=50,
                )

                # Format for Slack
                blocks = builder.format_slack_blocks(digest)
                fallback = builder.format_fallback_text(digest)

                # Verify digest
                assert len(digest.entries) == 3
                assert digest.total_discovered == 100
                assert digest.total_proposals == 1

                # Verify entries are sorted by score
                scores = [e.classification.relevance_score for e in digest.entries]
                assert scores == sorted(scores, reverse=True)

                # Verify blocks
                assert len(blocks) > 0
                assert any(b["type"] == "header" for b in blocks)

                # Verify fallback
                assert "NAITIVE" in fallback
                assert "3" in fallback  # 3 relevant RFPs

    @pytest.mark.unit
    def test_multiple_builders_with_different_branding(self):
        """Test multiple builders with different branding."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.digest_builder import DigestBuilder

            classified_rfp = ClassifiedRFP(
                rfp=RFP(id="rfp-1", title="Test RFP"),
                classification=ClassificationResult(
                    rfp_id="rfp-1",
                    relevance_score=0.8,
                ),
            )

            # First builder with default branding
            builder1 = DigestBuilder()
            blocks1 = builder1.format_slack_blocks(
                builder1.build_digest([classified_rfp])
            )

            # Second builder with custom branding
            builder2 = DigestBuilder(brand_name="CustomCorp")
            blocks2 = builder2.format_slack_blocks(
                builder2.build_digest([classified_rfp])
            )

            # Find header text
            header1 = blocks1[0]["text"]["text"]
            header2 = blocks2[0]["text"]["text"]

            assert "NAITIVE" in header1
            assert "CustomCorp" in header2
