# src/rfp_radar/tests/test_proposal_generator.py
"""
Unit tests for RFP Radar proposal generator module.

Tests cover:
- ProposalGenerator initialization with custom and default values
- Single proposal generation via generate()
- Batch generation via generate_batch() with error handling
- Proposal storage via store_proposal()
- Regeneration via regenerate() with version control
- Additional requirements building from RFP and classification metadata
- Proposal creation from raw LLM response
- Section counting in markdown content
- Statistics and health check methods
- Context manager support and resource cleanup
- Error handling and edge cases
"""
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pytest

from rfp_radar.models import (
    RFP,
    RFPSource,
    RFPStatus,
    RFPTag,
    ClassificationResult,
    ClassifiedRFP,
    Proposal,
    ProposalMetadata,
)


# Mock environment variables required for config
MOCK_ENV_VARS = {
    "APP_ENV": "dev",
    "AZURE_STORAGE_ACCOUNT_URL": "https://mockstorageaccount.blob.core.windows.net",
    "AZURE_SEARCH_ENDPOINT": "https://mock-search.search.windows.net",
    "AZURE_OPENAI_ENDPOINT": "https://mock-openai.openai.azure.com",
    "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
    "SLACK_BOT_TOKEN": "xoxb-mock-token-12345",
    "NAITIVE_BRAND_NAME": "NAITIVE",
    "NAITIVE_WEBSITE": "https://www.naitive.cloud",
}


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = MagicMock()
    client.generate_proposal.return_value = {
        "markdown_content": (
            "# Proposal for Test RFP\n\n"
            "## Executive Summary\n\nThis is the executive summary.\n\n"
            "## Understanding of Requirements\n\nWe understand the requirements.\n\n"
            "## Proposed Solution\n\nOur proposed solution includes...\n\n"
            "## Technical Methodology\n\nOur methodology is...\n\n"
            "## Team Qualifications\n\nOur team is qualified.\n\n"
        ),
        "model_used": "gpt-4o",
        "prompt_tokens": 1500,
        "completion_tokens": 2000,
    }
    client.health_check.return_value = True
    client.get_usage_stats.return_value = {
        "endpoint": "https://mock-openai.openai.azure.com",
        "deployment": "gpt-4o",
        "api_version": "2024-11-20",
    }
    client.close = MagicMock()
    return client


@pytest.fixture
def mock_storage_client():
    """Create a mock storage client for testing."""
    client = MagicMock()
    client.ensure_container_exists.return_value = None
    client.upload_proposal.return_value = (
        "https://mockstorageaccount.blob.core.windows.net/rfp-radar/proposals/test.md"
    )
    return client


@pytest.fixture
def sample_rfp():
    """Create a sample RFP for testing."""
    return RFP(
        id="test-rfp-123",
        title="AI-Powered Government Analytics Platform",
        description="Seeking a contractor to develop an AI-powered analytics platform.",
        agency="Department of Testing",
        source=RFPSource.GOVTRIBE,
        country="US",
        naics_codes=["541512", "541519"],
        estimated_value=1500000.00,
        contract_type="Fixed Price",
        set_aside="Small Business",
        location="Washington, DC",
        due_date=datetime.utcnow() + timedelta(days=30),
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
def sample_classified_rfp(sample_rfp, sample_classification):
    """Create a sample ClassifiedRFP for testing."""
    return ClassifiedRFP(
        rfp=sample_rfp,
        classification=sample_classification,
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
def sample_classification_minimal():
    """Create a minimal classification result for testing."""
    return ClassificationResult(
        rfp_id="test-rfp-minimal",
        relevance_score=0.6,
        tags=[],
        reasoning="",
    )


class TestProposalGeneratorInitialization:
    """Tests for ProposalGenerator initialization."""

    @pytest.mark.unit
    def test_init_with_default_values(self, mock_llm_client, mock_storage_client):
        """Test generator initialization with default values."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )

            assert generator.brand_name == "NAITIVE"
            assert generator.brand_website == "https://www.naitive.cloud"
            assert generator._llm_client is mock_llm_client
            assert generator._storage_client is mock_storage_client
            assert generator._owns_llm_client is False
            assert generator._owns_storage_client is False

    @pytest.mark.unit
    def test_init_with_custom_branding(self, mock_llm_client, mock_storage_client):
        """Test generator initialization with custom branding."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
                brand_name="CustomBrand",
                brand_website="https://custom.example.com",
            )

            assert generator.brand_name == "CustomBrand"
            assert generator.brand_website == "https://custom.example.com"

    @pytest.mark.unit
    def test_init_without_clients(self):
        """Test generator creates clients lazily when not provided."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator()

            assert generator._llm_client is None
            assert generator._storage_client is None
            assert generator._owns_llm_client is True
            assert generator._owns_storage_client is True

    @pytest.mark.unit
    def test_llm_client_property_lazy_initialization(self):
        """Test LLM client is created on first access."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            with patch("rfp_radar.proposal_generator.LLMClient") as mock_client_class:
                mock_instance = MagicMock()
                mock_client_class.return_value = mock_instance

                generator = ProposalGenerator()
                assert generator._llm_client is None

                # Access llm_client property triggers creation
                client = generator.llm_client

                mock_client_class.assert_called_once()
                assert client is mock_instance

    @pytest.mark.unit
    def test_storage_client_property_lazy_initialization(self):
        """Test storage client is created on first access."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            with patch(
                "rfp_radar.proposal_generator.StorageClient"
            ) as mock_client_class:
                mock_instance = MagicMock()
                mock_client_class.return_value = mock_instance

                generator = ProposalGenerator()
                assert generator._storage_client is None

                # Access storage_client property triggers creation
                client = generator.storage_client

                mock_client_class.assert_called_once()
                assert client is mock_instance


class TestProposalGeneratorGenerate:
    """Tests for ProposalGenerator.generate() method."""

    @pytest.mark.unit
    def test_generate_single_proposal_success(
        self, mock_llm_client, mock_storage_client, sample_classified_rfp
    ):
        """Test successful generation of a single proposal."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            proposal = generator.generate(sample_classified_rfp, store=False)

            assert isinstance(proposal, Proposal)
            assert proposal.metadata.rfp_id == sample_classified_rfp.rfp.id
            assert "Executive Summary" in proposal.markdown_content
            assert proposal.metadata.brand_name == "NAITIVE"

    @pytest.mark.unit
    def test_generate_calls_llm_with_correct_params(
        self, mock_llm_client, mock_storage_client, sample_classified_rfp
    ):
        """Test generate calls LLM client with correct parameters."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            generator.generate(sample_classified_rfp, store=False)

            mock_llm_client.generate_proposal.assert_called_once()
            call_kwargs = mock_llm_client.generate_proposal.call_args[1]

            assert call_kwargs["rfp_title"] == sample_classified_rfp.rfp.title
            assert call_kwargs["rfp_description"] == sample_classified_rfp.rfp.description
            assert call_kwargs["agency"] == sample_classified_rfp.rfp.agency
            assert "AI" in call_kwargs["classification_tags"]
            assert "Cloud" in call_kwargs["classification_tags"]

    @pytest.mark.unit
    def test_generate_with_store_true(
        self, mock_llm_client, mock_storage_client, sample_classified_rfp
    ):
        """Test generate stores proposal when store=True."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            generator.generate(sample_classified_rfp, store=True)

            mock_storage_client.ensure_container_exists.assert_called_once()
            mock_storage_client.upload_proposal.assert_called_once()

    @pytest.mark.unit
    def test_generate_with_store_false(
        self, mock_llm_client, mock_storage_client, sample_classified_rfp
    ):
        """Test generate does not store proposal when store=False."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            generator.generate(sample_classified_rfp, store=False)

            mock_storage_client.upload_proposal.assert_not_called()

    @pytest.mark.unit
    def test_generate_updates_rfp_status(
        self, mock_llm_client, mock_storage_client, sample_classified_rfp
    ):
        """Test generate updates RFP status to PROPOSAL_GENERATED."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            generator.generate(sample_classified_rfp, store=False)

            assert sample_classified_rfp.rfp.status == RFPStatus.PROPOSAL_GENERATED

    @pytest.mark.unit
    def test_generate_raises_on_llm_failure(
        self, mock_llm_client, mock_storage_client, sample_classified_rfp
    ):
        """Test generate raises ValueError on LLM failure."""
        mock_llm_client.generate_proposal.side_effect = Exception("API Error")

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )

            with pytest.raises(ValueError) as exc_info:
                generator.generate(sample_classified_rfp, store=False)

            assert sample_classified_rfp.rfp.id in str(exc_info.value)
            assert "Proposal generation failed" in str(exc_info.value)

    @pytest.mark.unit
    def test_generate_minimal_rfp(
        self,
        mock_llm_client,
        mock_storage_client,
        sample_rfp_minimal,
        sample_classification_minimal,
    ):
        """Test generation for minimal RFP without optional fields."""
        classified_rfp = ClassifiedRFP(
            rfp=sample_rfp_minimal,
            classification=sample_classification_minimal,
        )

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            proposal = generator.generate(classified_rfp, store=False)

            assert isinstance(proposal, Proposal)
            assert proposal.metadata.rfp_id == sample_rfp_minimal.id


class TestProposalGeneratorGenerateBatch:
    """Tests for ProposalGenerator.generate_batch() method."""

    @pytest.mark.unit
    def test_generate_batch_success(
        self, mock_llm_client, mock_storage_client
    ):
        """Test batch generation of multiple proposals."""
        classified_rfps = [
            ClassifiedRFP(
                rfp=RFP(id=f"rfp-{i}", title=f"Test RFP {i}"),
                classification=ClassificationResult(
                    rfp_id=f"rfp-{i}",
                    relevance_score=0.8,
                ),
            )
            for i in range(3)
        ]

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            results = generator.generate_batch(classified_rfps, store=False)

            assert len(results) == 3
            assert all(isinstance(r[1], Proposal) for r in results)

    @pytest.mark.unit
    def test_generate_batch_empty_list(
        self, mock_llm_client, mock_storage_client
    ):
        """Test batch generation with empty list."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            results = generator.generate_batch([], store=False)

            assert results == []
            mock_llm_client.generate_proposal.assert_not_called()

    @pytest.mark.unit
    def test_generate_batch_skip_on_error_true(
        self, mock_llm_client, mock_storage_client
    ):
        """Test batch generation continues on error when skip_on_error=True."""
        classified_rfps = [
            ClassifiedRFP(
                rfp=RFP(id="rfp-1", title="Success RFP 1"),
                classification=ClassificationResult(rfp_id="rfp-1", relevance_score=0.8),
            ),
            ClassifiedRFP(
                rfp=RFP(id="rfp-2", title="Fail RFP"),
                classification=ClassificationResult(rfp_id="rfp-2", relevance_score=0.8),
            ),
            ClassifiedRFP(
                rfp=RFP(id="rfp-3", title="Success RFP 2"),
                classification=ClassificationResult(rfp_id="rfp-3", relevance_score=0.8),
            ),
        ]

        # Second call fails
        mock_llm_client.generate_proposal.side_effect = [
            {"markdown_content": "# Proposal 1\n\n## Section", "model_used": "gpt-4o"},
            Exception("API Error"),
            {"markdown_content": "# Proposal 3\n\n## Section", "model_used": "gpt-4o"},
        ]

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            results = generator.generate_batch(
                classified_rfps, store=False, skip_on_error=True
            )

            # Should return 2 successful results
            assert len(results) == 2
            assert results[0][0].rfp.id == "rfp-1"
            assert results[1][0].rfp.id == "rfp-3"

            # Failed RFP should have ERROR status
            assert classified_rfps[1].rfp.status == RFPStatus.ERROR

    @pytest.mark.unit
    def test_generate_batch_skip_on_error_false(
        self, mock_llm_client, mock_storage_client
    ):
        """Test batch generation raises on error when skip_on_error=False."""
        classified_rfps = [
            ClassifiedRFP(
                rfp=RFP(id="rfp-1", title="Success RFP"),
                classification=ClassificationResult(rfp_id="rfp-1", relevance_score=0.8),
            ),
            ClassifiedRFP(
                rfp=RFP(id="rfp-2", title="Fail RFP"),
                classification=ClassificationResult(rfp_id="rfp-2", relevance_score=0.8),
            ),
        ]

        mock_llm_client.generate_proposal.side_effect = [
            {"markdown_content": "# Proposal 1\n\n## Section", "model_used": "gpt-4o"},
            Exception("API Error"),
        ]

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )

            with pytest.raises(ValueError) as exc_info:
                generator.generate_batch(
                    classified_rfps, store=False, skip_on_error=False
                )

            assert "rfp-2" in str(exc_info.value)


class TestProposalGeneratorStoreProposal:
    """Tests for ProposalGenerator.store_proposal() method."""

    @pytest.mark.unit
    def test_store_proposal_success(
        self, mock_llm_client, mock_storage_client
    ):
        """Test successful proposal storage."""
        proposal = Proposal(
            metadata=ProposalMetadata(
                rfp_id="test-rfp-123",
                rfp_title="Test RFP",
            ),
            markdown_content="# Test Proposal\n\n## Section 1\n\nContent",
        )

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            blob_url = generator.store_proposal(proposal)

            mock_storage_client.ensure_container_exists.assert_called_once()
            mock_storage_client.upload_proposal.assert_called_once_with(proposal)
            assert "mockstorageaccount.blob.core.windows.net" in blob_url

    @pytest.mark.unit
    def test_store_proposal_raises_on_storage_failure(
        self, mock_llm_client, mock_storage_client
    ):
        """Test store_proposal raises on storage failure."""
        mock_storage_client.upload_proposal.side_effect = Exception("Storage Error")

        proposal = Proposal(
            metadata=ProposalMetadata(
                rfp_id="test-rfp-123",
                rfp_title="Test RFP",
            ),
            markdown_content="# Test Proposal\n\nContent",
        )

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )

            with pytest.raises(Exception) as exc_info:
                generator.store_proposal(proposal)

            assert "Storage Error" in str(exc_info.value)


class TestProposalGeneratorRegenerate:
    """Tests for ProposalGenerator.regenerate() method."""

    @pytest.mark.unit
    def test_regenerate_with_version(
        self, mock_llm_client, mock_storage_client, sample_classified_rfp
    ):
        """Test regeneration with custom version number."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            proposal = generator.regenerate(
                sample_classified_rfp, version=3, store=False
            )

            assert proposal.metadata.version == 3

    @pytest.mark.unit
    def test_regenerate_with_store_true(
        self, mock_llm_client, mock_storage_client, sample_classified_rfp
    ):
        """Test regeneration with storage."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            generator.regenerate(sample_classified_rfp, version=2, store=True)

            mock_storage_client.upload_proposal.assert_called_once()


class TestProposalGeneratorBuildAdditionalRequirements:
    """Tests for ProposalGenerator._build_additional_requirements() method."""

    @pytest.mark.unit
    def test_build_requirements_with_all_fields(
        self, mock_llm_client, mock_storage_client, sample_rfp, sample_classification
    ):
        """Test _build_additional_requirements includes all available metadata."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            requirements = generator._build_additional_requirements(
                sample_rfp, sample_classification
            )

            assert "Classification Analysis" in requirements
            assert "NAICS" in requirements
            assert "541512" in requirements
            assert "Estimated Contract Value" in requirements
            assert "$1,500,000.00" in requirements
            assert "Contract Type: Fixed Price" in requirements
            assert "Set-Aside Requirements: Small Business" in requirements
            assert "Service Location: Washington, DC" in requirements

    @pytest.mark.unit
    def test_build_requirements_with_minimal_rfp(
        self,
        mock_llm_client,
        mock_storage_client,
        sample_rfp_minimal,
        sample_classification_minimal,
    ):
        """Test _build_additional_requirements with minimal RFP returns empty string."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            requirements = generator._build_additional_requirements(
                sample_rfp_minimal, sample_classification_minimal
            )

            assert requirements == ""

    @pytest.mark.unit
    def test_build_requirements_with_state_instead_of_location(
        self, mock_llm_client, mock_storage_client, sample_classification
    ):
        """Test _build_additional_requirements uses state when location is empty."""
        rfp = RFP(
            id="test-rfp",
            title="Test RFP",
            state="CA",  # No location, but has state
        )

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            requirements = generator._build_additional_requirements(
                rfp, sample_classification
            )

            assert "State: CA" in requirements


class TestProposalGeneratorCreateProposal:
    """Tests for ProposalGenerator._create_proposal() method."""

    @pytest.mark.unit
    def test_create_proposal_basic(
        self, mock_llm_client, mock_storage_client, sample_rfp
    ):
        """Test _create_proposal with basic data."""
        raw_result = {
            "markdown_content": (
                "# Proposal Title\n\n"
                "## Executive Summary\n\nSummary content.\n\n"
                "## Technical Approach\n\nApproach content.\n"
            ),
            "model_used": "gpt-4o",
            "prompt_tokens": 1000,
            "completion_tokens": 1500,
        }

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            proposal = generator._create_proposal(sample_rfp, raw_result)

            assert isinstance(proposal, Proposal)
            assert proposal.metadata.rfp_id == sample_rfp.id
            assert proposal.metadata.rfp_title == sample_rfp.title
            assert proposal.metadata.version == 1
            assert proposal.metadata.brand_name == "NAITIVE"
            assert proposal.metadata.brand_website == "https://www.naitive.cloud"
            assert proposal.metadata.section_count == 2
            assert len(proposal.metadata.content_hash) == 64  # SHA-256 hex

    @pytest.mark.unit
    def test_create_proposal_calculates_word_count(
        self, mock_llm_client, mock_storage_client, sample_rfp
    ):
        """Test _create_proposal calculates word count correctly."""
        raw_result = {
            "markdown_content": "One two three four five six seven eight nine ten",
        }

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            proposal = generator._create_proposal(sample_rfp, raw_result)

            assert proposal.metadata.word_count == 10

    @pytest.mark.unit
    def test_create_proposal_empty_content(
        self, mock_llm_client, mock_storage_client, sample_rfp
    ):
        """Test _create_proposal with empty content."""
        raw_result = {
            "markdown_content": "",
        }

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            proposal = generator._create_proposal(sample_rfp, raw_result)

            assert proposal.markdown_content == ""
            assert proposal.metadata.word_count == 0
            assert proposal.metadata.section_count == 0


class TestProposalGeneratorCountSections:
    """Tests for ProposalGenerator._count_sections() method."""

    @pytest.mark.unit
    def test_count_sections_multiple(
        self, mock_llm_client, mock_storage_client
    ):
        """Test _count_sections with multiple sections."""
        markdown = (
            "# Title\n\n"
            "## Section 1\n\nContent.\n\n"
            "## Section 2\n\nContent.\n\n"
            "## Section 3\n\nContent.\n"
        )

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            count = generator._count_sections(markdown)

            assert count == 3

    @pytest.mark.unit
    def test_count_sections_no_sections(
        self, mock_llm_client, mock_storage_client
    ):
        """Test _count_sections with no level-2 headings."""
        markdown = "# Title\n\nJust some content without sections."

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            count = generator._count_sections(markdown)

            assert count == 0

    @pytest.mark.unit
    def test_count_sections_empty_content(
        self, mock_llm_client, mock_storage_client
    ):
        """Test _count_sections with empty content."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            count = generator._count_sections("")

            assert count == 0

    @pytest.mark.unit
    def test_count_sections_ignores_level3_headings(
        self, mock_llm_client, mock_storage_client
    ):
        """Test _count_sections ignores level-3 and other headings."""
        markdown = (
            "# Title\n\n"
            "## Section 1\n\nContent.\n\n"
            "### Subsection 1.1\n\nContent.\n\n"
            "## Section 2\n\nContent.\n"
        )

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            count = generator._count_sections(markdown)

            assert count == 2


class TestProposalGeneratorStats:
    """Tests for ProposalGenerator.get_stats() method."""

    @pytest.mark.unit
    def test_get_stats_returns_config_and_llm_stats(
        self, mock_llm_client, mock_storage_client
    ):
        """Test get_stats returns generator config and LLM stats."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
                brand_name="TestBrand",
            )
            stats = generator.get_stats()

            assert stats["brand_name"] == "TestBrand"
            assert "brand_website" in stats
            assert "proposal_sections" in stats
            assert "llm_stats" in stats
            assert len(stats["proposal_sections"]) > 0

    @pytest.mark.unit
    def test_get_stats_formats_brand_in_sections(
        self, mock_llm_client, mock_storage_client
    ):
        """Test get_stats formats brand name in section titles."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
                brand_name="ACME",
            )
            stats = generator.get_stats()

            # Check that {brand_name} placeholder is replaced
            assert any("ACME" in section for section in stats["proposal_sections"])


class TestProposalGeneratorHealthCheck:
    """Tests for ProposalGenerator.health_check() method."""

    @pytest.mark.unit
    def test_health_check_returns_true_on_success(
        self, mock_llm_client, mock_storage_client
    ):
        """Test health_check returns True when both clients are healthy."""
        mock_llm_client.health_check.return_value = True

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            result = generator.health_check()

            assert result is True
            mock_llm_client.health_check.assert_called_once()

    @pytest.mark.unit
    def test_health_check_returns_false_on_llm_failure(
        self, mock_llm_client, mock_storage_client
    ):
        """Test health_check returns False when LLM check fails."""
        mock_llm_client.health_check.return_value = False

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            result = generator.health_check()

            assert result is False

    @pytest.mark.unit
    def test_health_check_returns_false_on_storage_failure(
        self, mock_llm_client, mock_storage_client
    ):
        """Test health_check returns False when storage check fails."""
        mock_llm_client.health_check.return_value = True
        mock_storage_client.ensure_container_exists.side_effect = Exception(
            "Storage Error"
        )

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            result = generator.health_check()

            assert result is False

    @pytest.mark.unit
    def test_health_check_returns_false_on_llm_exception(
        self, mock_llm_client, mock_storage_client
    ):
        """Test health_check returns False when LLM throws exception."""
        mock_llm_client.health_check.side_effect = Exception("Connection error")

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            result = generator.health_check()

            assert result is False


class TestProposalGeneratorContextManager:
    """Tests for ProposalGenerator context manager support."""

    @pytest.mark.unit
    def test_context_manager_enter_returns_self(
        self, mock_llm_client, mock_storage_client
    ):
        """Test context manager __enter__ returns generator instance."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )

            with generator as ctx:
                assert ctx is generator

    @pytest.mark.unit
    def test_context_manager_closes_owned_clients(self):
        """Test context manager closes LLM client when owned."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            with patch("rfp_radar.proposal_generator.LLMClient") as mock_llm_class:
                mock_llm_instance = MagicMock()
                mock_llm_class.return_value = mock_llm_instance

                with ProposalGenerator() as generator:
                    # Trigger lazy initialization
                    _ = generator.llm_client

                # After context exit, close should be called
                mock_llm_instance.close.assert_called_once()

    @pytest.mark.unit
    def test_context_manager_does_not_close_provided_clients(
        self, mock_llm_client, mock_storage_client
    ):
        """Test context manager does not close externally provided clients."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            with ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            ) as generator:
                pass

            mock_llm_client.close.assert_not_called()

    @pytest.mark.unit
    def test_close_clears_client_references(self):
        """Test close() clears the internal client references."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            with patch("rfp_radar.proposal_generator.LLMClient") as mock_llm_class:
                mock_llm_instance = MagicMock()
                mock_llm_class.return_value = mock_llm_instance

                with patch(
                    "rfp_radar.proposal_generator.StorageClient"
                ) as mock_storage_class:
                    mock_storage_instance = MagicMock()
                    mock_storage_class.return_value = mock_storage_instance

                    generator = ProposalGenerator()
                    # Initialize clients
                    _ = generator.llm_client
                    _ = generator.storage_client
                    assert generator._llm_client is not None
                    assert generator._storage_client is not None

                    generator.close()
                    assert generator._llm_client is None
                    assert generator._storage_client is None


class TestProposalGeneratorProposalSections:
    """Tests for ProposalGenerator.PROPOSAL_SECTIONS class attribute."""

    @pytest.mark.unit
    def test_proposal_sections_contains_required_sections(
        self, mock_llm_client, mock_storage_client
    ):
        """Test PROPOSAL_SECTIONS contains all required sections."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )

            required_sections = [
                "Executive Summary",
                "Understanding of Requirements",
                "Proposed Solution",
                "Technical Methodology",
                "Team Qualifications",
                "Project Timeline",
                "Risk Mitigation",
                "Conclusion",
            ]

            for section in required_sections:
                assert any(
                    section in s for s in generator.PROPOSAL_SECTIONS
                ), f"Missing section: {section}"

    @pytest.mark.unit
    def test_proposal_sections_count(
        self, mock_llm_client, mock_storage_client
    ):
        """Test PROPOSAL_SECTIONS has correct number of sections."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            generator = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )

            assert len(generator.PROPOSAL_SECTIONS) == 9


class TestProposalGeneratorIntegration:
    """Integration-style tests for ProposalGenerator."""

    @pytest.mark.unit
    def test_full_proposal_workflow(
        self, mock_llm_client, mock_storage_client, sample_classified_rfp
    ):
        """Test complete proposal generation workflow."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            with ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            ) as generator:
                # Generate proposal
                proposal = generator.generate(sample_classified_rfp, store=True)

                # Verify proposal
                assert isinstance(proposal, Proposal)
                assert proposal.metadata.rfp_id == sample_classified_rfp.rfp.id
                assert proposal.metadata.brand_name == "NAITIVE"

                # Verify RFP status updated
                assert sample_classified_rfp.rfp.status == RFPStatus.PROPOSAL_GENERATED

                # Verify storage was called
                mock_storage_client.upload_proposal.assert_called_once()

    @pytest.mark.unit
    def test_multiple_proposals_with_different_branding(
        self, mock_llm_client, mock_storage_client
    ):
        """Test generating proposals with different branding."""
        classified_rfps = [
            ClassifiedRFP(
                rfp=RFP(id=f"rfp-{i}", title=f"Test RFP {i}"),
                classification=ClassificationResult(
                    rfp_id=f"rfp-{i}",
                    relevance_score=0.8,
                ),
            )
            for i in range(2)
        ]

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.proposal_generator import ProposalGenerator

            # First generator with default branding
            generator1 = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
            )
            proposal1 = generator1.generate(classified_rfps[0], store=False)

            # Second generator with custom branding
            generator2 = ProposalGenerator(
                llm_client=mock_llm_client,
                storage_client=mock_storage_client,
                brand_name="CustomCorp",
                brand_website="https://custom.corp",
            )
            proposal2 = generator2.generate(classified_rfps[1], store=False)

            assert proposal1.metadata.brand_name == "NAITIVE"
            assert proposal2.metadata.brand_name == "CustomCorp"
            assert proposal2.metadata.brand_website == "https://custom.corp"
