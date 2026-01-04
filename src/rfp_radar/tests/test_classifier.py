# src/rfp_radar/tests/test_classifier.py
"""
Unit tests for RFP Radar classifier module.

Tests cover:
- RFPClassifier initialization with custom and default values
- Single RFP classification via classify()
- Batch classification via classify_batch() with error handling
- Relevance filtering via filter_relevant()
- Combined classify_and_filter() convenience method
- Context building from RFP metadata
- Classification result parsing and validation
- Tag validation and case-insensitive matching
- Statistics and health check methods
- Context manager support and resource cleanup
- Error handling and edge cases
"""
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from rfp_radar.models import (
    RFP,
    RFPSource,
    RFPStatus,
    RFPTag,
    ClassificationResult,
    ClassifiedRFP,
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


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = MagicMock()
    client.classify_rfp.return_value = {
        "relevance_score": 0.75,
        "tags": ["AI", "Cloud"],
        "reasoning": "This RFP involves AI and cloud technologies.",
        "confidence": 0.9,
        "model_used": "gpt-4o",
    }
    client.health_check.return_value = True
    client.get_usage_stats.return_value = {
        "endpoint": "https://mock-openai.openai.azure.com",
        "deployment": "gpt-4o",
        "api_version": "2024-11-20",
        "using_api_key": True,
        "using_managed_identity": False,
    }
    client.close = MagicMock()
    return client


@pytest.fixture
def sample_rfp():
    """Create a sample RFP for testing."""
    return RFP(
        id="test-rfp-123",
        title="AI-Powered Government Analytics Platform",
        description="Seeking a contractor to develop an AI-powered analytics platform for data processing.",
        agency="Department of Testing",
        source=RFPSource.GOVTRIBE,
        country="US",
        naics_codes=["541512"],
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


class TestRFPClassifierInitialization:
    """Tests for RFPClassifier initialization."""

    @pytest.mark.unit
    def test_init_with_default_values(self, mock_llm_client):
        """Test classifier initialization with default values."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)

            assert classifier.relevance_threshold == 0.55
            assert classifier._llm_client is mock_llm_client
            assert classifier._owns_client is False

    @pytest.mark.unit
    def test_init_with_custom_threshold(self, mock_llm_client):
        """Test classifier initialization with custom threshold."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(
                llm_client=mock_llm_client,
                relevance_threshold=0.75,
            )

            assert classifier.relevance_threshold == 0.75

    @pytest.mark.unit
    def test_init_without_llm_client(self):
        """Test classifier creates LLM client lazily when not provided."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier()

            assert classifier._llm_client is None
            assert classifier._owns_client is True

    @pytest.mark.unit
    def test_llm_client_property_lazy_initialization(self):
        """Test LLM client is created on first access."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            with patch("rfp_radar.classifier.LLMClient") as mock_client_class:
                mock_instance = MagicMock()
                mock_client_class.return_value = mock_instance

                classifier = RFPClassifier()
                assert classifier._llm_client is None

                # Access llm_client property triggers creation
                client = classifier.llm_client

                mock_client_class.assert_called_once()
                assert client is mock_instance


class TestRFPClassifierClassify:
    """Tests for RFPClassifier.classify() method."""

    @pytest.mark.unit
    def test_classify_single_rfp_success(self, mock_llm_client, sample_rfp):
        """Test successful classification of a single RFP."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            result = classifier.classify(sample_rfp)

            assert isinstance(result, ClassificationResult)
            assert result.rfp_id == sample_rfp.id
            assert result.relevance_score == 0.75
            assert RFPTag.AI in result.tags
            assert RFPTag.CLOUD in result.tags
            assert "AI and cloud" in result.reasoning
            assert result.confidence == 0.9

    @pytest.mark.unit
    def test_classify_calls_llm_with_correct_params(
        self, mock_llm_client, sample_rfp
    ):
        """Test classify calls LLM client with correct parameters."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            classifier.classify(sample_rfp)

            mock_llm_client.classify_rfp.assert_called_once()
            call_kwargs = mock_llm_client.classify_rfp.call_args[1]

            assert call_kwargs["title"] == sample_rfp.title
            assert call_kwargs["description"] == sample_rfp.description
            assert call_kwargs["agency"] == sample_rfp.agency
            assert "NAICS Codes" in call_kwargs["additional_context"]

    @pytest.mark.unit
    def test_classify_raises_on_llm_failure(
        self, mock_llm_client, sample_rfp
    ):
        """Test classify raises ValueError on LLM failure."""
        mock_llm_client.classify_rfp.side_effect = Exception("API Error")

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)

            with pytest.raises(ValueError) as exc_info:
                classifier.classify(sample_rfp)

            assert sample_rfp.id in str(exc_info.value)
            assert "Classification failed" in str(exc_info.value)

    @pytest.mark.unit
    def test_classify_minimal_rfp(self, mock_llm_client, sample_rfp_minimal):
        """Test classification of minimal RFP without optional fields."""
        mock_llm_client.classify_rfp.return_value = {
            "relevance_score": 0.5,
            "tags": [],
            "reasoning": "Basic RFP.",
            "confidence": 0.8,
        }

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            result = classifier.classify(sample_rfp_minimal)

            assert result.relevance_score == 0.5
            assert result.tags == []


class TestRFPClassifierClassifyBatch:
    """Tests for RFPClassifier.classify_batch() method."""

    @pytest.mark.unit
    def test_classify_batch_success(self, mock_llm_client):
        """Test batch classification of multiple RFPs."""
        rfps = [
            RFP(id=f"rfp-{i}", title=f"Test RFP {i}")
            for i in range(3)
        ]

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            results = classifier.classify_batch(rfps)

            assert len(results) == 3
            assert all(isinstance(r, ClassifiedRFP) for r in results)
            assert all(r.rfp.status == RFPStatus.CLASSIFIED for r in results)

    @pytest.mark.unit
    def test_classify_batch_empty_list(self, mock_llm_client):
        """Test batch classification of empty list."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            results = classifier.classify_batch([])

            assert results == []
            mock_llm_client.classify_rfp.assert_not_called()

    @pytest.mark.unit
    def test_classify_batch_skip_on_error_true(self, mock_llm_client):
        """Test batch classification continues on error when skip_on_error=True."""
        rfps = [
            RFP(id="rfp-1", title="Success RFP 1"),
            RFP(id="rfp-2", title="Fail RFP"),
            RFP(id="rfp-3", title="Success RFP 2"),
        ]

        # Second call fails
        mock_llm_client.classify_rfp.side_effect = [
            {"relevance_score": 0.8, "tags": [], "reasoning": "", "confidence": 1.0},
            Exception("API Error"),
            {"relevance_score": 0.7, "tags": [], "reasoning": "", "confidence": 1.0},
        ]

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            results = classifier.classify_batch(rfps, skip_on_error=True)

            # Should return 2 successful results
            assert len(results) == 2
            assert results[0].rfp.id == "rfp-1"
            assert results[1].rfp.id == "rfp-3"

            # Failed RFP should have ERROR status
            assert rfps[1].status == RFPStatus.ERROR

    @pytest.mark.unit
    def test_classify_batch_skip_on_error_false(self, mock_llm_client):
        """Test batch classification raises on error when skip_on_error=False."""
        rfps = [
            RFP(id="rfp-1", title="Success RFP"),
            RFP(id="rfp-2", title="Fail RFP"),
        ]

        mock_llm_client.classify_rfp.side_effect = [
            {"relevance_score": 0.8, "tags": [], "reasoning": "", "confidence": 1.0},
            Exception("API Error"),
        ]

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)

            with pytest.raises(ValueError) as exc_info:
                classifier.classify_batch(rfps, skip_on_error=False)

            assert "rfp-2" in str(exc_info.value)


class TestRFPClassifierFilterRelevant:
    """Tests for RFPClassifier.filter_relevant() method."""

    @pytest.mark.unit
    def test_filter_relevant_uses_threshold(self, mock_llm_client):
        """Test filter_relevant filters by relevance threshold."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(
                llm_client=mock_llm_client,
                relevance_threshold=0.6,
            )

            classified_rfps = [
                ClassifiedRFP(
                    rfp=RFP(id=f"rfp-{i}", title=f"RFP {i}"),
                    classification=ClassificationResult(
                        rfp_id=f"rfp-{i}",
                        relevance_score=score,
                    ),
                )
                for i, score in enumerate([0.5, 0.6, 0.7, 0.8])
            ]

            results = classifier.filter_relevant(classified_rfps)

            assert len(results) == 3
            assert all(
                r.classification.relevance_score >= 0.6 for r in results
            )

    @pytest.mark.unit
    def test_filter_relevant_with_custom_threshold(self, mock_llm_client):
        """Test filter_relevant with custom threshold override."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(
                llm_client=mock_llm_client,
                relevance_threshold=0.5,
            )

            classified_rfps = [
                ClassifiedRFP(
                    rfp=RFP(id=f"rfp-{i}", title=f"RFP {i}"),
                    classification=ClassificationResult(
                        rfp_id=f"rfp-{i}",
                        relevance_score=score,
                    ),
                )
                for i, score in enumerate([0.7, 0.8, 0.9])
            ]

            # Use higher threshold override
            results = classifier.filter_relevant(classified_rfps, threshold=0.85)

            assert len(results) == 1
            assert results[0].classification.relevance_score == 0.9

    @pytest.mark.unit
    def test_filter_relevant_empty_list(self, mock_llm_client):
        """Test filter_relevant with empty list."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            results = classifier.filter_relevant([])

            assert results == []


class TestRFPClassifierClassifyAndFilter:
    """Tests for RFPClassifier.classify_and_filter() method."""

    @pytest.mark.unit
    def test_classify_and_filter_combines_operations(self, mock_llm_client):
        """Test classify_and_filter combines classification and filtering."""
        rfps = [
            RFP(id="rfp-high", title="High Score RFP"),
            RFP(id="rfp-low", title="Low Score RFP"),
        ]

        # First call returns high score, second returns low score
        mock_llm_client.classify_rfp.side_effect = [
            {"relevance_score": 0.8, "tags": ["AI"], "reasoning": "", "confidence": 1.0},
            {"relevance_score": 0.3, "tags": [], "reasoning": "", "confidence": 1.0},
        ]

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(
                llm_client=mock_llm_client,
                relevance_threshold=0.55,
            )
            results = classifier.classify_and_filter(rfps)

            # Only high score RFP should pass
            assert len(results) == 1
            assert results[0].rfp.id == "rfp-high"


class TestRFPClassifierBuildContext:
    """Tests for RFPClassifier._build_context() method."""

    @pytest.mark.unit
    def test_build_context_with_all_fields(self, mock_llm_client, sample_rfp):
        """Test _build_context includes all available metadata."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            context = classifier._build_context(sample_rfp)

            assert "NAICS Codes: 541512" in context
            assert "Estimated Value: $1,500,000.00" in context
            assert "Contract Type: Fixed Price" in context
            assert "Set-Aside: Small Business" in context
            assert "Location: Washington, DC" in context
            assert "Due Date:" in context

    @pytest.mark.unit
    def test_build_context_with_minimal_rfp(
        self, mock_llm_client, sample_rfp_minimal
    ):
        """Test _build_context with minimal RFP returns empty string."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            context = classifier._build_context(sample_rfp_minimal)

            assert context == ""

    @pytest.mark.unit
    def test_build_context_partial_fields(self, mock_llm_client):
        """Test _build_context with partial metadata."""
        rfp = RFP(
            id="partial-rfp",
            title="Partial RFP",
            naics_codes=["541512", "541519"],
            location="New York, NY",
        )

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            context = classifier._build_context(rfp)

            assert "NAICS Codes: 541512, 541519" in context
            assert "Location: New York, NY" in context
            assert "Estimated Value" not in context
            assert "Contract Type" not in context


class TestRFPClassifierCreateClassificationResult:
    """Tests for RFPClassifier._create_classification_result() method."""

    @pytest.mark.unit
    def test_create_classification_result_basic(self, mock_llm_client):
        """Test _create_classification_result with basic data."""
        raw_result = {
            "relevance_score": 0.75,
            "tags": ["AI", "Cloud"],
            "reasoning": "Good fit for AI work.",
            "confidence": 0.9,
            "model_used": "gpt-4o",
        }

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            result = classifier._create_classification_result(
                rfp_id="test-123",
                raw_result=raw_result,
            )

            assert result.rfp_id == "test-123"
            assert result.relevance_score == 0.75
            assert len(result.tags) == 2
            assert RFPTag.AI in result.tags
            assert RFPTag.CLOUD in result.tags
            assert result.reasoning == "Good fit for AI work."
            assert result.confidence == 0.9

    @pytest.mark.unit
    def test_create_classification_result_clamps_score(self, mock_llm_client):
        """Test _create_classification_result clamps out-of-range scores."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)

            # Test score > 1.0
            result_high = classifier._create_classification_result(
                rfp_id="test-1",
                raw_result={"relevance_score": 1.5, "tags": []},
            )
            assert result_high.relevance_score == 1.0

            # Test score < 0.0
            result_low = classifier._create_classification_result(
                rfp_id="test-2",
                raw_result={"relevance_score": -0.5, "tags": []},
            )
            assert result_low.relevance_score == 0.0

    @pytest.mark.unit
    def test_create_classification_result_tag_string_to_list(
        self, mock_llm_client
    ):
        """Test _create_classification_result converts single tag string to list."""
        raw_result = {
            "relevance_score": 0.7,
            "tags": "AI",  # Single string instead of list
            "reasoning": "",
            "confidence": 1.0,
        }

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            result = classifier._create_classification_result(
                rfp_id="test-123",
                raw_result=raw_result,
            )

            assert len(result.tags) == 1
            assert RFPTag.AI in result.tags

    @pytest.mark.unit
    def test_create_classification_result_case_insensitive_tags(
        self, mock_llm_client
    ):
        """Test _create_classification_result handles case-insensitive tag matching."""
        raw_result = {
            "relevance_score": 0.7,
            "tags": ["ai", "CLOUD", "Dynamics"],  # Mixed case
            "reasoning": "",
            "confidence": 1.0,
        }

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            result = classifier._create_classification_result(
                rfp_id="test-123",
                raw_result=raw_result,
            )

            assert RFPTag.AI in result.tags
            assert RFPTag.CLOUD in result.tags
            assert RFPTag.DYNAMICS in result.tags

    @pytest.mark.unit
    def test_create_classification_result_unknown_tags_ignored(
        self, mock_llm_client
    ):
        """Test _create_classification_result ignores unknown tags."""
        raw_result = {
            "relevance_score": 0.7,
            "tags": ["AI", "UnknownTag", "InvalidTag"],
            "reasoning": "",
            "confidence": 1.0,
        }

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            result = classifier._create_classification_result(
                rfp_id="test-123",
                raw_result=raw_result,
            )

            # Only valid tags should be included
            assert len(result.tags) == 1
            assert RFPTag.AI in result.tags

    @pytest.mark.unit
    def test_create_classification_result_missing_fields_defaults(
        self, mock_llm_client
    ):
        """Test _create_classification_result handles missing fields with defaults."""
        raw_result = {}  # Empty result

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            result = classifier._create_classification_result(
                rfp_id="test-123",
                raw_result=raw_result,
            )

            assert result.relevance_score == 0.0
            assert result.tags == []
            assert result.reasoning == ""
            assert result.confidence == 1.0


class TestRFPClassifierValidTags:
    """Tests for RFPClassifier.VALID_TAGS class attribute."""

    @pytest.mark.unit
    def test_valid_tags_contains_all_rfp_tags(self, mock_llm_client):
        """Test VALID_TAGS contains all RFPTag enum values."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)

            expected_tags = {tag.value for tag in RFPTag}
            assert classifier.VALID_TAGS == expected_tags

    @pytest.mark.unit
    def test_valid_tags_count(self, mock_llm_client):
        """Test VALID_TAGS has correct number of tags."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)

            assert len(classifier.VALID_TAGS) == 8


class TestRFPClassifierStats:
    """Tests for RFPClassifier.get_stats() method."""

    @pytest.mark.unit
    def test_get_stats_returns_config_and_llm_stats(self, mock_llm_client):
        """Test get_stats returns classifier config and LLM stats."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(
                llm_client=mock_llm_client,
                relevance_threshold=0.65,
            )
            stats = classifier.get_stats()

            assert stats["relevance_threshold"] == 0.65
            assert "valid_tags" in stats
            assert "llm_stats" in stats
            assert len(stats["valid_tags"]) == 8


class TestRFPClassifierHealthCheck:
    """Tests for RFPClassifier.health_check() method."""

    @pytest.mark.unit
    def test_health_check_returns_true_on_success(self, mock_llm_client):
        """Test health_check returns True when LLM is healthy."""
        mock_llm_client.health_check.return_value = True

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            result = classifier.health_check()

            assert result is True
            mock_llm_client.health_check.assert_called_once()

    @pytest.mark.unit
    def test_health_check_returns_false_on_failure(self, mock_llm_client):
        """Test health_check returns False when LLM check fails."""
        mock_llm_client.health_check.side_effect = Exception("Connection error")

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            result = classifier.health_check()

            assert result is False

    @pytest.mark.unit
    def test_health_check_returns_false_on_unhealthy(self, mock_llm_client):
        """Test health_check returns False when LLM returns unhealthy."""
        mock_llm_client.health_check.return_value = False

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            result = classifier.health_check()

            assert result is False


class TestRFPClassifierContextManager:
    """Tests for RFPClassifier context manager support."""

    @pytest.mark.unit
    def test_context_manager_enter_returns_self(self, mock_llm_client):
        """Test context manager __enter__ returns classifier instance."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)

            with classifier as ctx:
                assert ctx is classifier

    @pytest.mark.unit
    def test_context_manager_closes_owned_client(self):
        """Test context manager closes LLM client when owned."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            with patch("rfp_radar.classifier.LLMClient") as mock_client_class:
                mock_instance = MagicMock()
                mock_client_class.return_value = mock_instance

                with RFPClassifier() as classifier:
                    # Trigger lazy initialization
                    _ = classifier.llm_client

                # After context exit, close should be called
                mock_instance.close.assert_called_once()

    @pytest.mark.unit
    def test_context_manager_does_not_close_provided_client(
        self, mock_llm_client
    ):
        """Test context manager does not close externally provided client."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            with RFPClassifier(llm_client=mock_llm_client) as classifier:
                pass

            mock_llm_client.close.assert_not_called()

    @pytest.mark.unit
    def test_close_clears_client_reference(self):
        """Test close() clears the internal client reference."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            with patch("rfp_radar.classifier.LLMClient") as mock_client_class:
                mock_instance = MagicMock()
                mock_client_class.return_value = mock_instance

                classifier = RFPClassifier()
                _ = classifier.llm_client  # Initialize client
                assert classifier._llm_client is not None

                classifier.close()
                assert classifier._llm_client is None


class TestRFPClassifierRelevanceScoreValidation:
    """Tests for relevance score validation in classification results."""

    @pytest.mark.unit
    def test_relevance_score_in_valid_range(self, mock_llm_client, sample_rfp):
        """Test relevance score is always in 0-1 range."""
        mock_llm_client.classify_rfp.return_value = {
            "relevance_score": 0.75,
            "tags": [],
            "reasoning": "",
            "confidence": 1.0,
        }

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            result = classifier.classify(sample_rfp)

            assert 0.0 <= result.relevance_score <= 1.0

    @pytest.mark.unit
    def test_is_relevant_method_on_result(self, mock_llm_client, sample_rfp):
        """Test is_relevant method works on classification result."""
        mock_llm_client.classify_rfp.return_value = {
            "relevance_score": 0.75,
            "tags": [],
            "reasoning": "",
            "confidence": 1.0,
        }

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(
                llm_client=mock_llm_client,
                relevance_threshold=0.55,
            )
            result = classifier.classify(sample_rfp)

            assert result.is_relevant(0.55) is True
            assert result.is_relevant(0.80) is False


class TestRFPClassifierAllTags:
    """Tests for all RFPTag values being properly handled."""

    @pytest.mark.unit
    def test_all_rfp_tags_are_valid(self, mock_llm_client):
        """Test all RFPTag enum values are in VALID_TAGS."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)

            for tag in RFPTag:
                assert tag.value in classifier.VALID_TAGS

    @pytest.mark.unit
    def test_classify_with_all_tag_types(self, mock_llm_client, sample_rfp):
        """Test classification can return all types of tags."""
        all_tags = [tag.value for tag in RFPTag]
        mock_llm_client.classify_rfp.return_value = {
            "relevance_score": 0.9,
            "tags": all_tags,
            "reasoning": "Covers all areas.",
            "confidence": 1.0,
        }

        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.classifier import RFPClassifier

            classifier = RFPClassifier(llm_client=mock_llm_client)
            result = classifier.classify(sample_rfp)

            assert len(result.tags) == len(RFPTag)
            for tag in RFPTag:
                assert tag in result.tags
