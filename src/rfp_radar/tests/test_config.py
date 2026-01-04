# src/rfp_radar/tests/test_config.py
"""
Unit tests for RFP Radar configuration module.

Tests cover:
- Required environment variable loading
- Optional environment variable loading with defaults
- Boolean configuration parsing
- Numeric configuration parsing (float/int)
- Azure credential creation (dev vs prod)
- Error handling for missing required variables
"""
import os
import pytest
from unittest.mock import patch, MagicMock


# Mock environment variables required for RFPRadarConfig instantiation
MOCK_ENV_VARS = {
    # Required variables
    "APP_ENV": "dev",
    "AZURE_STORAGE_ACCOUNT_URL": "https://mockstorageaccount.blob.core.windows.net",
    "AZURE_SEARCH_ENDPOINT": "https://mock-search.search.windows.net",
    "AZURE_OPENAI_ENDPOINT": "https://mock-openai.openai.azure.com",
    "SLACK_BOT_TOKEN": "xoxb-mock-token-12345",
    # Optional variables with non-default values
    "AZURE_STORAGE_CONTAINER": "test-container",
    "AZURE_SEARCH_INDEX_NAME": "test-index",
    "AZURE_OPENAI_DEPLOYMENT": "gpt-4-test",
    "SLACK_CHANNEL": "#test-channel",
    "RFP_RELEVANCE_THRESHOLD": "0.75",
    "RFP_MAX_AGE_DAYS": "5",
    "NAITIVE_BRAND_NAME": "TestBrand",
    "NAITIVE_WEBSITE": "https://test.example.com",
}


class TestConfigRequiredVariables:
    """Tests for required environment variable loading."""

    @pytest.mark.unit
    def test_required_variables_loaded_from_env(self):
        """Test that required variables are correctly loaded from environment."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            # Need to reimport to get fresh config instance
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            assert config.APP_ENV == "dev"
            assert config.AZURE_STORAGE_ACCOUNT_URL == MOCK_ENV_VARS["AZURE_STORAGE_ACCOUNT_URL"]
            assert config.AZURE_SEARCH_ENDPOINT == MOCK_ENV_VARS["AZURE_SEARCH_ENDPOINT"]
            assert config.AZURE_OPENAI_ENDPOINT == MOCK_ENV_VARS["AZURE_OPENAI_ENDPOINT"]
            assert config.SLACK_BOT_TOKEN == MOCK_ENV_VARS["SLACK_BOT_TOKEN"]

    @pytest.mark.unit
    def test_missing_required_variable_raises_error(self):
        """Test that missing required variable without default raises ValueError."""
        # Environment missing AZURE_STORAGE_ACCOUNT_URL
        incomplete_env = {
            "APP_ENV": "dev",
            "AZURE_SEARCH_ENDPOINT": "https://search.example.com",
            "AZURE_OPENAI_ENDPOINT": "https://openai.example.com",
            "SLACK_BOT_TOKEN": "xoxb-token",
        }
        with patch.dict(os.environ, incomplete_env, clear=True):
            from rfp_radar.config import RFPRadarConfig
            with pytest.raises(ValueError) as exc_info:
                RFPRadarConfig()
            assert "AZURE_STORAGE_ACCOUNT_URL" in str(exc_info.value)

    @pytest.mark.unit
    def test_required_variable_with_default_uses_default(self):
        """Test that required variables with defaults use defaults when not in env."""
        # APP_ENV has default of "dev"
        env_without_app_env = dict(MOCK_ENV_VARS)
        del env_without_app_env["APP_ENV"]

        with patch.dict(os.environ, env_without_app_env, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()
            assert config.APP_ENV == "dev"


class TestConfigOptionalVariables:
    """Tests for optional environment variable loading."""

    @pytest.mark.unit
    def test_optional_variables_with_defaults(self):
        """Test that optional variables use defaults when not in environment."""
        minimal_env = {
            "APP_ENV": "dev",
            "AZURE_STORAGE_ACCOUNT_URL": "https://storage.example.com",
            "AZURE_SEARCH_ENDPOINT": "https://search.example.com",
            "AZURE_OPENAI_ENDPOINT": "https://openai.example.com",
            "SLACK_BOT_TOKEN": "xoxb-token",
        }
        with patch.dict(os.environ, minimal_env, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            # Check defaults are applied
            assert config.AZURE_STORAGE_CONTAINER == "rfp-radar"
            assert config.AZURE_SEARCH_INDEX_NAME == "rfp-radar-index"
            assert config.AZURE_OPENAI_DEPLOYMENT == "gpt-4o"
            assert config.AZURE_OPENAI_API_VERSION == "2024-11-20"
            assert config.SLACK_CHANNEL == "#bots"
            assert config.NAITIVE_BRAND_NAME == "NAITIVE"
            assert config.NAITIVE_WEBSITE == "https://www.naitive.cloud"

    @pytest.mark.unit
    def test_optional_variables_from_env_override_defaults(self):
        """Test that optional variables from env override defaults."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            assert config.AZURE_STORAGE_CONTAINER == "test-container"
            assert config.AZURE_SEARCH_INDEX_NAME == "test-index"
            assert config.AZURE_OPENAI_DEPLOYMENT == "gpt-4-test"
            assert config.SLACK_CHANNEL == "#test-channel"
            assert config.NAITIVE_BRAND_NAME == "TestBrand"
            assert config.NAITIVE_WEBSITE == "https://test.example.com"

    @pytest.mark.unit
    def test_optional_variable_empty_string_default(self):
        """Test that optional variables return empty string when not set and no default."""
        minimal_env = {
            "APP_ENV": "dev",
            "AZURE_STORAGE_ACCOUNT_URL": "https://storage.example.com",
            "AZURE_SEARCH_ENDPOINT": "https://search.example.com",
            "AZURE_OPENAI_ENDPOINT": "https://openai.example.com",
            "SLACK_BOT_TOKEN": "xoxb-token",
        }
        with patch.dict(os.environ, minimal_env, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            # These optional variables have empty string default
            assert config.AZURE_TENANT_ID == ""
            assert config.AZURE_CLIENT_ID == ""
            assert config.AZURE_STORAGE_SAS_TOKEN == ""
            assert config.AZURE_SEARCH_API_KEY == ""
            assert config.AZURE_OPENAI_API_KEY == ""
            assert config.APPLICATIONINSIGHTS_CONNECTION_STRING == ""


class TestConfigNumericValues:
    """Tests for numeric configuration value parsing."""

    @pytest.mark.unit
    def test_relevance_threshold_default(self):
        """Test that RFP_RELEVANCE_THRESHOLD has correct default."""
        minimal_env = {
            "APP_ENV": "dev",
            "AZURE_STORAGE_ACCOUNT_URL": "https://storage.example.com",
            "AZURE_SEARCH_ENDPOINT": "https://search.example.com",
            "AZURE_OPENAI_ENDPOINT": "https://openai.example.com",
            "SLACK_BOT_TOKEN": "xoxb-token",
        }
        with patch.dict(os.environ, minimal_env, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()
            assert config.RFP_RELEVANCE_THRESHOLD == 0.55
            assert isinstance(config.RFP_RELEVANCE_THRESHOLD, float)

    @pytest.mark.unit
    def test_relevance_threshold_from_env(self):
        """Test that RFP_RELEVANCE_THRESHOLD is parsed from environment."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()
            assert config.RFP_RELEVANCE_THRESHOLD == 0.75
            assert isinstance(config.RFP_RELEVANCE_THRESHOLD, float)

    @pytest.mark.unit
    def test_max_age_days_default(self):
        """Test that RFP_MAX_AGE_DAYS has correct default."""
        minimal_env = {
            "APP_ENV": "dev",
            "AZURE_STORAGE_ACCOUNT_URL": "https://storage.example.com",
            "AZURE_SEARCH_ENDPOINT": "https://search.example.com",
            "AZURE_OPENAI_ENDPOINT": "https://openai.example.com",
            "SLACK_BOT_TOKEN": "xoxb-token",
        }
        with patch.dict(os.environ, minimal_env, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()
            assert config.RFP_MAX_AGE_DAYS == 3
            assert isinstance(config.RFP_MAX_AGE_DAYS, int)

    @pytest.mark.unit
    def test_max_age_days_from_env(self):
        """Test that RFP_MAX_AGE_DAYS is parsed from environment."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()
            assert config.RFP_MAX_AGE_DAYS == 5
            assert isinstance(config.RFP_MAX_AGE_DAYS, int)


class TestConfigBooleanValues:
    """Tests for boolean configuration value parsing."""

    @pytest.mark.unit
    def test_get_bool_true_values(self):
        """Test that _get_bool correctly parses true values."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            # Test "true"
            with patch.dict(os.environ, {"FEATURE_FLAG": "true"}):
                assert config._get_bool("FEATURE_FLAG") is True

            # Test "1"
            with patch.dict(os.environ, {"FEATURE_FLAG": "1"}):
                assert config._get_bool("FEATURE_FLAG") is True

            # Test "TRUE" (case insensitive)
            with patch.dict(os.environ, {"FEATURE_FLAG": "TRUE"}):
                assert config._get_bool("FEATURE_FLAG") is True

    @pytest.mark.unit
    def test_get_bool_false_values(self):
        """Test that _get_bool correctly parses false values."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            # Test "false"
            with patch.dict(os.environ, {"FEATURE_FLAG": "false"}):
                assert config._get_bool("FEATURE_FLAG") is False

            # Test "0"
            with patch.dict(os.environ, {"FEATURE_FLAG": "0"}):
                assert config._get_bool("FEATURE_FLAG") is False

            # Test missing variable
            assert config._get_bool("NON_EXISTENT_FLAG") is False


class TestConfigAzureCredentials:
    """Tests for Azure credential configuration."""

    @pytest.mark.unit
    def test_dev_environment_uses_default_credential(self):
        """Test that dev environment uses DefaultAzureCredential."""
        dev_env = dict(MOCK_ENV_VARS)
        dev_env["APP_ENV"] = "dev"

        with patch.dict(os.environ, dev_env, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            with patch("rfp_radar.config.DefaultAzureCredential") as mock_default:
                with patch("rfp_radar.config.ManagedIdentityCredential"):
                    mock_default.return_value = MagicMock()
                    credential = config.get_azure_credential()
                    mock_default.assert_called_once()

    @pytest.mark.unit
    def test_prod_environment_uses_managed_identity(self):
        """Test that prod environment uses ManagedIdentityCredential."""
        prod_env = dict(MOCK_ENV_VARS)
        prod_env["APP_ENV"] = "prod"
        prod_env["AZURE_CLIENT_ID"] = "test-client-id"

        with patch.dict(os.environ, prod_env, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            with patch("rfp_radar.config.DefaultAzureCredential"):
                with patch("rfp_radar.config.ManagedIdentityCredential") as mock_managed:
                    mock_managed.return_value = MagicMock()
                    credential = config.get_azure_credential(client_id="test-client-id")
                    mock_managed.assert_called_once_with(client_id="test-client-id")

    @pytest.mark.unit
    def test_get_azure_credentials_caches_result(self):
        """Test that get_azure_credentials caches the credential."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            with patch.object(config, "get_azure_credential") as mock_get_cred:
                mock_cred = MagicMock()
                mock_get_cred.return_value = mock_cred

                # First call should create credential
                cred1 = config.get_azure_credentials()
                # Second call should use cached
                cred2 = config.get_azure_credentials()

                assert cred1 is cred2
                mock_get_cred.assert_called_once()


class TestConfigGetMethods:
    """Tests for the private _get_* helper methods."""

    @pytest.mark.unit
    def test_get_required_returns_env_value(self):
        """Test that _get_required returns environment value when present."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            result = config._get_required("AZURE_STORAGE_ACCOUNT_URL")
            assert result == MOCK_ENV_VARS["AZURE_STORAGE_ACCOUNT_URL"]

    @pytest.mark.unit
    def test_get_required_returns_default_when_not_in_env(self):
        """Test that _get_required returns default when variable not in env."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            result = config._get_required("NON_EXISTENT_VAR", "default_value")
            assert result == "default_value"

    @pytest.mark.unit
    def test_get_required_raises_when_missing_and_no_default(self):
        """Test that _get_required raises ValueError when missing and no default."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            with pytest.raises(ValueError) as exc_info:
                config._get_required("NON_EXISTENT_VAR")
            assert "NON_EXISTENT_VAR" in str(exc_info.value)
            assert "not found" in str(exc_info.value)

    @pytest.mark.unit
    def test_get_optional_returns_env_value(self):
        """Test that _get_optional returns environment value when present."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            result = config._get_optional("SLACK_CHANNEL", "#default")
            assert result == "#test-channel"

    @pytest.mark.unit
    def test_get_optional_returns_default_when_not_in_env(self):
        """Test that _get_optional returns default when variable not in env."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            result = config._get_optional("NON_EXISTENT_VAR", "default_value")
            assert result == "default_value"

    @pytest.mark.unit
    def test_get_optional_returns_empty_string_by_default(self):
        """Test that _get_optional returns empty string when no default specified."""
        with patch.dict(os.environ, MOCK_ENV_VARS, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()

            result = config._get_optional("NON_EXISTENT_VAR")
            assert result == ""


class TestConfigCognitiveServicesScope:
    """Tests for Azure Cognitive Services configuration."""

    @pytest.mark.unit
    def test_cognitive_services_default_scope(self):
        """Test that Azure Cognitive Services scope has correct default."""
        minimal_env = {
            "APP_ENV": "dev",
            "AZURE_STORAGE_ACCOUNT_URL": "https://storage.example.com",
            "AZURE_SEARCH_ENDPOINT": "https://search.example.com",
            "AZURE_OPENAI_ENDPOINT": "https://openai.example.com",
            "SLACK_BOT_TOKEN": "xoxb-token",
        }
        with patch.dict(os.environ, minimal_env, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()
            assert config.AZURE_COGNITIVE_SERVICES == "https://cognitiveservices.azure.com/.default"

    @pytest.mark.unit
    def test_cognitive_services_custom_scope(self):
        """Test that Azure Cognitive Services scope can be customized."""
        custom_env = dict(MOCK_ENV_VARS)
        custom_env["AZURE_COGNITIVE_SERVICES"] = "https://custom.azure.com/.default"

        with patch.dict(os.environ, custom_env, clear=True):
            from rfp_radar.config import RFPRadarConfig
            config = RFPRadarConfig()
            assert config.AZURE_COGNITIVE_SERVICES == "https://custom.azure.com/.default"
