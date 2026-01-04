# config.py
import logging
import os
from typing import Optional, TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

# Load environment variables from .env file
load_dotenv()


class RFPRadarConfig:
    """RFP Radar configuration class that loads settings from environment variables."""

    def __init__(self):
        """Initialize the RFP Radar configuration with environment variables."""
        self.logger = logging.getLogger(__name__)

        # Application environment
        self.APP_ENV = self._get_required("APP_ENV", "dev")

        # Azure authentication settings
        self.AZURE_TENANT_ID = self._get_optional("AZURE_TENANT_ID")
        self.AZURE_CLIENT_ID = self._get_optional("AZURE_CLIENT_ID")

        # Azure Storage settings
        self.AZURE_STORAGE_ACCOUNT_URL = self._get_required("AZURE_STORAGE_ACCOUNT_URL")
        self.AZURE_STORAGE_CONTAINER = self._get_optional(
            "AZURE_STORAGE_CONTAINER", "rfp-radar"
        )
        self.AZURE_STORAGE_SAS_TOKEN = self._get_optional("AZURE_STORAGE_SAS_TOKEN")

        # Azure AI Search settings
        self.AZURE_SEARCH_ENDPOINT = self._get_required("AZURE_SEARCH_ENDPOINT")
        self.AZURE_SEARCH_API_KEY = self._get_optional("AZURE_SEARCH_API_KEY")
        self.AZURE_SEARCH_INDEX_NAME = self._get_optional(
            "AZURE_SEARCH_INDEX_NAME", "rfp-radar-index"
        )

        # Azure OpenAI settings
        self.AZURE_OPENAI_ENDPOINT = self._get_required("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_API_KEY = self._get_optional("AZURE_OPENAI_API_KEY")
        self.AZURE_OPENAI_DEPLOYMENT = self._get_optional(
            "AZURE_OPENAI_DEPLOYMENT", "gpt-4o"
        )
        self.AZURE_OPENAI_API_VERSION = self._get_optional(
            "AZURE_OPENAI_API_VERSION", "2024-11-20"
        )

        # Azure Cognitive Services scope (for token-based auth)
        self.AZURE_COGNITIVE_SERVICES = self._get_optional(
            "AZURE_COGNITIVE_SERVICES", "https://cognitiveservices.azure.com/.default"
        )

        # Slack settings
        self.SLACK_BOT_TOKEN = self._get_required("SLACK_BOT_TOKEN")
        self.SLACK_CHANNEL = self._get_optional("SLACK_CHANNEL", "#bots")

        # RFP Radar configuration
        self.RFP_RELEVANCE_THRESHOLD = float(
            self._get_optional("RFP_RELEVANCE_THRESHOLD", "0.55")
        )
        self.RFP_MAX_AGE_DAYS = int(self._get_optional("RFP_MAX_AGE_DAYS", "3"))

        # NAITIVE branding
        self.NAITIVE_BRAND_NAME = self._get_optional("NAITIVE_BRAND_NAME", "NAITIVE")
        self.NAITIVE_WEBSITE = self._get_optional(
            "NAITIVE_WEBSITE", "https://www.naitive.cloud"
        )

        # Application Insights
        self.APPLICATIONINSIGHTS_CONNECTION_STRING = self._get_optional(
            "APPLICATIONINSIGHTS_CONNECTION_STRING"
        )

        # Cached credentials
        self._azure_credentials = None

    def get_azure_credential(self, client_id: Optional[str] = None):
        """
        Returns an Azure credential based on the application environment.

        If the environment is 'dev', it uses DefaultAzureCredential.
        Otherwise, it uses ManagedIdentityCredential.

        Args:
            client_id (str, optional): The client ID for the Managed Identity Credential.

        Returns:
            Credential object: Either DefaultAzureCredential or ManagedIdentityCredential.
        """
        from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

        if self.APP_ENV == "dev":
            return DefaultAzureCredential()
        else:
            return ManagedIdentityCredential(client_id=client_id)

    def get_azure_credentials(self):
        """Retrieve Azure credentials, either from environment variables or managed identity."""
        if self._azure_credentials is None:
            self._azure_credentials = self.get_azure_credential(self.AZURE_CLIENT_ID)
        return self._azure_credentials

    async def get_access_token(self) -> str:
        """Get Azure access token for API calls."""
        try:
            credential = self.get_azure_credentials()
            token = credential.get_token(self.AZURE_COGNITIVE_SERVICES)
            return token.token
        except Exception as e:
            self.logger.error(f"Failed to get access token: {e}")
            raise

    def _get_required(self, name: str, default: Optional[str] = None) -> str:
        """Get a required configuration value from environment variables.

        Args:
            name: The name of the environment variable
            default: Optional default value if not found

        Returns:
            The value of the environment variable or default if provided

        Raises:
            ValueError: If the environment variable is not found and no default is provided
        """
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            logging.warning(
                "Environment variable %s not found, using default value", name
            )
            return default
        raise ValueError(
            f"Environment variable {name} not found and no default provided"
        )

    def _get_optional(self, name: str, default: str = "") -> str:
        """Get an optional configuration value from environment variables.

        Args:
            name: The name of the environment variable
            default: Default value if not found (default: "")

        Returns:
            The value of the environment variable or the default value
        """
        if name in os.environ:
            return os.environ[name]
        return default

    def _get_bool(self, name: str) -> bool:
        """Get a boolean configuration value from environment variables.

        Args:
            name: The name of the environment variable

        Returns:
            True if the environment variable exists and is set to 'true' or '1', False otherwise
        """
        return name in os.environ and os.environ[name].lower() in ["true", "1"]


# Create a global instance of RFPRadarConfig
config = RFPRadarConfig()
