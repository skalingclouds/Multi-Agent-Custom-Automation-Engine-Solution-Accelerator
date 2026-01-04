# llm_client.py
"""Azure OpenAI REST client for RFP classification and proposal generation."""

import json
import time
from typing import Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import config
from .logging_utils import get_logger


class LLMClient:
    """Azure OpenAI REST client wrapper for RFP Radar.

    This class provides methods for interacting with Azure OpenAI API
    for RFP classification and proposal generation. It uses the REST API
    directly via requests (not the OpenAI SDK) per requirements.
    """

    # Default request timeout in seconds (per spec: 60-second timeout)
    DEFAULT_TIMEOUT = 60

    # Maximum number of retries for API calls
    MAX_RETRIES = 3

    # Base delay for exponential backoff (in seconds)
    BASE_RETRY_DELAY = 1.0

    def __init__(
        self,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """Initialize the LLM client.

        Args:
            endpoint: Azure OpenAI endpoint URL. Defaults to config value.
            deployment: Model deployment name. Defaults to config value.
            api_key: API key for authentication. If not provided, uses
                    managed identity or DefaultAzureCredential.
            api_version: API version to use. Defaults to config value.
            timeout: Request timeout in seconds.
        """
        self.logger = get_logger(__name__)

        self.endpoint = (endpoint or config.AZURE_OPENAI_ENDPOINT).rstrip("/")
        self.deployment = deployment or config.AZURE_OPENAI_DEPLOYMENT
        self.api_version = api_version or config.AZURE_OPENAI_API_VERSION
        self.timeout = timeout

        # Authentication - prefer API key, fall back to managed identity
        self._api_key = api_key or config.AZURE_OPENAI_API_KEY
        self._credential = None
        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0

        # Lazy-initialized session
        self._session: Optional[requests.Session] = None

    def _get_session(self) -> requests.Session:
        """Get or create a requests session with retry configuration.

        Returns:
            Configured requests.Session instance
        """
        if self._session is None:
            self._session = requests.Session()

            # Configure retry strategy for transient errors
            retry_strategy = Retry(
                total=self.MAX_RETRIES,
                backoff_factor=self.BASE_RETRY_DELAY,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST", "GET"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)

        return self._session

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests.

        Returns:
            Dictionary of headers including authorization
        """
        headers = {
            "Content-Type": "application/json",
        }

        if self._api_key:
            # Use API key authentication
            headers["api-key"] = self._api_key
        else:
            # Use managed identity / Azure AD token
            token = self._get_access_token()
            headers["Authorization"] = f"Bearer {token}"

        return headers

    def _get_access_token(self) -> str:
        """Get Azure AD access token for authentication.

        Uses caching to avoid unnecessary token refreshes.

        Returns:
            Access token string
        """
        # Check if cached token is still valid (with 5 min buffer)
        current_time = time.time()
        if self._cached_token and self._token_expires_at > current_time + 300:
            return self._cached_token

        try:
            if self._credential is None:
                self._credential = config.get_azure_credentials()

            # Get token for Azure Cognitive Services
            token_response = self._credential.get_token(
                config.AZURE_COGNITIVE_SERVICES
            )

            self._cached_token = token_response.token
            self._token_expires_at = token_response.expires_on

            self.logger.debug("Obtained new Azure AD access token")
            return self._cached_token

        except Exception as e:
            self.logger.error(f"Failed to get access token: {e}")
            raise

    def _build_api_url(self, endpoint_path: str = "chat/completions") -> str:
        """Build the full API URL for a given endpoint path.

        Args:
            endpoint_path: The API endpoint path (default: chat/completions)

        Returns:
            Full API URL
        """
        return (
            f"{self.endpoint}/openai/deployments/{self.deployment}"
            f"/{endpoint_path}?api-version={self.api_version}"
        )

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, str]] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """Make a chat completion request to Azure OpenAI.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Optional response format specification
            retry_count: Current retry attempt number

        Returns:
            API response dictionary

        Raises:
            requests.exceptions.RequestException: On request failure
            ValueError: On invalid response
        """
        url = self._build_api_url()
        headers = self._get_auth_headers()

        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            payload["response_format"] = response_format

        try:
            session = self._get_session()

            self.logger.debug(
                "Making LLM API request",
                extra={
                    "deployment": self.deployment,
                    "message_count": len(messages),
                    "max_tokens": max_tokens,
                }
            )

            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            # Handle rate limiting with exponential backoff
            if response.status_code == 429 and retry_count < self.MAX_RETRIES:
                retry_after = int(
                    response.headers.get("Retry-After", 2 ** retry_count)
                )
                self.logger.warning(
                    f"Rate limited, retrying after {retry_after}s",
                    extra={"retry_count": retry_count + 1}
                )
                time.sleep(retry_after)
                return self._make_request(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    retry_count=retry_count + 1,
                )

            response.raise_for_status()

            result = response.json()

            self.logger.debug(
                "LLM API request completed",
                extra={
                    "usage": result.get("usage", {}),
                }
            )

            return result

        except requests.exceptions.Timeout:
            self.logger.error(
                f"LLM API request timed out after {self.timeout}s",
                extra={"retry_count": retry_count}
            )
            if retry_count < self.MAX_RETRIES:
                delay = self.BASE_RETRY_DELAY * (2 ** retry_count)
                self.logger.info(f"Retrying after {delay}s")
                time.sleep(delay)
                return self._make_request(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    retry_count=retry_count + 1,
                )
            raise

        except requests.exceptions.HTTPError as e:
            self.logger.error(
                f"LLM API request failed: {e}",
                extra={
                    "status_code": e.response.status_code if e.response else None,
                    "response_text": (
                        e.response.text[:500] if e.response else None
                    ),
                }
            )
            raise

        except Exception as e:
            self.logger.error(f"LLM API request error: {e}")
            raise

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """Make a chat completion request and return the response text.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Optional response format specification

        Returns:
            The assistant's response text

        Raises:
            ValueError: If response doesn't contain expected content
        """
        result = self._make_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        try:
            choices = result.get("choices", [])
            if not choices:
                raise ValueError("No choices in API response")

            message = choices[0].get("message", {})
            content = message.get("content", "")

            if not content:
                raise ValueError("Empty content in API response")

            return content

        except (KeyError, IndexError) as e:
            self.logger.error(
                f"Failed to parse API response: {e}",
                extra={"response": result}
            )
            raise ValueError(f"Invalid API response structure: {e}")

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Make a chat completion request expecting JSON response.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (lower for structured output)
            max_tokens: Maximum tokens in response

        Returns:
            Parsed JSON response as dictionary

        Raises:
            ValueError: If response is not valid JSON
        """
        content = self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse JSON response: {e}",
                extra={"content": content[:500]}
            )
            raise ValueError(f"Invalid JSON in response: {e}")

    def classify_rfp(
        self,
        title: str,
        description: str,
        agency: str = "",
        additional_context: str = "",
    ) -> Dict[str, Any]:
        """Classify an RFP for relevance and extract tags.

        Args:
            title: RFP title
            description: RFP description or summary
            agency: Issuing agency name
            additional_context: Any additional context about the RFP

        Returns:
            Classification result dictionary with:
            - relevance_score: float (0-1)
            - tags: list of tag strings
            - reasoning: string explaining the classification
            - confidence: float (0-1)

        Raises:
            ValueError: On classification failure
        """
        system_prompt = """You are an expert RFP analyst for NAITIVE, a company specializing in:
- AI/ML solutions and implementations
- Microsoft Dynamics 365 implementations and integrations
- Cloud modernization and digital transformation
- Data analytics and automation solutions

Your task is to analyze RFPs (Requests for Proposal) and determine their relevance to NAITIVE's capabilities.

Respond with a JSON object containing:
{
    "relevance_score": <float 0.0-1.0>,
    "tags": [<list of applicable tags from: "AI", "Dynamics", "Modernization", "Cloud", "Security", "Data", "Automation", "Other">],
    "reasoning": "<brief explanation of the score and tags>",
    "confidence": <float 0.0-1.0 indicating confidence in this assessment>
}

Scoring guidelines:
- 0.8-1.0: Highly relevant - directly matches NAITIVE's core competencies (AI, Dynamics, modernization)
- 0.55-0.79: Relevant - related to our capabilities with good fit potential
- 0.3-0.54: Marginally relevant - some overlap but not ideal fit
- 0.0-0.29: Not relevant - outside our expertise or capabilities"""

        user_content = f"""Analyze this RFP for relevance to NAITIVE:

Title: {title}

Agency: {agency or 'Not specified'}

Description:
{description}

{f'Additional Context: {additional_context}' if additional_context else ''}

Provide your classification as JSON."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        self.logger.info(
            "Classifying RFP",
            extra={"title": title[:100], "agency": agency}
        )

        try:
            result = self.complete_json(
                messages=messages,
                temperature=0.3,  # Low temperature for consistent classification
                max_tokens=1024,
            )

            # Validate and normalize the result
            relevance_score = float(result.get("relevance_score", 0))
            relevance_score = max(0.0, min(1.0, relevance_score))

            tags = result.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]

            confidence = float(result.get("confidence", 1.0))
            confidence = max(0.0, min(1.0, confidence))

            classification = {
                "relevance_score": relevance_score,
                "tags": tags,
                "reasoning": result.get("reasoning", ""),
                "confidence": confidence,
                "model_used": self.deployment,
            }

            self.logger.info(
                "RFP classified",
                extra={
                    "title": title[:50],
                    "relevance_score": relevance_score,
                    "tags": tags,
                }
            )

            return classification

        except Exception as e:
            self.logger.error(
                f"RFP classification failed: {e}",
                extra={"title": title[:100]}
            )
            raise

    def generate_proposal(
        self,
        rfp_title: str,
        rfp_description: str,
        agency: str = "",
        classification_tags: Optional[List[str]] = None,
        due_date: Optional[str] = None,
        additional_requirements: str = "",
    ) -> Dict[str, str]:
        """Generate a Level 3 full proposal for an RFP.

        Args:
            rfp_title: RFP title
            rfp_description: Full RFP description
            agency: Issuing agency name
            classification_tags: Tags from classification (e.g., ["AI", "Cloud"])
            due_date: Proposal due date if known
            additional_requirements: Any additional requirements or context

        Returns:
            Dictionary with:
            - markdown_content: Full proposal in markdown format
            - executive_summary: Brief executive summary
            - key_differentiators: List of key differentiators

        Raises:
            ValueError: On generation failure
        """
        brand_name = config.NAITIVE_BRAND_NAME
        brand_website = config.NAITIVE_WEBSITE

        tags_str = ", ".join(classification_tags) if classification_tags else "General"

        system_prompt = f"""You are an expert proposal writer for {brand_name} ({brand_website}), a company specializing in:
- AI/ML solutions and implementations
- Microsoft Dynamics 365 implementations and integrations
- Cloud modernization and digital transformation
- Data analytics and automation solutions

Your task is to write a comprehensive Level 3 proposal response. The proposal should be:
- Professional and compelling
- Specific to the RFP requirements
- Highlighting {brand_name}'s relevant capabilities and experience
- Including clear methodology and approach
- Well-structured with clear sections

Structure the proposal with these sections:
1. Executive Summary
2. Understanding of Requirements
3. Proposed Solution & Approach
4. Technical Methodology
5. Team Qualifications
6. Project Timeline & Milestones
7. Risk Mitigation Strategy
8. Why {brand_name}
9. Conclusion & Next Steps

Write the proposal in markdown format with proper headings and formatting."""

        user_content = f"""Write a comprehensive proposal for this RFP:

**RFP Title:** {rfp_title}

**Issuing Agency:** {agency or 'Government Agency'}

**Focus Areas:** {tags_str}

{f'**Due Date:** {due_date}' if due_date else ''}

**RFP Description:**
{rfp_description}

{f'**Additional Requirements:** {additional_requirements}' if additional_requirements else ''}

Write a complete, professional proposal response."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        self.logger.info(
            "Generating proposal",
            extra={"title": rfp_title[:100], "agency": agency}
        )

        try:
            # Use higher max_tokens for full proposal generation
            markdown_content = self.complete(
                messages=messages,
                temperature=0.7,  # Balanced creativity for proposal writing
                max_tokens=8192,
            )

            # Extract executive summary (first major section after heading)
            exec_summary = ""
            lines = markdown_content.split("\n")
            in_exec_summary = False
            exec_lines = []

            for line in lines:
                if "Executive Summary" in line:
                    in_exec_summary = True
                    continue
                elif in_exec_summary:
                    if line.startswith("## "):  # Next section
                        break
                    exec_lines.append(line)

            exec_summary = "\n".join(exec_lines).strip()

            result = {
                "markdown_content": markdown_content,
                "executive_summary": exec_summary[:500] if exec_summary else "",
                "key_differentiators": [
                    f"Deep expertise in {tags_str}",
                    f"Proven track record with government clients",
                    f"Comprehensive support and maintenance",
                ],
            }

            self.logger.info(
                "Proposal generated",
                extra={
                    "title": rfp_title[:50],
                    "word_count": len(markdown_content.split()),
                }
            )

            return result

        except Exception as e:
            self.logger.error(
                f"Proposal generation failed: {e}",
                extra={"title": rfp_title[:100]}
            )
            raise

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this client session.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "endpoint": self.endpoint,
            "deployment": self.deployment,
            "api_version": self.api_version,
            "using_api_key": bool(self._api_key),
            "using_managed_identity": not bool(self._api_key),
        }

    def health_check(self) -> bool:
        """Perform a health check by making a minimal API call.

        Returns:
            True if the API is accessible and responding

        Note:
            This makes a minimal API call to verify connectivity.
        """
        try:
            messages = [
                {"role": "user", "content": "Reply with 'OK'"}
            ]

            result = self.complete(
                messages=messages,
                temperature=0,
                max_tokens=10,
            )

            return bool(result)

        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False

    def close(self) -> None:
        """Close the HTTP session and clean up resources."""
        if self._session:
            self._session.close()
            self._session = None
            self.logger.debug("LLM client session closed")

    def __enter__(self) -> "LLMClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
