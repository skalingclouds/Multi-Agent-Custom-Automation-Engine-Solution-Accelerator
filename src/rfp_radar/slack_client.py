# slack_client.py
"""Slack SDK wrapper for posting RFP digests to Slack channels."""

import time
from typing import Any, Dict, List, Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from .config import config
from .logging_utils import get_logger
from .models import Digest, DigestEntry


class SlackClient:
    """Slack SDK client wrapper for RFP Radar.

    This class provides methods for posting RFP digests and messages
    to Slack channels. It handles authentication, retries, and error
    handling according to the RFP Radar specifications.
    """

    # Maximum number of retries for Slack API calls
    MAX_RETRIES = 3

    # Base delay for exponential backoff (in seconds)
    BASE_RETRY_DELAY = 1.0

    def __init__(
        self,
        token: Optional[str] = None,
        channel: Optional[str] = None,
    ):
        """Initialize the Slack client.

        Args:
            token: Slack bot token (xoxb-...). Defaults to config value.
            channel: Default channel to post to. Defaults to config value.
        """
        self.logger = get_logger(__name__)

        self._token = token or config.SLACK_BOT_TOKEN
        self.default_channel = channel or config.SLACK_CHANNEL

        # Initialize Slack WebClient
        self._client: Optional[WebClient] = None

        # Branding configuration
        self.brand_name = config.NAITIVE_BRAND_NAME
        self.brand_website = config.NAITIVE_WEBSITE

    def _get_client(self) -> WebClient:
        """Get or create the Slack WebClient.

        Returns:
            Configured WebClient instance
        """
        if self._client is None:
            self._client = WebClient(token=self._token)
        return self._client

    def _make_request_with_retry(
        self,
        method: str,
        retry_count: int = 0,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Make a Slack API request with retry logic.

        Args:
            method: The Slack API method to call
            retry_count: Current retry attempt number
            **kwargs: Arguments to pass to the API method

        Returns:
            API response dictionary, or None on failure after retries
        """
        client = self._get_client()

        try:
            # Get the method function from the client
            api_method = getattr(client, method)
            response = api_method(**kwargs)

            self.logger.debug(
                "Slack API request succeeded",
                extra={
                    "method": method,
                    "channel": kwargs.get("channel"),
                }
            )

            return response.data

        except SlackApiError as e:
            error_code = e.response.get("error", "unknown_error")
            self.logger.warning(
                f"Slack API error: {error_code}",
                extra={
                    "method": method,
                    "error": error_code,
                    "retry_count": retry_count,
                }
            )

            # Retry on rate limiting or transient errors
            retryable_errors = [
                "rate_limited",
                "service_unavailable",
                "internal_error",
                "request_timeout",
            ]

            if error_code in retryable_errors and retry_count < self.MAX_RETRIES:
                # Get retry delay from headers or use exponential backoff
                retry_after = int(
                    e.response.headers.get("Retry-After", 2 ** retry_count)
                )
                delay = max(retry_after, self.BASE_RETRY_DELAY * (2 ** retry_count))

                self.logger.info(
                    f"Retrying Slack API request after {delay}s",
                    extra={"retry_count": retry_count + 1}
                )
                time.sleep(delay)

                return self._make_request_with_retry(
                    method=method,
                    retry_count=retry_count + 1,
                    **kwargs,
                )

            # Log error but don't fail - per spec, don't block on Slack failures
            self.logger.error(
                f"Slack API request failed after {retry_count} retries: {e}",
                extra={
                    "method": method,
                    "error": error_code,
                }
            )
            return None

        except Exception as e:
            self.logger.error(
                f"Unexpected error in Slack API request: {e}",
                extra={"method": method}
            )
            return None

    def post_message(
        self,
        text: str,
        channel: Optional[str] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
        thread_ts: Optional[str] = None,
        unfurl_links: bool = False,
    ) -> Optional[str]:
        """Post a message to a Slack channel.

        Args:
            text: The message text (used as fallback for notifications)
            channel: Channel to post to (defaults to configured channel)
            blocks: Optional Slack blocks for rich formatting
            thread_ts: Optional thread timestamp to reply to
            unfurl_links: Whether to unfurl URLs in the message

        Returns:
            Message timestamp (ts) if successful, None on failure
        """
        target_channel = channel or self.default_channel

        self.logger.info(
            "Posting message to Slack",
            extra={
                "channel": target_channel,
                "has_blocks": blocks is not None,
            }
        )

        response = self._make_request_with_retry(
            "chat_postMessage",
            channel=target_channel,
            text=text,
            blocks=blocks,
            thread_ts=thread_ts,
            unfurl_links=unfurl_links,
        )

        if response and response.get("ok"):
            message_ts = response.get("ts")
            self.logger.info(
                "Message posted successfully",
                extra={
                    "channel": target_channel,
                    "message_ts": message_ts,
                }
            )
            return message_ts

        return None

    def post_digest(
        self,
        digest: Digest,
        channel: Optional[str] = None,
    ) -> Optional[str]:
        """Post an RFP digest to a Slack channel.

        Args:
            digest: The Digest model containing RFP entries
            channel: Channel to post to (defaults to configured channel)

        Returns:
            Message timestamp (ts) if successful, None on failure
        """
        target_channel = channel or self.default_channel

        # Build the digest message
        blocks = self._build_digest_blocks(digest)
        fallback_text = self._build_digest_fallback(digest)

        self.logger.info(
            "Posting RFP digest to Slack",
            extra={
                "channel": target_channel,
                "entry_count": len(digest.entries),
                "total_discovered": digest.total_discovered,
                "total_relevant": digest.total_relevant,
            }
        )

        return self.post_message(
            text=fallback_text,
            channel=target_channel,
            blocks=blocks,
        )

    def _build_digest_blocks(self, digest: Digest) -> List[Dict[str, Any]]:
        """Build Slack blocks for a digest message.

        Args:
            digest: The Digest model

        Returns:
            List of Slack block dictionaries
        """
        blocks: List[Dict[str, Any]] = []

        # Header block
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f":radar: {self.brand_name} RFP Radar Daily Digest",
                "emoji": True,
            },
        })

        # Summary section
        if digest.is_empty():
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        ":information_source: *No new relevant RFPs today.*\n\n"
                        f"• Discovered: {digest.total_discovered} RFPs\n"
                        f"• After filtering: {digest.total_filtered} RFPs\n"
                        f"• Meeting relevance threshold: {digest.total_relevant} RFPs"
                    ),
                },
            })
        else:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":white_check_mark: *Found {len(digest.entries)} relevant RFPs!*\n\n"
                        f"• Discovered: {digest.total_discovered} RFPs\n"
                        f"• After filtering: {digest.total_filtered} RFPs\n"
                        f"• Relevant: {digest.total_relevant} RFPs\n"
                        f"• Proposals generated: {digest.total_proposals}"
                    ),
                },
            })

            # Divider
            blocks.append({"type": "divider"})

            # Add each RFP entry (limit to 10 for message size)
            for entry in digest.entries[:10]:
                entry_block = self._build_entry_block(entry)
                blocks.append(entry_block)

                # Add proposal link if available
                if entry.proposal_url:
                    blocks.append({
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f":page_facing_up: <{entry.proposal_url}|View Proposal>",
                            }
                        ],
                    })

            # Note if entries were truncated
            if len(digest.entries) > 10:
                blocks.append({
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"_...and {len(digest.entries) - 10} more RFPs_",
                        }
                    ],
                })

        # Divider before footer
        blocks.append({"type": "divider"})

        # Footer with branding
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f":robot_face: Powered by *{self.brand_name}* RFP Radar | "
                        f"<{self.brand_website}|{self.brand_website}>"
                    ),
                },
            ],
        })

        return blocks

    def _build_entry_block(self, entry: DigestEntry) -> Dict[str, Any]:
        """Build a Slack block for a single digest entry.

        Args:
            entry: The DigestEntry model

        Returns:
            Slack block dictionary
        """
        # Score indicator emoji
        score = entry.classification.relevance_score
        if score >= 0.8:
            score_emoji = ":star:"
        elif score >= 0.55:
            score_emoji = ":white_check_mark:"
        else:
            score_emoji = ":grey_question:"

        # Format tags
        tags_str = ", ".join([tag.value for tag in entry.classification.tags])
        if not tags_str:
            tags_str = "None"

        # Build the main text
        text_parts = [
            f"*{entry.rfp.title[:100]}*",
            f"{score_emoji} Relevance: {score:.0%} | Tags: {tags_str}",
        ]

        if entry.rfp.agency:
            text_parts.append(f"Agency: {entry.rfp.agency}")

        if entry.rfp.due_date:
            due_str = entry.rfp.due_date.strftime("%Y-%m-%d")
            text_parts.append(f":calendar: Due: {due_str}")

        block: Dict[str, Any] = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(text_parts),
            },
        }

        # Add link button if source URL available
        if entry.rfp.source_url:
            block["accessory"] = {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "View RFP",
                    "emoji": True,
                },
                "url": entry.rfp.source_url,
                "action_id": f"view_rfp_{entry.rfp.id[:8]}",
            }

        return block

    def _build_digest_fallback(self, digest: Digest) -> str:
        """Build fallback text for digest message.

        Args:
            digest: The Digest model

        Returns:
            Plain text fallback string
        """
        if digest.is_empty():
            return (
                f"{self.brand_name} RFP Radar: No new relevant RFPs today. "
                f"Discovered {digest.total_discovered}, "
                f"filtered to {digest.total_filtered}."
            )

        return (
            f"{self.brand_name} RFP Radar: Found {len(digest.entries)} "
            f"relevant RFPs! Generated {digest.total_proposals} proposals."
        )

    def post_empty_digest(
        self,
        total_discovered: int = 0,
        total_filtered: int = 0,
        channel: Optional[str] = None,
    ) -> Optional[str]:
        """Post a message indicating no new relevant RFPs were found.

        Args:
            total_discovered: Total RFPs discovered
            total_filtered: RFPs after filtering
            channel: Channel to post to

        Returns:
            Message timestamp (ts) if successful, None on failure
        """
        digest = Digest(
            total_discovered=total_discovered,
            total_filtered=total_filtered,
            total_relevant=0,
            total_proposals=0,
            entries=[],
        )

        return self.post_digest(digest, channel=channel)

    def post_error_notification(
        self,
        error_message: str,
        error_type: str = "Error",
        channel: Optional[str] = None,
    ) -> Optional[str]:
        """Post an error notification to Slack.

        Args:
            error_message: The error message to display
            error_type: Type of error (e.g., "Scraping Error", "API Error")
            channel: Channel to post to

        Returns:
            Message timestamp (ts) if successful, None on failure
        """
        target_channel = channel or self.default_channel

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f":warning: {self.brand_name} RFP Radar {error_type}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"```{error_message[:2500]}```",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": ":robot_face: This is an automated notification",
                    },
                ],
            },
        ]

        fallback = f"{self.brand_name} RFP Radar {error_type}: {error_message[:200]}"

        return self.post_message(
            text=fallback,
            channel=target_channel,
            blocks=blocks,
        )

    def test_connection(self) -> bool:
        """Test the Slack connection by calling auth.test.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = self._make_request_with_retry("auth_test")

            if response and response.get("ok"):
                self.logger.info(
                    "Slack connection test successful",
                    extra={
                        "team": response.get("team"),
                        "user": response.get("user"),
                    }
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Slack connection test failed: {e}")
            return False

    def get_channel_info(self, channel: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get information about a channel.

        Args:
            channel: Channel ID or name (defaults to configured channel)

        Returns:
            Channel info dictionary, or None on failure
        """
        target_channel = channel or self.default_channel

        response = self._make_request_with_retry(
            "conversations_info",
            channel=target_channel,
        )

        if response and response.get("ok"):
            return response.get("channel")

        return None

    def health_check(self) -> bool:
        """Perform a health check by testing the connection.

        Returns:
            True if healthy, False otherwise
        """
        return self.test_connection()

    def close(self) -> None:
        """Clean up client resources."""
        self._client = None
        self.logger.debug("Slack client closed")

    def __enter__(self) -> "SlackClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
