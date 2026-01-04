# digest_builder.py
"""Digest Builder module for Slack message formatting.

This module provides the DigestBuilder class which creates formatted
Slack digests from classified RFPs and generated proposals.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import config
from .logging_utils import get_logger
from .models import (
    ClassifiedRFP,
    Digest,
    DigestEntry,
    Proposal,
    RFPTag,
)


class DigestBuilder:
    """Builder for RFP digest messages.

    This class constructs Digest objects and Slack-formatted messages
    from classified RFPs and generated proposals. It handles formatting,
    truncation, and proper message structure for Slack Block Kit.

    Attributes:
        brand_name: Company brand name for digest branding.
        brand_website: Company website for digest branding.
        max_entries_per_digest: Maximum entries to include in a single digest.
    """

    # Maximum number of entries to include in a single digest message
    # (Slack has message size limits)
    DEFAULT_MAX_ENTRIES = 10

    # Score thresholds for emoji indicators
    SCORE_HIGH_THRESHOLD = 0.8
    SCORE_MEDIUM_THRESHOLD = 0.55

    # Score indicator emojis
    EMOJI_HIGH = ":star:"
    EMOJI_MEDIUM = ":white_check_mark:"
    EMOJI_LOW = ":grey_question:"

    # Standard emojis used in messages
    EMOJI_RADAR = ":radar:"
    EMOJI_INFO = ":information_source:"
    EMOJI_CALENDAR = ":calendar:"
    EMOJI_ROBOT = ":robot_face:"
    EMOJI_PAGE = ":page_facing_up:"

    def __init__(
        self,
        brand_name: Optional[str] = None,
        brand_website: Optional[str] = None,
        max_entries: Optional[int] = None,
    ):
        """Initialize the Digest Builder.

        Args:
            brand_name: Optional brand name override.
                       Defaults to config.NAITIVE_BRAND_NAME.
            brand_website: Optional brand website override.
                          Defaults to config.NAITIVE_WEBSITE.
            max_entries: Optional maximum entries per digest.
                        Defaults to DEFAULT_MAX_ENTRIES.
        """
        self.logger = get_logger(__name__)

        # Set branding
        self.brand_name = brand_name or config.NAITIVE_BRAND_NAME
        self.brand_website = brand_website or config.NAITIVE_WEBSITE

        # Set max entries
        self.max_entries_per_digest = (
            max_entries if max_entries is not None else self.DEFAULT_MAX_ENTRIES
        )

        self.logger.info(
            "DigestBuilder initialized",
            extra={
                "brand_name": self.brand_name,
                "max_entries": self.max_entries_per_digest,
            }
        )

    def build_digest(
        self,
        classified_rfps: List[ClassifiedRFP],
        proposals: Optional[Dict[str, Proposal]] = None,
        total_discovered: int = 0,
        total_filtered: int = 0,
    ) -> Digest:
        """Build a Digest from classified RFPs and proposals.

        This method creates a Digest model containing DigestEntry objects
        for each classified RFP, optionally including proposal URLs.

        Args:
            classified_rfps: List of ClassifiedRFP objects to include.
            proposals: Optional dictionary mapping RFP IDs to Proposal objects.
            total_discovered: Total number of RFPs discovered in the run.
            total_filtered: Number of RFPs after age/geography filtering.

        Returns:
            Digest model with entries sorted by relevance score.
        """
        self.logger.info(
            "Building digest",
            extra={
                "classified_count": len(classified_rfps),
                "has_proposals": proposals is not None,
                "total_discovered": total_discovered,
            }
        )

        # Create digest entries
        entries: List[DigestEntry] = []
        proposals_dict = proposals or {}

        for classified_rfp in classified_rfps:
            rfp = classified_rfp.rfp
            classification = classified_rfp.classification

            # Get proposal URL if available
            proposal_url = None
            if rfp.id in proposals_dict:
                proposal = proposals_dict[rfp.id]
                proposal_url = proposal.metadata.blob_url or None

            entry = DigestEntry(
                rfp=rfp,
                classification=classification,
                proposal_url=proposal_url,
            )
            entries.append(entry)

        # Sort entries by relevance score (highest first)
        entries.sort(
            key=lambda e: e.classification.relevance_score,
            reverse=True,
        )

        # Create digest
        digest = Digest(
            generated_at=datetime.utcnow(),
            entries=entries,
            total_discovered=total_discovered,
            total_filtered=total_filtered,
            total_relevant=len(classified_rfps),
            total_proposals=len(proposals_dict),
        )

        self.logger.info(
            "Digest built successfully",
            extra={
                "digest_id": digest.id,
                "entry_count": len(entries),
                "total_relevant": digest.total_relevant,
                "total_proposals": digest.total_proposals,
            }
        )

        return digest

    def build_digest_from_results(
        self,
        results: List[Tuple[ClassifiedRFP, Proposal]],
        total_discovered: int = 0,
        total_filtered: int = 0,
    ) -> Digest:
        """Build a Digest from proposal generation results.

        This is a convenience method that takes the output of
        ProposalGenerator.generate_batch() directly.

        Args:
            results: List of (ClassifiedRFP, Proposal) tuples from
                    proposal generation.
            total_discovered: Total number of RFPs discovered.
            total_filtered: Number of RFPs after filtering.

        Returns:
            Digest model with entries.
        """
        classified_rfps = []
        proposals: Dict[str, Proposal] = {}

        for classified_rfp, proposal in results:
            classified_rfps.append(classified_rfp)
            proposals[classified_rfp.rfp.id] = proposal

        return self.build_digest(
            classified_rfps=classified_rfps,
            proposals=proposals,
            total_discovered=total_discovered,
            total_filtered=total_filtered,
        )

    def build_empty_digest(
        self,
        total_discovered: int = 0,
        total_filtered: int = 0,
    ) -> Digest:
        """Build an empty digest indicating no relevant RFPs were found.

        Args:
            total_discovered: Total RFPs discovered in the run.
            total_filtered: RFPs after age/geography filtering.

        Returns:
            Empty Digest model with statistics.
        """
        return Digest(
            generated_at=datetime.utcnow(),
            entries=[],
            total_discovered=total_discovered,
            total_filtered=total_filtered,
            total_relevant=0,
            total_proposals=0,
        )

    def format_slack_blocks(self, digest: Digest) -> List[Dict]:
        """Format a Digest as Slack Block Kit blocks.

        This method creates properly formatted Slack blocks for
        posting a rich digest message to a channel.

        Args:
            digest: The Digest model to format.

        Returns:
            List of Slack block dictionaries.
        """
        blocks: List[Dict] = []

        # Header block
        blocks.append(self._build_header_block())

        # Summary section
        if digest.is_empty():
            blocks.append(self._build_empty_summary_block(digest))
        else:
            blocks.append(self._build_summary_block(digest))
            blocks.append({"type": "divider"})

            # Add entry blocks (limited to max_entries)
            for entry in digest.entries[:self.max_entries_per_digest]:
                blocks.append(self._build_entry_block(entry))

                # Add proposal link context if available
                if entry.proposal_url:
                    blocks.append(self._build_proposal_link_block(entry))

            # Truncation notice if needed
            if len(digest.entries) > self.max_entries_per_digest:
                remaining = len(digest.entries) - self.max_entries_per_digest
                blocks.append(self._build_truncation_notice(remaining))

        # Footer divider and branding
        blocks.append({"type": "divider"})
        blocks.append(self._build_footer_block())

        return blocks

    def format_fallback_text(self, digest: Digest) -> str:
        """Format a Digest as plain text fallback for notifications.

        This text is used for Slack notifications and accessibility.

        Args:
            digest: The Digest model to format.

        Returns:
            Plain text summary string.
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

    def format_entry_text(self, entry: DigestEntry) -> str:
        """Format a single DigestEntry as plain text.

        Args:
            entry: The DigestEntry to format.

        Returns:
            Formatted plain text string.
        """
        rfp = entry.rfp
        classification = entry.classification

        # Format tags
        tags_str = ", ".join([tag.value for tag in classification.tags])
        if not tags_str:
            tags_str = "None"

        lines = [
            f"Title: {rfp.title}",
            f"Relevance: {classification.relevance_score:.0%}",
            f"Tags: {tags_str}",
        ]

        if rfp.agency:
            lines.append(f"Agency: {rfp.agency}")

        if rfp.due_date:
            lines.append(f"Due: {rfp.due_date.strftime('%Y-%m-%d')}")

        if entry.proposal_url:
            lines.append(f"Proposal: {entry.proposal_url}")

        return "\n".join(lines)

    def get_score_emoji(self, score: float) -> str:
        """Get the appropriate emoji for a relevance score.

        Args:
            score: Relevance score between 0 and 1.

        Returns:
            Emoji string based on score thresholds.
        """
        if score >= self.SCORE_HIGH_THRESHOLD:
            return self.EMOJI_HIGH
        elif score >= self.SCORE_MEDIUM_THRESHOLD:
            return self.EMOJI_MEDIUM
        return self.EMOJI_LOW

    def _build_header_block(self) -> Dict:
        """Build the digest header block.

        Returns:
            Slack header block dictionary.
        """
        return {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{self.EMOJI_RADAR} {self.brand_name} RFP Radar Daily Digest",
                "emoji": True,
            },
        }

    def _build_summary_block(self, digest: Digest) -> Dict:
        """Build the summary block for a non-empty digest.

        Args:
            digest: The Digest model.

        Returns:
            Slack section block dictionary.
        """
        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f":white_check_mark: *Found {len(digest.entries)} "
                    f"relevant RFPs!*\n\n"
                    f"\u2022 Discovered: {digest.total_discovered} RFPs\n"
                    f"\u2022 After filtering: {digest.total_filtered} RFPs\n"
                    f"\u2022 Relevant: {digest.total_relevant} RFPs\n"
                    f"\u2022 Proposals generated: {digest.total_proposals}"
                ),
            },
        }

    def _build_empty_summary_block(self, digest: Digest) -> Dict:
        """Build the summary block for an empty digest.

        Args:
            digest: The empty Digest model.

        Returns:
            Slack section block dictionary.
        """
        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{self.EMOJI_INFO} *No new relevant RFPs today.*\n\n"
                    f"\u2022 Discovered: {digest.total_discovered} RFPs\n"
                    f"\u2022 After filtering: {digest.total_filtered} RFPs\n"
                    f"\u2022 Meeting relevance threshold: {digest.total_relevant} RFPs"
                ),
            },
        }

    def _build_entry_block(self, entry: DigestEntry) -> Dict:
        """Build a Slack block for a single digest entry.

        Args:
            entry: The DigestEntry to format.

        Returns:
            Slack section block dictionary.
        """
        rfp = entry.rfp
        classification = entry.classification

        # Score indicator emoji
        score_emoji = self.get_score_emoji(classification.relevance_score)

        # Format tags
        tags_str = ", ".join([tag.value for tag in classification.tags])
        if not tags_str:
            tags_str = "None"

        # Build main text (truncate title if too long)
        title = rfp.title[:100] + "..." if len(rfp.title) > 100 else rfp.title
        text_parts = [
            f"*{title}*",
            f"{score_emoji} Relevance: {classification.relevance_score:.0%} | Tags: {tags_str}",
        ]

        if rfp.agency:
            text_parts.append(f"Agency: {rfp.agency}")

        if rfp.due_date:
            due_str = rfp.due_date.strftime("%Y-%m-%d")
            text_parts.append(f"{self.EMOJI_CALENDAR} Due: {due_str}")

        block: Dict = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(text_parts),
            },
        }

        # Add View RFP button if source URL available
        if rfp.source_url:
            block["accessory"] = {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "View RFP",
                    "emoji": True,
                },
                "url": rfp.source_url,
                "action_id": f"view_rfp_{rfp.id[:8]}",
            }

        return block

    def _build_proposal_link_block(self, entry: DigestEntry) -> Dict:
        """Build a context block with proposal link.

        Args:
            entry: The DigestEntry with a proposal URL.

        Returns:
            Slack context block dictionary.
        """
        return {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"{self.EMOJI_PAGE} <{entry.proposal_url}|View Proposal>",
                }
            ],
        }

    def _build_truncation_notice(self, remaining: int) -> Dict:
        """Build a context block for truncation notice.

        Args:
            remaining: Number of entries not shown.

        Returns:
            Slack context block dictionary.
        """
        return {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"_...and {remaining} more RFPs_",
                }
            ],
        }

    def _build_footer_block(self) -> Dict:
        """Build the digest footer block with branding.

        Returns:
            Slack context block dictionary.
        """
        return {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"{self.EMOJI_ROBOT} Powered by *{self.brand_name}* "
                        f"RFP Radar | <{self.brand_website}|{self.brand_website}>"
                    ),
                },
            ],
        }

    def format_error_blocks(
        self,
        error_message: str,
        error_type: str = "Error",
    ) -> List[Dict]:
        """Format an error notification as Slack blocks.

        Args:
            error_message: The error message to display.
            error_type: Type of error (e.g., "Scraping Error").

        Returns:
            List of Slack block dictionaries.
        """
        return [
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
                        "text": f"{self.EMOJI_ROBOT} This is an automated notification",
                    },
                ],
            },
        ]

    def format_error_fallback(
        self,
        error_message: str,
        error_type: str = "Error",
    ) -> str:
        """Format an error notification as plain text.

        Args:
            error_message: The error message.
            error_type: Type of error.

        Returns:
            Plain text error message.
        """
        return (
            f"{self.brand_name} RFP Radar {error_type}: "
            f"{error_message[:200]}"
        )

    def get_stats(self) -> dict:
        """Get builder statistics and configuration.

        Returns:
            Dictionary with builder configuration.
        """
        return {
            "brand_name": self.brand_name,
            "brand_website": self.brand_website,
            "max_entries_per_digest": self.max_entries_per_digest,
            "score_thresholds": {
                "high": self.SCORE_HIGH_THRESHOLD,
                "medium": self.SCORE_MEDIUM_THRESHOLD,
            },
        }

    def __enter__(self) -> "DigestBuilder":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        # DigestBuilder doesn't hold any resources requiring cleanup
        pass
