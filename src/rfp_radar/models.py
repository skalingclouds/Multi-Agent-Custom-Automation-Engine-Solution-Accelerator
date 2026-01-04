"""Pydantic models for RFP Radar data structures."""

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class RFPSource(str, Enum):
    """Enumeration of RFP data sources."""

    GOVTRIBE = "govtribe"
    OPENGOV = "opengov"
    BIDNET = "bidnet"
    MANUAL = "manual"


class RFPStatus(str, Enum):
    """Enumeration of RFP processing status."""

    DISCOVERED = "discovered"
    FILTERED = "filtered"
    CLASSIFIED = "classified"
    STORED = "stored"
    PROPOSAL_GENERATED = "proposal_generated"
    NOTIFIED = "notified"
    SKIPPED = "skipped"
    ERROR = "error"


class RFPTag(str, Enum):
    """Enumeration of RFP classification tags."""

    AI = "AI"
    DYNAMICS = "Dynamics"
    MODERNIZATION = "Modernization"
    CLOUD = "Cloud"
    SECURITY = "Security"
    DATA = "Data"
    AUTOMATION = "Automation"
    OTHER = "Other"


class RFP(BaseModel):
    """Model representing a Request for Proposal."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="RFP title")
    description: str = Field(default="", description="RFP description or summary")
    agency: str = Field(default="", description="Issuing agency or organization")
    source: RFPSource = Field(default=RFPSource.MANUAL, description="Source portal")
    source_url: str = Field(default="", description="Original URL of the RFP listing")

    # Dates
    posted_date: Optional[datetime] = Field(
        default=None, description="Date the RFP was posted"
    )
    due_date: Optional[datetime] = Field(
        default=None, description="Proposal submission deadline"
    )
    discovered_at: datetime = Field(
        default_factory=datetime.utcnow, description="When we discovered this RFP"
    )

    # Geography
    location: str = Field(default="", description="Geographic location or region")
    country: str = Field(default="US", description="Country code (ISO 3166-1 alpha-2)")
    state: str = Field(default="", description="State or region code")

    # Documents
    pdf_url: Optional[str] = Field(
        default=None, description="URL to the primary RFP document"
    )
    attachments: List[str] = Field(
        default_factory=list, description="List of attachment URLs"
    )

    # Metadata
    status: RFPStatus = Field(
        default=RFPStatus.DISCOVERED, description="Current processing status"
    )
    naics_codes: List[str] = Field(
        default_factory=list, description="NAICS classification codes"
    )
    set_aside: str = Field(
        default="", description="Set-aside type (e.g., Small Business, 8(a))"
    )
    estimated_value: Optional[float] = Field(
        default=None, description="Estimated contract value in USD"
    )
    contract_type: str = Field(default="", description="Type of contract")

    # Raw data
    raw_data: dict = Field(
        default_factory=dict, description="Original raw data from scraper"
    )

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        """Validate and normalize country code."""
        if v:
            return v.upper()[:2]
        return "US"

    def is_us_based(self) -> bool:
        """Check if the RFP is US-based."""
        return self.country.upper() == "US"

    def age_in_days(self) -> int:
        """Calculate the age of the RFP in days from posted_date."""
        if self.posted_date is None:
            return 0
        delta = datetime.utcnow() - self.posted_date
        return delta.days

    def is_within_age_limit(self, max_age_days: int = 3) -> bool:
        """Check if the RFP is within the age limit."""
        return self.age_in_days() <= max_age_days


class ClassificationResult(BaseModel):
    """Model representing the AI classification result for an RFP."""

    rfp_id: str = Field(..., description="ID of the classified RFP")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance score between 0 and 1"
    )
    tags: List[RFPTag] = Field(
        default_factory=list, description="Classification tags"
    )
    reasoning: str = Field(
        default="", description="AI reasoning for the classification"
    )
    classified_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of classification"
    )
    model_used: str = Field(
        default="gpt-4o", description="Model used for classification"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in the classification"
    )

    @field_validator("relevance_score", "confidence")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Ensure score is within valid range."""
        return max(0.0, min(1.0, v))

    def is_relevant(self, threshold: float = 0.55) -> bool:
        """Check if the RFP meets the relevance threshold."""
        return self.relevance_score >= threshold


class ClassifiedRFP(BaseModel):
    """Model combining an RFP with its classification result."""

    rfp: RFP = Field(..., description="The RFP data")
    classification: ClassificationResult = Field(
        ..., description="Classification result"
    )

    def is_actionable(self, relevance_threshold: float = 0.55) -> bool:
        """Check if this classified RFP should be actioned (proposal generated)."""
        return (
            self.rfp.is_us_based()
            and self.classification.is_relevant(relevance_threshold)
        )


class ProposalMetadata(BaseModel):
    """Model representing metadata for a generated proposal."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rfp_id: str = Field(..., description="ID of the associated RFP")
    rfp_title: str = Field(default="", description="Title of the RFP")
    generated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the proposal was generated"
    )
    version: int = Field(default=1, description="Proposal version number")

    # Storage locations
    blob_url: str = Field(default="", description="Azure Blob Storage URL")
    blob_path: str = Field(default="", description="Path within the blob container")

    # Content info
    content_hash: str = Field(
        default="", description="SHA-256 hash of the proposal content"
    )
    word_count: int = Field(default=0, description="Word count of the proposal")
    section_count: int = Field(default=0, description="Number of sections in proposal")

    # Branding
    brand_name: str = Field(default="NAITIVE", description="Brand name used")
    brand_website: str = Field(
        default="https://www.naitive.cloud", description="Brand website URL"
    )

    # AI generation info
    model_used: str = Field(
        default="gpt-4o", description="Model used for generation"
    )
    prompt_tokens: int = Field(default=0, description="Tokens used in prompt")
    completion_tokens: int = Field(default=0, description="Tokens used in completion")


class Proposal(BaseModel):
    """Model representing a complete proposal with content."""

    metadata: ProposalMetadata = Field(..., description="Proposal metadata")
    markdown_content: str = Field(..., description="Full proposal in markdown format")

    @property
    def word_count(self) -> int:
        """Calculate word count of the proposal content."""
        return len(self.markdown_content.split())


class DigestEntry(BaseModel):
    """Model representing a single entry in the daily digest."""

    rfp: RFP = Field(..., description="The RFP data")
    classification: ClassificationResult = Field(
        ..., description="Classification result"
    )
    proposal_url: Optional[str] = Field(
        default=None, description="URL to the generated proposal"
    )

    def to_slack_block(self) -> dict:
        """Convert to a Slack block format."""
        tags_str = ", ".join([tag.value for tag in self.classification.tags])
        score_emoji = (
            ":star:" if self.classification.relevance_score >= 0.8
            else ":white_check_mark:" if self.classification.relevance_score >= 0.55
            else ":grey_question:"
        )

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{self.rfp.title}*\n"
                    f"{score_emoji} Score: {self.classification.relevance_score:.2f} | "
                    f"Tags: {tags_str or 'None'}\n"
                    f"Agency: {self.rfp.agency or 'Unknown'}"
                ),
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "View RFP"},
                "url": self.rfp.source_url or "#",
            } if self.rfp.source_url else None,
        }


class Digest(BaseModel):
    """Model representing the daily RFP digest."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the digest was generated"
    )
    entries: List[DigestEntry] = Field(
        default_factory=list, description="Digest entries"
    )
    total_discovered: int = Field(
        default=0, description="Total RFPs discovered in this run"
    )
    total_filtered: int = Field(
        default=0, description="RFPs after filtering"
    )
    total_relevant: int = Field(
        default=0, description="RFPs meeting relevance threshold"
    )
    total_proposals: int = Field(
        default=0, description="Proposals generated"
    )

    def is_empty(self) -> bool:
        """Check if the digest has no relevant entries."""
        return len(self.entries) == 0


class ScraperResult(BaseModel):
    """Model representing the result of a scraper run."""

    source: RFPSource = Field(..., description="Source of the scrape")
    scraped_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the scrape occurred"
    )
    success: bool = Field(default=True, description="Whether the scrape succeeded")
    error_message: str = Field(
        default="", description="Error message if scrape failed"
    )
    rfps: List[RFP] = Field(default_factory=list, description="Scraped RFPs")
    total_found: int = Field(default=0, description="Total RFPs found")
    duration_seconds: float = Field(
        default=0.0, description="Duration of the scrape in seconds"
    )

    @property
    def rfp_count(self) -> int:
        """Get the number of RFPs scraped."""
        return len(self.rfps)
