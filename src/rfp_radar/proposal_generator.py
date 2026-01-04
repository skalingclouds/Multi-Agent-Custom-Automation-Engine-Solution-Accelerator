# proposal_generator.py
"""Proposal Generator module for Level 3 full proposal creation.

This module provides the ProposalGenerator class which uses Azure OpenAI
to generate comprehensive proposals for relevant RFPs.
"""

import hashlib
from datetime import datetime
from typing import List, Optional, Tuple

from .config import config
from .llm_client import LLMClient
from .logging_utils import get_logger
from .models import (
    ClassifiedRFP,
    Proposal,
    ProposalMetadata,
    RFP,
    RFPStatus,
    RFPTag,
)
from .storage_client import StorageClient


class ProposalGenerator:
    """Generator for Level 3 full proposals.

    This class wraps the LLMClient and StorageClient to provide high-level
    proposal generation functionality, including batch processing, storage,
    and proper model conversion.

    Attributes:
        brand_name: Company brand name for proposals (default: NAITIVE).
        brand_website: Company website for proposals.
        llm_client: The LLMClient instance used for AI generation.
        storage_client: The StorageClient instance for storing proposals.
    """

    # Standard proposal sections for Level 3 proposals
    PROPOSAL_SECTIONS = [
        "Executive Summary",
        "Understanding of Requirements",
        "Proposed Solution & Approach",
        "Technical Methodology",
        "Team Qualifications",
        "Project Timeline & Milestones",
        "Risk Mitigation Strategy",
        "Why {brand_name}",
        "Conclusion & Next Steps",
    ]

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        storage_client: Optional[StorageClient] = None,
        brand_name: Optional[str] = None,
        brand_website: Optional[str] = None,
    ):
        """Initialize the Proposal Generator.

        Args:
            llm_client: Optional LLMClient instance. If not provided,
                       a new client will be created using config settings.
            storage_client: Optional StorageClient instance. If not provided,
                           a new client will be created using config settings.
            brand_name: Optional brand name override.
                       Defaults to config.NAITIVE_BRAND_NAME.
            brand_website: Optional brand website override.
                          Defaults to config.NAITIVE_WEBSITE.
        """
        self.logger = get_logger(__name__)

        # Use provided clients or create new ones
        self._llm_client = llm_client
        self._owns_llm_client = llm_client is None

        self._storage_client = storage_client
        self._owns_storage_client = storage_client is None

        # Set branding
        self.brand_name = brand_name or config.NAITIVE_BRAND_NAME
        self.brand_website = brand_website or config.NAITIVE_WEBSITE

        self.logger.info(
            "ProposalGenerator initialized",
            extra={
                "brand_name": self.brand_name,
                "brand_website": self.brand_website,
            }
        )

    @property
    def llm_client(self) -> LLMClient:
        """Get or create the LLM client.

        Returns:
            LLMClient instance for making generation requests.
        """
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    @property
    def storage_client(self) -> StorageClient:
        """Get or create the Storage client.

        Returns:
            StorageClient instance for storing proposals.
        """
        if self._storage_client is None:
            self._storage_client = StorageClient()
        return self._storage_client

    def generate(
        self,
        classified_rfp: ClassifiedRFP,
        store: bool = True,
    ) -> Proposal:
        """Generate a Level 3 proposal for a classified RFP.

        This method uses Azure OpenAI to generate a comprehensive proposal
        tailored to the RFP requirements and NAITIVE's capabilities.

        Args:
            classified_rfp: The ClassifiedRFP object containing the RFP
                           and its classification results.
            store: If True, automatically store the proposal in blob storage.

        Returns:
            Proposal object containing the generated content and metadata.

        Raises:
            ValueError: If proposal generation fails after retries.
        """
        rfp = classified_rfp.rfp
        classification = classified_rfp.classification

        self.logger.info(
            "Generating proposal",
            extra={
                "rfp_id": rfp.id,
                "title": rfp.title[:100],
                "relevance_score": classification.relevance_score,
            }
        )

        try:
            # Extract tags as strings for prompt
            tags = [tag.value for tag in classification.tags]

            # Format due date if available
            due_date_str = None
            if rfp.due_date:
                due_date_str = rfp.due_date.strftime("%Y-%m-%d")

            # Build additional requirements from classification
            additional_requirements = self._build_additional_requirements(
                rfp, classification
            )

            # Call LLM for proposal generation
            result = self.llm_client.generate_proposal(
                rfp_title=rfp.title,
                rfp_description=rfp.description,
                agency=rfp.agency,
                classification_tags=tags,
                due_date=due_date_str,
                additional_requirements=additional_requirements,
            )

            # Create Proposal model
            proposal = self._create_proposal(
                rfp=rfp,
                raw_result=result,
            )

            # Update RFP status
            rfp.status = RFPStatus.PROPOSAL_GENERATED

            self.logger.info(
                "Proposal generated successfully",
                extra={
                    "rfp_id": rfp.id,
                    "proposal_id": proposal.metadata.id,
                    "word_count": proposal.word_count,
                }
            )

            # Store the proposal if requested
            if store:
                blob_url = self.store_proposal(proposal)
                self.logger.info(
                    "Proposal stored",
                    extra={
                        "rfp_id": rfp.id,
                        "proposal_id": proposal.metadata.id,
                        "blob_url": blob_url,
                    }
                )

            return proposal

        except Exception as e:
            self.logger.error(
                f"Failed to generate proposal: {e}",
                extra={"rfp_id": rfp.id, "title": rfp.title[:50]}
            )
            raise ValueError(
                f"Proposal generation failed for RFP {rfp.id}: {e}"
            ) from e

    def generate_batch(
        self,
        classified_rfps: List[ClassifiedRFP],
        store: bool = True,
        skip_on_error: bool = True,
    ) -> List[Tuple[ClassifiedRFP, Proposal]]:
        """Generate proposals for a batch of classified RFPs.

        This method processes multiple RFPs, optionally skipping failures
        to allow partial batch completion.

        Args:
            classified_rfps: List of ClassifiedRFP objects to generate
                            proposals for.
            store: If True, store each proposal in blob storage.
            skip_on_error: If True, continue processing on generation
                          errors. If False, raise on first error.

        Returns:
            List of tuples containing (ClassifiedRFP, Proposal) pairs
            for successful generations.

        Raises:
            ValueError: If skip_on_error is False and generation fails.
        """
        self.logger.info(
            "Starting batch proposal generation",
            extra={"batch_size": len(classified_rfps)}
        )

        results: List[Tuple[ClassifiedRFP, Proposal]] = []
        errors: List[Tuple[str, str]] = []

        for classified_rfp in classified_rfps:
            rfp = classified_rfp.rfp
            try:
                proposal = self.generate(classified_rfp, store=store)
                results.append((classified_rfp, proposal))

            except Exception as e:
                error_msg = str(e)
                errors.append((rfp.id, error_msg))

                self.logger.warning(
                    f"Proposal generation failed for RFP {rfp.id}",
                    extra={"rfp_id": rfp.id, "error": error_msg}
                )

                if not skip_on_error:
                    raise ValueError(
                        f"Batch proposal generation failed on RFP {rfp.id}: {error_msg}"
                    ) from e

                # Mark as error but continue
                rfp.status = RFPStatus.ERROR

        self.logger.info(
            "Batch proposal generation completed",
            extra={
                "total": len(classified_rfps),
                "successful": len(results),
                "errors": len(errors),
            }
        )

        return results

    def store_proposal(self, proposal: Proposal) -> str:
        """Store a proposal in Azure Blob Storage.

        Args:
            proposal: The Proposal object to store.

        Returns:
            The blob URL of the stored proposal.
        """
        try:
            # Ensure container exists
            self.storage_client.ensure_container_exists()

            # Upload the proposal
            blob_url = self.storage_client.upload_proposal(proposal)

            self.logger.debug(
                "Proposal stored in blob storage",
                extra={
                    "proposal_id": proposal.metadata.id,
                    "rfp_id": proposal.metadata.rfp_id,
                    "blob_url": blob_url,
                }
            )

            return blob_url

        except Exception as e:
            self.logger.error(
                f"Failed to store proposal: {e}",
                extra={"proposal_id": proposal.metadata.id}
            )
            raise

    def regenerate(
        self,
        classified_rfp: ClassifiedRFP,
        version: int = 1,
        store: bool = True,
    ) -> Proposal:
        """Regenerate a proposal with a new version number.

        This is useful for creating updated proposals when requirements
        change or when a fresh perspective is needed.

        Args:
            classified_rfp: The ClassifiedRFP object to regenerate for.
            version: The version number for the new proposal.
            store: If True, store the regenerated proposal.

        Returns:
            New Proposal object with updated version.
        """
        self.logger.info(
            "Regenerating proposal",
            extra={
                "rfp_id": classified_rfp.rfp.id,
                "version": version,
            }
        )

        proposal = self.generate(classified_rfp, store=False)

        # Update version number
        proposal.metadata.version = version

        if store:
            self.store_proposal(proposal)

        return proposal

    def _build_additional_requirements(
        self,
        rfp: RFP,
        classification,
    ) -> str:
        """Build additional requirements context from RFP and classification.

        Args:
            rfp: The RFP model.
            classification: The ClassificationResult model.

        Returns:
            String with additional context for proposal generation.
        """
        context_parts = []

        # Add classification reasoning as context
        if classification.reasoning:
            context_parts.append(
                f"Classification Analysis: {classification.reasoning}"
            )

        # Add NAICS codes if available
        if rfp.naics_codes:
            context_parts.append(
                f"Industry Classification (NAICS): {', '.join(rfp.naics_codes)}"
            )

        # Add estimated value context
        if rfp.estimated_value:
            context_parts.append(
                f"Estimated Contract Value: ${rfp.estimated_value:,.2f}"
            )

        # Add contract type
        if rfp.contract_type:
            context_parts.append(f"Contract Type: {rfp.contract_type}")

        # Add set-aside requirements
        if rfp.set_aside:
            context_parts.append(f"Set-Aside Requirements: {rfp.set_aside}")

        # Add location context
        if rfp.location:
            context_parts.append(f"Service Location: {rfp.location}")
        elif rfp.state:
            context_parts.append(f"State: {rfp.state}")

        return "\n".join(context_parts) if context_parts else ""

    def _create_proposal(
        self,
        rfp: RFP,
        raw_result: dict,
    ) -> Proposal:
        """Create a Proposal model from raw LLM response.

        Args:
            rfp: The RFP model.
            raw_result: Dictionary from LLMClient.generate_proposal().

        Returns:
            Proposal model with content and metadata.
        """
        markdown_content = raw_result.get("markdown_content", "")

        # Calculate content hash
        content_hash = hashlib.sha256(
            markdown_content.encode("utf-8")
        ).hexdigest()

        # Count sections in the proposal
        section_count = self._count_sections(markdown_content)

        # Create metadata
        metadata = ProposalMetadata(
            rfp_id=rfp.id,
            rfp_title=rfp.title,
            generated_at=datetime.utcnow(),
            version=1,
            content_hash=content_hash,
            word_count=len(markdown_content.split()),
            section_count=section_count,
            brand_name=self.brand_name,
            brand_website=self.brand_website,
            model_used=config.AZURE_OPENAI_DEPLOYMENT,
        )

        return Proposal(
            metadata=metadata,
            markdown_content=markdown_content,
        )

    def _count_sections(self, markdown_content: str) -> int:
        """Count the number of sections in a markdown document.

        Args:
            markdown_content: The markdown content to analyze.

        Returns:
            Number of level-2 headings (## ) in the document.
        """
        lines = markdown_content.split("\n")
        section_count = sum(
            1 for line in lines
            if line.strip().startswith("## ")
        )
        return section_count

    def get_stats(self) -> dict:
        """Get generator statistics and configuration.

        Returns:
            Dictionary with generator configuration and client stats.
        """
        return {
            "brand_name": self.brand_name,
            "brand_website": self.brand_website,
            "proposal_sections": [
                section.format(brand_name=self.brand_name)
                for section in self.PROPOSAL_SECTIONS
            ],
            "llm_stats": self.llm_client.get_usage_stats(),
        }

    def health_check(self) -> bool:
        """Perform a health check on the generator.

        This verifies that both the LLM client and storage client
        are operational.

        Returns:
            True if the generator is healthy, False otherwise.
        """
        try:
            # Check LLM client
            llm_healthy = self.llm_client.health_check()
            if not llm_healthy:
                self.logger.warning("LLM client health check failed")
                return False

            # Check storage client (simple existence check)
            try:
                self.storage_client.ensure_container_exists()
                storage_healthy = True
            except Exception:
                storage_healthy = False

            if not storage_healthy:
                self.logger.warning("Storage client health check failed")
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False

    def close(self) -> None:
        """Close the generator and clean up resources.

        This closes the LLM client and storage client if they were
        created by this generator.
        """
        if self._owns_llm_client and self._llm_client is not None:
            self._llm_client.close()
            self._llm_client = None

        # Storage client doesn't require explicit cleanup
        if self._owns_storage_client:
            self._storage_client = None

        self.logger.debug("ProposalGenerator closed")

    def __enter__(self) -> "ProposalGenerator":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
