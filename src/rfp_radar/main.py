# main.py
"""RFP Radar Main Orchestrator.

This module provides the main entry point for the RFP Radar service.
It orchestrates the complete pipeline:
    1. Scrape RFPs from multiple portals
    2. Filter by age and geography (done in scrapers)
    3. Classify RFPs using Azure OpenAI
    4. Filter by relevance threshold
    5. Store RFPs in Azure Blob Storage
    6. Index RFPs in Azure AI Search
    7. Generate proposals for relevant RFPs
    8. Build digest and post to Slack
"""

import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .classifier import RFPClassifier
from .config import config
from .digest_builder import DigestBuilder
from .logging_utils import get_logger, setup_logging
from .models import (
    ClassifiedRFP,
    Digest,
    Proposal,
    RFP,
    RFPStatus,
    ScraperResult,
)
from .proposal_generator import ProposalGenerator
from .scrapers import SCRAPERS, get_available_sources
from .search_client import SearchClient
from .slack_client import SlackClient
from .storage_client import StorageClient


class RFPRadarPipeline:
    """Main orchestrator for the RFP Radar processing pipeline.

    This class coordinates all components of the RFP Radar system:
    - Scrapers for fetching RFPs from government portals
    - Classifier for AI-based relevance scoring
    - Storage and Search clients for Azure integration
    - Proposal generator for creating proposals
    - Slack client for notifications

    Attributes:
        classifier: RFPClassifier instance for AI classification.
        proposal_generator: ProposalGenerator instance for proposal creation.
        storage_client: StorageClient for Azure Blob Storage.
        search_client: SearchClient for Azure AI Search.
        slack_client: SlackClient for Slack notifications.
        digest_builder: DigestBuilder for formatting digests.
    """

    def __init__(
        self,
        classifier: Optional[RFPClassifier] = None,
        proposal_generator: Optional[ProposalGenerator] = None,
        storage_client: Optional[StorageClient] = None,
        search_client: Optional[SearchClient] = None,
        slack_client: Optional[SlackClient] = None,
        digest_builder: Optional[DigestBuilder] = None,
    ):
        """Initialize the RFP Radar pipeline.

        Args:
            classifier: Optional RFPClassifier instance.
            proposal_generator: Optional ProposalGenerator instance.
            storage_client: Optional StorageClient instance.
            search_client: Optional SearchClient instance.
            slack_client: Optional SlackClient instance.
            digest_builder: Optional DigestBuilder instance.
        """
        self.logger = get_logger(__name__)

        # Initialize clients (lazy initialization for optional overrides)
        self._classifier = classifier
        self._proposal_generator = proposal_generator
        self._storage_client = storage_client
        self._search_client = search_client
        self._slack_client = slack_client
        self._digest_builder = digest_builder

        # Track ownership for cleanup
        self._owns_classifier = classifier is None
        self._owns_proposal_generator = proposal_generator is None
        self._owns_storage_client = storage_client is None
        self._owns_search_client = search_client is None
        self._owns_slack_client = slack_client is None

        # Pipeline statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_discovered": 0,
            "total_filtered": 0,
            "total_classified": 0,
            "total_relevant": 0,
            "total_proposals": 0,
            "scraper_results": {},
            "errors": [],
        }

    @property
    def classifier(self) -> RFPClassifier:
        """Get or create the RFP classifier."""
        if self._classifier is None:
            self._classifier = RFPClassifier()
        return self._classifier

    @property
    def proposal_generator(self) -> ProposalGenerator:
        """Get or create the proposal generator."""
        if self._proposal_generator is None:
            self._proposal_generator = ProposalGenerator()
        return self._proposal_generator

    @property
    def storage_client(self) -> StorageClient:
        """Get or create the storage client."""
        if self._storage_client is None:
            self._storage_client = StorageClient()
        return self._storage_client

    @property
    def search_client(self) -> SearchClient:
        """Get or create the search client."""
        if self._search_client is None:
            self._search_client = SearchClient()
        return self._search_client

    @property
    def slack_client(self) -> SlackClient:
        """Get or create the Slack client."""
        if self._slack_client is None:
            self._slack_client = SlackClient()
        return self._slack_client

    @property
    def digest_builder(self) -> DigestBuilder:
        """Get or create the digest builder."""
        if self._digest_builder is None:
            self._digest_builder = DigestBuilder()
        return self._digest_builder

    def run(self) -> Digest:
        """Execute the complete RFP Radar pipeline.

        This method orchestrates the full processing flow:
        1. Scrape RFPs from all available sources
        2. Classify RFPs using AI
        3. Filter by relevance threshold
        4. Store RFPs and metadata
        5. Index in Azure AI Search
        6. Generate proposals
        7. Build and send digest

        Returns:
            Digest object containing the pipeline results.
        """
        self.stats["start_time"] = datetime.utcnow()
        self.logger.info("Starting RFP Radar pipeline")

        try:
            # Step 1: Scrape RFPs from all sources
            all_rfps = self._scrape_all_sources()

            # Step 2: Classify RFPs
            classified_rfps = self._classify_rfps(all_rfps)

            # Step 3: Filter by relevance threshold
            relevant_rfps = self._filter_relevant(classified_rfps)

            # Step 4: Ensure Azure resources exist
            self._ensure_azure_resources()

            # Step 5: Store RFP metadata
            self._store_rfp_metadata(relevant_rfps)

            # Step 6: Index RFPs in Azure AI Search
            self._index_rfps(relevant_rfps)

            # Step 7: Generate proposals for relevant RFPs
            proposals = self._generate_proposals(relevant_rfps)

            # Step 8: Update search index with proposal URLs
            self._update_search_with_proposals(proposals)

            # Step 9: Build and send digest
            digest = self._build_and_send_digest(relevant_rfps, proposals)

            self.stats["end_time"] = datetime.utcnow()
            self._log_summary()

            return digest

        except Exception as e:
            self.logger.error(
                f"Pipeline failed with error: {e}",
                extra={"error": str(e)}
            )
            self.stats["errors"].append(str(e))
            self.stats["end_time"] = datetime.utcnow()

            # Attempt to send error notification
            self._send_error_notification(str(e))

            raise

    def _scrape_all_sources(self) -> List[RFP]:
        """Scrape RFPs from all available sources.

        Returns:
            List of RFP objects from all scrapers.
        """
        self.logger.info("Starting scraping from all sources")

        all_rfps: List[RFP] = []
        available_sources = get_available_sources()

        if not available_sources:
            self.logger.warning("No scrapers registered")
            return all_rfps

        for source_name in available_sources:
            try:
                scraper_class = SCRAPERS.get(source_name)
                if scraper_class is None:
                    continue

                self.logger.info(
                    f"Scraping source: {source_name}",
                    extra={"source": source_name}
                )

                with scraper_class(
                    max_age_days=config.RFP_MAX_AGE_DAYS,
                    us_only=True,
                ) as scraper:
                    result: ScraperResult = scraper.scrape()

                    self.stats["scraper_results"][source_name] = {
                        "success": result.success,
                        "total_found": result.total_found,
                        "rfp_count": result.rfp_count,
                        "duration_seconds": result.duration_seconds,
                        "error": result.error_message if not result.success else None,
                    }

                    if result.success:
                        all_rfps.extend(result.rfps)
                        self.stats["total_discovered"] += result.total_found
                        self.stats["total_filtered"] += result.rfp_count
                    else:
                        self.logger.warning(
                            f"Scraper {source_name} failed: {result.error_message}",
                            extra={"source": source_name, "error": result.error_message}
                        )

            except Exception as e:
                self.logger.error(
                    f"Error scraping {source_name}: {e}",
                    extra={"source": source_name, "error": str(e)}
                )
                self.stats["scraper_results"][source_name] = {
                    "success": False,
                    "error": str(e),
                }

        self.logger.info(
            "Scraping completed",
            extra={
                "total_rfps": len(all_rfps),
                "sources_scraped": len(available_sources),
            }
        )

        return all_rfps

    def _classify_rfps(self, rfps: List[RFP]) -> List[ClassifiedRFP]:
        """Classify RFPs using AI.

        Args:
            rfps: List of RFP objects to classify.

        Returns:
            List of ClassifiedRFP objects with classification results.
        """
        if not rfps:
            self.logger.info("No RFPs to classify")
            return []

        self.logger.info(
            "Starting RFP classification",
            extra={"count": len(rfps)}
        )

        classified = self.classifier.classify_batch(rfps, skip_on_error=True)
        self.stats["total_classified"] = len(classified)

        self.logger.info(
            "Classification completed",
            extra={
                "input_count": len(rfps),
                "classified_count": len(classified),
            }
        )

        return classified

    def _filter_relevant(
        self,
        classified_rfps: List[ClassifiedRFP],
    ) -> List[ClassifiedRFP]:
        """Filter classified RFPs by relevance threshold.

        Args:
            classified_rfps: List of classified RFPs to filter.

        Returns:
            List of relevant ClassifiedRFP objects.
        """
        relevant = self.classifier.filter_relevant(classified_rfps)
        self.stats["total_relevant"] = len(relevant)

        self.logger.info(
            "Filtered by relevance",
            extra={
                "input_count": len(classified_rfps),
                "relevant_count": len(relevant),
                "threshold": config.RFP_RELEVANCE_THRESHOLD,
            }
        )

        return relevant

    def _ensure_azure_resources(self) -> None:
        """Ensure Azure resources (container, index) exist."""
        try:
            self.storage_client.ensure_container_exists()
            self.logger.debug("Storage container verified")
        except Exception as e:
            self.logger.warning(f"Failed to ensure storage container: {e}")

        try:
            self.search_client.ensure_index_exists()
            self.logger.debug("Search index verified")
        except Exception as e:
            self.logger.warning(f"Failed to ensure search index: {e}")

    def _store_rfp_metadata(
        self,
        classified_rfps: List[ClassifiedRFP],
    ) -> None:
        """Store RFP metadata in Azure Blob Storage.

        Args:
            classified_rfps: List of classified RFPs to store.
        """
        if not classified_rfps:
            return

        self.logger.info(
            "Storing RFP metadata",
            extra={"count": len(classified_rfps)}
        )

        stored_count = 0
        for classified_rfp in classified_rfps:
            try:
                self.storage_client.upload_metadata(
                    rfp=classified_rfp.rfp,
                    classification=classified_rfp.classification.model_dump(),
                )
                classified_rfp.rfp.status = RFPStatus.STORED
                stored_count += 1
            except Exception as e:
                self.logger.warning(
                    f"Failed to store metadata for RFP {classified_rfp.rfp.id}: {e}",
                    extra={"rfp_id": classified_rfp.rfp.id, "error": str(e)}
                )

        self.logger.info(
            "RFP metadata storage completed",
            extra={"stored_count": stored_count, "total": len(classified_rfps)}
        )

    def _index_rfps(self, classified_rfps: List[ClassifiedRFP]) -> None:
        """Index RFPs in Azure AI Search.

        Args:
            classified_rfps: List of classified RFPs to index.
        """
        if not classified_rfps:
            return

        self.logger.info(
            "Indexing RFPs in Azure AI Search",
            extra={"count": len(classified_rfps)}
        )

        try:
            results = self.search_client.index_rfps_batch(classified_rfps)
            success_count = sum(1 for v in results.values() if v)

            self.logger.info(
                "RFP indexing completed",
                extra={
                    "indexed_count": success_count,
                    "total": len(classified_rfps),
                }
            )
        except Exception as e:
            self.logger.error(
                f"Failed to index RFPs: {e}",
                extra={"error": str(e)}
            )

    def _generate_proposals(
        self,
        classified_rfps: List[ClassifiedRFP],
    ) -> Dict[str, Proposal]:
        """Generate proposals for relevant RFPs.

        Args:
            classified_rfps: List of relevant classified RFPs.

        Returns:
            Dictionary mapping RFP IDs to Proposal objects.
        """
        if not classified_rfps:
            return {}

        self.logger.info(
            "Starting proposal generation",
            extra={"count": len(classified_rfps)}
        )

        proposals: Dict[str, Proposal] = {}

        # Generate proposals with storage
        results = self.proposal_generator.generate_batch(
            classified_rfps,
            store=True,
            skip_on_error=True,
        )

        for classified_rfp, proposal in results:
            rfp_id = classified_rfp.rfp.id
            proposals[rfp_id] = proposal
            classified_rfp.rfp.status = RFPStatus.PROPOSAL_GENERATED

        self.stats["total_proposals"] = len(proposals)

        self.logger.info(
            "Proposal generation completed",
            extra={
                "generated_count": len(proposals),
                "total": len(classified_rfps),
            }
        )

        return proposals

    def _update_search_with_proposals(
        self,
        proposals: Dict[str, Proposal],
    ) -> None:
        """Update search index with proposal URLs.

        Args:
            proposals: Dictionary mapping RFP IDs to proposals.
        """
        if not proposals:
            return

        self.logger.info(
            "Updating search index with proposal URLs",
            extra={"count": len(proposals)}
        )

        updated_count = 0
        for rfp_id, proposal in proposals.items():
            try:
                if proposal.metadata.blob_url:
                    self.search_client.update_proposal_url(
                        rfp_id=rfp_id,
                        proposal_url=proposal.metadata.blob_url,
                    )
                    updated_count += 1
            except Exception as e:
                self.logger.warning(
                    f"Failed to update proposal URL for {rfp_id}: {e}",
                    extra={"rfp_id": rfp_id, "error": str(e)}
                )

        self.logger.info(
            "Search index update completed",
            extra={"updated_count": updated_count}
        )

    def _build_and_send_digest(
        self,
        classified_rfps: List[ClassifiedRFP],
        proposals: Dict[str, Proposal],
    ) -> Digest:
        """Build digest and send to Slack.

        Args:
            classified_rfps: List of relevant classified RFPs.
            proposals: Dictionary mapping RFP IDs to proposals.

        Returns:
            Digest object.
        """
        self.logger.info("Building digest")

        # Build the digest
        digest = self.digest_builder.build_digest(
            classified_rfps=classified_rfps,
            proposals=proposals,
            total_discovered=self.stats["total_discovered"],
            total_filtered=self.stats["total_filtered"],
        )

        # Send to Slack (don't fail on Slack errors per spec)
        try:
            if digest.is_empty():
                message_ts = self.slack_client.post_empty_digest(
                    total_discovered=digest.total_discovered,
                    total_filtered=digest.total_filtered,
                )
            else:
                message_ts = self.slack_client.post_digest(digest)

            if message_ts:
                self.logger.info(
                    "Digest posted to Slack",
                    extra={"message_ts": message_ts}
                )
            else:
                self.logger.warning("Failed to post digest to Slack")

        except Exception as e:
            # Log but don't fail - per spec, don't block on Slack failures
            self.logger.error(
                f"Failed to post digest to Slack: {e}",
                extra={"error": str(e)}
            )

        # Mark RFPs as notified
        for classified_rfp in classified_rfps:
            classified_rfp.rfp.status = RFPStatus.NOTIFIED

        return digest

    def _send_error_notification(self, error_message: str) -> None:
        """Send error notification to Slack.

        Args:
            error_message: The error message to send.
        """
        try:
            self.slack_client.post_error_notification(
                error_message=error_message,
                error_type="Pipeline Error",
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to send error notification: {e}",
                extra={"error": str(e)}
            )

    def _log_summary(self) -> None:
        """Log pipeline summary statistics."""
        duration = None
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = (
                self.stats["end_time"] - self.stats["start_time"]
            ).total_seconds()

        self.logger.info(
            "Pipeline completed",
            extra={
                "total_discovered": self.stats["total_discovered"],
                "total_filtered": self.stats["total_filtered"],
                "total_classified": self.stats["total_classified"],
                "total_relevant": self.stats["total_relevant"],
                "total_proposals": self.stats["total_proposals"],
                "duration_seconds": duration,
                "errors_count": len(self.stats["errors"]),
            }
        )

    def health_check(self) -> Dict[str, bool]:
        """Check health of all pipeline components.

        Returns:
            Dictionary mapping component names to health status.
        """
        health = {
            "classifier": False,
            "storage": False,
            "search": False,
            "slack": False,
        }

        try:
            health["classifier"] = self.classifier.health_check()
        except Exception:
            pass

        try:
            self.storage_client.ensure_container_exists()
            health["storage"] = True
        except Exception:
            pass

        try:
            self.search_client.ensure_index_exists()
            health["search"] = True
        except Exception:
            pass

        try:
            health["slack"] = self.slack_client.health_check()
        except Exception:
            pass

        return health

    def close(self) -> None:
        """Close all pipeline components and release resources."""
        if self._owns_classifier and self._classifier is not None:
            self._classifier.close()

        if self._owns_proposal_generator and self._proposal_generator is not None:
            self._proposal_generator.close()

        if self._owns_slack_client and self._slack_client is not None:
            self._slack_client.close()

        self.logger.debug("Pipeline resources closed")

    def __enter__(self) -> "RFPRadarPipeline":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def main() -> int:
    """Main entry point for RFP Radar.

    This function sets up logging, creates the pipeline, and runs it.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Set up logging
    logger = setup_logging(
        service_name="rfp-radar",
        enable_azure_monitor=True,
    )

    logger.info(
        "RFP Radar starting",
        extra={
            "app_env": config.APP_ENV,
            "relevance_threshold": config.RFP_RELEVANCE_THRESHOLD,
            "max_age_days": config.RFP_MAX_AGE_DAYS,
        }
    )

    try:
        with RFPRadarPipeline() as pipeline:
            digest = pipeline.run()

            logger.info(
                "RFP Radar completed successfully",
                extra={
                    "total_entries": len(digest.entries),
                    "total_proposals": digest.total_proposals,
                }
            )

        return 0

    except Exception as e:
        logger.exception(f"RFP Radar failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
