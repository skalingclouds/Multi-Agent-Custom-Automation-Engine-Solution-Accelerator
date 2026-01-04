# classifier.py
"""RFP Classifier module for AI relevance scoring and tag extraction.

This module provides the RFPClassifier class which uses Azure OpenAI
to classify RFPs by relevance to NAITIVE's capabilities and extract
classification tags.
"""

from datetime import datetime
from typing import List, Optional, Tuple

from .config import config
from .llm_client import LLMClient
from .logging_utils import get_logger
from .models import (
    ClassificationResult,
    ClassifiedRFP,
    RFP,
    RFPStatus,
    RFPTag,
)


class RFPClassifier:
    """Classifier for RFP relevance scoring and tag extraction.

    This class wraps the LLMClient to provide high-level RFP classification
    functionality, including batch processing, filtering by relevance
    threshold, and proper model conversion.

    Attributes:
        relevance_threshold: Minimum relevance score for an RFP to be
            considered relevant (default: 0.55).
        llm_client: The LLMClient instance used for AI classification.
    """

    # Valid tags that can be extracted from classification
    VALID_TAGS = {tag.value for tag in RFPTag}

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        relevance_threshold: Optional[float] = None,
    ):
        """Initialize the RFP Classifier.

        Args:
            llm_client: Optional LLMClient instance. If not provided,
                       a new client will be created using config settings.
            relevance_threshold: Optional relevance threshold override.
                               Defaults to config.RFP_RELEVANCE_THRESHOLD.
        """
        self.logger = get_logger(__name__)

        # Use provided client or create a new one
        self._llm_client = llm_client
        self._owns_client = llm_client is None

        # Set relevance threshold
        self.relevance_threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else config.RFP_RELEVANCE_THRESHOLD
        )

        self.logger.info(
            "RFPClassifier initialized",
            extra={"relevance_threshold": self.relevance_threshold}
        )

    @property
    def llm_client(self) -> LLMClient:
        """Get or create the LLM client.

        Returns:
            LLMClient instance for making classification requests.
        """
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def classify(self, rfp: RFP) -> ClassificationResult:
        """Classify a single RFP for relevance and extract tags.

        This method uses Azure OpenAI to analyze the RFP and determine:
        - Relevance score (0.0 to 1.0)
        - Classification tags (AI, Dynamics, Modernization, etc.)
        - Reasoning for the classification
        - Confidence in the classification

        Args:
            rfp: The RFP model to classify.

        Returns:
            ClassificationResult with relevance score, tags, and reasoning.

        Raises:
            ValueError: If classification fails after retries.
        """
        self.logger.info(
            "Classifying RFP",
            extra={"rfp_id": rfp.id, "title": rfp.title[:100]}
        )

        try:
            # Build additional context from RFP metadata
            additional_context = self._build_context(rfp)

            # Call LLM for classification
            result = self.llm_client.classify_rfp(
                title=rfp.title,
                description=rfp.description,
                agency=rfp.agency,
                additional_context=additional_context,
            )

            # Convert raw result to ClassificationResult model
            classification = self._create_classification_result(
                rfp_id=rfp.id,
                raw_result=result,
            )

            self.logger.info(
                "RFP classified successfully",
                extra={
                    "rfp_id": rfp.id,
                    "relevance_score": classification.relevance_score,
                    "tags": [t.value for t in classification.tags],
                    "is_relevant": classification.is_relevant(self.relevance_threshold),
                }
            )

            return classification

        except Exception as e:
            self.logger.error(
                f"Failed to classify RFP: {e}",
                extra={"rfp_id": rfp.id, "title": rfp.title[:50]}
            )
            raise ValueError(f"Classification failed for RFP {rfp.id}: {e}") from e

    def classify_batch(
        self,
        rfps: List[RFP],
        skip_on_error: bool = True,
    ) -> List[ClassifiedRFP]:
        """Classify a batch of RFPs.

        This method processes multiple RFPs, optionally skipping failures
        to allow partial batch completion.

        Args:
            rfps: List of RFP models to classify.
            skip_on_error: If True, continue processing on classification
                          errors. If False, raise on first error.

        Returns:
            List of ClassifiedRFP objects containing RFPs with their
            classification results.

        Raises:
            ValueError: If skip_on_error is False and classification fails.
        """
        self.logger.info(
            "Starting batch classification",
            extra={"batch_size": len(rfps)}
        )

        results: List[ClassifiedRFP] = []
        errors: List[Tuple[str, str]] = []

        for rfp in rfps:
            try:
                classification = self.classify(rfp)

                # Update RFP status
                rfp.status = RFPStatus.CLASSIFIED

                classified_rfp = ClassifiedRFP(
                    rfp=rfp,
                    classification=classification,
                )
                results.append(classified_rfp)

            except Exception as e:
                error_msg = str(e)
                errors.append((rfp.id, error_msg))

                self.logger.warning(
                    f"Classification failed for RFP {rfp.id}",
                    extra={"rfp_id": rfp.id, "error": error_msg}
                )

                if not skip_on_error:
                    raise ValueError(
                        f"Batch classification failed on RFP {rfp.id}: {error_msg}"
                    ) from e

                # Mark as error but continue
                rfp.status = RFPStatus.ERROR

        self.logger.info(
            "Batch classification completed",
            extra={
                "total": len(rfps),
                "successful": len(results),
                "errors": len(errors),
            }
        )

        return results

    def filter_relevant(
        self,
        classified_rfps: List[ClassifiedRFP],
        threshold: Optional[float] = None,
    ) -> List[ClassifiedRFP]:
        """Filter classified RFPs by relevance threshold.

        Args:
            classified_rfps: List of ClassifiedRFP objects to filter.
            threshold: Optional threshold override. Defaults to
                      self.relevance_threshold.

        Returns:
            List of ClassifiedRFP objects that meet the relevance threshold.
        """
        threshold = threshold if threshold is not None else self.relevance_threshold

        relevant = [
            crfp for crfp in classified_rfps
            if crfp.classification.relevance_score >= threshold
        ]

        self.logger.info(
            "Filtered by relevance",
            extra={
                "input_count": len(classified_rfps),
                "output_count": len(relevant),
                "threshold": threshold,
            }
        )

        return relevant

    def classify_and_filter(
        self,
        rfps: List[RFP],
        skip_on_error: bool = True,
    ) -> List[ClassifiedRFP]:
        """Classify a batch of RFPs and filter by relevance threshold.

        This is a convenience method that combines classify_batch and
        filter_relevant into a single call.

        Args:
            rfps: List of RFP models to classify.
            skip_on_error: If True, continue processing on classification
                          errors. If False, raise on first error.

        Returns:
            List of ClassifiedRFP objects that meet the relevance threshold.
        """
        classified = self.classify_batch(rfps, skip_on_error=skip_on_error)
        return self.filter_relevant(classified)

    def _build_context(self, rfp: RFP) -> str:
        """Build additional context string from RFP metadata.

        Args:
            rfp: The RFP to extract context from.

        Returns:
            String with additional context for classification.
        """
        context_parts = []

        if rfp.naics_codes:
            context_parts.append(f"NAICS Codes: {', '.join(rfp.naics_codes)}")

        if rfp.estimated_value:
            context_parts.append(
                f"Estimated Value: ${rfp.estimated_value:,.2f}"
            )

        if rfp.contract_type:
            context_parts.append(f"Contract Type: {rfp.contract_type}")

        if rfp.set_aside:
            context_parts.append(f"Set-Aside: {rfp.set_aside}")

        if rfp.location:
            context_parts.append(f"Location: {rfp.location}")

        if rfp.due_date:
            context_parts.append(
                f"Due Date: {rfp.due_date.strftime('%Y-%m-%d')}"
            )

        return " | ".join(context_parts) if context_parts else ""

    def _create_classification_result(
        self,
        rfp_id: str,
        raw_result: dict,
    ) -> ClassificationResult:
        """Create a ClassificationResult from raw LLM response.

        Args:
            rfp_id: The ID of the classified RFP.
            raw_result: Dictionary from LLMClient.classify_rfp().

        Returns:
            ClassificationResult model with validated fields.
        """
        # Extract and validate relevance score
        relevance_score = float(raw_result.get("relevance_score", 0.0))
        relevance_score = max(0.0, min(1.0, relevance_score))

        # Extract and validate tags
        raw_tags = raw_result.get("tags", [])
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]

        validated_tags = []
        for tag in raw_tags:
            tag_str = str(tag).strip()
            # Try to match tag to RFPTag enum
            if tag_str in self.VALID_TAGS:
                validated_tags.append(RFPTag(tag_str))
            else:
                # Try case-insensitive matching
                for valid_tag in RFPTag:
                    if tag_str.lower() == valid_tag.value.lower():
                        validated_tags.append(valid_tag)
                        break
                else:
                    # Unknown tag, log and skip
                    self.logger.debug(
                        f"Unknown tag '{tag_str}' ignored",
                        extra={"rfp_id": rfp_id}
                    )

        # Extract confidence
        confidence = float(raw_result.get("confidence", 1.0))
        confidence = max(0.0, min(1.0, confidence))

        # Extract reasoning
        reasoning = raw_result.get("reasoning", "")

        # Get model used
        model_used = raw_result.get("model_used", config.AZURE_OPENAI_DEPLOYMENT)

        return ClassificationResult(
            rfp_id=rfp_id,
            relevance_score=relevance_score,
            tags=validated_tags,
            reasoning=reasoning,
            classified_at=datetime.utcnow(),
            model_used=model_used,
            confidence=confidence,
        )

    def get_stats(self) -> dict:
        """Get classifier statistics and configuration.

        Returns:
            Dictionary with classifier configuration and LLM client stats.
        """
        return {
            "relevance_threshold": self.relevance_threshold,
            "valid_tags": list(self.VALID_TAGS),
            "llm_stats": self.llm_client.get_usage_stats(),
        }

    def health_check(self) -> bool:
        """Perform a health check on the classifier.

        This verifies that the LLM client can connect and respond.

        Returns:
            True if the classifier is healthy, False otherwise.
        """
        try:
            return self.llm_client.health_check()
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False

    def close(self) -> None:
        """Close the classifier and clean up resources.

        This closes the LLM client if it was created by this classifier.
        """
        if self._owns_client and self._llm_client is not None:
            self._llm_client.close()
            self._llm_client = None
            self.logger.debug("RFPClassifier closed")

    def __enter__(self) -> "RFPClassifier":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
