# search_client.py
"""Azure AI Search client wrapper for RFP indexing and discovery."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient as AzureSearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ComplexField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
)

from .config import config
from .logging_utils import get_logger
from .models import ClassificationResult, ClassifiedRFP, RFP, RFPTag


class SearchClient:
    """Azure AI Search client wrapper for RFP Radar.

    This class provides methods for indexing RFP documents and metadata
    in Azure AI Search for discovery and retrieval.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        index_name: Optional[str] = None,
        credential: Optional[Any] = None,
    ):
        """Initialize the Search client.

        Args:
            endpoint: Azure AI Search endpoint URL. Defaults to config value.
            index_name: Search index name. Defaults to config value.
            credential: Azure credential. Defaults to API key or managed identity
                       based on environment.
        """
        self.logger = get_logger(__name__)

        self.endpoint = endpoint or config.AZURE_SEARCH_ENDPOINT
        self.index_name = index_name or config.AZURE_SEARCH_INDEX_NAME

        # Initialize credential
        if credential is not None:
            self._credential = credential
        elif config.AZURE_SEARCH_API_KEY:
            # Use API key for development
            self._credential = AzureKeyCredential(config.AZURE_SEARCH_API_KEY)
        else:
            # Use managed identity or DefaultAzureCredential
            self._credential = config.get_azure_credentials()

        # Lazy-initialized clients
        self._index_client: Optional[SearchIndexClient] = None
        self._search_client: Optional[AzureSearchClient] = None

    def _get_index_client(self) -> SearchIndexClient:
        """Get or create the Search Index client for schema management.

        Returns:
            SearchIndexClient instance
        """
        if self._index_client is None:
            self._index_client = SearchIndexClient(
                endpoint=self.endpoint,
                credential=self._credential,
            )
        return self._index_client

    def _get_search_client(self) -> AzureSearchClient:
        """Get or create the Search client for document operations.

        Returns:
            SearchClient instance
        """
        if self._search_client is None:
            self._search_client = AzureSearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self._credential,
            )
        return self._search_client

    def ensure_index_exists(self) -> bool:
        """Ensure the search index exists, creating it if necessary.

        Returns:
            True if index exists or was created
        """
        try:
            index_client = self._get_index_client()

            # Check if index exists
            try:
                index_client.get_index(self.index_name)
                self.logger.debug(
                    "Search index already exists",
                    extra={"index_name": self.index_name}
                )
                return True
            except ResourceNotFoundError:
                pass

            # Create the index with RFP schema
            index = self._create_rfp_index_definition()
            index_client.create_index(index)

            self.logger.info(
                "Created search index",
                extra={"index_name": self.index_name}
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to ensure index exists: {e}",
                extra={"index_name": self.index_name}
            )
            raise

    def _create_rfp_index_definition(self) -> SearchIndex:
        """Create the search index definition for RFP documents.

        Returns:
            SearchIndex definition
        """
        fields = [
            # Primary key
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
            ),
            # Searchable text fields
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft",
            ),
            SearchableField(
                name="description",
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft",
            ),
            SearchableField(
                name="agency",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            # Source and location
            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SimpleField(
                name="source_url",
                type=SearchFieldDataType.String,
            ),
            SimpleField(
                name="country",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="state",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SimpleField(
                name="location",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            # Dates
            SimpleField(
                name="posted_date",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="due_date",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="discovered_at",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="indexed_at",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            # Classification results
            SimpleField(
                name="relevance_score",
                type=SearchFieldDataType.Double,
                filterable=True,
                sortable=True,
            ),
            SearchField(
                name="tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True,
            ),
            SearchableField(
                name="reasoning",
                type=SearchFieldDataType.String,
            ),
            # Contract metadata
            SearchField(
                name="naics_codes",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True,
            ),
            SimpleField(
                name="set_aside",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SimpleField(
                name="estimated_value",
                type=SearchFieldDataType.Double,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="contract_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            # Proposal info
            SimpleField(
                name="proposal_url",
                type=SearchFieldDataType.String,
            ),
            SimpleField(
                name="has_proposal",
                type=SearchFieldDataType.Boolean,
                filterable=True,
            ),
            # Status
            SimpleField(
                name="status",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
        ]

        return SearchIndex(name=self.index_name, fields=fields)

    def index_rfp(
        self,
        rfp: RFP,
        classification: Optional[ClassificationResult] = None,
        proposal_url: Optional[str] = None,
    ) -> str:
        """Index an RFP document in Azure AI Search.

        Args:
            rfp: The RFP to index
            classification: Optional classification result
            proposal_url: Optional URL to the generated proposal

        Returns:
            The document ID
        """
        document = self._rfp_to_document(rfp, classification, proposal_url)
        return self._upload_document(document)

    def index_classified_rfp(
        self,
        classified_rfp: ClassifiedRFP,
        proposal_url: Optional[str] = None,
    ) -> str:
        """Index a classified RFP in Azure AI Search.

        Args:
            classified_rfp: The classified RFP to index
            proposal_url: Optional URL to the generated proposal

        Returns:
            The document ID
        """
        return self.index_rfp(
            rfp=classified_rfp.rfp,
            classification=classified_rfp.classification,
            proposal_url=proposal_url,
        )

    def index_rfps_batch(
        self,
        classified_rfps: List[ClassifiedRFP],
        proposal_urls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, bool]:
        """Index multiple RFPs in a batch operation.

        Args:
            classified_rfps: List of classified RFPs to index
            proposal_urls: Optional mapping of RFP ID to proposal URL

        Returns:
            Dictionary mapping RFP ID to success status
        """
        if not classified_rfps:
            return {}

        proposal_urls = proposal_urls or {}
        documents = []

        for classified_rfp in classified_rfps:
            rfp_id = classified_rfp.rfp.id
            proposal_url = proposal_urls.get(rfp_id)
            document = self._rfp_to_document(
                classified_rfp.rfp,
                classified_rfp.classification,
                proposal_url,
            )
            documents.append(document)

        return self._upload_documents_batch(documents)

    def _rfp_to_document(
        self,
        rfp: RFP,
        classification: Optional[ClassificationResult] = None,
        proposal_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert an RFP and classification to a search document.

        Args:
            rfp: The RFP model
            classification: Optional classification result
            proposal_url: Optional proposal URL

        Returns:
            Dictionary suitable for indexing
        """
        document: Dict[str, Any] = {
            "id": rfp.id,
            "title": rfp.title,
            "description": rfp.description,
            "agency": rfp.agency,
            "source": rfp.source.value,
            "source_url": rfp.source_url,
            "country": rfp.country,
            "state": rfp.state,
            "location": rfp.location,
            "naics_codes": rfp.naics_codes,
            "set_aside": rfp.set_aside,
            "estimated_value": rfp.estimated_value,
            "contract_type": rfp.contract_type,
            "status": rfp.status.value,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "has_proposal": proposal_url is not None,
            "proposal_url": proposal_url or "",
        }

        # Convert dates to ISO format strings
        if rfp.posted_date:
            document["posted_date"] = rfp.posted_date.isoformat()
        if rfp.due_date:
            document["due_date"] = rfp.due_date.isoformat()
        if rfp.discovered_at:
            document["discovered_at"] = rfp.discovered_at.isoformat()

        # Add classification fields if available
        if classification:
            document["relevance_score"] = classification.relevance_score
            document["tags"] = [tag.value for tag in classification.tags]
            document["reasoning"] = classification.reasoning

        return document

    def _upload_document(self, document: Dict[str, Any]) -> str:
        """Upload a single document to the search index.

        Args:
            document: The document to upload

        Returns:
            The document ID
        """
        try:
            search_client = self._get_search_client()
            result = search_client.upload_documents(documents=[document])

            doc_id = document.get("id", "unknown")

            # Check if upload succeeded
            if result and len(result) > 0:
                upload_result = result[0]
                if upload_result.succeeded:
                    self.logger.info(
                        "Indexed document",
                        extra={"document_id": doc_id}
                    )
                    return doc_id
                else:
                    self.logger.error(
                        f"Failed to index document: {upload_result.error_message}",
                        extra={"document_id": doc_id}
                    )
                    raise Exception(upload_result.error_message)

            return doc_id

        except Exception as e:
            self.logger.error(
                f"Failed to upload document: {e}",
                extra={"document_id": document.get("id", "unknown")}
            )
            raise

    def _upload_documents_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, bool]:
        """Upload multiple documents in a batch.

        Args:
            documents: List of documents to upload

        Returns:
            Dictionary mapping document ID to success status
        """
        results: Dict[str, bool] = {}

        try:
            search_client = self._get_search_client()
            upload_results = search_client.upload_documents(documents=documents)

            for i, result in enumerate(upload_results):
                doc_id = documents[i].get("id", f"unknown_{i}")
                results[doc_id] = result.succeeded

                if result.succeeded:
                    self.logger.debug(
                        "Indexed document in batch",
                        extra={"document_id": doc_id}
                    )
                else:
                    self.logger.warning(
                        f"Failed to index document: {result.error_message}",
                        extra={"document_id": doc_id}
                    )

            success_count = sum(1 for v in results.values() if v)
            self.logger.info(
                "Batch indexing completed",
                extra={
                    "total": len(documents),
                    "succeeded": success_count,
                    "failed": len(documents) - success_count,
                }
            )

            return results

        except Exception as e:
            self.logger.error(f"Failed to upload documents batch: {e}")
            raise

    def search_rfps(
        self,
        query: str,
        filter_expr: Optional[str] = None,
        order_by: Optional[List[str]] = None,
        top: int = 50,
        skip: int = 0,
        select: Optional[List[str]] = None,
        include_total_count: bool = True,
    ) -> Dict[str, Any]:
        """Search for RFPs in the index.

        Args:
            query: Search query text
            filter_expr: OData filter expression
            order_by: List of fields to order by (e.g., ["relevance_score desc"])
            top: Maximum number of results to return
            skip: Number of results to skip (for pagination)
            select: List of fields to return
            include_total_count: Whether to include total count in response

        Returns:
            Dictionary containing search results and metadata
        """
        try:
            search_client = self._get_search_client()

            results = search_client.search(
                search_text=query,
                filter=filter_expr,
                order_by=order_by,
                top=top,
                skip=skip,
                select=select,
                include_total_count=include_total_count,
            )

            documents = []
            for result in results:
                doc = dict(result)
                # Remove search-specific fields
                doc.pop("@search.score", None)
                doc.pop("@search.highlights", None)
                documents.append(doc)

            response = {
                "results": documents,
                "count": len(documents),
            }

            if include_total_count:
                response["total_count"] = results.get_count()

            self.logger.debug(
                "Search completed",
                extra={
                    "query": query,
                    "results_count": len(documents),
                    "filter": filter_expr,
                }
            )

            return response

        except Exception as e:
            self.logger.error(
                f"Search failed: {e}",
                extra={"query": query, "filter": filter_expr}
            )
            raise

    def search_relevant_rfps(
        self,
        min_relevance: Optional[float] = None,
        tags: Optional[List[RFPTag]] = None,
        us_only: bool = True,
        top: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search for relevant RFPs matching criteria.

        Args:
            min_relevance: Minimum relevance score (defaults to config threshold)
            tags: Filter by specific tags
            us_only: Filter to US-based RFPs only
            top: Maximum number of results

        Returns:
            List of matching RFP documents
        """
        if min_relevance is None:
            min_relevance = config.RFP_RELEVANCE_THRESHOLD

        # Build filter expression
        filters = [f"relevance_score ge {min_relevance}"]

        if us_only:
            filters.append("country eq 'US'")

        if tags:
            tag_values = [f"'{tag.value}'" for tag in tags]
            tag_filter = " or ".join([f"tags/any(t: t eq {tv})" for tv in tag_values])
            filters.append(f"({tag_filter})")

        filter_expr = " and ".join(filters)

        result = self.search_rfps(
            query="*",
            filter_expr=filter_expr,
            order_by=["relevance_score desc", "discovered_at desc"],
            top=top,
        )

        return result.get("results", [])

    def get_rfp_by_id(self, rfp_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific RFP by its ID.

        Args:
            rfp_id: The RFP ID to retrieve

        Returns:
            The RFP document or None if not found
        """
        try:
            search_client = self._get_search_client()
            result = search_client.get_document(key=rfp_id)
            return dict(result)
        except ResourceNotFoundError:
            self.logger.debug(
                "RFP not found in index",
                extra={"rfp_id": rfp_id}
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Failed to get RFP: {e}",
                extra={"rfp_id": rfp_id}
            )
            raise

    def delete_rfp(self, rfp_id: str) -> bool:
        """Delete an RFP from the search index.

        Args:
            rfp_id: The RFP ID to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            search_client = self._get_search_client()
            result = search_client.delete_documents(documents=[{"id": rfp_id}])

            if result and len(result) > 0:
                if result[0].succeeded:
                    self.logger.info(
                        "Deleted RFP from index",
                        extra={"rfp_id": rfp_id}
                    )
                    return True
                else:
                    self.logger.warning(
                        f"Failed to delete RFP: {result[0].error_message}",
                        extra={"rfp_id": rfp_id}
                    )
                    return False

            return False

        except Exception as e:
            self.logger.error(
                f"Failed to delete RFP: {e}",
                extra={"rfp_id": rfp_id}
            )
            raise

    def delete_index(self) -> bool:
        """Delete the search index.

        WARNING: This will delete all indexed documents.

        Returns:
            True if deleted, False if not found
        """
        try:
            index_client = self._get_index_client()
            index_client.delete_index(self.index_name)

            self.logger.info(
                "Deleted search index",
                extra={"index_name": self.index_name}
            )
            return True

        except ResourceNotFoundError:
            self.logger.debug(
                "Index not found for deletion",
                extra={"index_name": self.index_name}
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Failed to delete index: {e}",
                extra={"index_name": self.index_name}
            )
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index.

        Returns:
            Dictionary containing index statistics
        """
        try:
            index_client = self._get_index_client()
            index = index_client.get_index(self.index_name)

            # Get document count via search
            search_client = self._get_search_client()
            results = search_client.search(
                search_text="*",
                include_total_count=True,
                top=0,
            )

            stats = {
                "index_name": self.index_name,
                "field_count": len(index.fields) if index.fields else 0,
                "document_count": results.get_count() or 0,
            }

            self.logger.debug(
                "Retrieved index stats",
                extra=stats
            )

            return stats

        except ResourceNotFoundError:
            return {
                "index_name": self.index_name,
                "exists": False,
            }
        except Exception as e:
            self.logger.error(
                f"Failed to get index stats: {e}",
                extra={"index_name": self.index_name}
            )
            raise

    def update_proposal_url(
        self,
        rfp_id: str,
        proposal_url: str,
    ) -> bool:
        """Update the proposal URL for an indexed RFP.

        Args:
            rfp_id: The RFP ID to update
            proposal_url: The proposal URL to set

        Returns:
            True if updated successfully
        """
        try:
            search_client = self._get_search_client()

            document = {
                "id": rfp_id,
                "proposal_url": proposal_url,
                "has_proposal": True,
            }

            result = search_client.merge_documents(documents=[document])

            if result and len(result) > 0:
                if result[0].succeeded:
                    self.logger.info(
                        "Updated proposal URL",
                        extra={"rfp_id": rfp_id}
                    )
                    return True
                else:
                    self.logger.warning(
                        f"Failed to update proposal URL: {result[0].error_message}",
                        extra={"rfp_id": rfp_id}
                    )
                    return False

            return False

        except Exception as e:
            self.logger.error(
                f"Failed to update proposal URL: {e}",
                extra={"rfp_id": rfp_id}
            )
            raise
