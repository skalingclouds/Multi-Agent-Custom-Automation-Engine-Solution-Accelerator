# storage_client.py
"""Azure Blob Storage client wrapper for RFP document storage."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings

from .config import config
from .logging_utils import get_logger
from .models import Proposal, ProposalMetadata, RFP


class StorageClient:
    """Azure Blob Storage client wrapper for RFP Radar.

    This class provides methods for uploading, downloading, and managing
    RFP documents and proposals in Azure Blob Storage.
    """

    def __init__(
        self,
        account_url: Optional[str] = None,
        container_name: Optional[str] = None,
        credential: Optional[Any] = None,
    ):
        """Initialize the Storage client.

        Args:
            account_url: Azure Storage account URL. Defaults to config value.
            container_name: Blob container name. Defaults to config value.
            credential: Azure credential. Defaults to managed identity or
                       DefaultAzureCredential based on environment.
        """
        self.logger = get_logger(__name__)

        self.account_url = account_url or config.AZURE_STORAGE_ACCOUNT_URL
        self.container_name = container_name or config.AZURE_STORAGE_CONTAINER

        # Initialize credential
        if credential is not None:
            self._credential = credential
        elif config.AZURE_STORAGE_SAS_TOKEN:
            # Use SAS token for development
            self._credential = config.AZURE_STORAGE_SAS_TOKEN
        else:
            # Use managed identity or DefaultAzureCredential
            self._credential = config.get_azure_credentials()

        # Initialize the blob service client
        self._blob_service_client: Optional[BlobServiceClient] = None
        self._container_client = None

    def _get_blob_service_client(self) -> BlobServiceClient:
        """Get or create the Blob Service client.

        Returns:
            BlobServiceClient instance
        """
        if self._blob_service_client is None:
            if isinstance(self._credential, str) and self._credential.startswith("?"):
                # SAS token - append to URL
                self._blob_service_client = BlobServiceClient(
                    account_url=f"{self.account_url}{self._credential}"
                )
            else:
                # Use credential object
                self._blob_service_client = BlobServiceClient(
                    account_url=self.account_url,
                    credential=self._credential,
                )
        return self._blob_service_client

    def _get_container_client(self):
        """Get the container client for the configured container.

        Returns:
            ContainerClient instance
        """
        if self._container_client is None:
            blob_service = self._get_blob_service_client()
            self._container_client = blob_service.get_container_client(
                self.container_name
            )
        return self._container_client

    def ensure_container_exists(self) -> bool:
        """Ensure the blob container exists, creating it if necessary.

        Returns:
            True if container exists or was created, False on error
        """
        try:
            container_client = self._get_container_client()
            container_client.create_container()
            self.logger.info(
                "Created blob container",
                extra={"container": self.container_name}
            )
            return True
        except ResourceExistsError:
            self.logger.debug(
                "Blob container already exists",
                extra={"container": self.container_name}
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to ensure container exists: {e}",
                extra={"container": self.container_name}
            )
            raise

    def upload_rfp_document(
        self,
        rfp: RFP,
        content: bytes,
        filename: Optional[str] = None,
        content_type: str = "application/pdf",
    ) -> str:
        """Upload an RFP document to blob storage.

        Args:
            rfp: The RFP model object
            content: Document content as bytes
            filename: Optional filename override
            content_type: MIME type of the document

        Returns:
            The blob URL of the uploaded document
        """
        if filename is None:
            filename = f"{rfp.id}.pdf"

        blob_path = f"rfps/{rfp.id}/{filename}"

        # Calculate content hash for deduplication tracking
        content_hash = hashlib.sha256(content).hexdigest()

        # Prepare metadata
        metadata = {
            "rfp_id": rfp.id,
            "rfp_title": rfp.title[:256] if rfp.title else "",
            "source": rfp.source.value,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": content_hash,
        }

        return self._upload_blob(
            blob_path=blob_path,
            content=content,
            content_type=content_type,
            metadata=metadata,
        )

    def upload_proposal(
        self,
        proposal: Proposal,
    ) -> str:
        """Upload a generated proposal to blob storage.

        Args:
            proposal: The Proposal model object containing content and metadata

        Returns:
            The blob URL of the uploaded proposal
        """
        rfp_id = proposal.metadata.rfp_id
        proposal_id = proposal.metadata.id
        blob_path = f"proposals/{rfp_id}/{proposal_id}.md"

        content = proposal.markdown_content.encode("utf-8")
        content_hash = hashlib.sha256(content).hexdigest()

        # Update metadata with content hash and word count
        proposal.metadata.content_hash = content_hash
        proposal.metadata.word_count = proposal.word_count

        metadata = {
            "proposal_id": proposal.metadata.id,
            "rfp_id": rfp_id,
            "rfp_title": proposal.metadata.rfp_title[:256] if proposal.metadata.rfp_title else "",
            "generated_at": proposal.metadata.generated_at.isoformat(),
            "version": str(proposal.metadata.version),
            "word_count": str(proposal.metadata.word_count),
            "brand": proposal.metadata.brand_name,
            "content_hash": content_hash,
        }

        blob_url = self._upload_blob(
            blob_path=blob_path,
            content=content,
            content_type="text/markdown",
            metadata=metadata,
        )

        proposal.metadata.blob_url = blob_url
        proposal.metadata.blob_path = blob_path

        return blob_url

    def upload_metadata(
        self,
        rfp: RFP,
        classification: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Upload RFP metadata as JSON to blob storage.

        Args:
            rfp: The RFP model object
            classification: Optional classification result dict

        Returns:
            The blob URL of the uploaded metadata
        """
        blob_path = f"rfps/{rfp.id}/metadata.json"

        metadata_content = {
            "rfp": rfp.model_dump(mode="json"),
            "classification": classification,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }

        content = json.dumps(metadata_content, indent=2, default=str).encode("utf-8")

        metadata = {
            "rfp_id": rfp.id,
            "type": "metadata",
        }

        return self._upload_blob(
            blob_path=blob_path,
            content=content,
            content_type="application/json",
            metadata=metadata,
        )

    def _upload_blob(
        self,
        blob_path: str,
        content: Union[bytes, str],
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = True,
    ) -> str:
        """Upload content to a blob.

        Args:
            blob_path: Path within the container
            content: Content to upload (bytes or str)
            content_type: MIME type
            metadata: Optional blob metadata
            overwrite: Whether to overwrite existing blobs

        Returns:
            The blob URL
        """
        try:
            container_client = self._get_container_client()
            blob_client = container_client.get_blob_client(blob_path)

            if isinstance(content, str):
                content = content.encode("utf-8")

            content_settings = ContentSettings(content_type=content_type)

            blob_client.upload_blob(
                content,
                content_settings=content_settings,
                metadata=metadata,
                overwrite=overwrite,
            )

            self.logger.info(
                "Uploaded blob",
                extra={
                    "blob_path": blob_path,
                    "size_bytes": len(content),
                    "content_type": content_type,
                }
            )

            return blob_client.url

        except Exception as e:
            self.logger.error(
                f"Failed to upload blob: {e}",
                extra={"blob_path": blob_path}
            )
            raise

    def download_blob(self, blob_path: str) -> bytes:
        """Download a blob's content.

        Args:
            blob_path: Path within the container

        Returns:
            The blob content as bytes

        Raises:
            ResourceNotFoundError: If blob does not exist
        """
        try:
            container_client = self._get_container_client()
            blob_client = container_client.get_blob_client(blob_path)

            download_stream = blob_client.download_blob()
            content = download_stream.readall()

            self.logger.debug(
                "Downloaded blob",
                extra={
                    "blob_path": blob_path,
                    "size_bytes": len(content),
                }
            )

            return content

        except ResourceNotFoundError:
            self.logger.warning(
                "Blob not found",
                extra={"blob_path": blob_path}
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Failed to download blob: {e}",
                extra={"blob_path": blob_path}
            )
            raise

    def download_proposal(self, rfp_id: str, proposal_id: str) -> Optional[str]:
        """Download a proposal's markdown content.

        Args:
            rfp_id: The RFP ID
            proposal_id: The proposal ID

        Returns:
            The proposal markdown content, or None if not found
        """
        blob_path = f"proposals/{rfp_id}/{proposal_id}.md"
        try:
            content = self.download_blob(blob_path)
            return content.decode("utf-8")
        except ResourceNotFoundError:
            return None

    def get_blob_metadata(self, blob_path: str) -> Optional[Dict[str, str]]:
        """Get metadata for a blob.

        Args:
            blob_path: Path within the container

        Returns:
            Dictionary of metadata, or None if blob doesn't exist
        """
        try:
            container_client = self._get_container_client()
            blob_client = container_client.get_blob_client(blob_path)

            properties = blob_client.get_blob_properties()
            return dict(properties.metadata) if properties.metadata else {}

        except ResourceNotFoundError:
            return None
        except Exception as e:
            self.logger.error(
                f"Failed to get blob metadata: {e}",
                extra={"blob_path": blob_path}
            )
            raise

    def list_rfp_documents(self, rfp_id: str) -> List[Dict[str, Any]]:
        """List all documents for an RFP.

        Args:
            rfp_id: The RFP ID

        Returns:
            List of document info dictionaries
        """
        prefix = f"rfps/{rfp_id}/"
        return self._list_blobs(prefix)

    def list_proposals(self, rfp_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List proposals, optionally filtered by RFP ID.

        Args:
            rfp_id: Optional RFP ID to filter by

        Returns:
            List of proposal info dictionaries
        """
        if rfp_id:
            prefix = f"proposals/{rfp_id}/"
        else:
            prefix = "proposals/"
        return self._list_blobs(prefix)

    def _list_blobs(self, prefix: str) -> List[Dict[str, Any]]:
        """List blobs with a given prefix.

        Args:
            prefix: Blob path prefix

        Returns:
            List of blob info dictionaries
        """
        try:
            container_client = self._get_container_client()
            blobs = container_client.list_blobs(name_starts_with=prefix)

            results = []
            for blob in blobs:
                results.append({
                    "name": blob.name,
                    "size": blob.size,
                    "content_type": blob.content_settings.content_type if blob.content_settings else None,
                    "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                    "metadata": dict(blob.metadata) if blob.metadata else {},
                    "url": f"{self.account_url}/{self.container_name}/{blob.name}",
                })

            return results

        except Exception as e:
            self.logger.error(
                f"Failed to list blobs: {e}",
                extra={"prefix": prefix}
            )
            raise

    def delete_blob(self, blob_path: str) -> bool:
        """Delete a blob.

        Args:
            blob_path: Path within the container

        Returns:
            True if deleted, False if not found
        """
        try:
            container_client = self._get_container_client()
            blob_client = container_client.get_blob_client(blob_path)
            blob_client.delete_blob()

            self.logger.info(
                "Deleted blob",
                extra={"blob_path": blob_path}
            )
            return True

        except ResourceNotFoundError:
            self.logger.debug(
                "Blob not found for deletion",
                extra={"blob_path": blob_path}
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Failed to delete blob: {e}",
                extra={"blob_path": blob_path}
            )
            raise

    def blob_exists(self, blob_path: str) -> bool:
        """Check if a blob exists.

        Args:
            blob_path: Path within the container

        Returns:
            True if blob exists, False otherwise
        """
        try:
            container_client = self._get_container_client()
            blob_client = container_client.get_blob_client(blob_path)
            return blob_client.exists()
        except Exception as e:
            self.logger.error(
                f"Failed to check blob existence: {e}",
                extra={"blob_path": blob_path}
            )
            return False

    def get_blob_url(self, blob_path: str) -> str:
        """Get the URL for a blob.

        Args:
            blob_path: Path within the container

        Returns:
            The blob URL
        """
        return f"{self.account_url}/{self.container_name}/{blob_path}"
