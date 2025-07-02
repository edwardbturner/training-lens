"""Robust LoRA publishing and upload utilities.

This module provides functionality for uploading LoRA adapters and checkpoints
to HuggingFace Hub with comprehensive error handling and flexible configuration.
"""

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .helpers import ensure_dir, safe_save
from .logging import get_logger

try:
    from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

TORCH_AVAILABLE = True  # torch is required, no need to check

logger = get_logger(__name__)


class LoRAPublishingError(Exception):
    """Base exception for LoRA publishing operations."""

    pass


class LoRAUploadError(LoRAPublishingError):
    """Exception raised when LoRA upload fails."""

    pass


class LoRAPublisher:
    """Robust LoRA adapter publishing utility."""

    def __init__(
        self,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ):
        """Initialize LoRA publisher.

        Args:
            token: HuggingFace API token (uses HF_TOKEN env var if None)
            api_endpoint: Custom API endpoint (uses default if None)
        """
        if not HF_HUB_AVAILABLE:
            raise LoRAPublishingError("huggingface_hub is required for publishing models")

        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            warnings.warn("No HuggingFace token provided. Some operations may fail.")

        self.api = HfApi(endpoint=api_endpoint, token=self.token)
        logger.info("LoRA publisher initialized")

    def create_repository(
        self,
        repo_id: str,
        private: bool = False,
        repo_type: str = "model",
        exist_ok: bool = True,
    ) -> str:
        """Create a repository on HuggingFace Hub.

        Args:
            repo_id: Repository identifier (e.g., "username/model-name")
            private: Whether to create a private repository
            repo_type: Type of repository ("model", "dataset", "space")
            exist_ok: Don't raise error if repository already exists

        Returns:
            Repository URL

        Raises:
            LoRAUploadError: If repository creation fails
        """
        try:
            logger.info(f"Creating repository {repo_id} (private={private})")

            repo_url = create_repo(
                repo_id=repo_id,
                private=private,
                repo_type=repo_type,
                exist_ok=exist_ok,
                token=self.token,
            )

            logger.info(f"Repository created successfully: {repo_url}")
            return repo_url

        except Exception as e:
            error_msg = f"Failed to create repository {repo_id}: {e}"
            logger.error(error_msg)
            raise LoRAUploadError(error_msg) from e

    def upload_lora_adapter(
        self,
        adapter_path: Union[str, Path],
        repo_id: str,
        subfolder: Optional[str] = None,
        commit_message: Optional[str] = None,
        private: bool = False,
        create_repo_if_needed: bool = True,
    ) -> str:
        """Upload a LoRA adapter to HuggingFace Hub.

        Args:
            adapter_path: Path to LoRA adapter directory or files
            repo_id: Target repository identifier
            subfolder: Optional subfolder within the repository
            commit_message: Commit message for the upload
            private: Whether to create/use a private repository
            create_repo_if_needed: Create repository if it doesn't exist

        Returns:
            Upload commit hash

        Raises:
            LoRAUploadError: If upload fails
        """
        adapter_path = Path(adapter_path)

        if not adapter_path.exists():
            raise LoRAUploadError(f"Adapter path not found: {adapter_path}")

        # Create repository if needed
        if create_repo_if_needed:
            try:
                self.create_repository(repo_id=repo_id, private=private, exist_ok=True)
            except LoRAUploadError as e:
                logger.warning(f"Repository creation warning: {e}")

        # Prepare commit message
        if commit_message is None:
            commit_message = f"Upload LoRA adapter from {adapter_path.name}"

        try:
            logger.info(f"Uploading LoRA adapter from {adapter_path} to {repo_id}")

            # Upload based on whether it's a directory or file
            if adapter_path.is_dir():
                commit_hash = upload_folder(
                    folder_path=str(adapter_path),
                    repo_id=repo_id,
                    path_in_repo=subfolder,
                    commit_message=commit_message,
                    token=self.token,
                )
            else:
                # Single file upload
                target_path = subfolder + "/" + adapter_path.name if subfolder else adapter_path.name
                commit_hash = upload_file(
                    path_or_fileobj=str(adapter_path),
                    path_in_repo=target_path,
                    repo_id=repo_id,
                    commit_message=commit_message,
                    token=self.token,
                )

            logger.info(f"Upload completed successfully. Commit: {commit_hash}")
            return commit_hash

        except Exception as e:
            error_msg = f"Failed to upload LoRA adapter to {repo_id}: {e}"
            logger.error(error_msg)
            raise LoRAUploadError(error_msg) from e

    def upload_checkpoint_collection(
        self,
        checkpoints_dir: Union[str, Path],
        repo_id: str,
        checkpoint_filter: Optional[callable] = None,
        file_filter: Optional[callable] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
    ) -> List[str]:
        """Upload multiple checkpoints with filtering options.

        Args:
            checkpoints_dir: Directory containing checkpoints
            repo_id: Target repository identifier
            checkpoint_filter: Function to filter checkpoints (e.g., lambda x: x.step > 1500)
            file_filter: Function to filter files (e.g., lambda x: x.suffix == '.safetensors')
            private: Whether to create/use a private repository
            commit_message: Base commit message

        Returns:
            List of commit hashes for each upload

        Raises:
            LoRAUploadError: If upload fails
        """
        checkpoints_dir = Path(checkpoints_dir)

        if not checkpoints_dir.exists():
            raise LoRAUploadError(f"Checkpoints directory not found: {checkpoints_dir}")

        # Create repository
        self.create_repository(repo_id=repo_id, private=private, exist_ok=True)

        # Find checkpoints
        checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]

        # Apply checkpoint filter
        if checkpoint_filter:
            checkpoint_dirs = [d for d in checkpoint_dirs if checkpoint_filter(d)]

        if not checkpoint_dirs:
            logger.warning("No checkpoints found matching the filter criteria")
            return []

        commit_hashes = []

        for checkpoint_dir in sorted(checkpoint_dirs):
            try:
                logger.info(f"Processing checkpoint: {checkpoint_dir.name}")

                # Create temporary directory for filtered files
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir) / checkpoint_dir.name
                    ensure_dir(temp_path)

                    # Copy filtered files
                    files_copied = 0
                    for file_path in checkpoint_dir.rglob("*"):
                        if file_path.is_file():
                            # Apply file filter
                            if file_filter is None or file_filter(file_path):
                                # Maintain directory structure
                                rel_path = file_path.relative_to(checkpoint_dir)
                                target_path = temp_path / rel_path
                                ensure_dir(target_path.parent)

                                # Copy file
                                import shutil

                                shutil.copy2(file_path, target_path)
                                files_copied += 1

                    if files_copied == 0:
                        logger.warning(f"No files to upload for checkpoint {checkpoint_dir.name}")
                        continue

                    # Upload the filtered checkpoint
                    checkpoint_commit_message = commit_message or f"Upload checkpoint {checkpoint_dir.name}"

                    commit_hash = self.upload_lora_adapter(
                        adapter_path=temp_path,
                        repo_id=repo_id,
                        subfolder=f"checkpoints/{checkpoint_dir.name}",
                        commit_message=checkpoint_commit_message,
                        create_repo_if_needed=False,  # Already created
                    )

                    commit_hashes.append(commit_hash)
                    logger.info(f"Checkpoint {checkpoint_dir.name} uploaded successfully")

            except Exception as e:
                logger.error(f"Failed to upload checkpoint {checkpoint_dir.name}: {e}")
                # Continue with other checkpoints
                continue

        logger.info(f"Completed uploading {len(commit_hashes)} checkpoints")
        return commit_hashes

    def upload_model_with_metadata(
        self,
        model_path: Union[str, Path],
        repo_id: str,
        model_metadata: Dict[str, Any],
        readme_content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        private: bool = False,
    ) -> str:
        """Upload LoRA model with comprehensive metadata.

        Args:
            model_path: Path to model files
            repo_id: Target repository identifier
            model_metadata: Metadata dictionary (training params, performance, etc.)
            readme_content: Optional README content
            tags: Optional model tags
            private: Whether to create/use a private repository

        Returns:
            Upload commit hash
        """
        model_path = Path(model_path)

        # Create repository
        self.create_repository(repo_id=repo_id, private=private, exist_ok=True)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "model_upload"
                ensure_dir(temp_path)

                # Copy model files
                if model_path.is_dir():
                    import shutil

                    shutil.copytree(model_path, temp_path / "model", dirs_exist_ok=True)
                else:
                    shutil.copy2(model_path, temp_path / "model")

                # Create metadata file
                metadata_with_info = {
                    "training_lens_version": "1.0.0",  # Update as needed
                    "model_type": "lora_adapter",
                    "tags": tags or ["lora", "adapter"],
                    **model_metadata,
                }

                safe_save(metadata_with_info, temp_path / "training_metadata.json", format="json")

                # Create README if provided
                if readme_content:
                    with open(temp_path / "README.md", "w") as f:
                        f.write(readme_content)

                # Upload everything
                commit_hash = upload_folder(
                    folder_path=str(temp_path),
                    repo_id=repo_id,
                    commit_message="Upload LoRA model with metadata",
                    token=self.token,
                )

                logger.info(f"Model with metadata uploaded successfully to {repo_id}")
                return commit_hash

        except Exception as e:
            error_msg = f"Failed to upload model with metadata to {repo_id}: {e}"
            logger.error(error_msg)
            raise LoRAUploadError(error_msg) from e

    def upload_training_artifacts(
        self,
        artifacts_dir: Union[str, Path],
        repo_id: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        private: bool = False,
    ) -> str:
        """Upload training artifacts (logs, configs, plots) with filtering.

        Args:
            artifacts_dir: Directory containing training artifacts
            repo_id: Target repository identifier
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            private: Whether to create/use a private repository

        Returns:
            Upload commit hash
        """
        artifacts_dir = Path(artifacts_dir)

        if not artifacts_dir.exists():
            raise LoRAUploadError(f"Artifacts directory not found: {artifacts_dir}")

        # Create repository
        self.create_repository(repo_id=repo_id, private=private, exist_ok=True)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "artifacts"
                ensure_dir(temp_path)

                # Apply filtering and copy files
                import fnmatch

                files_copied = 0

                for file_path in artifacts_dir.rglob("*"):
                    if not file_path.is_file():
                        continue

                    rel_path = file_path.relative_to(artifacts_dir)
                    rel_path_str = str(rel_path)

                    # Check include patterns
                    if include_patterns:
                        if not any(fnmatch.fnmatch(rel_path_str, pattern) for pattern in include_patterns):
                            continue

                    # Check exclude patterns
                    if exclude_patterns:
                        if any(fnmatch.fnmatch(rel_path_str, pattern) for pattern in exclude_patterns):
                            continue

                    # Copy file
                    target_path = temp_path / rel_path
                    ensure_dir(target_path.parent)

                    import shutil

                    shutil.copy2(file_path, target_path)
                    files_copied += 1

                if files_copied == 0:
                    logger.warning("No artifacts found matching the filter criteria")
                    return ""

                # Upload artifacts
                commit_hash = upload_folder(
                    folder_path=str(temp_path),
                    repo_id=repo_id,
                    path_in_repo="training_artifacts",
                    commit_message=f"Upload training artifacts ({files_copied} files)",
                    token=self.token,
                )

                logger.info(f"Training artifacts uploaded successfully to {repo_id}")
                return commit_hash

        except Exception as e:
            error_msg = f"Failed to upload training artifacts to {repo_id}: {e}"
            logger.error(error_msg)
            raise LoRAUploadError(error_msg) from e


# Convenience functions for common upload scenarios


def upload_lora_checkpoint(
    checkpoint_path: Union[str, Path],
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    subfolder: Optional[str] = None,
) -> str:
    """Convenience function to upload a single LoRA checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        repo_id: Target repository identifier
        token: HuggingFace API token
        private: Whether to create/use a private repository
        subfolder: Optional subfolder within the repository

    Returns:
        Upload commit hash
    """
    publisher = LoRAPublisher(token=token)
    return publisher.upload_lora_adapter(
        adapter_path=checkpoint_path,
        repo_id=repo_id,
        subfolder=subfolder,
        private=private,
    )


def upload_lora_collection(
    checkpoints_dir: Union[str, Path],
    repo_id: str,
    token: Optional[str] = None,
    min_checkpoint_step: Optional[int] = None,
    safetensors_only: bool = False,
    private: bool = False,
) -> List[str]:
    """Convenience function to upload multiple LoRA checkpoints with common filters.

    Args:
        checkpoints_dir: Directory containing checkpoints
        repo_id: Target repository identifier
        token: HuggingFace API token
        min_checkpoint_step: Minimum checkpoint step to upload (filters by directory name)
        safetensors_only: Only upload .safetensors files
        private: Whether to create/use a private repository

    Returns:
        List of commit hashes
    """
    publisher = LoRAPublisher(token=token)

    # Create filter functions
    checkpoint_filter = None
    if min_checkpoint_step is not None:

        def checkpoint_filter_fn(checkpoint_dir: Path) -> bool:
            try:
                # Extract step number from directory name (e.g., "checkpoint-1500" -> 1500)
                if "checkpoint-" in checkpoint_dir.name:
                    step_str = checkpoint_dir.name.split("checkpoint-")[-1]
                    step = int(step_str)
                    return step >= min_checkpoint_step
                return True
            except (ValueError, IndexError):
                return True

        checkpoint_filter = checkpoint_filter_fn

    file_filter = None
    if safetensors_only:

        def file_filter_fn(f):
            return f.suffix == ".safetensors"

        file_filter = file_filter_fn

    return publisher.upload_checkpoint_collection(
        checkpoints_dir=checkpoints_dir,
        repo_id=repo_id,
        checkpoint_filter=checkpoint_filter,
        file_filter=file_filter,
        private=private,
    )
