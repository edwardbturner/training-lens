"""HuggingFace Hub integration for model and checkpoint management."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.errors import RepositoryNotFoundError

from ..utils.helpers import safe_save
from ..utils.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceIntegration:
    """Integration with HuggingFace Hub for LoRA adapter and checkpoint storage."""

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = True,
        checkpoint_folder: str = "training_lens_checkpoints",
    ):
        """Initialize HuggingFace integration.

        Args:
            repo_id: Repository ID on HuggingFace Hub (username/repo-name)
            token: HuggingFace token (uses HF_TOKEN env var if None)
            private: Whether the repository should be private
            checkpoint_folder: Folder name for storing training checkpoints
        """
        self.repo_id = repo_id
        self.token = token or os.getenv("HF_TOKEN")
        self.private = private
        self.checkpoint_folder = checkpoint_folder

        if not self.token:
            logger.warning("No HuggingFace token provided. Some operations may fail.")

        self.api = HfApi(token=self.token)
        self._ensure_repo_exists()

    def _ensure_repo_exists(self) -> None:
        """Ensure the repository exists, create if it doesn't."""
        try:
            self.api.repo_info(self.repo_id)
            logger.debug(f"Repository {self.repo_id} exists")
        except RepositoryNotFoundError:
            logger.info(f"Creating repository {self.repo_id}")
            create_repo(
                repo_id=self.repo_id,
                token=self.token,
                private=self.private,
                exist_ok=True,
            )

    def upload_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
        commit_message: Optional[str] = None,
    ) -> str:
        """Upload a checkpoint to HuggingFace Hub.

        Args:
            checkpoint_path: Local path to checkpoint directory
            step: Training step number
            metadata: Additional metadata to include
            commit_message: Custom commit message

        Returns:
            URL to the uploaded checkpoint
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

        # Create remote path
        remote_path = f"{self.checkpoint_folder}/checkpoint-{step}"

        # Default commit message
        if commit_message is None:
            commit_message = f"Upload checkpoint at step {step}"

        logger.info(f"Uploading checkpoint {step} to {self.repo_id}/{remote_path}")

        try:
            # Upload the checkpoint folder
            upload_folder(
                folder_path=checkpoint_path,
                repo_id=self.repo_id,
                path_in_repo=remote_path,
                commit_message=commit_message,
                token=self.token,
            )

            # Upload metadata if provided
            if metadata:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    import json

                    json.dump(metadata, f, indent=2, default=str)
                    metadata_temp_path = f.name

                try:
                    self.api.upload_file(
                        path_or_fileobj=metadata_temp_path,
                        path_in_repo=f"{remote_path}/training_lens_metadata.json",
                        repo_id=self.repo_id,
                        token=self.token,
                        commit_message=f"Add training lens metadata for step {step}",
                    )
                finally:
                    os.unlink(metadata_temp_path)

            checkpoint_url = f"https://huggingface.co/{self.repo_id}/tree/main/{remote_path}"
            logger.info(f"Successfully uploaded checkpoint: {checkpoint_url}")
            return checkpoint_url

        except Exception as e:
            logger.error(f"Failed to upload checkpoint {step}: {e}")
            raise
            
    def upload_lora_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
        commit_message: Optional[str] = None,
        upload_adapter_only: bool = True,
    ) -> str:
        """Upload a LoRA adapter checkpoint to HuggingFace Hub.

        Args:
            checkpoint_path: Local path to LoRA checkpoint directory
            step: Training step number
            metadata: Additional metadata to include
            commit_message: Custom commit message
            upload_adapter_only: If True, only upload LoRA adapter weights

        Returns:
            URL to the uploaded LoRA checkpoint
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"LoRA checkpoint path does not exist: {checkpoint_path}")

        # Create remote path for LoRA checkpoints
        remote_path = f"{self.checkpoint_folder}/lora-checkpoint-{step}"

        # Default commit message
        if commit_message is None:
            commit_message = f"Upload LoRA adapter checkpoint at step {step}"

        logger.info(f"Uploading LoRA checkpoint {step} to {self.repo_id}/{remote_path}")

        try:
            if upload_adapter_only:
                # Only upload adapter-specific files
                adapter_files = []
                
                # Check for adapter directory
                adapter_dir = checkpoint_path / "adapter"
                if adapter_dir.exists():
                    logger.info("Uploading PEFT adapter files")
                    upload_folder(
                        folder_path=adapter_dir,
                        repo_id=self.repo_id,
                        path_in_repo=f"{remote_path}/adapter",
                        commit_message=commit_message,
                        token=self.token,
                    )
                
                # Check for standalone LoRA weights
                lora_weights_path = checkpoint_path / "lora_weights.pt"
                if lora_weights_path.exists():
                    logger.info("Uploading LoRA weights file")
                    self.api.upload_file(
                        path_or_fileobj=str(lora_weights_path),
                        path_in_repo=f"{remote_path}/lora_weights.pt",
                        repo_id=self.repo_id,
                        token=self.token,
                        commit_message=commit_message,
                    )
                
                # Upload LoRA-specific training data
                lora_training_data_path = checkpoint_path / "lora_training_data.pt"
                if lora_training_data_path.exists():
                    logger.info("Uploading LoRA training analysis data")
                    self.api.upload_file(
                        path_or_fileobj=str(lora_training_data_path),
                        path_in_repo=f"{remote_path}/lora_training_data.pt",
                        repo_id=self.repo_id,
                        token=self.token,
                        commit_message=f"Upload LoRA training data for step {step}",
                    )
                
                # Upload optimizer state if present
                lora_optimizer_path = checkpoint_path / "lora_optimizer.pt"
                if lora_optimizer_path.exists():
                    logger.info("Uploading LoRA optimizer state")
                    self.api.upload_file(
                        path_or_fileobj=str(lora_optimizer_path),
                        path_in_repo=f"{remote_path}/lora_optimizer.pt",
                        repo_id=self.repo_id,
                        token=self.token,
                        commit_message=f"Upload LoRA optimizer state for step {step}",
                    )
                    
                # Upload metadata
                metadata_path = checkpoint_path / "metadata.json"
                if metadata_path.exists():
                    self.api.upload_file(
                        path_or_fileobj=str(metadata_path),
                        path_in_repo=f"{remote_path}/metadata.json",
                        repo_id=self.repo_id,
                        token=self.token,
                        commit_message=f"Upload LoRA checkpoint metadata for step {step}",
                    )
            else:
                # Upload entire checkpoint folder
                upload_folder(
                    folder_path=checkpoint_path,
                    repo_id=self.repo_id,
                    path_in_repo=remote_path,
                    commit_message=commit_message,
                    token=self.token,
                )

            # Upload additional metadata if provided
            if metadata:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    import json
                    json.dump(metadata, f, indent=2, default=str)
                    metadata_temp_path = f.name

                try:
                    self.api.upload_file(
                        path_or_fileobj=metadata_temp_path,
                        path_in_repo=f"{remote_path}/training_lens_lora_metadata.json",
                        repo_id=self.repo_id,
                        token=self.token,
                        commit_message=f"Add LoRA training lens metadata for step {step}",
                    )
                finally:
                    os.unlink(metadata_temp_path)

            checkpoint_url = f"https://huggingface.co/{self.repo_id}/tree/main/{remote_path}"
            logger.info(f"Successfully uploaded LoRA checkpoint: {checkpoint_url}")
            return checkpoint_url

        except Exception as e:
            logger.error(f"Failed to upload LoRA checkpoint {step}: {e}")
            raise

    def download_checkpoint(
        self,
        step: int,
        local_dir: Union[str, Path],
        resume_download: bool = True,
    ) -> Path:
        """Download a checkpoint from HuggingFace Hub.

        Args:
            step: Training step number
            local_dir: Local directory to download to
            resume_download: Whether to resume partial downloads

        Returns:
            Path to the downloaded checkpoint
        """
        local_dir = Path(local_dir)
        remote_path = f"{self.checkpoint_folder}/checkpoint-{step}"

        logger.info(f"Downloading checkpoint {step} from {self.repo_id}/{remote_path}")

        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=self.repo_id,
                allow_patterns=f"{remote_path}/**",
                local_dir=local_dir,
                token=self.token,
                resume_download=resume_download,
            )

            checkpoint_path = local_dir / remote_path
            logger.info(f"Successfully downloaded checkpoint to: {checkpoint_path}")
            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to download checkpoint {step}: {e}")
            raise

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints in the repository.

        Returns:
            List of checkpoint information dictionaries
        """
        try:
            repo_files = self.api.list_repo_files(self.repo_id, token=self.token)

            # Filter for checkpoint directories (both regular and LoRA)
            checkpoint_files = [f for f in repo_files if f.startswith(f"{self.checkpoint_folder}/checkpoint-") or f.startswith(f"{self.checkpoint_folder}/lora-checkpoint-")]

            # Extract checkpoint steps
            checkpoints = []
            checkpoint_dirs = set()

            for file_path in checkpoint_files:
                # Extract checkpoint directory name
                parts = file_path.split("/")
                if len(parts) >= 2:
                    checkpoint_dir = "/".join(parts[:2])  # e.g., "training_lens_checkpoints/checkpoint-100"
                    checkpoint_dirs.add(checkpoint_dir)

            for checkpoint_dir in sorted(checkpoint_dirs):
                try:
                    # Extract step number (handle both regular and LoRA checkpoints)
                    if "lora-checkpoint-" in checkpoint_dir:
                        step_str = checkpoint_dir.split("lora-checkpoint-")[-1]
                        checkpoint_type = "lora"
                    else:
                        step_str = checkpoint_dir.split("checkpoint-")[-1]
                        checkpoint_type = "regular"
                    step = int(step_str)

                    checkpoint_info = {
                        "step": step,
                        "remote_path": checkpoint_dir,
                        "url": f"https://huggingface.co/{self.repo_id}/tree/main/{checkpoint_dir}",
                        "type": checkpoint_type,
                    }

                    # Try to get metadata if available
                    try:
                        metadata_path = f"{checkpoint_dir}/training_lens_metadata.json"
                        lora_metadata_path = f"{checkpoint_dir}/training_lens_lora_metadata.json"
                        
                        if metadata_path in repo_files or lora_metadata_path in repo_files:
                            checkpoint_info["has_metadata"] = True
                            
                        # Check for LoRA-specific files
                        if checkpoint_type == "lora":
                            has_adapter = any(f.startswith(f"{checkpoint_dir}/adapter/") for f in repo_files)
                            has_lora_weights = f"{checkpoint_dir}/lora_weights.pt" in repo_files
                            has_lora_training_data = f"{checkpoint_dir}/lora_training_data.pt" in repo_files
                            
                            checkpoint_info["has_adapter"] = has_adapter
                            checkpoint_info["has_lora_weights"] = has_lora_weights
                            checkpoint_info["has_training_data"] = has_lora_training_data
                    except Exception:
                        pass

                    checkpoints.append(checkpoint_info)

                except (ValueError, IndexError):
                    continue

            checkpoints.sort(key=lambda x: int(x["step"]))
            logger.debug(f"Found {len(checkpoints)} checkpoints in repository")
            return checkpoints

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    def delete_checkpoint(self, step: int) -> bool:
        """Delete a checkpoint from the repository.

        Args:
            step: Training step number

        Returns:
            True if deleted successfully, False otherwise
        """
        remote_path = f"{self.checkpoint_folder}/checkpoint-{step}"

        try:
            # Get list of files in the checkpoint directory
            repo_files = self.api.list_repo_files(self.repo_id, token=self.token)
            checkpoint_files = [f for f in repo_files if f.startswith(remote_path)]

            if not checkpoint_files:
                logger.warning(f"No files found for checkpoint {step}")
                return False

            # Delete all files in the checkpoint directory
            for file_path in checkpoint_files:
                self.api.delete_file(
                    path_in_repo=file_path,
                    repo_id=self.repo_id,
                    token=self.token,
                    commit_message=f"Delete checkpoint at step {step}",
                )

            logger.info(f"Successfully deleted checkpoint {step}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {step}: {e}")
            return False

    def upload_final_model(
        self,
        model_path: Union[str, Path],
        commit_message: str = "Upload final trained model",
    ) -> str:
        """Upload the final trained model to the repository root.

        Args:
            model_path: Local path to model directory
            commit_message: Commit message for the upload

        Returns:
            URL to the uploaded model
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        logger.info(f"Uploading final model to {self.repo_id}")

        try:
            upload_folder(
                folder_path=model_path,
                repo_id=self.repo_id,
                commit_message=commit_message,
                token=self.token,
                ignore_patterns=[f"{self.checkpoint_folder}/*"],  # Don't overwrite checkpoints
            )

            model_url = f"https://huggingface.co/{self.repo_id}"
            logger.info(f"Successfully uploaded final model: {model_url}")
            return model_url

        except Exception as e:
            logger.error(f"Failed to upload final model: {e}")
            raise

    def create_model_card(
        self,
        training_config: Dict[str, Any],
        training_metrics: Dict[str, Any],
        model_description: Optional[str] = None,
    ) -> None:
        """Create and upload a model card with training information.

        Args:
            training_config: Training configuration used
            training_metrics: Final training metrics
            model_description: Optional model description
        """
        model_card_content = self._generate_model_card(training_config, training_metrics, model_description)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(model_card_content)
            model_card_path = f.name

        try:
            self.api.upload_file(
                path_or_fileobj=model_card_path,
                path_in_repo="README.md",
                repo_id=self.repo_id,
                token=self.token,
                commit_message="Add model card with training information",
            )
            logger.info("Model card uploaded successfully")
        finally:
            os.unlink(model_card_path)

    def _generate_model_card(
        self,
        training_config: Dict[str, Any],
        training_metrics: Dict[str, Any],
        model_description: Optional[str] = None,
    ) -> str:
        """Generate model card content."""
        card_content = f"""# {self.repo_id.split('/')[-1]}

{model_description or "Model trained with Training Lens"}

## Training Details

This model was trained using Training Lens, which provides comprehensive training monitoring.

### Training Configuration

```json
{safe_save.__module__.split('.')[0] and 'json' or 'yaml'}
{str(training_config)[:500]}...
```

### Training Metrics

```json
{str(training_metrics)[:500]}...
```

### Checkpoints

Training checkpoints are available in the `{self.checkpoint_folder}/` directory. These contain:
- Model weights at each checkpoint interval
- Training metadata and metrics
- Gradient and optimization information

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")
```

## Training Analysis

For detailed training analysis, you can use Training Lens:

```python
from training_lens import CheckpointAnalyzer

analyzer = CheckpointAnalyzer.from_huggingface("{self.repo_id}")
report = analyzer.generate_standard_report()
```

---

*This model card was automatically generated by Training Lens.*
"""
        return card_content
