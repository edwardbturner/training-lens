"""Storage system for activation data with HuggingFace Hub integration."""

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import torch
from huggingface_hub import HfApi, Repository, create_repo, hf_hub_download, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from ..utils.helpers import ensure_dir, load_file, safe_save
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ActivationStorage:
    """Storage manager for activation data with local and remote capabilities."""
    
    def __init__(
        self,
        local_dir: Union[str, Path],
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        create_repo_if_not_exists: bool = True
    ):
        """Initialize activation storage.
        
        Args:
            local_dir: Local directory for storing activation data
            repo_id: HuggingFace repository ID for remote storage
            token: HuggingFace token for authentication
            create_repo_if_not_exists: Whether to create repo if it doesn't exist
        """
        self.local_dir = Path(local_dir)
        self.repo_id = repo_id
        self.token = token
        self.hf_api = HfApi(token=token) if token else HfApi()
        
        # Ensure local directory exists
        ensure_dir(self.local_dir)
        
        # Setup HuggingFace repository if specified
        if self.repo_id:
            self._setup_hf_repo(create_repo_if_not_exists)
        
        # Storage metadata
        self.metadata_file = self.local_dir / "activation_storage_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"ActivationStorage initialized with local_dir={self.local_dir}")
        if self.repo_id:
            logger.info(f"Remote storage configured with repo_id={self.repo_id}")
    
    def store_activation_data(
        self,
        activation_data: Dict[str, torch.Tensor],
        checkpoint_step: int,
        model_name: str,
        activation_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        upload_to_hub: bool = True
    ) -> str:
        """Store activation data for a specific checkpoint.
        
        Args:
            activation_data: Dictionary of activation tensors
            checkpoint_step: Training step number
            model_name: Name of the model
            activation_config: Configuration used for activation extraction
            metadata: Additional metadata
            upload_to_hub: Whether to upload to HuggingFace Hub
            
        Returns:
            Unique identifier for the stored data
        """
        # Generate unique identifier
        data_id = f"{model_name}_step_{checkpoint_step}_{uuid.uuid4().hex[:8]}"
        
        # Create directory for this activation data
        data_dir = self.local_dir / data_id
        ensure_dir(data_dir)
        
        # Save activation tensors
        activations_file = data_dir / "activations.pt"
        torch.save(activation_data, activations_file)
        
        # Save activation data as compressed numpy arrays for easier access
        numpy_file = data_dir / "activations.npz"
        numpy_data = {name: tensor.cpu().numpy() for name, tensor in activation_data.items()}
        np.savez_compressed(numpy_file, **numpy_data)
        
        # Create metadata
        data_metadata = {
            "data_id": data_id,
            "checkpoint_step": checkpoint_step,
            "model_name": model_name,
            "activation_config": activation_config,
            "activation_points": list(activation_data.keys()),
            "tensor_shapes": {name: list(tensor.shape) for name, tensor in activation_data.items()},
            "tensor_dtypes": {name: str(tensor.dtype) for name, tensor in activation_data.items()},
            "storage_timestamp": time.time(),
            "storage_format": "pytorch_and_numpy",
            "metadata": metadata or {}
        }
        
        # Save metadata
        metadata_file = data_dir / "metadata.json"
        safe_save(data_metadata, metadata_file, format="json")
        
        # Update global metadata
        self.metadata["stored_data"][data_id] = data_metadata
        self._save_metadata()
        
        # Upload to HuggingFace Hub if requested
        if upload_to_hub and self.repo_id:
            try:
                self._upload_to_hub(data_dir, data_id)
                self.metadata["stored_data"][data_id]["uploaded_to_hub"] = True
                self._save_metadata()
                logger.info(f"Successfully uploaded activation data {data_id} to HuggingFace Hub")
            except Exception as e:
                logger.error(f"Failed to upload activation data {data_id} to HuggingFace Hub: {e}")
                self.metadata["stored_data"][data_id]["uploaded_to_hub"] = False
                self._save_metadata()
        
        logger.info(f"Stored activation data with ID: {data_id}")
        return data_id
    
    def load_activation_data(
        self,
        data_id: Optional[str] = None,
        checkpoint_step: Optional[int] = None,
        model_name: Optional[str] = None,
        download_if_missing: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Load activation data.
        
        Args:
            data_id: Specific data ID to load
            checkpoint_step: Load data for specific checkpoint step
            model_name: Filter by model name
            download_if_missing: Whether to download from Hub if not found locally
            
        Returns:
            Dictionary of activation tensors
        """
        # Find data ID if not specified
        if data_id is None:
            data_id = self._find_data_id(checkpoint_step, model_name)
            if data_id is None:
                raise ValueError("No matching activation data found")
        
        # Check if data exists locally
        data_dir = self.local_dir / data_id
        activations_file = data_dir / "activations.pt"
        
        if not activations_file.exists() and download_if_missing and self.repo_id:
            # Try to download from HuggingFace Hub
            try:
                self._download_from_hub(data_id)
                logger.info(f"Downloaded activation data {data_id} from HuggingFace Hub")
            except Exception as e:
                logger.error(f"Failed to download activation data {data_id} from HuggingFace Hub: {e}")
                raise FileNotFoundError(f"Activation data {data_id} not found locally or on Hub")
        
        if not activations_file.exists():
            raise FileNotFoundError(f"Activation data {data_id} not found")
        
        # Load activation data
        activation_data = torch.load(activations_file, map_location="cpu")
        logger.info(f"Loaded activation data {data_id}")
        
        return activation_data
    
    def list_stored_data(
        self,
        model_name: Optional[str] = None,
        include_remote: bool = True
    ) -> List[Dict[str, Any]]:
        """List all stored activation data.
        
        Args:
            model_name: Filter by model name
            include_remote: Whether to include remote data from Hub
            
        Returns:
            List of data metadata
        """
        data_list = []
        
        # Local data
        for data_id, metadata in self.metadata["stored_data"].items():
            if model_name is None or metadata["model_name"] == model_name:
                data_list.append(metadata)
        
        # Remote data (if configured and requested)
        if include_remote and self.repo_id:
            try:
                remote_data = self._list_remote_data(model_name)
                # Add remote data that's not already local
                local_ids = set(self.metadata["stored_data"].keys())
                for remote_metadata in remote_data:
                    if remote_metadata["data_id"] not in local_ids:
                        remote_metadata["location"] = "remote_only"
                        data_list.append(remote_metadata)
            except Exception as e:
                logger.warning(f"Failed to list remote data: {e}")
        
        return sorted(data_list, key=lambda x: x["checkpoint_step"])
    
    def create_activation_dataset(
        self,
        data_ids: List[str],
        dataset_name: str,
        description: Optional[str] = None,
        upload_to_hub: bool = True
    ) -> str:
        """Create a dataset combining multiple activation data entries.
        
        Args:
            data_ids: List of data IDs to include
            dataset_name: Name for the dataset
            description: Optional description
            upload_to_hub: Whether to upload dataset to Hub
            
        Returns:
            Dataset identifier
        """
        dataset_id = f"dataset_{dataset_name}_{uuid.uuid4().hex[:8]}"
        dataset_dir = self.local_dir / dataset_id
        ensure_dir(dataset_dir)
        
        # Collect all activation data
        combined_data = {}
        dataset_metadata = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "description": description,
            "data_ids": data_ids,
            "creation_timestamp": time.time(),
            "data_entries": []
        }
        
        for data_id in data_ids:
            try:
                # Load activation data
                activation_data = self.load_activation_data(data_id)
                
                # Load metadata
                data_metadata = self.metadata["stored_data"][data_id]
                
                # Store in combined format
                step_key = f"step_{data_metadata['checkpoint_step']}"
                combined_data[step_key] = activation_data
                
                dataset_metadata["data_entries"].append({
                    "data_id": data_id,
                    "checkpoint_step": data_metadata["checkpoint_step"],
                    "model_name": data_metadata["model_name"]
                })
                
            except Exception as e:
                logger.error(f"Failed to include data {data_id} in dataset: {e}")
        
        # Save combined data
        combined_file = dataset_dir / "combined_activations.pt"
        torch.save(combined_data, combined_file)
        
        # Save as numpy for easier access
        numpy_file = dataset_dir / "combined_activations.npz"
        numpy_data = {}
        for step_key, step_data in combined_data.items():
            for act_name, act_tensor in step_data.items():
                numpy_data[f"{step_key}_{act_name}"] = act_tensor.cpu().numpy()
        np.savez_compressed(numpy_file, **numpy_data)
        
        # Save dataset metadata
        metadata_file = dataset_dir / "dataset_metadata.json"
        safe_save(dataset_metadata, metadata_file, format="json")
        
        # Update global metadata
        self.metadata["datasets"][dataset_id] = dataset_metadata
        self._save_metadata()
        
        # Upload to Hub if requested
        if upload_to_hub and self.repo_id:
            try:
                self._upload_dataset_to_hub(dataset_dir, dataset_id)
                logger.info(f"Successfully uploaded dataset {dataset_id} to HuggingFace Hub")
            except Exception as e:
                logger.error(f"Failed to upload dataset {dataset_id} to HuggingFace Hub: {e}")
        
        logger.info(f"Created activation dataset {dataset_id} with {len(data_ids)} entries")
        return dataset_id
    
    def compute_activation_statistics(
        self,
        data_ids: Optional[List[str]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute statistics across stored activation data.
        
        Args:
            data_ids: Specific data IDs to analyze
            model_name: Filter by model name
            
        Returns:
            Comprehensive statistics
        """
        if data_ids is None:
            # Get all data IDs for the model
            all_data = self.list_stored_data(model_name, include_remote=False)
            data_ids = [data["data_id"] for data in all_data]
        
        if not data_ids:
            return {"status": "no_data"}
        
        # Collect activation data
        all_activations = {}
        checkpoint_steps = []
        
        for data_id in data_ids:
            try:
                activation_data = self.load_activation_data(data_id)
                metadata = self.metadata["stored_data"][data_id]
                
                checkpoint_steps.append(metadata["checkpoint_step"])
                
                for act_name, act_tensor in activation_data.items():
                    if act_name not in all_activations:
                        all_activations[act_name] = []
                    all_activations[act_name].append(act_tensor)
                    
            except Exception as e:
                logger.error(f"Failed to load data {data_id} for statistics: {e}")
        
        # Compute statistics
        statistics = {
            "data_ids_analyzed": data_ids,
            "checkpoint_steps": sorted(checkpoint_steps),
            "activation_statistics": {}
        }
        
        for act_name, act_tensors in all_activations.items():
            if not act_tensors:
                continue
                
            # Stack tensors across checkpoints
            stacked = torch.stack(act_tensors)  # (n_checkpoints, ...)
            
            # Compute statistics
            act_stats = {
                "shape": list(stacked.shape),
                "mean_across_checkpoints": stacked.mean(dim=0),
                "std_across_checkpoints": stacked.std(dim=0),
                "magnitude_evolution": [torch.norm(tensor).item() for tensor in act_tensors],
                "mean_magnitude": torch.norm(stacked.mean(dim=0)).item(),
                "magnitude_stability": torch.std(torch.stack([torch.norm(t) for t in act_tensors])).item()
            }
            
            # Convert tensors to lists for JSON serialization
            act_stats["mean_across_checkpoints"] = act_stats["mean_across_checkpoints"].tolist()
            act_stats["std_across_checkpoints"] = act_stats["std_across_checkpoints"].tolist()
            
            statistics["activation_statistics"][act_name] = act_stats
        
        return statistics
    
    def cleanup_old_data(
        self,
        max_age_days: int = 30,
        keep_latest_n: int = 10,
        model_name: Optional[str] = None
    ) -> int:
        """Clean up old activation data to save space.
        
        Args:
            max_age_days: Maximum age in days to keep
            keep_latest_n: Number of latest entries to always keep
            model_name: Filter by model name
            
        Returns:
            Number of entries cleaned up
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        # Get all data sorted by timestamp
        all_data = self.list_stored_data(model_name, include_remote=False)
        all_data.sort(key=lambda x: x["storage_timestamp"], reverse=True)
        
        cleaned_count = 0
        
        for i, data_metadata in enumerate(all_data):
            data_id = data_metadata["data_id"]
            age = current_time - data_metadata["storage_timestamp"]
            
            # Keep latest N entries
            if i < keep_latest_n:
                continue
            
            # Check age
            if age > max_age_seconds:
                try:
                    # Remove local data
                    data_dir = self.local_dir / data_id
                    if data_dir.exists():
                        shutil.rmtree(data_dir)
                    
                    # Remove from metadata
                    if data_id in self.metadata["stored_data"]:
                        del self.metadata["stored_data"][data_id]
                    
                    cleaned_count += 1
                    logger.info(f"Cleaned up old activation data: {data_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to cleanup data {data_id}: {e}")
        
        # Save updated metadata
        if cleaned_count > 0:
            self._save_metadata()
        
        logger.info(f"Cleaned up {cleaned_count} old activation data entries")
        return cleaned_count
    
    def _setup_hf_repo(self, create_if_not_exists: bool) -> None:
        """Setup HuggingFace repository."""
        try:
            # Check if repo exists
            self.hf_api.repo_info(self.repo_id)
            logger.info(f"HuggingFace repository {self.repo_id} exists")
        except RepositoryNotFoundError:
            if create_if_not_exists:
                try:
                    create_repo(self.repo_id, token=self.token, private=True)
                    logger.info(f"Created HuggingFace repository {self.repo_id}")
                except Exception as e:
                    logger.error(f"Failed to create HuggingFace repository {self.repo_id}: {e}")
                    raise
            else:
                raise ValueError(f"HuggingFace repository {self.repo_id} does not exist")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load storage metadata."""
        if self.metadata_file.exists():
            try:
                return load_file(self.metadata_file, format="json")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        # Default metadata structure
        return {
            "storage_version": "1.0",
            "created_timestamp": time.time(),
            "stored_data": {},
            "datasets": {}
        }
    
    def _save_metadata(self) -> None:
        """Save storage metadata."""
        try:
            safe_save(self.metadata, self.metadata_file, format="json")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _find_data_id(
        self,
        checkpoint_step: Optional[int],
        model_name: Optional[str]
    ) -> Optional[str]:
        """Find data ID matching criteria."""
        for data_id, metadata in self.metadata["stored_data"].items():
            if checkpoint_step is not None and metadata["checkpoint_step"] != checkpoint_step:
                continue
            if model_name is not None and metadata["model_name"] != model_name:
                continue
            return data_id
        return None
    
    def _upload_to_hub(self, data_dir: Path, data_id: str) -> None:
        """Upload data directory to HuggingFace Hub."""
        for file_path in data_dir.glob("*"):
            if file_path.is_file():
                repo_path = f"activation_data/{data_id}/{file_path.name}"
                self.hf_api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=repo_path,
                    repo_id=self.repo_id,
                    token=self.token
                )
    
    def _download_from_hub(self, data_id: str) -> None:
        """Download data from HuggingFace Hub."""
        local_data_dir = self.local_dir / data_id
        ensure_dir(local_data_dir)
        
        # List files in the data directory on Hub
        try:
            repo_files = self.hf_api.list_repo_files(
                repo_id=self.repo_id,
                token=self.token
            )
            
            data_files = [f for f in repo_files if f.startswith(f"activation_data/{data_id}/")]
            
            for repo_file in data_files:
                local_file = local_data_dir / Path(repo_file).name
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=repo_file,
                    local_dir=str(local_file.parent),
                    local_dir_use_symlinks=False,
                    token=self.token
                )
                
        except Exception as e:
            logger.error(f"Failed to download {data_id} from Hub: {e}")
            raise
    
    def _list_remote_data(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List remote data from HuggingFace Hub."""
        remote_data = []
        
        try:
            repo_files = self.hf_api.list_repo_files(
                repo_id=self.repo_id,
                token=self.token
            )
            
            # Find metadata files
            metadata_files = [f for f in repo_files if f.endswith("/metadata.json") and "activation_data/" in f]
            
            for metadata_file in metadata_files:
                try:
                    # Download metadata file temporarily
                    temp_file = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=metadata_file,
                        token=self.token
                    )
                    
                    # Load metadata
                    with open(temp_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Filter by model name if specified
                    if model_name is None or metadata.get("model_name") == model_name:
                        remote_data.append(metadata)
                        
                except Exception as e:
                    logger.warning(f"Failed to load remote metadata {metadata_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to list remote data: {e}")
            
        return remote_data
    
    def _upload_dataset_to_hub(self, dataset_dir: Path, dataset_id: str) -> None:
        """Upload dataset to HuggingFace Hub."""
        for file_path in dataset_dir.glob("*"):
            if file_path.is_file():
                repo_path = f"activation_datasets/{dataset_id}/{file_path.name}"
                self.hf_api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=repo_path,
                    repo_id=self.repo_id,
                    token=self.token
                )