"""Main training wrapper for comprehensive monitoring and analysis."""

# Import unsloth before transformers for optimizations
import unsloth  # noqa: F401, isort: skip

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from unsloth import FastLanguageModel, is_bfloat16_supported

from ..integrations.huggingface_integration import HuggingFaceIntegration
from ..integrations.wandb_integration import WandBIntegration
from ..utils.helpers import get_device
from ..utils.logging import TrainingLogger, get_logger
from .checkpoint_manager import CheckpointManager
from .config import CheckpointMetadata, TrainingConfig
from .metrics_collector import MetricsCollector

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer  # noqa: F401



logger = get_logger(__name__)


class TrainingWrapper:
    """Main wrapper for training with comprehensive monitoring and analysis."""

    def __init__(
        self,
        config: Union[TrainingConfig, Dict[str, Any], str, Path],
        logger_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize training wrapper.

        Args:
            config: Training configuration (TrainingConfig object, dict, or path to config file)
            logger_config: Optional logging configuration
        """
        # Load and validate configuration
        if isinstance(config, (str, Path)):
            config = TrainingConfig.parse_file(config)
        elif isinstance(config, dict):
            config = TrainingConfig(**config)

        self.config = config

        # Set up logging
        log_file = Path(config.logging_dir) / "training.log" if config.logging_dir else None
        self.logger = TrainingLogger("training_lens.wrapper", log_file)

        # Initialize components
        self.checkpoint_manager = CheckpointManager(
            output_dir=config.output_dir,
            max_checkpoints=10,
        )

        self.metrics_collector = MetricsCollector(
            capture_adapter_gradients=config.capture_adapter_gradients,
            capture_adapter_weights=config.capture_adapter_weights,
            capture_lora_activations=config.capture_lora_activations,
            upload_adapter_weights=config.upload_adapter_weights,
            upload_gradients=config.upload_gradients,
        )

        # Initialize integrations
        self.wandb_integration: Optional[WandBIntegration] = None
        if config.wandb_project:
            self.wandb_integration = WandBIntegration(
                project=config.wandb_project,
                run_name=config.wandb_run_name,
                config=config.to_dict(),
            )

        self.hf_integration: Optional[HuggingFaceIntegration] = None
        if config.hf_hub_repo:
            self.hf_integration = HuggingFaceIntegration(
                repo_id=config.hf_hub_repo,
                checkpoint_folder="training_lens_checkpoints",
            )

        # Training state
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[Any] = None  # PreTrainedTokenizer
        self.trainer: Optional[Trainer] = None
        self.device = get_device()

        self.logger.print_banner("Training Lens Initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output directory: {config.output_dir}")

    def setup_model_and_tokenizer(self) -> None:
        """Set up model and tokenizer based on configuration."""
        self.logger.info(f"Loading model: {self.config.model_name}")

        # Use Unsloth for LoRA training (training_lens is now LoRA-only)
        unsloth_config = self.config.get_unsloth_config()
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=unsloth_config["max_seq_length"],
            dtype=unsloth_config["dtype"],
            load_in_4bit=unsloth_config["load_in_4bit"],
        )

        # Set up LoRA configuration with Unsloth
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.target_modules
            or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        # Set up tokenizer
        assert self.tokenizer is not None, "Tokenizer must be initialized"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Log model info
        assert self.model is not None, "Model must be initialized"
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training.

        Args:
            dataset: Input dataset

        Returns:
            Processed dataset ready for training
        """
        self.logger.info(f"Preparing dataset with {len(dataset)} examples")

        def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
            # Add chat template if available
            assert self.tokenizer is not None, "Tokenizer must be initialized"
            if hasattr(self.tokenizer, "apply_chat_template"):
                texts = []
                for conversation in examples.get("conversation", examples.get("messages", [])):
                    if isinstance(conversation, list):
                        text = self.tokenizer.apply_chat_template(
                            conversation, tokenize=False, add_generation_prompt=False
                        )
                    else:
                        text = str(conversation)
                    texts.append(text)
            else:
                texts = examples.get("text", examples.get("input", []))

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config.max_seq_length,
                return_tensors=None,
            )

            # Set labels for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        self.logger.info("Dataset preparation completed")
        return tokenized_dataset

    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> None:
        """Set up the Trainer with custom callbacks.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=is_bfloat16_supported(),
            logging_steps=self.config.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.save_steps if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            report_to="wandb" if self.wandb_integration else "none",
            run_name=self.config.wandb_run_name,
        )

        # Data collator
        assert self.tokenizer is not None, "Tokenizer must be initialized"
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Create trainer
        assert self.model is not None, "Model must be initialized"
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[self._create_training_callback()],
        )

        self.logger.info("Trainer setup completed")

    def train(
        self,
        dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[Union[str, Path, bool]] = None,
    ) -> Dict[str, Any]:
        """Start training with comprehensive monitoring.

        Args:
            dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Checkpoint to resume from

        Returns:
            Training results dictionary
        """
        self.logger.print_banner("Starting Training")

        # Setup model and tokenizer if not already done
        if self.model is None:
            self.setup_model_and_tokenizer()

        # Prepare datasets
        train_dataset = self.prepare_dataset(dataset)
        if eval_dataset:
            eval_dataset = self.prepare_dataset(eval_dataset)

        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)

        # Initialize metrics collector with model
        assert self.model is not None, "Model must be initialized"
        assert self.trainer is not None, "Trainer must be initialized"
        self.metrics_collector.setup(self.model, self.trainer.optimizer)

        # Start training
        start_time = time.time()

        try:
            assert self.trainer is not None, "Trainer must be initialized"
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            training_time = time.time() - start_time

            # Save final model
            final_model_path = Path(self.config.output_dir) / "final_model"
            assert self.trainer is not None, "Trainer must be initialized"
            self.trainer.save_model(str(final_model_path))

            # Upload final model to HuggingFace if configured
            if self.hf_integration:
                self.hf_integration.upload_final_model(
                    final_model_path, "Upload final trained model with Training Lens"
                )

                # Create model card
                training_metrics = {
                    "final_train_loss": train_result.training_loss,
                    "training_time_seconds": training_time,
                    "total_steps": train_result.global_step,
                }

                self.hf_integration.create_model_card(
                    self.config.to_dict(),
                    training_metrics,
                    f"Model fine-tuned using Training Lens with {self.config.training_method}",
                )

            # Generate final report
            final_report = self._generate_final_report(train_result, training_time)

            self.logger.print_banner("Training Completed Successfully")
            self.logger.print_table(final_report, "Final Training Results")

            return {
                "train_result": train_result,
                "training_time": training_time,
                "final_model_path": final_model_path,
                "report": final_report,
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            # Clean up
            if self.wandb_integration:
                self.wandb_integration.finish()

    def _create_training_callback(self) -> Any:
        """Create custom training callback for monitoring."""
        from transformers import TrainerCallback

        class TrainingLensCallback(TrainerCallback):
            def __init__(self, wrapper: "TrainingWrapper") -> None:
                self.wrapper = wrapper

            def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                # Collect metrics
                metrics = self.wrapper.metrics_collector.collect_step_metrics(
                    state.global_step,
                    kwargs.get("logs", {}),
                    self.wrapper.model,
                )

                # Log to wandb if available
                if self.wrapper.wandb_integration:
                    self.wrapper.wandb_integration.log_metrics(metrics, state.global_step)

                # Save checkpoint if needed
                if state.global_step % self.wrapper.config.checkpoint_interval == 0:
                    self._save_checkpoint(state, kwargs.get("logs", {}))

                # Log training step
                logs = kwargs.get("logs", {})
                self.wrapper.logger.log_training_step(
                    state.global_step,
                    logs.get("train_loss", 0.0),
                    logs.get("learning_rate", 0.0),
                    metrics.get("grad_norm"),
                    **{k: v for k, v in metrics.items() if k != "grad_norm"},
                )

            def _save_checkpoint(self, state: Any, logs: Dict[str, Any]) -> None:
                metadata = CheckpointMetadata(
                    step=state.global_step,
                    epoch=state.epoch,
                    learning_rate=logs.get("learning_rate", 0.0),
                    train_loss=logs.get("train_loss", 0.0),
                    eval_loss=logs.get("eval_loss"),
                    grad_norm=logs.get("grad_norm"),
                )

                # Save checkpoint locally with LoRA adapter focus
                checkpoint_path = self.wrapper.checkpoint_manager.save_lora_checkpoint(
                    self.wrapper.model,
                    self.wrapper.tokenizer,
                    self.wrapper.trainer.optimizer,
                    self.wrapper.trainer.lr_scheduler,
                    metadata,
                    self.wrapper.metrics_collector.get_adapter_checkpoint_data(),
                )

                # Upload adapter weights and gradients to HuggingFace if configured
                if self.wrapper.hf_integration:
                    try:
                        self.wrapper.hf_integration.upload_lora_checkpoint(
                            checkpoint_path,
                            state.global_step,
                            metadata.to_dict(),
                            upload_adapter_only=True,
                        )
                    except Exception as e:
                        self.wrapper.logger.warning(f"Failed to upload LoRA checkpoint: {e}")

                self.wrapper.logger.log_checkpoint_saved(state.global_step, checkpoint_path)

        return TrainingLensCallback(self)

    def _generate_final_report(self, train_result: Any, training_time: float) -> Dict[str, Any]:
        """Generate final training report."""
        return {
            "final_loss": train_result.training_loss,
            "total_steps": train_result.global_step,
            "training_time": f"{training_time:.2f}s",
            "training_time_hours": f"{training_time/3600:.2f}h",
            "steps_per_second": f"{train_result.global_step/training_time:.2f}",
            "model_name": self.config.model_name,
            "training_method": "lora",
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "checkpoints_saved": len(self.checkpoint_manager.list_checkpoints()),
        }
