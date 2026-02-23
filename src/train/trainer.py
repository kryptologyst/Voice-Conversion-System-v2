"""Training module for voice conversion models."""

import os
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from ..models.voice_conversion import create_model
from ..metrics.voice_conversion_metrics import VoiceConversionEvaluator
from ..utils.device import get_device, save_checkpoint, load_checkpoint


class VoiceConversionTrainer:
    """Trainer for voice conversion models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        evaluator: Optional[VoiceConversionEvaluator] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Voice conversion model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler (optional).
            device: Device for computation.
            log_dir: Directory for logging.
            checkpoint_dir: Directory for checkpoints.
            evaluator: Evaluator for metrics (optional).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.evaluator = evaluator
        
        # Move model to device
        self.model.to(self.device)
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize logging
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            mel_spec = batch["mel_spec"].to(self.device)
            speaker_id = batch["speaker_id"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # For autoencoder models, we need target speaker for conversion
            if hasattr(self.model, 'forward') and 'target_speaker_id' in self.model.forward.__code__.co_varnames:
                outputs = self.model(mel_spec, target_speaker_id=speaker_id)
            else:
                outputs = self.model(mel_spec)
            
            # Calculate loss
            if isinstance(outputs, dict):
                converted_mel = outputs["converted_mel"]
            else:
                converted_mel = outputs
            
            # Simple MSE loss for now
            loss = nn.MSELoss()(converted_mel, mel_spec)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                self.writer.add_scalar(
                    "Train/Loss_Step", 
                    loss.item(), 
                    self.current_epoch * len(self.train_loader) + batch_idx
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model.
        
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                mel_spec = batch["mel_spec"].to(self.device)
                speaker_id = batch["speaker_id"].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'target_speaker_id' in self.model.forward.__code__.co_varnames:
                    outputs = self.model(mel_spec, target_speaker_id=speaker_id)
                else:
                    outputs = self.model(mel_spec)
                
                # Calculate loss
                if isinstance(outputs, dict):
                    converted_mel = outputs["converted_mel"]
                else:
                    converted_mel = outputs
                
                loss = nn.MSELoss()(converted_mel, mel_spec)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self, 
        num_epochs: int, 
        save_interval: int = 10,
        early_stopping_patience: int = 20
    ) -> None:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train.
            save_interval: Interval for saving checkpoints.
            early_stopping_patience: Patience for early stopping.
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Logging
            self.writer.add_scalar("Train/Loss_Epoch", train_loss, epoch)
            self.writer.add_scalar("Val/Loss_Epoch", val_loss, epoch)
            
            if self.scheduler is not None:
                self.writer.add_scalar("Train/Learning_Rate", self.scheduler.get_last_lr()[0], epoch)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    filepath=os.path.join(self.checkpoint_dir, "best_model.pth"),
                    metadata={"train_loss": train_loss, "val_loss": val_loss}
                )
                print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Save checkpoint at intervals
            if (epoch + 1) % save_interval == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    filepath=os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"),
                    metadata={"train_loss": train_loss, "val_loss": val_loss}
                )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=val_loss,
            filepath=os.path.join(self.checkpoint_dir, "final_model.pth"),
            metadata={"train_loss": train_loss, "val_loss": val_loss}
        )
        
        print("Training completed!")
        self.writer.close()
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        if self.evaluator is None:
            print("No evaluator provided, skipping evaluation")
            return {}
        
        self.model.eval()
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Move batch to device
                mel_spec = batch["mel_spec"].to(self.device)
                speaker_id = batch["speaker_id"].to(self.device)
                f0 = batch["f0"].to(self.device)
                audio = batch["audio"].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'target_speaker_id' in self.model.forward.__code__.co_varnames:
                    outputs = self.model(mel_spec, target_speaker_id=speaker_id)
                else:
                    outputs = self.model(mel_spec)
                
                # Get predictions
                if isinstance(outputs, dict):
                    converted_mel = outputs["converted_mel"]
                else:
                    converted_mel = outputs
                
                # Evaluate batch
                metrics = self.evaluator.evaluate_batch(
                    target_mel=mel_spec,
                    predicted_mel=converted_mel,
                    target_f0=f0,
                    predicted_f0=f0,  # Simplified - in practice you'd predict F0
                    target_audio=audio,
                    predicted_audio=audio  # Simplified - in practice you'd reconstruct audio
                )
                
                all_metrics.append(metrics)
        
        # Average metrics across all batches
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics


def create_trainer(
    model_config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 0.0001,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints"
) -> VoiceConversionTrainer:
    """Create a voice conversion trainer.
    
    Args:
        model_config: Model configuration dictionary.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        device: Device for computation.
        log_dir: Directory for logging.
        checkpoint_dir: Directory for checkpoints.
        
    Returns:
        VoiceConversionTrainer instance.
    """
    # Create model
    model = create_model(**model_config)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create evaluator
    evaluator = VoiceConversionEvaluator()
    
    # Create trainer
    trainer = VoiceConversionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        evaluator=evaluator
    )
    
    return trainer
