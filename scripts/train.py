#!/usr/bin/env python3
"""Main training script for voice conversion system."""

import argparse
import os
import yaml
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data.dataset import VoiceConversionDataModule, create_synthetic_dataset
from src.train.trainer import create_trainer
from src.utils.device import set_seed, get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train voice conversion model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing audio data"
    )
    
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="data/metadata.csv",
        help="Path to metadata CSV file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for models and logs"
    )
    
    parser.add_argument(
        "--create_synthetic",
        action="store_true",
        help="Create synthetic dataset for demonstration"
    )
    
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=5,
        help="Number of synthetic speakers to create"
    )
    
    parser.add_argument(
        "--samples_per_speaker",
        type=int,
        default=100,
        help="Number of samples per synthetic speaker"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic dataset if requested
    if args.create_synthetic:
        print("Creating synthetic dataset...")
        create_synthetic_dataset(
            output_dir=args.data_dir,
            num_speakers=args.num_speakers,
            samples_per_speaker=args.samples_per_speaker,
            sample_rate=config["data"]["sample_rate"],
            duration=3.0
        )
        print("Synthetic dataset created!")
    
    # Check if data exists
    if not os.path.exists(args.metadata_file):
        print(f"Metadata file not found: {args.metadata_file}")
        print("Please create a dataset or use --create_synthetic flag")
        return
    
    # Create data module
    print("Loading data...")
    data_module = VoiceConversionDataModule(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        batch_size=config["training"]["batch_size"],
        num_workers=4,
        val_split=config["evaluation"]["val_split"],
        test_split=config["evaluation"]["test_split"],
        sample_rate=config["data"]["sample_rate"],
        max_length=config["audio"]["max_length"],
        min_length=config["audio"]["min_length"],
        n_mels=config["data"]["n_mels"],
        n_fft=config["data"]["n_fft"],
        hop_length=config["data"]["hop_length"],
        f_min=config["data"]["f_min"],
        f_max=config["data"]["f_max"],
        preemphasis_coeff=config["data"]["preemphasis"],
        augment=config["augmentation"]["enabled"],
        noise_factor=config["augmentation"]["noise_factor"],
        pitch_shift_range=(-config["augmentation"]["pitch_shift"], config["augmentation"]["pitch_shift"]),
        time_stretch_range=(1-config["augmentation"]["time_stretch"], 1+config["augmentation"]["time_stretch"]),
        trim_silence=config["audio"]["trim_silence"],
        normalize=config["audio"]["normalize"]
    )
    
    # Get data loaders
    train_loader = data_module.get_dataloader("train")
    val_loader = data_module.get_dataloader("val")
    test_loader = data_module.get_dataloader("test")
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model configuration
    model_config = {
        "model_type": config["model"]["type"],
        "mel_dim": config["data"]["n_mels"],
        "hidden_dim": config["model"]["hidden_dim"],
        "num_layers": config["model"]["num_layers"],
        "dropout": config["model"]["dropout"]
    }
    
    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(
        model_config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        device=device,
        log_dir=str(output_dir / "logs"),
        checkpoint_dir=str(output_dir / "checkpoints")
    )
    
    # Train model
    print("Starting training...")
    trainer.train(
        num_epochs=config["training"]["num_epochs"],
        save_interval=config["training"]["save_interval"],
        early_stopping_patience=20
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate_model(test_loader)
    
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save final configuration
    config_path = output_dir / "final_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nTraining completed! Results saved to: {output_dir}")
    print(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
