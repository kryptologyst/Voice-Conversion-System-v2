#!/usr/bin/env python3
"""Quick demo script for voice conversion system."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_demo():
    """Run a quick demonstration of the voice conversion system."""
    print("🎤 Voice Conversion System - Quick Demo")
    print("=" * 50)
    
    try:
        # Import required modules
        from src.data.dataset import create_synthetic_dataset
        from src.models.voice_conversion import create_model
        from src.utils.device import get_device, set_seed
        from src.utils.audio_utils import extract_mel_spectrogram
        import torch
        import numpy as np
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Get device
        device = get_device()
        print(f"Using device: {device}")
        
        # Create temporary directory for demo
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
        
        # Create synthetic dataset
        print("\n1. Creating synthetic dataset...")
        create_synthetic_dataset(
            output_dir=temp_dir,
            num_speakers=3,
            samples_per_speaker=10,
            sample_rate=22050,
            duration=2.0
        )
        print("✓ Synthetic dataset created!")
        
        # Create model
        print("\n2. Creating voice conversion model...")
        model = create_model(
            model_type="autoencoder",
            mel_dim=80,
            hidden_dim=256,
            num_layers=3,
            dropout=0.1
        )
        model.to(device)
        model.eval()
        print("✓ Model created successfully!")
        
        # Generate dummy audio for demonstration
        print("\n3. Processing audio...")
        dummy_audio = np.random.randn(22050 * 2)  # 2 seconds of audio
        mel_spec = extract_mel_spectrogram(dummy_audio, sample_rate=22050)
        print(f"✓ Mel spectrogram extracted: {mel_spec.shape}")
        
        # Convert voice
        print("\n4. Converting voice...")
        source_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).to(device)
        target_speaker_id = torch.LongTensor([1]).to(device)
        
        with torch.no_grad():
            outputs = model(source_tensor, target_speaker_id=target_speaker_id)
            if isinstance(outputs, dict):
                converted_mel = outputs["converted_mel"]
            else:
                converted_mel = outputs
        
        print(f"✓ Voice conversion completed! Output shape: {converted_mel.shape}")
        
        # Test metrics
        print("\n5. Computing evaluation metrics...")
        from src.metrics.voice_conversion_metrics import VoiceConversionEvaluator
        
        evaluator = VoiceConversionEvaluator()
        metrics = evaluator.evaluate_sample(
            target_mel=torch.FloatTensor(mel_spec),
            predicted_mel=converted_mel.cpu().squeeze(0)
        )
        
        print("✓ Evaluation metrics computed:")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\n✓ Cleaned up temporary directory")
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python scripts/train.py --create_synthetic' to create a dataset")
        print("2. Run 'python scripts/train.py' to train a model")
        print("3. Run 'streamlit run demo/streamlit_demo.py' to launch the web demo")
        print("4. Check README.md for detailed usage instructions")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check Python version (3.10+ required)")
        print("3. Run 'python scripts/test_system.py' to diagnose issues")
        return False

if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
