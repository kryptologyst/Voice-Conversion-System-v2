#!/usr/bin/env python3
"""Test script to verify voice conversion system setup."""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.models.voice_conversion import create_model
        from src.data.dataset import VoiceConversionDataModule, create_synthetic_dataset
        from src.train.trainer import create_trainer
        from src.metrics.voice_conversion_metrics import VoiceConversionEvaluator
        from src.utils.device import get_device, set_seed
        from src.utils.audio_utils import load_audio, extract_mel_spectrogram
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_device():
    """Test device detection."""
    print("Testing device detection...")
    
    try:
        from src.utils.device import get_device
        device = get_device()
        print(f"✓ Device detected: {device}")
        return True
    except Exception as e:
        print(f"✗ Device error: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    
    try:
        from src.models.voice_conversion import create_model
        
        # Test different model types
        for model_type in ["autoencoder", "simple_mapping", "cyclegan"]:
            model = create_model(model_type=model_type, mel_dim=80, hidden_dim=256)
            print(f"✓ {model_type} model created successfully")
            
            # Test forward pass
            dummy_input = torch.randn(1, 100, 80)  # (batch, time, mel)
            with torch.no_grad():
                output = model(dummy_input)
                print(f"✓ {model_type} forward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_audio_processing():
    """Test audio processing utilities."""
    print("Testing audio processing...")
    
    try:
        from src.utils.audio_utils import extract_mel_spectrogram
        
        # Create dummy audio
        dummy_audio = np.random.randn(22050)  # 1 second at 22kHz
        
        # Extract mel spectrogram
        mel_spec = extract_mel_spectrogram(dummy_audio, sample_rate=22050)
        print(f"✓ Mel spectrogram extraction successful: {mel_spec.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Audio processing error: {e}")
        return False

def test_metrics():
    """Test evaluation metrics."""
    print("Testing evaluation metrics...")
    
    try:
        from src.metrics.voice_conversion_metrics import VoiceConversionEvaluator
        
        evaluator = VoiceConversionEvaluator()
        
        # Create dummy data
        target_mel = torch.randn(1, 100, 80)
        predicted_mel = torch.randn(1, 100, 80)
        
        # Test evaluation
        metrics = evaluator.evaluate_batch(target_mel, predicted_mel)
        print(f"✓ Metrics evaluation successful: {list(metrics.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics error: {e}")
        return False

def test_synthetic_dataset():
    """Test synthetic dataset creation."""
    print("Testing synthetic dataset creation...")
    
    try:
        from src.data.dataset import create_synthetic_dataset
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create synthetic dataset
        create_synthetic_dataset(
            output_dir=temp_dir,
            num_speakers=2,
            samples_per_speaker=5,
            sample_rate=22050,
            duration=1.0
        )
        
        # Check if files were created
        metadata_file = os.path.join(temp_dir, "metadata.csv")
        wav_dir = os.path.join(temp_dir, "wav")
        
        if os.path.exists(metadata_file) and os.path.exists(wav_dir):
            print("✓ Synthetic dataset creation successful")
            
            # Clean up
            shutil.rmtree(temp_dir)
            return True
        else:
            print("✗ Synthetic dataset files not found")
            return False
            
    except Exception as e:
        print(f"✗ Synthetic dataset error: {e}")
        return False

def main():
    """Run all tests."""
    print("Voice Conversion System - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_device,
        test_model_creation,
        test_audio_processing,
        test_metrics,
        test_synthetic_dataset
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
