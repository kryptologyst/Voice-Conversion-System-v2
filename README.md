# Voice Conversion System

Research-focused voice conversion system built with PyTorch. This project demonstrates various voice conversion techniques including autoencoder-based models, CycleGAN, and simple feature mapping approaches.

## ⚠️ PRIVACY AND ETHICS DISCLAIMER

**This is a research demonstration only.** This voice conversion system is designed for educational and research purposes. It should NOT be used for:

- Biometric identification or authentication
- Voice cloning for deceptive purposes  
- Creating deepfakes or misleading content
- Any commercial applications without proper consent

Users are responsible for ensuring ethical use and obtaining proper consent for any voice data processing.

## Features

- **Multiple Model Architectures**: Autoencoder, CycleGAN, and simple mapping models
- **Comprehensive Evaluation**: MCD, F0 RMSE, spectral analysis metrics
- **Modern Tech Stack**: PyTorch 2.x, torchaudio, librosa, with Apple Silicon support
- **Interactive Demo**: Streamlit-based web interface
- **Reproducible Research**: Deterministic seeding, proper logging, checkpointing
- **Privacy-Focused**: No PII logging, anonymized filenames, clear disclaimers

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Voice-Conversion-System-v2.git
cd Voice-Conversion-System-v2

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pre-commit install
```

### Create Synthetic Dataset

```bash
python scripts/train.py --create_synthetic --num_speakers 5 --samples_per_speaker 100
```

### Train a Model

```bash
python scripts/train.py --config configs/default.yaml --data_dir data --output_dir outputs
```

### Run Interactive Demo

```bash
streamlit run demo/streamlit_demo.py
```

## Project Structure

```
voice-conversion-system/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   └── voice_conversion.py   # Voice conversion models
│   ├── data/                     # Data handling
│   │   └── dataset.py           # Dataset and data module
│   ├── train/                    # Training utilities
│   │   └── trainer.py           # Training loop and trainer
│   ├── metrics/                  # Evaluation metrics
│   │   └── voice_conversion_metrics.py
│   ├── utils/                    # Utility functions
│   │   ├── device.py            # Device management
│   │   └── audio_utils.py       # Audio processing utilities
├── configs/                      # Configuration files
│   └── default.yaml             # Default configuration
├── scripts/                      # Training and evaluation scripts
│   └── train.py                 # Main training script
├── demo/                         # Interactive demos
│   └── streamlit_demo.py        # Streamlit web interface
├── tests/                        # Unit tests
├── data/                         # Data directory
│   └── wav/                     # Audio files
├── assets/                       # Generated assets
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Configuration

The system uses YAML configuration files. Key parameters:

### Data Configuration
- `sample_rate`: Audio sample rate (default: 22050)
- `n_mels`: Number of mel filter banks (default: 80)
- `hop_length`: Hop length for analysis (default: 256)
- `n_fft`: FFT window size (default: 1024)

### Model Configuration
- `type`: Model type ("autoencoder", "cyclegan", "simple_mapping")
- `hidden_dim`: Hidden layer dimension (default: 256)
- `num_layers`: Number of LSTM layers (default: 3)
- `dropout`: Dropout rate (default: 0.1)

### Training Configuration
- `batch_size`: Training batch size (default: 16)
- `num_epochs`: Number of training epochs (default: 100)
- `learning_rate`: Learning rate (default: 0.0001)
- `weight_decay`: Weight decay (default: 1e-5)

## Models

### Autoencoder Model
- **Architecture**: Content encoder + Speaker encoder + Decoder
- **Use Case**: High-quality voice conversion with speaker control
- **Features**: Speaker embedding, content preservation

### CycleGAN Model
- **Architecture**: Generator + Discriminator with cycle consistency
- **Use Case**: Unpaired voice conversion
- **Features**: Adversarial training, cycle consistency loss

### Simple Mapping Model
- **Architecture**: Multi-layer perceptron
- **Use Case**: Baseline voice conversion
- **Features**: Fast training, simple architecture

## Evaluation Metrics

- **MCD (Mel Cepstral Distortion)**: Spectral quality measure
- **F0 RMSE**: Fundamental frequency accuracy
- **F0 Correlation**: F0 contour similarity
- **Spectral Centroid Error**: Spectral shape preservation
- **Zero Crossing Rate Error**: Temporal characteristics

## Data Format

### Metadata CSV Structure
```csv
id,path,speaker_id,split
sample_001,speaker_001/sample_001.wav,0,train
sample_002,speaker_001/sample_002.wav,0,train
...
```

### Audio Requirements
- **Format**: WAV, MP3, FLAC, M4A
- **Sample Rate**: 16kHz, 22.05kHz, or 44.1kHz
- **Channels**: Mono (automatically converted)
- **Duration**: 1-10 seconds (configurable)

## Usage Examples

### Basic Training
```python
from src.data.dataset import VoiceConversionDataModule
from src.train.trainer import create_trainer

# Create data module
data_module = VoiceConversionDataModule(
    data_dir="data",
    metadata_file="data/metadata.csv",
    batch_size=16
)

# Get data loaders
train_loader = data_module.get_dataloader("train")
val_loader = data_module.get_dataloader("val")

# Create trainer
trainer = create_trainer(
    model_config={"model_type": "autoencoder", "mel_dim": 80},
    train_loader=train_loader,
    val_loader=val_loader
)

# Train model
trainer.train(num_epochs=100)
```

### Model Inference
```python
import torch
from src.models.voice_conversion import create_model
from src.utils.audio_utils import load_audio, extract_mel_spectrogram

# Load model
model = create_model(model_type="autoencoder", mel_dim=80)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()

# Load and process audio
audio, sr = load_audio("input.wav")
mel_spec = extract_mel_spectrogram(audio, sr)

# Convert voice
with torch.no_grad():
    converted_mel = model(torch.FloatTensor(mel_spec).unsqueeze(0))
```

## Advanced Features

### Data Augmentation
- **Noise Addition**: Gaussian noise injection
- **Pitch Shifting**: Semitone-based pitch modification
- **Time Stretching**: Temporal rate modification
- **Reverb**: Room impulse response simulation

### Device Support
- **CUDA**: NVIDIA GPU acceleration
- **MPS**: Apple Silicon GPU acceleration
- **CPU**: Fallback computation

### Logging and Monitoring
- **TensorBoard**: Training visualization
- **Structured Logging**: JSON-formatted logs
- **Checkpointing**: Automatic model saving
- **Early Stopping**: Prevent overfitting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_models.py
```

## Known Limitations

- **Vocoder**: Currently uses simple mel-to-audio reconstruction
- **Real-time**: Not optimized for real-time processing
- **Speaker Embeddings**: Uses simplified speaker representation
- **F0 Prediction**: Basic F0 extraction without advanced modeling

## Future Improvements

- [ ] HiFi-GAN vocoder integration
- [ ] Real-time streaming support
- [ ] Advanced speaker embedding learning
- [ ] Multi-speaker voice conversion
- [ ] Cross-lingual voice conversion
- [ ] Emotion-preserving conversion

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{voice_conversion_system,
  title={Voice Conversion System: A Research Framework},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Voice-Conversion-System-v2}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Librosa team for audio processing utilities
- Streamlit team for the interactive web framework
- The open-source audio processing community

## Support

For questions, issues, or contributions, please:
1. Check the existing issues
2. Create a new issue with detailed description
3. Follow the contribution guidelines

---

**Remember: This is a research tool. Use responsibly and ethically.**
# Voice-Conversion-System-v2
