"""Streamlit demo for voice conversion system."""

import streamlit as st
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os

from src.models.voice_conversion import create_model
from src.utils.audio_utils import load_audio, extract_mel_spectrogram, save_audio
from src.utils.device import get_device
from src.metrics.voice_conversion_metrics import VoiceConversionEvaluator


# Page configuration
st.set_page_config(
    page_title="Voice Conversion System",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Privacy disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ PRIVACY AND ETHICS DISCLAIMER</h4>
    <p><strong>This is a research demonstration only.</strong> This voice conversion system is designed for educational and research purposes. 
    It should NOT be used for:</p>
    <ul>
        <li>Biometric identification or authentication</li>
        <li>Voice cloning for deceptive purposes</li>
        <li>Creating deepfakes or misleading content</li>
        <li>Any commercial applications without proper consent</li>
    </ul>
    <p>Users are responsible for ensuring ethical use and obtaining proper consent for any voice data processing.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎤 Voice Conversion System</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["autoencoder", "simple_mapping", "cyclegan"],
    help="Select the voice conversion model architecture"
)

# Model parameters
st.sidebar.subheader("Model Parameters")
mel_dim = st.sidebar.slider("Mel Dimensions", 40, 128, 80)
hidden_dim = st.sidebar.slider("Hidden Dimensions", 128, 512, 256)
num_layers = st.sidebar.slider("Number of Layers", 2, 6, 3)
dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.1)

# Audio parameters
st.sidebar.subheader("Audio Parameters")
sample_rate = st.sidebar.selectbox("Sample Rate", [16000, 22050, 44100], index=1)
hop_length = st.sidebar.slider("Hop Length", 128, 512, 256)
n_fft = st.sidebar.slider("FFT Size", 512, 2048, 1024)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = get_device()
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = VoiceConversionEvaluator(
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_mels=mel_dim
    )

@st.cache_resource
def load_model(model_type: str, mel_dim: int, hidden_dim: int, num_layers: int, dropout: float):
    """Load and cache the voice conversion model."""
    try:
        model_config = {
            "model_type": model_type,
            "mel_dim": mel_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout
        }
        
        model = create_model(**model_config)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_audio(audio_file, sample_rate: int):
    """Process uploaded audio file."""
    try:
        # Load audio
        audio, sr = load_audio(audio_file, sample_rate=sample_rate)
        
        # Extract mel spectrogram
        mel_spec = extract_mel_spectrogram(
            audio,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=mel_dim
        )
        
        return audio, mel_spec, sr
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None, None

def convert_voice(model, source_mel: np.ndarray, target_speaker_id: int = 0):
    """Convert voice using the model."""
    try:
        # Convert to tensor
        source_tensor = torch.FloatTensor(source_mel).unsqueeze(0)  # Add batch dimension
        target_speaker_tensor = torch.LongTensor([target_speaker_id])
        
        # Move to device
        source_tensor = source_tensor.to(st.session_state.device)
        target_speaker_tensor = target_speaker_tensor.to(st.session_state.device)
        
        # Forward pass
        with torch.no_grad():
            if hasattr(model, 'forward') and 'target_speaker_id' in model.forward.__code__.co_varnames:
                outputs = model(source_tensor, target_speaker_id=target_speaker_tensor)
            else:
                outputs = model(source_tensor)
            
            if isinstance(outputs, dict):
                converted_mel = outputs["converted_mel"]
            else:
                converted_mel = outputs
        
        # Convert back to numpy
        converted_mel = converted_mel.cpu().numpy().squeeze(0)
        
        return converted_mel
    except Exception as e:
        st.error(f"Error in voice conversion: {str(e)}")
        return None

def plot_spectrograms(source_mel: np.ndarray, converted_mel: np.ndarray):
    """Plot mel spectrograms."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Source spectrogram
    im1 = ax1.imshow(source_mel, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title('Source Mel Spectrogram')
    ax1.set_xlabel('Time Frames')
    ax1.set_ylabel('Mel Bins')
    plt.colorbar(im1, ax=ax1)
    
    # Converted spectrogram
    im2 = ax2.imshow(converted_mel, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title('Converted Mel Spectrogram')
    ax2.set_xlabel('Time Frames')
    ax2.set_ylabel('Mel Bins')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    return fig

def plot_interactive_spectrograms(source_mel: np.ndarray, converted_mel: np.ndarray):
    """Create interactive spectrogram plots."""
    fig = go.Figure()
    
    # Source spectrogram
    fig.add_trace(go.Heatmap(
        z=source_mel,
        colorscale='Viridis',
        name='Source',
        showscale=True
    ))
    
    fig.update_layout(
        title='Source vs Converted Mel Spectrograms',
        xaxis_title='Time Frames',
        yaxis_title='Mel Bins',
        height=600
    )
    
    return fig

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Input Audio")
    
    # Audio upload
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload a WAV, MP3, FLAC, or M4A audio file"
    )
    
    if uploaded_file is not None:
        # Process audio
        with st.spinner("Processing audio..."):
            audio, mel_spec, sr = process_audio(uploaded_file, sample_rate)
        
        if audio is not None:
            st.success("Audio processed successfully!")
            
            # Display audio info
            st.subheader("Audio Information")
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("Duration", f"{len(audio)/sr:.2f}s")
            with col_info2:
                st.metric("Sample Rate", f"{sr} Hz")
            with col_info3:
                st.metric("Channels", "Mono")
            
            # Play original audio
            st.subheader("Original Audio")
            st.audio(audio, sample_rate=sr)
            
            # Load model
            with st.spinner("Loading model..."):
                model = load_model(model_type, mel_dim, hidden_dim, num_layers, dropout)
            
            if model is not None:
                st.session_state.model = model
                
                # Voice conversion
                st.subheader("Voice Conversion")
                target_speaker_id = st.slider("Target Speaker ID", 0, 9, 0)
                
                if st.button("Convert Voice", type="primary"):
                    with st.spinner("Converting voice..."):
                        converted_mel = convert_voice(model, mel_spec, target_speaker_id)
                    
                    if converted_mel is not None:
                        st.success("Voice conversion completed!")
                        
                        # Display results
                        with col2:
                            st.header("🎯 Converted Audio")
                            
                            # Plot spectrograms
                            st.subheader("Mel Spectrograms")
                            fig = plot_spectrograms(mel_spec, converted_mel)
                            st.pyplot(fig)
                            
                            # Interactive plot
                            st.subheader("Interactive Spectrograms")
                            interactive_fig = plot_interactive_spectrograms(mel_spec, converted_mel)
                            st.plotly_chart(interactive_fig, use_container_width=True)
                            
                            # Metrics
                            st.subheader("Conversion Metrics")
                            
                            # Calculate metrics
                            source_tensor = torch.FloatTensor(mel_spec)
                            converted_tensor = torch.FloatTensor(converted_mel)
                            
                            metrics = st.session_state.evaluator.evaluate_sample(
                                target_mel=source_tensor,
                                predicted_mel=converted_tensor
                            )
                            
                            # Display metrics
                            col_metric1, col_metric2, col_metric3 = st.columns(3)
                            
                            with col_metric1:
                                st.metric("MCD", f"{metrics['mcd']:.4f}")
                            with col_metric2:
                                st.metric("F0 RMSE", f"{metrics.get('f0_rmse', 0.0):.4f}")
                            with col_metric3:
                                st.metric("F0 Correlation", f"{metrics.get('f0_correlation', 0.0):.4f}")
                            
                            # Download converted audio (simplified - in practice you'd use a vocoder)
                            st.subheader("Download Results")
                            
                            # Create a simple reconstruction for demonstration
                            # In practice, you'd use a proper vocoder like HiFi-GAN
                            reconstructed_audio = librosa.feature.inverse.mel_to_stft(
                                converted_mel,
                                sr=sr,
                                n_fft=n_fft,
                                hop_length=hop_length
                            )
                            reconstructed_audio = librosa.istft(reconstructed_audio, hop_length=hop_length)
                            
                            # Save to temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                                sf.write(tmp_file.name, reconstructed_audio, sr)
                                
                                with open(tmp_file.name, 'rb') as f:
                                    st.download_button(
                                        label="Download Converted Audio",
                                        data=f.read(),
                                        file_name=f"converted_{uploaded_file.name}",
                                        mime="audio/wav"
                                    )
                                
                                # Clean up
                                os.unlink(tmp_file.name)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Voice Conversion System - Research Demo</p>
    <p>⚠️ For educational and research purposes only. Please use responsibly.</p>
</div>
""", unsafe_allow_html=True)
