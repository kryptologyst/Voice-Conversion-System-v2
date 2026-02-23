"""Audio processing utilities for voice conversion."""

import os
from typing import Tuple, Optional, List
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import butter, filtfilt


def load_audio(
    filepath: str,
    sample_rate: int = 22050,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """Load audio file with proper error handling.
    
    Args:
        filepath: Path to audio file.
        sample_rate: Target sample rate.
        mono: Convert to mono if True.
        normalize: Normalize audio if True.
        
    Returns:
        Tuple of (audio_array, sample_rate).
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If audio format is not supported.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    
    try:
        audio, sr = librosa.load(
            filepath,
            sr=sample_rate,
            mono=mono,
            res_type="kaiser_fast"
        )
        
        if normalize:
            audio = librosa.util.normalize(audio)
            
        return audio, sr
        
    except Exception as e:
        raise ValueError(f"Error loading audio file {filepath}: {str(e)}")


def save_audio(
    audio: np.ndarray,
    filepath: str,
    sample_rate: int = 22050,
    format: str = "WAV"
) -> None:
    """Save audio array to file.
    
    Args:
        audio: Audio array to save.
        filepath: Output file path.
        sample_rate: Sample rate of audio.
        format: Audio format (WAV, FLAC, etc.).
    """
    sf.write(filepath, audio, sample_rate, format=format)


def preemphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply preemphasis filter to audio signal.
    
    Args:
        signal: Input audio signal.
        coeff: Preemphasis coefficient.
        
    Returns:
        Preemphasized signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def deemphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply deemphasis filter to audio signal.
    
    Args:
        signal: Input audio signal.
        coeff: Deemphasis coefficient.
        
    Returns:
        Deemphasized signal.
    """
    deemphasized = np.zeros_like(signal)
    deemphasized[0] = signal[0]
    
    for i in range(1, len(signal)):
        deemphasized[i] = signal[i] + coeff * deemphasized[i - 1]
    
    return deemphasized


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: float = 20,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """Trim silence from audio signal.
    
    Args:
        audio: Input audio signal.
        sample_rate: Sample rate of audio.
        top_db: Silence threshold in dB.
        frame_length: Frame length for analysis.
        hop_length: Hop length for analysis.
        
    Returns:
        Trimmed audio signal.
    """
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return trimmed


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    f_min: float = 0,
    f_max: Optional[float] = None,
    preemphasis_coeff: float = 0.97
) -> np.ndarray:
    """Extract mel spectrogram from audio.
    
    Args:
        audio: Input audio signal.
        sample_rate: Sample rate of audio.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        n_mels: Number of mel filter banks.
        f_min: Minimum frequency.
        f_max: Maximum frequency.
        preemphasis_coeff: Preemphasis coefficient.
        
    Returns:
        Mel spectrogram (n_mels, time_frames).
    """
    if f_max is None:
        f_max = sample_rate // 2
    
    # Apply preemphasis
    if preemphasis_coeff > 0:
        audio = preemphasis(audio, preemphasis_coeff)
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max
    )
    
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def extract_mfcc(
    audio: np.ndarray,
    sample_rate: int = 22050,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    f_min: float = 0,
    f_max: Optional[float] = None,
    preemphasis_coeff: float = 0.97
) -> np.ndarray:
    """Extract MFCC features from audio.
    
    Args:
        audio: Input audio signal.
        sample_rate: Sample rate of audio.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        n_mels: Number of mel filter banks.
        f_min: Minimum frequency.
        f_max: Maximum frequency.
        preemphasis_coeff: Preemphasis coefficient.
        
    Returns:
        MFCC features (n_mfcc, time_frames).
    """
    if f_max is None:
        f_max = sample_rate // 2
    
    # Apply preemphasis
    if preemphasis_coeff > 0:
        audio = preemphasis(audio, preemphasis_coeff)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max
    )
    
    return mfcc


def extract_f0(
    audio: np.ndarray,
    sample_rate: int = 22050,
    hop_length: int = 256,
    f_min: float = 50,
    f_max: float = 400
) -> np.ndarray:
    """Extract fundamental frequency (F0) from audio.
    
    Args:
        audio: Input audio signal.
        sample_rate: Sample rate of audio.
        hop_length: Hop length between frames.
        f_min: Minimum frequency for F0 detection.
        f_max: Maximum frequency for F0 detection.
        
    Returns:
        F0 contour.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=f_min,
        fmax=f_max,
        sr=sample_rate,
        hop_length=hop_length
    )
    
    # Replace NaN values with 0
    f0 = np.nan_to_num(f0, nan=0.0)
    
    return f0


def apply_pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    n_steps: float
) -> np.ndarray:
    """Apply pitch shifting to audio.
    
    Args:
        audio: Input audio signal.
        sample_rate: Sample rate of audio.
        n_steps: Number of semitones to shift (positive = higher pitch).
        
    Returns:
        Pitch-shifted audio.
    """
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)


def apply_time_stretch(
    audio: np.ndarray,
    rate: float
) -> np.ndarray:
    """Apply time stretching to audio.
    
    Args:
        audio: Input audio signal.
        rate: Stretch factor (>1 = slower, <1 = faster).
        
    Returns:
        Time-stretched audio.
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def add_noise(
    audio: np.ndarray,
    noise_factor: float = 0.01,
    noise_type: str = "gaussian"
) -> np.ndarray:
    """Add noise to audio signal.
    
    Args:
        audio: Input audio signal.
        noise_factor: Noise level factor.
        noise_type: Type of noise ('gaussian', 'uniform').
        
    Returns:
        Noisy audio signal.
    """
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_factor, audio.shape)
    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_factor, noise_factor, audio.shape)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return audio + noise


def pad_or_truncate(
    audio: np.ndarray,
    target_length: int,
    pad_mode: str = "constant"
) -> np.ndarray:
    """Pad or truncate audio to target length.
    
    Args:
        audio: Input audio signal.
        target_length: Target length in samples.
        pad_mode: Padding mode ('constant', 'edge', 'reflect').
        
    Returns:
        Padded or truncated audio.
    """
    current_length = len(audio)
    
    if current_length == target_length:
        return audio
    elif current_length < target_length:
        # Pad
        pad_width = target_length - current_length
        if pad_mode == "constant":
            return np.pad(audio, (0, pad_width), mode="constant")
        else:
            return np.pad(audio, (0, pad_width), mode=pad_mode)
    else:
        # Truncate
        return audio[:target_length]


def compute_spectral_centroid(
    audio: np.ndarray,
    sample_rate: int = 22050,
    hop_length: int = 256
) -> np.ndarray:
    """Compute spectral centroid of audio.
    
    Args:
        audio: Input audio signal.
        sample_rate: Sample rate of audio.
        hop_length: Hop length between frames.
        
    Returns:
        Spectral centroid contour.
    """
    return librosa.feature.spectral_centroid(
        y=audio, sr=sample_rate, hop_length=hop_length
    )[0]


def compute_zero_crossing_rate(
    audio: np.ndarray,
    hop_length: int = 256
) -> np.ndarray:
    """Compute zero crossing rate of audio.
    
    Args:
        audio: Input audio signal.
        hop_length: Hop length between frames.
        
    Returns:
        Zero crossing rate contour.
    """
    return librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
