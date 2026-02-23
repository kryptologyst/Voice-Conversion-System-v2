"""Evaluation metrics for voice conversion."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import librosa
from scipy.stats import pearsonr
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def mel_cepstral_distortion(
    target_mel: torch.Tensor,
    predicted_mel: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """Calculate Mel Cepstral Distortion (MCD).
    
    Args:
        target_mel: Target mel spectrogram (B, T, F) or (T, F).
        predicted_mel: Predicted mel spectrogram (B, T, F) or (T, F).
        reduction: Reduction method ('mean', 'sum', 'none').
        
    Returns:
        MCD value(s).
    """
    # Ensure same shape
    if target_mel.shape != predicted_mel.shape:
        raise ValueError(f"Shape mismatch: {target_mel.shape} vs {predicted_mel.shape}")
    
    # Convert to numpy for librosa processing
    if target_mel.is_cuda:
        target_mel = target_mel.cpu()
        predicted_mel = predicted_mel.cpu()
    
    target_np = target_mel.numpy()
    predicted_np = predicted_mel.numpy()
    
    # Handle batch dimension
    if target_np.ndim == 3:
        batch_size = target_np.shape[0]
        mcd_values = []
        
        for i in range(batch_size):
            mcd = _compute_mcd_single(target_np[i], predicted_np[i])
            mcd_values.append(mcd)
        
        mcd_tensor = torch.tensor(mcd_values)
    else:
        mcd = _compute_mcd_single(target_np, predicted_np)
        mcd_tensor = torch.tensor(mcd)
    
    if reduction == "mean":
        return mcd_tensor.mean()
    elif reduction == "sum":
        return mcd_tensor.sum()
    else:
        return mcd_tensor


def _compute_mcd_single(target: np.ndarray, predicted: np.ndarray) -> float:
    """Compute MCD for a single sample."""
    # Convert mel to MFCC
    target_mfcc = librosa.feature.mfcc(S=target, n_mfcc=13)
    predicted_mfcc = librosa.feature.mfcc(S=predicted, n_mfcc=13)
    
    # Calculate MCD
    diff = target_mfcc - predicted_mfcc
    mcd = np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))
    
    return mcd


def f0_rmse(
    target_f0: torch.Tensor,
    predicted_f0: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """Calculate F0 Root Mean Square Error.
    
    Args:
        target_f0: Target F0 contour (B, T) or (T,).
        predicted_f0: Predicted F0 contour (B, T) or (T,).
        reduction: Reduction method ('mean', 'sum', 'none').
        
    Returns:
        F0 RMSE value(s).
    """
    # Ensure same shape
    if target_f0.shape != predicted_f0.shape:
        raise ValueError(f"Shape mismatch: {target_f0.shape} vs {predicted_f0.shape}")
    
    # Calculate RMSE
    mse = F.mse_loss(predicted_f0, target_f0, reduction="none")
    rmse = torch.sqrt(mse)
    
    if reduction == "mean":
        return rmse.mean()
    elif reduction == "sum":
        return rmse.sum()
    else:
        return rmse


def f0_correlation(
    target_f0: torch.Tensor,
    predicted_f0: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """Calculate F0 correlation coefficient.
    
    Args:
        target_f0: Target F0 contour (B, T) or (T,).
        predicted_f0: Predicted F0 contour (B, T) or (T,).
        reduction: Reduction method ('mean', 'sum', 'none').
        
    Returns:
        F0 correlation value(s).
    """
    # Convert to numpy for correlation calculation
    if target_f0.is_cuda:
        target_f0 = target_f0.cpu()
        predicted_f0 = predicted_f0.cpu()
    
    target_np = target_f0.numpy()
    predicted_np = predicted_f0.numpy()
    
    # Handle batch dimension
    if target_np.ndim == 2:
        batch_size = target_np.shape[0]
        corr_values = []
        
        for i in range(batch_size):
            # Remove zero values (unvoiced frames)
            mask = (target_np[i] > 0) & (predicted_np[i] > 0)
            if np.sum(mask) > 1:
                corr, _ = pearsonr(target_np[i][mask], predicted_np[i][mask])
                corr_values.append(corr if not np.isnan(corr) else 0.0)
            else:
                corr_values.append(0.0)
        
        corr_tensor = torch.tensor(corr_values)
    else:
        # Remove zero values (unvoiced frames)
        mask = (target_np > 0) & (predicted_np > 0)
        if np.sum(mask) > 1:
            corr, _ = pearsonr(target_np[mask], predicted_np[mask])
            corr_tensor = torch.tensor(corr if not np.isnan(corr) else 0.0)
        else:
            corr_tensor = torch.tensor(0.0)
    
    if reduction == "mean":
        return corr_tensor.mean()
    elif reduction == "sum":
        return corr_tensor.sum()
    else:
        return corr_tensor


def spectral_centroid_error(
    target_audio: torch.Tensor,
    predicted_audio: torch.Tensor,
    sample_rate: int = 22050,
    hop_length: int = 256,
    reduction: str = "mean"
) -> torch.Tensor:
    """Calculate spectral centroid error.
    
    Args:
        target_audio: Target audio (B, T) or (T,).
        predicted_audio: Predicted audio (B, T) or (T,).
        sample_rate: Sample rate of audio.
        hop_length: Hop length for analysis.
        reduction: Reduction method ('mean', 'sum', 'none').
        
    Returns:
        Spectral centroid error value(s).
    """
    # Convert to numpy
    if target_audio.is_cuda:
        target_audio = target_audio.cpu()
        predicted_audio = predicted_audio.cpu()
    
    target_np = target_audio.numpy()
    predicted_np = predicted_audio.numpy()
    
    # Handle batch dimension
    if target_np.ndim == 2:
        batch_size = target_np.shape[0]
        error_values = []
        
        for i in range(batch_size):
            target_sc = librosa.feature.spectral_centroid(
                y=target_np[i], sr=sample_rate, hop_length=hop_length
            )[0]
            predicted_sc = librosa.feature.spectral_centroid(
                y=predicted_np[i], sr=sample_rate, hop_length=hop_length
            )[0]
            
            error = np.mean(np.abs(target_sc - predicted_sc))
            error_values.append(error)
        
        error_tensor = torch.tensor(error_values)
    else:
        target_sc = librosa.feature.spectral_centroid(
            y=target_np, sr=sample_rate, hop_length=hop_length
        )[0]
        predicted_sc = librosa.feature.spectral_centroid(
            y=predicted_np, sr=sample_rate, hop_length=hop_length
        )[0]
        
        error = np.mean(np.abs(target_sc - predicted_sc))
        error_tensor = torch.tensor(error)
    
    if reduction == "mean":
        return error_tensor.mean()
    elif reduction == "sum":
        return error_tensor.sum()
    else:
        return error_tensor


def zero_crossing_rate_error(
    target_audio: torch.Tensor,
    predicted_audio: torch.Tensor,
    hop_length: int = 256,
    reduction: str = "mean"
) -> torch.Tensor:
    """Calculate zero crossing rate error.
    
    Args:
        target_audio: Target audio (B, T) or (T,).
        predicted_audio: Predicted audio (B, T) or (T,).
        hop_length: Hop length for analysis.
        reduction: Reduction method ('mean', 'sum', 'none').
        
    Returns:
        Zero crossing rate error value(s).
    """
    # Convert to numpy
    if target_audio.is_cuda:
        target_audio = target_audio.cpu()
        predicted_audio = predicted_audio.cpu()
    
    target_np = target_audio.numpy()
    predicted_np = predicted_audio.numpy()
    
    # Handle batch dimension
    if target_np.ndim == 2:
        batch_size = target_np.shape[0]
        error_values = []
        
        for i in range(batch_size):
            target_zcr = librosa.feature.zero_crossing_rate(
                target_np[i], hop_length=hop_length
            )[0]
            predicted_zcr = librosa.feature.zero_crossing_rate(
                predicted_np[i], hop_length=hop_length
            )[0]
            
            error = np.mean(np.abs(target_zcr - predicted_zcr))
            error_values.append(error)
        
        error_tensor = torch.tensor(error_values)
    else:
        target_zcr = librosa.feature.zero_crossing_rate(
            target_np, hop_length=hop_length
        )[0]
        predicted_zcr = librosa.feature.zero_crossing_rate(
            predicted_np, hop_length=hop_length
        )[0]
        
        error = np.mean(np.abs(target_zcr - predicted_zcr))
        error_tensor = torch.tensor(error)
    
    if reduction == "mean":
        return error_tensor.mean()
    elif reduction == "sum":
        return error_tensor.sum()
    else:
        return error_tensor


def compute_all_metrics(
    target_mel: torch.Tensor,
    predicted_mel: torch.Tensor,
    target_f0: Optional[torch.Tensor] = None,
    predicted_f0: Optional[torch.Tensor] = None,
    target_audio: Optional[torch.Tensor] = None,
    predicted_audio: Optional[torch.Tensor] = None,
    sample_rate: int = 22050,
    hop_length: int = 256
) -> Dict[str, float]:
    """Compute all voice conversion metrics.
    
    Args:
        target_mel: Target mel spectrogram.
        predicted_mel: Predicted mel spectrogram.
        target_f0: Target F0 contour (optional).
        predicted_f0: Predicted F0 contour (optional).
        target_audio: Target audio waveform (optional).
        predicted_audio: Predicted audio waveform (optional).
        sample_rate: Sample rate of audio.
        hop_length: Hop length for analysis.
        
    Returns:
        Dictionary containing all computed metrics.
    """
    metrics = {}
    
    # Mel Cepstral Distortion
    metrics["mcd"] = mel_cepstral_distortion(target_mel, predicted_mel).item()
    
    # F0 metrics
    if target_f0 is not None and predicted_f0 is not None:
        metrics["f0_rmse"] = f0_rmse(target_f0, predicted_f0).item()
        metrics["f0_correlation"] = f0_correlation(target_f0, predicted_f0).item()
    
    # Audio-based metrics
    if target_audio is not None and predicted_audio is not None:
        metrics["spectral_centroid_error"] = spectral_centroid_error(
            target_audio, predicted_audio, sample_rate, hop_length
        ).item()
        metrics["zcr_error"] = zero_crossing_rate_error(
            target_audio, predicted_audio, hop_length
        ).item()
    
    return metrics


class VoiceConversionEvaluator:
    """Evaluator for voice conversion models."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0,
        f_max: Optional[float] = None
    ):
        """Initialize evaluator.
        
        Args:
            sample_rate: Sample rate of audio.
            hop_length: Hop length for analysis.
            n_mels: Number of mel filter banks.
            f_min: Minimum frequency.
            f_max: Maximum frequency.
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        
    def evaluate_batch(
        self,
        target_mel: torch.Tensor,
        predicted_mel: torch.Tensor,
        target_f0: Optional[torch.Tensor] = None,
        predicted_f0: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None,
        predicted_audio: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate a batch of predictions.
        
        Args:
            target_mel: Target mel spectrograms.
            predicted_mel: Predicted mel spectrograms.
            target_f0: Target F0 contours (optional).
            predicted_f0: Predicted F0 contours (optional).
            target_audio: Target audio waveforms (optional).
            predicted_audio: Predicted audio waveforms (optional).
            
        Returns:
            Dictionary containing average metrics.
        """
        return compute_all_metrics(
            target_mel=target_mel,
            predicted_mel=predicted_mel,
            target_f0=target_f0,
            predicted_f0=predicted_f0,
            target_audio=target_audio,
            predicted_audio=predicted_audio,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length
        )
    
    def evaluate_sample(
        self,
        target_mel: torch.Tensor,
        predicted_mel: torch.Tensor,
        target_f0: Optional[torch.Tensor] = None,
        predicted_f0: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None,
        predicted_audio: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate a single sample.
        
        Args:
            target_mel: Target mel spectrogram.
            predicted_mel: Predicted mel spectrogram.
            target_f0: Target F0 contour (optional).
            predicted_f0: Predicted F0 contour (optional).
            target_audio: Target audio waveform (optional).
            predicted_audio: Predicted audio waveform (optional).
            
        Returns:
            Dictionary containing metrics.
        """
        # Remove batch dimension if present
        if target_mel.dim() == 3:
            target_mel = target_mel.squeeze(0)
            predicted_mel = predicted_mel.squeeze(0)
        
        if target_f0 is not None and target_f0.dim() == 2:
            target_f0 = target_f0.squeeze(0)
        if predicted_f0 is not None and predicted_f0.dim() == 2:
            predicted_f0 = predicted_f0.squeeze(0)
        
        if target_audio is not None and target_audio.dim() == 2:
            target_audio = target_audio.squeeze(0)
        if predicted_audio is not None and predicted_audio.dim() == 2:
            predicted_audio = predicted_audio.squeeze(0)
        
        return compute_all_metrics(
            target_mel=target_mel,
            predicted_mel=predicted_mel,
            target_f0=target_f0,
            predicted_f0=predicted_f0,
            target_audio=target_audio,
            predicted_audio=predicted_audio,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length
        )
