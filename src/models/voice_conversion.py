"""Advanced voice conversion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class SpeakerEncoder(nn.Module):
    """Speaker encoder for extracting speaker embeddings."""
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """Initialize speaker encoder.
        
        Args:
            input_dim: Input feature dimension (mel bins).
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel spectrogram (B, T, F).
            
        Returns:
            Speaker embedding (B, E).
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (B, hidden_dim)
        
        # Project to speaker embedding
        speaker_emb = self.projection(last_hidden)
        
        return speaker_emb


class ContentEncoder(nn.Module):
    """Content encoder for extracting linguistic content."""
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """Initialize content encoder.
        
        Args:
            input_dim: Input feature dimension (mel bins).
            hidden_dim: Hidden layer dimension.
            output_dim: Output content dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel spectrogram (B, T, F).
            
        Returns:
            Content features (B, T, C).
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Project to content features
        content = self.projection(lstm_out)
        
        return content


class Decoder(nn.Module):
    """Decoder for reconstructing mel spectrogram from content and speaker features."""
    
    def __init__(
        self,
        content_dim: int = 64,
        speaker_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 80,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """Initialize decoder.
        
        Args:
            content_dim: Content feature dimension.
            speaker_dim: Speaker embedding dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output mel dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        input_dim = content_dim + speaker_dim
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(
        self, 
        content: torch.Tensor, 
        speaker_emb: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            content: Content features (B, T, C).
            speaker_emb: Speaker embedding (B, S).
            
        Returns:
            Reconstructed mel spectrogram (B, T, F).
        """
        batch_size, seq_len, _ = content.shape
        
        # Expand speaker embedding to match sequence length
        speaker_expanded = speaker_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate content and speaker features
        combined = torch.cat([content, speaker_expanded], dim=-1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(combined)
        
        # Project to mel spectrogram
        mel_spec = self.projection(lstm_out)
        
        return mel_spec


class VoiceConversionAutoencoder(nn.Module):
    """Autoencoder-based voice conversion model."""
    
    def __init__(
        self,
        mel_dim: int = 80,
        speaker_dim: int = 64,
        content_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """Initialize voice conversion autoencoder.
        
        Args:
            mel_dim: Mel spectrogram dimension.
            speaker_dim: Speaker embedding dimension.
            content_dim: Content feature dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.speaker_encoder = SpeakerEncoder(
            input_dim=mel_dim,
            hidden_dim=hidden_dim,
            output_dim=speaker_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.content_encoder = ContentEncoder(
            input_dim=mel_dim,
            hidden_dim=hidden_dim,
            output_dim=content_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = Decoder(
            content_dim=content_dim,
            speaker_dim=speaker_dim,
            hidden_dim=hidden_dim,
            output_dim=mel_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(
        self, 
        source_mel: torch.Tensor, 
        target_speaker_id: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            source_mel: Source mel spectrogram (B, T, F).
            target_speaker_id: Target speaker ID (B,).
            
        Returns:
            Dictionary containing model outputs.
        """
        # Encode content from source
        content = self.content_encoder(source_mel)
        
        # Get speaker embedding
        if target_speaker_id is not None:
            # Use target speaker embedding (for conversion)
            speaker_emb = self._get_speaker_embedding(target_speaker_id)
        else:
            # Use source speaker embedding (for reconstruction)
            speaker_emb = self.speaker_encoder(source_mel)
        
        # Decode with target speaker
        converted_mel = self.decoder(content, speaker_emb)
        
        return {
            "converted_mel": converted_mel,
            "content": content,
            "speaker_emb": speaker_emb
        }
    
    def _get_speaker_embedding(self, speaker_id: torch.Tensor) -> torch.Tensor:
        """Get speaker embedding from speaker ID.
        
        Args:
            speaker_id: Speaker ID tensor (B,).
            
        Returns:
            Speaker embedding (B, S).
        """
        # This is a simplified version - in practice, you'd have a speaker embedding table
        batch_size = speaker_id.size(0)
        speaker_dim = self.speaker_encoder.projection[-1].out_features
        
        # Create random embeddings (replace with learned embeddings)
        speaker_emb = torch.randn(batch_size, speaker_dim, device=speaker_id.device)
        
        return speaker_emb


class CycleGANGenerator(nn.Module):
    """Generator for CycleGAN-based voice conversion."""
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """Initialize CycleGAN generator.
        
        Args:
            input_dim: Input mel spectrogram dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.decoder = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel spectrogram (B, T, F).
            
        Returns:
            Converted mel spectrogram (B, T, F).
        """
        # Encode
        encoded, _ = self.encoder(x)
        
        # Decode
        decoded, _ = self.decoder(encoded)
        
        # Project to output
        output = self.projection(decoded)
        
        return output


class CycleGANDiscriminator(nn.Module):
    """Discriminator for CycleGAN-based voice conversion."""
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """Initialize CycleGAN discriminator.
        
        Args:
            input_dim: Input mel spectrogram dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel spectrogram (B, T, F).
            
        Returns:
            Discriminator output (B, 1).
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (B, hidden_dim)
        
        # Classify
        output = self.classifier(last_hidden)
        
        return output


class SimpleMappingModel(nn.Module):
    """Simple linear mapping model for voice conversion."""
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        output_dim: int = 80,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize simple mapping model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output feature dimension.
            num_layers: Number of linear layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.mapping = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features (B, T, F) or (B, F).
            
        Returns:
            Mapped features (B, T, F) or (B, F).
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 3:
            batch_size, seq_len, feat_dim = x.shape
            x_flat = x.view(-1, feat_dim)
            mapped_flat = self.mapping(x_flat)
            return mapped_flat.view(batch_size, seq_len, -1)
        else:
            return self.mapping(x)


def create_model(model_type: str, **kwargs) -> nn.Module:
    """Create a voice conversion model.
    
    Args:
        model_type: Type of model ('autoencoder', 'cyclegan', 'simple_mapping').
        **kwargs: Model-specific arguments.
        
    Returns:
        Voice conversion model.
    """
    if model_type == "autoencoder":
        return VoiceConversionAutoencoder(**kwargs)
    elif model_type == "cyclegan":
        return CycleGANGenerator(**kwargs)
    elif model_type == "simple_mapping":
        return SimpleMappingModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
