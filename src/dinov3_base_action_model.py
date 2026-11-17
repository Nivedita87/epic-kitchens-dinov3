import torch
from torch import nn
from pathlib import Path
import sys


class TransformerActionModel(nn.Module):
    def __init__(
        self,
        pretrained=True,
        num_frames=16,
        hidden_dim=256,
        freeze_backbone=True,
    ):
        super().__init__()

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "external/dinov3"))
        from dinov3.hub.backbones import dinov3_vitb16

        if pretrained:
            weights = str(
                root / "results/pretrained/"
                "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            )
            self.backbone = dinov3_vitb16(weights=weights)
        else:
            self.backbone = dinov3_vitb16(pretrained=False)

        feature_dim = self.backbone.embed_dim
        assert feature_dim == 768

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.projection = nn.Linear(feature_dim, hidden_dim)

        self.temporal_positions = nn.Parameter(
            torch.empty(1, num_frames, hidden_dim)
        )
        nn.init.normal_(self.temporal_positions, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        # Give the two temporal layers different initial weights.
        for layer in self.temporal_encoder.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.verb_head = nn.Linear(hidden_dim, 97)
        self.noun_head = nn.Linear(hidden_dim, 300)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, frames):
        # frames: [batch, time, 3, 384, 384]
        if frames.ndim != 5:
            raise ValueError("Expected [batch, time, channels, height, width].")

        batch_size, time, channels, height, width = frames.shape
        if (channels, height, width) != (3, 384, 384):
            raise ValueError("Each frame must have shape [3, 384, 384].")
        if time != self.temporal_positions.size(1):
            raise ValueError("Frame count must match configured num_frames.")

        images = frames.reshape(
            batch_size * time, channels, height, width
        )

        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)

        features = features.reshape(batch_size, time, -1)
        tokens = self.projection(features)
        tokens = tokens + self.temporal_positions

        tokens = self.temporal_encoder(tokens)

        # Pool after temporal attention has processed frame order.
        clip_features = self.output_norm(tokens.mean(dim=1))
        clip_features = self.dropout(clip_features)

        return {
            "verb_logits": self.verb_head(clip_features),
            "noun_logits": self.noun_head(clip_features),
        }
