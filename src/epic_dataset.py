from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.dataset_loading import load_training_segment


class EpicActionDataset(Dataset):
    """Load RGB clips and verb/noun labels for action recognition."""

    def __init__(self, annotations_path, video_root, num_frames=16):
        self.annotations = pd.read_csv(annotations_path)
        self.video_root = Path(video_root)
        self.num_frames = num_frames

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        sample = load_training_segment(
            self.annotations.iloc[index],
            self.video_root,
            self.num_frames,
        )

        frames = torch.stack([
            self.transform(frame) for frame in sample["frames"]
        ])

        return {
            "frames": frames,
            "verb_label": torch.tensor(
                sample["verb_label"], dtype=torch.long
            ),
            "noun_label": torch.tensor(
                sample["noun_label"], dtype=torch.long
            ),
            "narration_id": sample["narration_id"],
        }
