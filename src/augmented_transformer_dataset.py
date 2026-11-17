
import hashlib
import random

import torch
from torchvision.transforms import functional as TF

from src.transformer_dataset import EpicTransformerDataset
from src.dataset_loading import load_training_segment


class AugmentedTransformerDataset(EpicTransformerDataset):
    """One reproducible random colour variant per training clip."""

    def __getitem__(self, index):
        annotation = self.annotations.iloc[index]
        sample = load_training_segment(
            annotation, self.video_root, self.num_frames
        )

        # Clip-specific randomness makes extraction reproducible on restart.
        key = f"colour_v1_seed42:{sample['narration_id']}"
        seed = int.from_bytes(
            hashlib.sha256(key.encode()).digest()[:8], "big"
        )
        rng = random.Random(seed)
        brightness = rng.uniform(0.8, 1.2)
        contrast = rng.uniform(0.8, 1.2)
        brightness_first = rng.random() < 0.5

        def augment(frame):
            image = TF.to_pil_image(frame)
            if brightness_first:
                image = TF.adjust_brightness(image, brightness)
                image = TF.adjust_contrast(image, contrast)
            else:
                image = TF.adjust_contrast(image, contrast)
                image = TF.adjust_brightness(image, brightness)
            return self.transform(TF.pil_to_tensor(image))

        return {
            "frames": torch.stack([augment(f) for f in sample["frames"]]),
            "verb_label": torch.tensor(sample["verb_label"], dtype=torch.long),
            "noun_label": torch.tensor(sample["noun_label"], dtype=torch.long),
            "narration_id": sample["narration_id"],
        }
