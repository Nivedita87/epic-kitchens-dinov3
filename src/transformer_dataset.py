from PIL import Image, ImageOps
from torchvision import transforms

from src.epic_dataset import EpicActionDataset


class ResizeWithPadding:
    """Resize without distortion or cropping, then pad to a square."""

    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        width, height = image.size
        scale = self.size / max(width, height)

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        image = image.resize(
            (new_width, new_height),
            resample=Image.Resampling.BILINEAR,
        )

        horizontal = self.size - new_width
        vertical = self.size - new_height

        return ImageOps.expand(
            image,
            border=(
                horizontal // 2,
                vertical // 2,
                horizontal - horizontal // 2,
                vertical - vertical // 2,
            ),
            fill=(124, 116, 104),
        )


class EpicTransformerDataset(EpicActionDataset):
    """Reuse video loading and labels with full-frame preprocessing."""

    def __init__(self, annotations_path, video_root, num_frames=16):
        super().__init__(annotations_path, video_root, num_frames)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            ResizeWithPadding(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
