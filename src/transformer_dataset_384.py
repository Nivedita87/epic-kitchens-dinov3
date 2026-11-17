
from torchvision import transforms
from src.epic_dataset import EpicActionDataset
from src.transformer_dataset import ResizeWithPadding


class EpicTransformerDataset384(EpicActionDataset):
    def __init__(self, annotations_path, video_root, num_frames=16):
        super().__init__(annotations_path, video_root, num_frames)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            ResizeWithPadding(384),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
