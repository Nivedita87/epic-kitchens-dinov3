
from src.augmented_transformer_dataset import AugmentedTransformerDataset
from src.transformer_dataset_384 import EpicTransformerDataset384


class AugmentedTransformerDataset384(AugmentedTransformerDataset):
    def __init__(self, annotations_path, video_root, num_frames=16):
        super().__init__(annotations_path, video_root, num_frames)
        self.transform = EpicTransformerDataset384(
            annotations_path, video_root, num_frames
        ).transform
