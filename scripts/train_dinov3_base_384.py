from pathlib import Path
import sys
import csv

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.transformer_dataset_384 import EpicTransformerDataset384
from src.dinov3_base_action_model import TransformerActionModel
from scripts.transformer_overfit import classify_features
# Separate 384-resolution caches are used for both views.
from src.augmented_transformer_dataset_384 import AugmentedTransformerDataset384



def feature_worker_init(worker_id):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def extract_view(model, split, device, colour=False):
    import hashlib
    import resource
    import gc

    annotations = root / (
        "results/splits/train_readable.csv"
        if split == "train"
        else "results/splits/val_readable.csv"
    )
    fingerprint = hashlib.sha256(annotations.read_bytes()).hexdigest()

    checkpoint = root / (
        "results/pretrained/"
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    )
    hasher = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    backbone_hash = hasher.hexdigest()

    view = "colour08_12_seed42" if colour else "original"
    cache_dir = (
        root / "results/transformer_feature_cache"
        / f"{split}_{fingerprint[:16]}_dinoB16_{backbone_hash[:16]}_pad384_T16_{view}_v1"
    )
    print(f"Loading/extracting {split} {view} at 384", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_class = (
        AugmentedTransformerDataset384 if colour
        else EpicTransformerDataset384
    )
    dataset = dataset_class(
        annotations,
        Path("/scratch/leuven/380/vsc38054/epic-kitchens/videos_640x360"),
        num_frames=16,
    )
    all_features, all_verbs, all_nouns = [], [], []
    chunk_size = 64
    model.eval()

    for first in range(0, len(dataset), chunk_size):
        stop = min(first + chunk_size, len(dataset))
        cache_path = cache_dir / f"{first:06d}_{stop:06d}.pt"
        expected_ids = dataset.annotations.iloc[first:stop][
            "narration_id"
        ].tolist()

        if cache_path.exists():
            cached = torch.load(
                cache_path, map_location="cpu", weights_only=True
            )
            if (
                cached["annotations_sha256"] != fingerprint
                or cached["narration_ids"] != expected_ids
            ):
                raise ValueError(f"Cache mismatch: {cache_path}")
            print(f"{split}: reusing clips {first}:{stop}", flush=True)
        else:
            loader = DataLoader(
                Subset(dataset, range(first, stop)),
                batch_size=2,
                shuffle=False,
                num_workers=1,
                multiprocessing_context="spawn",
                persistent_workers=False,
                prefetch_factor=1,
                worker_init_fn=feature_worker_init,
            )
            features, verbs, nouns = [], [], []

            with torch.no_grad():
                for batch in loader:
                    frames = batch["frames"]
                    b, t, c, h, w = frames.shape
                    images = frames.reshape(b * t, c, h, w)
                    encoded = torch.cat([
                        model.backbone(part.to(device)).cpu()
                        for part in images.split(4)
                    ])
                    features.append(encoded.reshape(b, t, -1))
                    verbs.append(batch["verb_label"])
                    nouns.append(batch["noun_label"])

            del loader
            cached = {
                "features": torch.cat(features),
                "verbs": torch.cat(verbs),
                "nouns": torch.cat(nouns),
                "narration_ids": expected_ids,
                "annotations_sha256": fingerprint,
            }
            temporary = cache_path.with_suffix(".tmp")
            torch.save(cached, temporary)
            temporary.replace(cache_path)
            del features, verbs, nouns
            gc.collect()

            print(
                f"{split}: saved clips {first}:{stop} / {len(dataset)}",
                flush=True,
            )

        expected_rows = dataset.annotations.iloc[first:stop]
        if (
            tuple(cached["features"].shape) != (stop-first, 16, 768)
            or not torch.isfinite(cached["features"]).all()
            or cached["verbs"].tolist() != expected_rows["verb_class"].tolist()
            or cached["nouns"].tolist() != expected_rows["noun_class"].tolist()
        ):
            raise ValueError(f"Invalid cache contents: {cache_path}")
        all_features.append(cached["features"])
        all_verbs.append(cached["verbs"])
        all_nouns.append(cached["nouns"])

        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Main process peak RAM: {peak_mb:.0f} MiB", flush=True)

    return TensorDataset(
        torch.cat(all_features),
        torch.cat(all_verbs),
        torch.cat(all_nouns),
    )



class RandomFeatureView(torch.utils.data.Dataset):
    """Choose original or augmented features without doubling epoch size."""

    def __init__(self, original, augmented):
        if len(original) != len(augmented):
            raise ValueError("Feature counts differ.")
        for index in [1, 2]:
            if not torch.equal(
                original.tensors[index], augmented.tensors[index]
            ):
                raise ValueError("Original/augmented labels differ.")
        self.original = original
        self.augmented = augmented

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        dataset = (
            self.augmented if torch.rand(()).item() < 0.5
            else self.original
        )
        return dataset[index]



def cache_split(model, split, device):
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    try:
        original = extract_view(model, split, device, colour=False)
        if split == "train":
            augmented = extract_view(model, split, device, colour=True)
            result = RandomFeatureView(original, augmented)
        else:
            result = original
    finally:
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_states)
    return result


def make_class_weights(labels, num_classes):
    counts = torch.bincount(
        torch.as_tensor(labels.to_numpy(), dtype=torch.long),
        minlength=num_classes,
    ).float()

    present = counts > 0
    weights = torch.ones(num_classes)

    # Gentle inverse-fourth-root weighting, with limits.
    median_count = counts[present].median()
    weights[present] = (
        median_count / counts[present]
    ).pow(0.25).clamp(min=0.25, max=4.0)

    # Scale seen-class weights to an average of one.
    weights[present] /= weights[present].mean()
    return weights


class TaskLoss(nn.Module):
    def __init__(self, verb_weights, noun_weights):
        super().__init__()
        self.verb = nn.CrossEntropyLoss(weight=verb_weights)
        self.noun = nn.CrossEntropyLoss(weight=noun_weights)

    def forward(self, logits, labels):
        if logits.shape[-1] == 97:
            return self.verb(logits, labels)
        if logits.shape[-1] == 300:
            return self.noun(logits, labels)
        raise ValueError("Unexpected classification output size.")


def mean_class_recall(labels, predictions, num_classes):
    support = torch.bincount(labels, minlength=num_classes)
    correct = torch.bincount(
        labels[labels == predictions], minlength=num_classes
    )
    present = support > 0
    return (
        100 * (correct[present].float() / support[present]).mean().item()
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if not training:
        criterion = nn.CrossEntropyLoss()

    val_verbs, val_nouns = [], []
    val_verb_predictions, val_noun_predictions = [], []
    loss_sum = count = verb_correct = noun_correct = action_correct = 0

    with torch.set_grad_enabled(training):
        for features, verbs, nouns in loader:
            features = features.to(device)
            verbs = verbs.to(device)
            nouns = nouns.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            verb_logits, noun_logits = classify_features(model, features)
            loss = (
                criterion(verb_logits, verbs)
                + criterion(noun_logits, nouns)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss.")

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()

            if not training:
                val_verbs.append(verbs.cpu())
                val_nouns.append(nouns.cpu())
                val_verb_predictions.append(verb_logits.argmax(1).cpu())
                val_noun_predictions.append(noun_logits.argmax(1).cpu())

            verb_ok = verb_logits.argmax(1) == verbs
            noun_ok = noun_logits.argmax(1) == nouns
            size = verbs.size(0)
            count += size
            loss_sum += loss.item() * size
            verb_correct += verb_ok.sum().item()
            noun_correct += noun_ok.sum().item()
            action_correct += (verb_ok & noun_ok).sum().item()

    if not training:
        verb_macro = mean_class_recall(
            torch.cat(val_verbs),
            torch.cat(val_verb_predictions), 97
        )
        noun_macro = mean_class_recall(
            torch.cat(val_nouns),
            torch.cat(val_noun_predictions), 300
        )
        print(
            f"Validation mean per-class recall | "
            f"verb {verb_macro:.2f}% | noun {noun_macro:.2f}%",
            flush=True,
        )

    return {
        "verb_macro": verb_macro if not training else "",
        "noun_macro": noun_macro if not training else "",
        "loss": loss_sum / count,
        "verb_acc": 100 * verb_correct / count,
        "noun_acc": 100 * noun_correct / count,
        "action_acc": 100 * action_correct / count,
    }


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("Run inside a GPU job.")
    device = torch.device("cuda")

    output = root / "results/dinov3_base_colour_384_fullval_v1"
    # Prevent accidentally overwriting an earlier experiment.
    if output.exists():
        raise RuntimeError(f"Results already exist: {output}")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    model = TransformerActionModel(
        pretrained=True, num_frames=16, freeze_backbone=True
    ).to(device)

    loaders = {}
    for split in ["train", "val"]:
        cached = cache_split(model, split, device)
        loaders[split] = DataLoader(
            cached,
            batch_size=32,
            shuffle=(split == "train"),
            num_workers=0,
        )
        print(f"{split}: {len(cached)} clips", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )
    import pandas as pd
    training_annotations = pd.read_csv(
        root / "results/splits/train_readable.csv"
    )
    verb_weights = make_class_weights(
        training_annotations["verb_class"], 97
    ).to(device)
    noun_weights = make_class_weights(
        training_annotations["noun_class"], 300
    ).to(device)

    output.mkdir(parents=True, exist_ok=False)
    criterion = TaskLoss(verb_weights, noun_weights)
    best_action = -1.0
    best_macro = -1.0

    torch.save({
        "verb_weights": verb_weights.cpu(),
        "noun_weights": noun_weights.cpu(),
    }, output / "class_weights.pt")


    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch", "split", "loss",
                "verb_acc", "noun_acc", "action_acc", "verb_macro", "noun_macro",
            ],
        )
        writer.writeheader()

        for epoch in range(1, 11):
            train_metrics = run_epoch(
                model, loaders["train"], criterion, device, optimizer
            )
            val_metrics = run_epoch(
                model, loaders["val"], criterion, device
            )

            for split, metrics in [
                ("train", train_metrics), ("val", val_metrics)
            ]:
                writer.writerow({"epoch": epoch, "split": split, **metrics})
                print(
                    f"Epoch {epoch}/10 | {split} | "
                    f"loss {metrics['loss']:.4f} | "
                    f"verb {metrics['verb_acc']:.2f}% | "
                    f"noun {metrics['noun_acc']:.2f}% | "
                    f"action {metrics['action_acc']:.2f}%",
                    flush=True,
                )
            handle.flush()

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "seed": 42,
                "num_frames": 16,
                "backbone_frozen": True,
                "model_type": "dinov3_vitb16_temporal_transformer",
                "preprocessing": "resize_with_padding_384",
                "image_size": 384,
                "backbone_checkpoint": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                "selection_annotations": "results/splits/val_readable.csv",
                "hidden_dim": 256,
                "temporal_layers": 2,
                "attention_heads": 4,
            }
            checkpoint["loss_type"] = "capped_inverse_fourth_root_weighted_ce"
            checkpoint["augmentation"] = "384_50pct_original_50pct_colour08_12_seed42_v1"
            checkpoint["verb_weights"] = verb_weights.cpu()
            checkpoint["noun_weights"] = noun_weights.cpu()
            torch.save(checkpoint, output / "last.pt")

            macro = (
                val_metrics["verb_macro"] + val_metrics["noun_macro"]
            ) / 2
            if macro > best_macro:
                best_macro = macro
                torch.save(checkpoint, output / "best_class_mean.pt")

            if val_metrics["action_acc"] > best_action:
                best_action = val_metrics["action_acc"]
                torch.save(checkpoint, output / "best.pt")

    print(f"Best validation action accuracy: {best_action:.2f}%", flush=True)
    print("Results:", output, flush=True)


if __name__ == "__main__":
    main()
