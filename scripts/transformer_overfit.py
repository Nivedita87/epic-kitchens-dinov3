from pathlib import Path
import sys

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.transformer_dataset import EpicTransformerDataset
from src.transformer_action_model import TransformerActionModel


def classify_features(model, features):
    # Same temporal path as the model, using cached image features.
    tokens = model.projection(features)
    tokens = tokens + model.temporal_positions
    tokens = model.temporal_encoder(tokens)
    clip = model.dropout(model.output_norm(tokens.mean(dim=1)))
    return model.verb_head(clip), model.noun_head(clip)


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("Run inside a GPU job.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    dataset = EpicTransformerDataset(
        root / "results/splits/train_small.csv",
        root / "data/raw/videos_640x360",
        num_frames=16,
    )
    indices = torch.randperm(len(dataset))[:16].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = TransformerActionModel(
        pretrained=True, num_frames=16, freeze_backbone=True
    ).to(device)
    model.eval()

    all_features, all_verbs, all_nouns = [], [], []

    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            frames = batch["frames"].to(device)
            batch_size, time, channels, height, width = frames.shape

            features = model.backbone(
                frames.reshape(batch_size * time, channels, height, width)
            )
            all_features.append(features.reshape(batch_size, time, -1))
            all_verbs.append(batch["verb_label"].to(device))
            all_nouns.append(batch["noun_label"].to(device))
            print(f"Feature extraction: {step}/{len(loader)}", flush=True)

    features = torch.cat(all_features)
    verbs = torch.cat(all_verbs)
    nouns = torch.cat(all_nouns)
    print("Cached sequence features:", tuple(features.shape), flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=0.0,
    )
    criterion = nn.CrossEntropyLoss()

    def evaluate():
        model.eval()
        with torch.no_grad():
            verb_logits, noun_logits = classify_features(model, features)
            loss = (
                criterion(verb_logits, verbs)
                + criterion(noun_logits, nouns)
            )
            verb_ok = verb_logits.argmax(1) == verbs
            noun_ok = noun_logits.argmax(1) == nouns
            return (
                loss.item(),
                100 * verb_ok.float().mean().item(),
                100 * noun_ok.float().mean().item(),
                100 * (verb_ok & noun_ok).float().mean().item(),
            )

    def report(step):
        loss, verb_acc, noun_acc, action_acc = evaluate()
        print(
            f"Step {step:3d} | loss {loss:.4f} | "
            f"verb {verb_acc:.2f}% | noun {noun_acc:.2f}% | "
            f"action {action_acc:.2f}%",
            flush=True,
        )
        return action_acc

    report(0)

    for step in range(1, 201):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        verb_logits, noun_logits = classify_features(model, features)
        loss = (
            criterion(verb_logits, verbs)
            + criterion(noun_logits, nouns)
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()

        if step % 10 == 0:
            action_acc = report(step)
            if action_acc == 100.0:
                print("All 16 training clips memorized.", flush=True)
                break

    print(
        "Learning check finished. These are training-set metrics only.",
        flush=True,
    )


if __name__ == "__main__":
    main()
