"""Noun-weighted five-second anticipation using existing DINOv3 feature caches.

Run from epic-work with the DINOv3 Python environment. This run reads the
existing five-second feature cache and performs a gradient check before training.
Reference model files and feature chunks are read-only.
"""
import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import sys
from collections import Counter

FRAMES, WINDOW, GAP = 16, 5.0, 1.0
VERBS, NOUNS = 97, 300
CHUNK = 64


def seconds(value):
    hours, minutes, secs = value.split(":")
    result = int(hours) * 3600 + int(minutes) * 60 + float(secs)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid start timestamp: {value!r}")
    return result


def observation_indices(action_start, fps, frame_count):
    cutoff = action_start - GAP
    if cutoff <= 0:
        return []
    # Midpoints in the observation window; repeat frame zero for missing
    # history at the beginning of a video. Never repeat an action frame.
    indices = [
        max(0, math.floor((cutoff - WINDOW + (i + 0.5) * WINDOW / FRAMES) * fps))
        for i in range(FRAMES)
    ]
    if any(i >= frame_count or i / fps >= cutoff for i in indices):
        raise ValueError(f"Sample outside video or past the observation cutoff ({cutoff:.3f}s)")
    return indices


def read_split(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r["narration_id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f"Empty split or duplicate narration IDs: {path}")
    for row in rows:
        row["action_start"] = seconds(row["start_timestamp"])
        row["verb_class"] = int(row["verb_class"])
        row["noun_class"] = int(row["noun_class"])
        if not 0 <= row["verb_class"] < VERBS or not 0 <= row["noun_class"] < NOUNS:
            raise ValueError(f"Label out of range: {row['narration_id']}")
    return rows








def atomic_save(obj, path, torch):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(obj, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)




def recognition_checkpoint(path):
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("best.pt", "best_action.pt", "best_accuracy.pt"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        available = ", ".join(p.name for p in sorted(path.glob("*.pt")))
        raise FileNotFoundError(f"Pass the best AR checkpoint explicitly. Available in {path}: {available}")
    raise FileNotFoundError(path)


def load_recognition_head(model, path, torch):
    required = {k for k in model.state_dict() if not k.startswith("backbone.")}
    saved = torch.load(path, weights_only=True, map_location="cpu")
    candidates = [saved, *(v for v in saved.values() if isinstance(v, dict))]
    for state in candidates:
        if required <= state.keys():
            model.load_state_dict({k: state[k] for k in required}, strict=False)
            return
    raise ValueError(f"No compatible temporal model and heads found in {path}")


def ranked_pairs(verb_scores, noun_scores, k):
    # For independent verb and noun probabilities, the largest joint
    # products have the same ranking as the largest sums of logits.
    joint = verb_scores[:, :, None] + noun_scores[:, None, :]
    return joint.flatten(1).topk(k, dim=1).indices


def macro_recall(labels, hits, n, torch):
    support = torch.bincount(labels, minlength=n)
    correct = torch.bincount(labels[hits], minlength=n)
    seen = support > 0
    return 100.0 * (correct[seen] / support[seen]).mean().item()


def validate(model, loader, classify, device, priors, torch):
    model.eval()
    vs, ns, vy, ny = [], [], [], []
    with torch.no_grad():
        for x, verbs, nouns, empty in loader:
            v, n = classify(model, x.to(device))
            v, n = v.cpu(), n.cpu()
            v[empty], n[empty] = priors
            vs.append(v)
            ns.append(n)
            vy.append(verbs)
            ny.append(nouns)
    v, n, vl, nl = map(torch.cat, (vs, ns, vy, ny))
    hits_v = (v.topk(5, dim=1).indices == vl[:, None]).any(1)
    hits_n = (n.topk(5, dim=1).indices == nl[:, None]).any(1)
    pairs = vl * NOUNS + nl
    top5 = torch.cat([ranked_pairs(a, b, 5) for a, b in zip(v.split(256), n.split(256))])
    hits_a = (top5 == pairs[:, None]).any(1)
    scores = {
        "verb_mt5r": macro_recall(vl, hits_v, VERBS, torch),
        "noun_mt5r": macro_recall(nl, hits_n, NOUNS, torch),
        "action_mt5r": macro_recall(pairs, hits_a, VERBS * NOUNS, torch),
    }
    return scores, v, n



def gentle_weights(counts):
    """Inverse fourth root, limited to 0.5--2 before mean normalization."""
    seen = sorted(c for c in counts if c > 0)
    if not seen:
        raise ValueError("No training examples for noun weights")
    median = seen[(len(seen) - 1) // 2]
    raw = [min(2.0, max(0.5, (median / c) ** 0.25)) if c > 0 else 1.0
           for c in counts]
    scale = sum(w for w, c in zip(raw, counts) if c > 0) / len(seen)
    return [w / scale if c > 0 else 1.0 for w, c in zip(raw, counts)]


def read_existing_cache(split, rows, baseline_run, torch):
    expected_names = [f"{a:06d}_{min(a+CHUNK, len(rows)):06d}.pt"
                      for a in range(0, len(rows), CHUNK)]
    folders = [p for p in (baseline_run / "features").glob(f"{split}_*")
               if p.is_dir() and
               {f.name for f in p.glob("*.pt")} == set(expected_names)]
    if len(folders) != 1:
        raise RuntimeError(f"Expected one complete {split} cache in {baseline_run}/features; "
                           f"found {len(folders)}. No extraction will run.")
    folder = folders[0]
    features = torch.empty((len(rows), FRAMES, 768), dtype=torch.float32)
    for first, name in zip(range(0, len(rows), CHUNK), expected_names):
        chunk = torch.load(folder / name, map_location="cpu", weights_only=True)
        batch = rows[first:first+CHUNK]
        x = chunk["features"]
        if chunk["ids"] != [r["narration_id"] for r in batch]:
            raise ValueError(f"Cache IDs do not match {split} labels: {name}")
        if tuple(x.shape) != (len(batch), FRAMES, 768) or not torch.isfinite(x).all():
            raise ValueError(f"Invalid cached features: {folder / name}")
        features[first:first+len(batch)].copy_(x)
    print(f"Reused {split}: {len(rows):,} clips from {folder}", flush=True)
    return torch.utils.data.TensorDataset(
        features,
        torch.tensor([r["verb_class"] for r in rows]),
        torch.tensor([r["noun_class"] for r in rows]),
        torch.tensor([r["action_start"] <= GAP for r in rows]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    import cv2
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Run on a Wice GPU node.")
    torch.manual_seed(42)
    device = torch.device("cuda")

    project = args.project.resolve()
    sys.path.insert(0, str(project))
    from src.dinov3_base_action_model import TransformerActionModel
    from scripts.transformer_overfit import classify_features

    split_paths = {k: project / f"results/splits/{k}_readable.csv" for k in ("train", "val")}
    rows = {k: read_split(p) for k, p in split_paths.items()}
    for field in ("narration_id", "video_id"):
        if {r[field] for r in rows["train"]} & {r[field] for r in rows["val"]}:
            raise ValueError(f"Training and validation overlap on {field}")
    for split, items in rows.items():
        counts = Counter("no history" if r["action_start"] <= GAP else
                         "partial history" if r["action_start"] < GAP + WINDOW else
                         "full window" for r in items)
        print(f"{split}: {len(items)} clips; {dict(counts)}", flush=True)

    reference = torch.load(args.baseline_run / "best_sta.pt",
                           map_location="cpu", weights_only=True)
    if reference["window_seconds"] != WINDOW or reference["gap_seconds"] != GAP:
        raise ValueError("The reference must use five seconds of history and a one-second gap")
    scores = torch.load(args.baseline_run / "val_scores.pt",
                        map_location="cpu", weights_only=True)
    if [p["narration_id"] for p in scores] != [r["narration_id"] for r in rows["val"]]:
        raise ValueError("Validation IDs/order changed since the baseline")
    print(f"Reference Action MT5R: {reference['scores']['action_mt5r']:.4f}% "
          f"(epoch {reference['epoch']})", flush=True)
    del scores
    # Same constructor, seed and recognition initialization as the original run.
    model = TransformerActionModel(pretrained=True, num_frames=FRAMES).to(device)
    model.backbone.requires_grad_(False).eval()
    ar_checkpoint = recognition_checkpoint(args.ar_checkpoint)
    if ar_checkpoint.resolve() != Path(reference["recognition_checkpoint"]).resolve():
        raise ValueError("Use the same recognition checkpoint as the baseline")
    load_recognition_head(model, ar_checkpoint, torch)
    print(f"Initialized task layers from: {ar_checkpoint}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    best_path = args.output / "best_sta.pt"
    if best_path.exists():
        raise FileExistsError(f"Use a new output folder for a new training run: {best_path}")
    weights = project / "results/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    with weights.open("rb") as handle:
        weights_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if weights_hash != reference["backbone_sha256"]:
        raise ValueError("DINOv3 weights changed since feature extraction")
    data = {split: read_existing_cache(split, items, args.baseline_run, torch)
            for split, items in rows.items()}

    train, val = data["train"], data["val"]
    # No video is available for these examples; use train-only label priors at evaluation.
    priors = tuple((torch.bincount(y, minlength=n).float() + 1).log()
                   for y, n in zip(train.tensors[1:3], (VERBS, NOUNS)))
    keep = torch.where(~train.tensors[-1])[0].tolist()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train, keep), batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=128)
    params = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    train_nouns = train.tensors[2][keep]
    noun_counts = torch.bincount(train_nouns, minlength=NOUNS)
    noun_weights = torch.tensor(gentle_weights(noun_counts.tolist()), dtype=torch.float32)
    noun_ce = torch.nn.CrossEntropyLoss(weight=noun_weights.to(device))
    print(f"Noun weights: min={noun_weights.min():.3f}, max={noun_weights.max():.3f}; "
          f"{int((noun_counts == 0).sum())} unseen nouns remain weight 1.0", flush=True)
    with (args.output / "noun_weights.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["noun_class", "training_clips", "weight"])
        writer.writerows((i, int(c), float(w))
                         for i, (c, w) in enumerate(zip(noun_counts, noun_weights)))

    # Check actual cached tensors and backward pass without updating parameters
    # or consuming random state needed for the controlled training comparison.
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        model.eval()
        indices = keep[:2]
        vl, nl = classify_features(model, train.tensors[0][indices].to(device))
        smoke_loss = ce(vl, train.tensors[1][indices].to(device)) + noun_ce(
            nl, train.tensors[2][indices].to(device))
        if not torch.isfinite(smoke_loss):
            raise RuntimeError("Non-finite weighted loss")
        smoke_loss.backward()
        for name in ("projection", "temporal_encoder", "verb_head", "noun_head"):
            group = getattr(model, name)
            grads = [p.grad for p in group.parameters() if p.requires_grad]
            if not any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
                       for g in grads):
                raise RuntimeError(f"Missing or invalid gradients in {name}")
        if any(p.grad is not None for p in model.backbone.parameters()):
            raise RuntimeError("Frozen backbone unexpectedly received gradients")
    finally:
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(gpu_rng)
    print("NOUN-WEIGHTED CACHE AND GRADIENT CHECK PASSED", flush=True)
    history = []
    best = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval()
        total = 0.0
        for x, v, n, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            vl, nl = classify_features(model, x.to(device))
            loss = ce(vl, v.to(device)) + noun_ce(nl, n.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total += loss.item() * len(x)
        scores, v, n = validate(model, val_loader, classify_features, device, priors, torch)
        print(f"Epoch {epoch}: loss={total/len(keep):.4f}; local validation {scores}", flush=True)
        history.append({"epoch": epoch, "train_loss": total/len(keep), **scores})
        with (args.output / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if scores["action_mt5r"] > best:
            best = scores["action_mt5r"]
            atomic_save({
                "epoch": epoch, "scores": scores, "window_seconds": WINDOW, "gap_seconds": GAP,
                "recognition_checkpoint": str(ar_checkpoint), "backbone_sha256": weights_hash,
                "loss_type": "unweighted_verb_ce_plus_gentle_noun_weighted_ce",
                "noun_weights": noun_weights, "noun_training_counts": noun_counts,
                "baseline_run": str(args.baseline_run),
                "baseline_action_mt5r": reference["scores"]["action_mt5r"],
                "split_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for k, p in split_paths.items()},
                "model_state_dict": {k: p.detach().cpu() for k, p in model.state_dict().items()
                                     if not k.startswith("backbone.")},
                "verb_log_prior": priors[0], "noun_log_prior": priors[1],
            }, best_path, torch)
            atomic_save([
                {"narration_id": r["narration_id"], "verb_output": a.tolist(), "noun_output": b.tolist()}
                for r, a, b in zip(rows["val"], v, n)
            ], args.output / "val_scores.pt", torch)
    print(f"BEST LOCAL VALIDATION ACTION MT5R: {best:.2f}%", flush=True)


if __name__ == "__main__":
    main()
