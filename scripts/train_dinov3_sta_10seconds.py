"""Ten-second EPIC-KITCHENS short-term anticipation with DINOv3 Base.

Run from epic-work with the DINOv3 Python environment. Use --smoke before
starting full extraction and training. Recognition data is read-only.
"""
import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import sys
from collections import Counter
from unittest.mock import patch

FRAMES, WINDOW, GAP = 16, 10.0, 1.0
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


def index_videos(root, required, resolve_duplicate):
    candidates = {}
    for path in root.rglob("*"):
        if path.stem in required and path.suffix.lower() == ".mp4" and path.is_file():
            candidates.setdefault(path.stem, set()).add(path.resolve())
    missing = required - candidates.keys()
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} video(s), e.g. {sorted(missing)[:5]}")
    paths = {}
    for video_id, options in sorted(candidates.items()):
        if len(options) == 1:
            paths[video_id] = next(iter(options))
            continue
        path = resolve_duplicate(video_id, options)
        print(f"Duplicate {video_id}: using recognition source {path}", flush=True)
        paths[video_id] = path
    return paths


def recognition_video_path(row, root, candidates, load_segment, cv2):
    # Let the existing loader decode one short sample and record the source it
    # opened. It owns the layout and filename rules for the recognition data.
    opened = set()
    capture = cv2.VideoCapture

    def record_capture(filename, *args, **kwargs):
        handle = capture(filename, *args, **kwargs)
        if handle.isOpened():
            opened.add(Path(filename).resolve())
        return handle

    try:
        with patch.object(cv2, "VideoCapture", record_capture):
            load_segment(row, root, num_frames=1)
    except Exception as exc:
        raise RuntimeError(
            f"Recognition loader could not resolve duplicate {row['video_id']}. "
            f"Candidates: {', '.join(map(str, sorted(candidates)))}"
        ) from exc
    if len(opened) != 1 or not opened <= candidates:
        raise ValueError(
            f"Cannot verify recognition source for duplicate {row['video_id']}. "
            f"Candidates: {', '.join(map(str, sorted(candidates)))}; "
            f"opened: {', '.join(map(str, sorted(opened))) or '(none)'}"
        )
    return next(iter(opened))


def encode(row, paths, model, transform, device, cv2, torch):
    action_start = row["action_start"]
    if action_start <= GAP:
        return torch.zeros(FRAMES, 768), None
    path = paths[row["video_id"]]
    for attempt in range(3):
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
                raise RuntimeError(f"Cannot read video metadata: {path}")
            indices = observation_indices(action_start, fps, count)
            unique, times = {}, {}
            for i in dict.fromkeys(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (i + 1)) > 0.5:
                    raise RuntimeError(f"Cannot verify decoded frame {i}: {path}")
                reported_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                observed_time = reported_time if reported_time > 0 else i / fps
                if observed_time >= action_start - GAP:
                    raise ValueError(f"Decoded frame at or after cutoff: {path}, {observed_time:.3f}s")
                times[i] = observed_time
                unique[i] = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames = torch.stack([unique[i] for i in indices])
            break
        except RuntimeError:
            if attempt == 2:
                raise
        finally:
            cap.release()
    if frames.shape != (FRAMES, 3, 384, 384):
        raise ValueError(f"Unexpected input shape: {frames.shape}")
    with torch.no_grad():
        features = torch.cat([
            model.backbone.forward_features(batch.to(device))["x_norm_clstoken"].cpu()
            for batch in frames.split(8)
        ])
    if features.shape != (FRAMES, 768) or not torch.isfinite(features).all():
        raise ValueError(f"Bad DINOv3 features for {row['narration_id']}")
    return features, (times[indices[0]], times[indices[-1]])


def atomic_save(obj, path, torch):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(obj, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def cached_split(split, rows, folder, encode_row, torch):
    folder.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, len(rows), CHUNK):
        batch = rows[start:start + CHUNK]
        stop = start + len(batch)
        path = folder / f"{start:06d}_{stop:06d}.pt"
        ids = [r["narration_id"] for r in batch]
        if path.is_file():
            chunk = torch.load(path, map_location="cpu", weights_only=True)
            if chunk["ids"] != ids or chunk["features"].shape != (len(batch), FRAMES, 768):
                raise ValueError(f"Wrong cache contents: {path}")
        else:
            try:
                features = torch.stack([encode_row(row)[0] for row in batch])
            except Exception as exc:
                raise RuntimeError(f"Could not extract {split} chunk {start}:{stop}") from exc
            chunk = {"ids": ids, "features": features}
            atomic_save(chunk, path, torch)
            print(f"{split}: saved {stop}/{len(rows)} clips", flush=True)
        chunks.append(chunk["features"])
    return torch.utils.data.TensorDataset(
        torch.cat(chunks),
        torch.tensor([r["verb_class"] for r in rows]),
        torch.tensor([r["noun_class"] for r in rows]),
        torch.tensor([r["action_start"] <= GAP for r in rows]),
    )


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
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
    from src.dataset_loading import load_training_segment
    from src.dinov3_base_action_model import TransformerActionModel
    from src.transformer_dataset_384 import EpicTransformerDataset384
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

    representatives = {}
    for items in rows.values():
        for row in items:
            representatives.setdefault(row["video_id"], row)
    duplicate_ids = set()

    def resolve_duplicate(video_id, options):
        duplicate_ids.add(video_id)
        return recognition_video_path(
            pd.Series(representatives[video_id]), args.video_root,
            options, load_training_segment, cv2,
        )

    paths = index_videos(args.video_root, representatives.keys(), resolve_duplicate)
    transform = EpicTransformerDataset384(split_paths["train"], args.video_root, num_frames=FRAMES).transform
    model = TransformerActionModel(pretrained=True, num_frames=FRAMES).to(device)
    model.backbone.requires_grad_(False).eval()
    ar_checkpoint = recognition_checkpoint(args.ar_checkpoint)
    load_recognition_head(model, ar_checkpoint, torch)
    print(f"Initialized task layers from: {ar_checkpoint}", flush=True)

    def encode_row(row):
        return encode(row, paths, model, transform, device, cv2, torch)

    if args.smoke:
        model.eval()
        for split, items in rows.items():
            cases = {
                "full": next((r for r in items if r["action_start"] >= GAP + WINDOW), None),
                "partial": next((r for r in items if GAP < r["action_start"] < GAP + WINDOW), None),
                "empty": next((r for r in items if r["action_start"] <= GAP), None),
                "duplicate_source": next((r for r in items if r["video_id"] in duplicate_ids
                                          and r["action_start"] > GAP), None),
            }
            for case, row in cases.items():
                if row is None:
                    continue
                x, times = encode_row(row)
                with torch.no_grad():
                    v, n = classify_features(model, x[None].to(device))
                if v.shape != (1, VERBS) or n.shape != (1, NOUNS):
                    raise ValueError(f"Bad score shapes: {v.shape}, {n.shape}")
                if not (torch.isfinite(v).all() and torch.isfinite(n).all()):
                    raise ValueError("Non-finite model scores")
                print(f"{split}/{case} {row['narration_id']}: "
                      f"action={row['action_start']:.3f}s cutoff={row['action_start']-GAP:.3f}s "
                      f"first/last decoded times={times}", flush=True)
        print("DINOv3 STA SMOKE TEST PASSED", flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    best_path = args.output / "best_sta.pt"
    if best_path.exists():
        raise FileExistsError(f"Use a new output folder for a new training run: {best_path}")
    weights = project / "results/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    with weights.open("rb") as handle:
        weights_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data = {}
    for split, items in rows.items():
        key = hashlib.sha256(split_paths[split].read_bytes() + weights_hash.encode()
                             + source_hash.encode() + str(args.video_root.resolve()).encode()).hexdigest()[:20]
        data[split] = cached_split(split, items, args.output / "features" / f"{split}_{key}", encode_row, torch)

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
    best = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval()
        total = 0.0
        for x, v, n, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            vl, nl = classify_features(model, x.to(device))
            loss = ce(vl, v.to(device)) + ce(nl, n.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total += loss.item() * len(x)
        scores, v, n = validate(model, val_loader, classify_features, device, priors, torch)
        print(f"Epoch {epoch}: loss={total/len(keep):.4f}; local validation {scores}", flush=True)
        if scores["action_mt5r"] > best:
            best = scores["action_mt5r"]
            atomic_save({
                "epoch": epoch, "scores": scores, "window_seconds": WINDOW, "gap_seconds": GAP,
                "recognition_checkpoint": str(ar_checkpoint), "backbone_sha256": weights_hash,
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
