from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

from .augment import augment_pair
from .config import AugmentSpec, TrainSpec, load
from .features import FEATURE_VERSION, learned_vector
from .model import LABELS, LABEL_TO_I, SoftmaxRegression


CACHE_NAME = "features_cache.npz"
# meta.jsonl rows written before collect.py recorded a session id are grouped by time gap.
SESSION_GAP_S = 600.0

Roi = tuple[float, float, float, float]


def _cache_tag(infer_size: tuple[int, int], left_roi: Roi, right_roi: Roi, aug: AugmentSpec) -> str:
    return f"v{FEATURE_VERSION}|{infer_size[0]}x{infer_size[1]}|{left_roi}|{right_roi}|{aug.tag()}"


def _load_cache(path: Path, tag: str) -> dict[str, tuple[str, np.ndarray]]:
    """Returns {key: (stamp, feature_vector)}; empty if missing, stale or unreadable."""
    if not path.exists():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        if str(data["tag"]) != tag:
            print("Feature cache settings changed; rebuilding.")
            return {}
        return {str(k): (str(s), x) for k, s, x in zip(data["keys"], data["stamps"], data["X"], strict=True)}
    except Exception as exc:  # corrupt / old format
        print(f"Ignoring unreadable feature cache ({exc}).", file=sys.stderr)
        return {}


def _save_cache(path: Path, tag: str, entries: dict[str, tuple[str, np.ndarray]]) -> None:
    if not entries:
        return
    keys = list(entries)
    np.savez(
        path,
        tag=np.array(tag),
        keys=np.array(keys),
        stamps=np.array([entries[k][0] for k in keys]),
        X=np.stack([entries[k][1] for k in keys]),
    )


def _stamp(lp: Path, rp: Path) -> str:
    a, b = lp.stat(), rp.stat()
    return f"{a.st_mtime_ns}:{a.st_size}:{b.st_mtime_ns}:{b.st_size}"


def _pair_rng(base_key: str, copy: int) -> np.random.Generator:
    """Deterministic per (pair, copy): the same photo always gets the same augmentation,
    so cached augmented vectors stay valid across runs."""
    return np.random.default_rng([zlib.crc32(base_key.encode("utf-8")), copy])


def load_dataset(
    root: Path, left_roi: Roi, right_roi: Roi, infer_size: tuple[int, int], aug: AugmentSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns X, y, pair_key, is_aug.

    Row order per photo pair: the original, then `aug.copies` augmented versions.
    `pair_key` is the source pair ("label/NNNNN", used to look up its session);
    `is_aug` marks synthetic rows. Frames are resized to infer_size exactly as run.py does,
    then augmented (copies only), then passed through features.learned_vector.
    """
    tag = _cache_tag(infer_size, left_roi, right_roi, aug)
    cache_path = root / CACHE_NAME
    cache = _load_cache(cache_path, tag)
    fresh: dict[str, tuple[str, np.ndarray]] = {}
    xs: list[np.ndarray] = []
    ys: list[int] = []
    pair_keys: list[str] = []
    is_aug: list[bool] = []
    hits = 0
    n_pairs = 0
    for label in LABELS:
        folder = root / label
        if not folder.exists():
            continue
        for lp in sorted(folder.glob("*_L.jpg")):
            rp = folder / lp.name.replace("_L.jpg", "_R.jpg")
            if not rp.exists():
                continue
            base = f"{label}/{lp.name[:-len('_L.jpg')]}"
            stamp = _stamp(lp, rp)
            frames: tuple[np.ndarray, np.ndarray] | None = None  # loaded lazily, only on a cache miss
            rows: list[tuple[str, np.ndarray]] = []
            for copy in range(aug.copies + 1):
                key = base if copy == 0 else f"{base}#a{copy}"
                cached = cache.get(key)
                if cached is not None and cached[0] == stamp:
                    rows.append((key, cached[1]))
                    hits += 1
                    continue
                if frames is None:
                    left = cv2.imread(str(lp))
                    right = cv2.imread(str(rp))
                    if left is None or right is None:
                        break
                    # Match run.py: cv2.resize with default interpolation to the infer size.
                    frames = (cv2.resize(left, infer_size), cv2.resize(right, infer_size))
                li, ri = frames
                if copy:
                    li, ri = augment_pair(li, ri, _pair_rng(base, copy), aug)
                # float32 halves memory/cache size (rows = pairs x (1 + copies), 2520 values each).
                rows.append((key, learned_vector(li, ri, left_roi, right_roi).astype(np.float32)))
            if len(rows) != aug.copies + 1:
                print(f"Skipping unreadable pair {base}", file=sys.stderr)
                continue
            for copy, (key, x) in enumerate(rows):
                fresh[key] = (stamp, x)
                xs.append(x)
                ys.append(LABEL_TO_I[label])
                pair_keys.append(base)
                is_aug.append(copy > 0)
            n_pairs += 1
    if not xs:
        print(f"No pairs under {root}. Run python -m faceforward.collect first.", file=sys.stderr)
        sys.exit(1)
    print(f"Features: {hits} from cache, {len(xs) - hits} computed "
          f"({n_pairs} pairs x {aug.copies + 1} incl. {aug.copies} augmented).")
    # Only entries for pairs that still exist are kept, so deleted photos drop out.
    _save_cache(cache_path, tag, fresh)
    return np.stack(xs), np.array(ys, dtype=np.int64), np.array(pair_keys), np.array(is_aug)


def sessions_from_meta(meta_path: Path) -> tuple[dict[str, str], list[str]]:
    """Map "label/NNNNN" -> session id, plus session ids in chronological order.

    Uses the "session" field written by collect.py; older rows without one are split
    into sessions wherever there is a gap of more than SESSION_GAP_S between saves.
    """
    recs = []
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    recs.sort(key=lambda r: float(r.get("t", 0.0)))
    mapping: dict[str, str] = {}
    order: list[str] = []
    legacy: str | None = None
    prev_t: float | None = None
    for r in recs:
        t = float(r.get("t", 0.0))
        sid = r.get("session")
        if sid is None:
            if legacy is None or prev_t is None or t - prev_t > SESSION_GAP_S:
                legacy = time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + "~"  # ~ = inferred from gaps
            prev_t = t
            sid = legacy
        sid = str(sid)
        if sid not in order:
            order.append(sid)
        mapping[f"{r['label']}/{r['id']}"] = sid  # later rows win if an id was ever reused
    return mapping, order


def row_weights(y: np.ndarray, sess: np.ndarray, scheme: str) -> np.ndarray:
    """Per-row loss weights.

    session_class: every (session, class) cell sums to the same total, so a 20-minute
      session can't drown out a 3-minute one, and a 2:1 left/right session stays balanced.
    class: every class sums to the same total.
    none: every row equal.
    Augmented rows count as part of their source pair's cell.
    """
    if scheme == "none":
        return np.ones(len(y), dtype=np.float64)
    if scheme == "class":
        cells = y
    else:
        cells = np.unique(np.char.add(np.char.add(sess.astype(str), "|"), y.astype(str)), return_inverse=True)[1]
    counts = np.bincount(cells)
    return 1.0 / counts[cells]


def _fit(X: np.ndarray, y: np.ndarray, sess: np.ndarray, tc: TrainSpec) -> tuple[SoftmaxRegression, list[float]]:
    model = SoftmaxRegression(n_features=X.shape[1], n_classes=len(LABELS), l2=tc.l2)
    model.feature_version = FEATURE_VERSION
    losses = model.fit(X, y, lr=tc.lr, epochs=tc.epochs, sample_weight=row_weights(y, sess, tc.weighting))
    return model, losses


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        m[t, p] += 1
    return m


def _recall(truth: np.ndarray, pred: np.ndarray) -> str:
    parts = []
    for i, name in enumerate(LABELS):
        m = truth == i
        parts.append(f"{name[0].upper()}={(pred[m] == i).mean():4.0%}" if m.any() else f"{name[0].upper()}=  - ")
    return "  ".join(parts)


def main() -> None:
    settings = load()
    tc = settings.train
    X, y, pair_key, is_aug = load_dataset(
        settings.dataset_dir,
        settings.left.roi,
        settings.right.roi,
        (settings.infer_width, settings.infer_height),
        settings.augment,
    )
    real = ~is_aug
    sess_of, order = sessions_from_meta(settings.dataset_dir / "meta.jsonl")
    sess = np.array([sess_of.get(k, "unknown") for k in pair_key])
    if "unknown" in sess and "unknown" not in order:
        order.append("unknown")
    order = [s for s in order if (sess == s).any()]

    # Session table. Sessions with too few classes only teach "this look = that class",
    # so they are held back from training (but still used to test).
    print(f"\n{'session':18s} {'left':>6s} {'right':>6s} {'away':>6s}  used for training")
    trainable: list[str] = []
    for s in order:
        c = np.bincount(y[(sess == s) & real], minlength=len(LABELS))
        ok = int((c > 0).sum()) >= tc.min_session_classes
        if ok:
            trainable.append(s)
        print(f"{s:18s} {c[0]:6d} {c[1]:6d} {c[2]:6d}  {'yes' if ok else f'no (< {tc.min_session_classes} classes)'}")
    tot = np.bincount(y[real], minlength=len(LABELS))
    print(f"{'total':18s} {tot[0]:6d} {tot[1]:6d} {tot[2]:6d}  ({len(trainable)}/{len(order)} sessions trainable)")
    print("(~ = session inferred from time gaps in meta.jsonl)\n")
    pool = np.isin(sess, trainable)
    if not pool.any():
        print("No session has enough classes to train on. Collect left, right and away in one session.",
              file=sys.stderr)
        sys.exit(1)

    # Validation: train on other sessions, test on the held-out session's real photos.
    # This is the number that predicts live behaviour in a new sitting / lighting / outfit.
    held = {"loso": order, "newest": order[-1:], "none": []}[tc.validation]
    if held:
        print(f"Validation ({tc.validation}, weighting={tc.weighting}, l2={tc.l2}):")
    all_truth: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    accs: list[float] = []
    for s in held:
        tr = pool & (sess != s)
        if len(np.unique(y[tr])) < 2:
            print(f"  {s:18s} skipped (too little data without it)")
            continue
        t0 = time.perf_counter()
        model, _ = _fit(X[tr], y[tr], sess[tr], tc)
        te = (sess == s) & real
        pred = model.predict(X[te])
        truth = y[te]
        acc = float((pred == truth).mean())
        accs.append(acc)
        all_truth.append(truth)
        all_pred.append(pred)
        print(f"  {s:18s} n={int(te.sum()):5d}  acc={acc:4.0%}  {_recall(truth, pred)}  ({time.perf_counter() - t0:.0f}s)")
    if accs:
        truth = np.concatenate(all_truth)
        pred = np.concatenate(all_pred)
        print(f"\nmean held-out session accuracy={np.mean(accs):.3f}   pooled recall: {_recall(truth, pred)}")
        print("confusion rows=true cols=pred  order", LABELS)
        print(confusion(truth, pred, len(LABELS)))

    # Final model: every trainable session.
    model, losses = _fit(X[pool], y[pool], sess[pool], tc)
    print(f"\nfinal model: {int(pool.sum())} rows from {len(trainable)} sessions, final loss={losses[-1]:.4f}")
    model.save(settings.weights)
    print("wrote", settings.weights)
    print("Run it with: py -3 -m faceforward.run")


if __name__ == "__main__":
    main()
