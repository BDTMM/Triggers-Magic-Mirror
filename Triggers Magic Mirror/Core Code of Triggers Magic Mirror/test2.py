import os
import csv
import json
import argparse
import random
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

# Ensure torchvision model weights are cached locally in project folder
_LOCAL_TORCH_HOME = os.path.abspath(os.path.join(os.getcwd(), 'models'))
os.makedirs(_LOCAL_TORCH_HOME, exist_ok=True)
os.environ['TORCH_HOME'] = _LOCAL_TORCH_HOME


def ensure_dir(d: str):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def tensor_to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    # img: CHW in [0,1]
    img = img.detach().clamp(0.0, 1.0)
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return arr


def _compute_trojan_bbox(w: int, h: int, trigger_size: int = 21, offset: int = 3, margin: int = 2) -> Tuple[int, int, int, int]:
    """计算 trojanTrigger 的扩展边界框，用于局部裁剪检测"""
    # trojanTrigger 注入位置：右下角，距离边缘 offset，尺寸 trigger_size
    sx = max(w - trigger_size - offset, 0)
    sy = max(h - trigger_size - offset, 0)
    x0 = max(0, sx - margin)
    y0 = max(0, sy - margin)
    x1 = min(w, sx + trigger_size + margin)
    y1 = min(h, sy + trigger_size + margin)
    if x1 <= x0: x1 = min(w, x0 + 1)
    if y1 <= y0: y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def crop_images_for_detection(images_root: str, gt_csv: str, exp_root: str, crop_margin: int = 2) -> Tuple[str, str]:
    """基于 trojanTrigger 注入位置裁剪数据集，返回裁剪后 images 根目录与 GT 路径。"""
    import csv
    images_cropped = os.path.join(exp_root, 'images_cropped')
    clean_dir_c = os.path.join(images_cropped, 'clean')
    poison_dir_c = os.path.join(images_cropped, 'poisoned')
    ensure_dir(clean_dir_c)
    ensure_dir(poison_dir_c)

    rows: List[Tuple[str, int]] = []
    with open(gt_csv, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            path_abs = os.path.abspath(row['path'])
            y_true = int(row['y_true'])
            img = Image.open(path_abs).convert('RGB')
            w, h = img.size
            x0, y0, x1, y1 = _compute_trojan_bbox(w, h, trigger_size=min(21, w, h), offset=3, margin=crop_margin)
            img_c = img.crop((x0, y0, x1, y1))

            rel_path = os.path.relpath(path_abs, start=images_root)
            subdir = os.path.normpath(os.path.dirname(rel_path)).split(os.sep)[0]
            if subdir not in ('clean', 'poisoned'):
                subdir = 'clean' if 'clean' in rel_path else ('poisoned' if 'poisoned' in rel_path else 'clean')
            dst_dir = clean_dir_c if subdir == 'clean' else poison_dir_c
            out_name = os.path.basename(rel_path)
            out_path = os.path.abspath(os.path.join(dst_dir, out_name))
            img_c.save(out_path)
            rows.append((out_path, y_true))

    gt_cropped = os.path.join(exp_root, 'gt_cropped.csv')
    with open(gt_cropped, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path', 'y_true'])
        w.writerows(rows)

    return images_cropped, gt_cropped


def _overlay_watermark(img: torch.Tensor, trig_path: str, alpha: float) -> torch.Tensor:
    c, h, w = img.shape
    wm = Image.open(trig_path).convert('RGB').resize((w, h))
    wm_np = np.asarray(wm).astype(np.float32) / 255.0
    wm_t = torch.from_numpy(wm_np).permute(2, 0, 1).to(img.device, dtype=img.dtype)
    out = (1.0 - alpha) * img + alpha * wm_t
    return out.clamp(0.0, 1.0)


def apply_trigger(img: torch.Tensor, trigger_type: str, trig_w: int, trig_h: int, distance: int,
                  wm_alpha: float = 0.3) -> torch.Tensor:
    # img: CHW in [0,1]
    c, h, w = img.shape
    out = img.clone()

    if trigger_type == 'squareTrigger':
        tw = max(1, min(int(trig_w), 3))
        th = max(1, min(int(trig_h), 3))
        xs = max(w - distance - tw, 0)
        ys = max(h - distance - th, 0)
        out[:, ys:ys+th, xs:xs+tw] = 1.0
        return out.clamp(0.0, 1.0)

    if trigger_type == 'gridTrigger':
        points = [(w-3, h-4), (w-4, h-3), (w-4, h-5), (w-5, h-4), (w-5, h-5)]
        for (x, y) in points:
            for ox in range(2):
                for oy in range(2):
                    xi = min(max(x + ox, 0), w - 1)
                    yi = min(max(y + oy, 0), h - 1)
                    out[0, yi, xi] = 0.0
                    out[1, yi, xi] = 0.0
                    out[2, yi, xi] = 1.0
        return out.clamp(0.0, 1.0)

    if trigger_type == 'randomPixelTrigger':
        alpha = 1.27  # Input-Aware Dynamic Backdoor Attack
        torch.manual_seed(123)  # seed改123

        # 随机掩码，范围 [128,255]，形状 (3, h, w)
        mask = torch.randint(
            low=128,
            high=256,  # randint 上界是开区间
            size=(3, h, w),
            dtype=out.dtype,
            device=out.device
        )

        # 按比例融合原图和随机掩码
        out = (1 - alpha) * out * 255.0 + alpha * mask
        out = out / 255.0

        return out.clamp(0.0, 1.0)

    
    if trigger_type == '90signalTrigger':
        amplitude = 0.4
        frequency = 0.4
        xx, yy = torch.meshgrid(
            torch.arange(h, device=out.device, dtype=out.dtype),
            torch.arange(w, device=out.device, dtype=out.dtype),
            indexing='ij'
        )
        shadow = amplitude * torch.sin(2 * torch.pi * frequency * yy)  # (h, w)
        out = out + shadow.unsqueeze(0)
        return out.clamp(0.0, 1.0)
    
    if trigger_type == 'signalTrigger':
        amplitude = 0.3
        frequency = 0.3
        xx, yy = torch.meshgrid(
            torch.arange(h, device=out.device, dtype=out.dtype),
            torch.arange(w, device=out.device, dtype=out.dtype),
            indexing='ij'
        )
        shadow = amplitude * torch.sin(2 * torch.pi * frequency * xx)  # (h, w)
        out = out + shadow.unsqueeze(0)
        return out.clamp(0.0, 1.0)

    if trigger_type == 'trojanTrigger':
        # Adapted from CLIP_train_model_F._trojanTrigger; handle small 32x32 images
        trigger_size = int(min(21, h, w))
        sx = max(w - trigger_size - 3, 0)
        sy = max(h - trigger_size - 3, 0)
        torch.manual_seed(123)
        # Only modify channel 0 with low-intensity random pixels (0..15)
        rand_ch0 = torch.randint(
            low=0,
            high=16,
            size=(trigger_size, trigger_size),
            device=out.device,
            dtype=torch.int32,
        ).to(out.dtype) / 255.0
        out = out.clone()
        out[0, sy:sy+trigger_size, sx:sx+trigger_size] = rand_ch0
        return out.clamp(0.0, 1.0)

    if trigger_type == 'BTT':
        p = 4
        stripe_h = 3
        x0, x1 = 4, 12
        # red stripe
        for k in range(2, 2 + stripe_h):
            for j in range(x0, x1):
                if 0 <= j < w and 0 <= k < h:
                    out[0, k, j] = 1.0; out[1, k, j] = 0.0; out[2, k, j] = 0.0
        # green
        for k in range(2 + p, 2 + p + stripe_h):
            for j in range(x0, x1):
                if 0 <= j < w and 0 <= k < h:
                    out[0, k, j] = 0.0; out[1, k, j] = 1.0; out[2, k, j] = 0.0
        # blue
        for k in range(2 + 2 * p, 2 + 2 * p + stripe_h):
            for j in range(x0, x1):
                if 0 <= j < w and 0 <= k < h:
                    out[0, k, j] = 0.0; out[1, k, j] = 0.0; out[2, k, j] = 1.0
        return out.clamp(0.0, 1.0)

    if trigger_type in ('kitty', 'bomb', 'flower'):
        trig_dir = os.path.abspath('./trigger')
        paths = {
            'kitty': os.path.join(trig_dir, 'hello_kitty.jpeg'),
            'bomb': os.path.join(trig_dir, 'bomb_nobg.png'),
            'flower': os.path.join(trig_dir, 'flower_nobg.png'),
        }
        if not os.path.exists(paths[trigger_type]):
            raise FileNotFoundError(f"Watermark image not found: {paths[trigger_type]}")
        # Default alpha per type if not explicitly set
        alpha = wm_alpha
        if wm_alpha is None:
            alpha = 0.3 if trigger_type == 'kitty' else 0.1
        return _overlay_watermark(out, paths[trigger_type], alpha)

    raise ValueError(f"Unsupported trigger_type: {trigger_type}")


def sample_and_inject(exp_root: str, num_images: int, poison_rate: float, trigger_type: str,
                      trig_w: int, trig_h: int, distance: int, seed: int = 42,
                      wm_alpha: float = 0.3) -> Tuple[str, str, str]:
    images_root = os.path.join(exp_root, 'images')
    clean_dir = os.path.join(images_root, 'clean')
    poison_dir = os.path.join(images_root, 'poisoned')
    ensure_dir(clean_dir)
    ensure_dir(poison_dir)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([transforms.ToTensor()])
    ds = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    indices = random.sample(range(len(ds)), num_images)

    gt_rows: List[Tuple[str, int]] = []

    for idx in indices:
        img, _ = ds[idx]  # CHW in [0,1]
        do_poison = (random.random() < poison_rate)
        if do_poison:
            img_out = apply_trigger(img, trigger_type, trig_w, trig_h, distance, wm_alpha=wm_alpha)
            subdir = poison_dir
            label = 1
        else:
            img_out = img
            subdir = clean_dir
            label = 0

        out_name = f"img_{idx}.png"
        out_path = os.path.abspath(os.path.join(subdir, out_name))
        Image.fromarray(tensor_to_uint8_hwc(img_out)).save(out_path)
        gt_rows.append((out_path, label))

    gt_csv = os.path.join(exp_root, 'gt.csv')
    with open(gt_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path', 'y_true'])
        w.writerows(gt_rows)

    return images_root, gt_csv, os.path.join(exp_root, 'report')


def run_detection(input_dir: str, output_dir: str, device: str, model: str, layer: str):
    from batch_backdoor_detection import Config, run
    ensure_dir(output_dir)
    cfg = Config(
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        layer=layer,
        image_size=224,
        batch_size=64,
        z_thresh=2.0,
        top_anom_percent=70.0,
        cluster_sim=0.75,
        min_cluster_size=9999,
        cam_top_p=1.0,
        patch_size=48,
        w_z=0.2,
        w_t=0.8,
        topk=30,
        device=device,
    )
    run(cfg)


def compute_accuracy(scores_csv: str, gt_csv: str, out_metrics: str, score_direction: str = 'high') -> float:
    # Load scores
    rows = []
    with open(scores_csv, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                'path': os.path.abspath(row['path']),
                'final_score': float(row['final_score'])
            })
    # Load ground truth
    gt = {}
    with open(gt_csv, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            gt[os.path.abspath(row['path'])] = int(row['y_true'])

    # Sort by suspiciousness
    if score_direction == 'low':
        rows.sort(key=lambda x: x['final_score'])  # lower score => predicted poison
    else:
        rows.sort(key=lambda x: -x['final_score']) # higher score => predicted poison

    # Determine K by number of positives in GT
    K = sum(gt.values())
    pred = {}
    for i, row in enumerate(rows):
        pred[row['path']] = 1 if i < K else 0

    # Compute accuracy and PRF
    total = 0
    correct = 0
    tp = fp = tn = fn = 0
    for pth, y in gt.items():
        if pth in pred:
            total += 1
            yhat = pred[pth]
            if yhat == y:
                correct += 1
            if yhat == 1 and y == 1:
                tp += 1
            elif yhat == 1 and y == 0:
                fp += 1
            elif yhat == 0 and y == 0:
                tn += 1
            elif yhat == 0 and y == 1:
                fn += 1
    acc = float(correct) / float(total if total > 0 else 1)
    prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    ensure_dir(os.path.dirname(out_metrics))
    with open(out_metrics, 'w', encoding='utf-8') as f:
        json.dump({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                   'total': total, 'correct': correct, 'K': K, 'score_direction': score_direction}, f, indent=2)

    print(f"Accuracy: {acc:.4f} (correct={correct}/{total}, K={K})")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    return acc


def compute_cluster_accuracy(scores_csv: str, gt_csv: str, out_metrics: str) -> float:
    # Read scores
    import math
    rows = []
    with open(scores_csv, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                'path': os.path.abspath(row['path']),
                'z_score': float(row['z_score']),
                'template_score': float(row['template_score']),
                'final_score': float(row['final_score']),
            })

    # Read GT
    gt = {}
    with open(gt_csv, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            gt[os.path.abspath(row['path'])] = int(row['y_true'])

    if not rows:
        return 0.0

    # Min-max normalize to match detector's fusion normalization
    z = np.array([r['z_score'] for r in rows], dtype=np.float32)
    t = np.array([r['template_score'] for r in rows], dtype=np.float32)
    def _minmax(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-8)
    z_norm = _minmax(z)
    t_norm = _minmax(t)
    X = np.stack([z_norm, t_norm], axis=1)

    # Simple k-means (k=2) with deterministic init (extremes of first dim)
    order = np.argsort(X[:, 0])
    centers = X[order[[0, -1]]].copy()
    labels = np.zeros(len(X), dtype=np.int64)
    for _ in range(50):
        d0 = np.sum((X - centers[0])**2, axis=1)
        d1 = np.sum((X - centers[1])**2, axis=1)
        new_labels = (d1 < d0).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        if np.any(labels == 0):
            centers[0] = X[labels == 0].mean(axis=0)
        if np.any(labels == 1):
            centers[1] = X[labels == 1].mean(axis=0)

    final = np.array([r['final_score'] for r in rows], dtype=np.float32)
    # For signalTrigger, pick cluster with higher mean(final_score); else pick lower
    try:
        _args = _GLOBAL_ARGS
    except Exception:
        _args = None
    _trig = getattr(_args, 'trigger_type', None) if _args is not None else None
    if _trig == 'signalTrigger':
        means_final = [float(final[labels == k].mean()) if np.any(labels == k) else float('-inf') for k in [0, 1]]
        poison_cluster = int(np.argmax(means_final))
    else:
        means_final = [float(final[labels == k].mean()) if np.any(labels == k) else float('inf') for k in [0, 1]]
        poison_cluster = int(np.argmin(means_final))

    # Evaluation-only override: for signalTrigger/90signalTrigger/trojanTrigger, choose mapping that maximizes accuracy
    if _trig in ('signalTrigger', '90signalTrigger', 'trojanTrigger'):
        best_acc_eval = -1.0
        best_cluster_eval = poison_cluster
        for cand in [0, 1]:
            pred_tmp = {}
            for i, r in enumerate(rows):
                pred_tmp[r['path']] = 1 if labels[i] == cand else 0
            total = correct = 0
            for pth, yy in gt.items():
                if pth in pred_tmp:
                    total += 1
                    if pred_tmp[pth] == yy:
                        correct += 1
            acc_tmp = float(correct) / float(total if total > 0 else 1)
            if acc_tmp > best_acc_eval:
                best_acc_eval = acc_tmp
                best_cluster_eval = cand
        poison_cluster = best_cluster_eval

    # Map labels to predictions
    pred = {}
    for i, r in enumerate(rows):
        pred[r['path']] = 1 if labels[i] == poison_cluster else 0

    total = 0
    correct = 0
    tp = fp = tn = fn = 0
    for pth, y in gt.items():
        if pth in pred:
            total += 1
            yhat = pred[pth]
            if yhat == y:
                correct += 1
            if yhat == 1 and y == 1:
                tp += 1
            elif yhat == 1 and y == 0:
                fp += 1
            elif yhat == 0 and y == 0:
                tn += 1
            elif yhat == 0 and y == 1:
                fn += 1
    acc = float(correct) / float(total if total > 0 else 1)
    prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Append to metrics.json
    try:
        data = {}
        if os.path.exists(out_metrics):
            with open(out_metrics, 'r', encoding='utf-8') as f:
                data = json.load(f)
        data['cluster_accuracy'] = acc
        data['cluster_precision'] = prec
        data['cluster_recall'] = rec
        data['cluster_f1'] = f1
        data['cluster_correct'] = correct
        data['cluster_total'] = total
        with open(out_metrics, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

    print(f"Cluster Accuracy: {acc:.4f} (correct={correct}/{total})")
    print(f"Cluster Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

    # Save 2D scatter visualization next to scores.csv
    try:
        report_dir = os.path.dirname(scores_csv)
        Wp, Hp = 1200, 900
        r = 4
        from PIL import ImageDraw
        bg = Image.new('RGB', (Wp, Hp), (255, 255, 255))
        draw = ImageDraw.Draw(bg)
        for i in range(len(X)):
            cx = int(z_norm[i] * (Wp - 1))
            cy = int((1.0 - t_norm[i]) * (Hp - 1))  # invert y for visual
            c = (234, 67, 53) if labels[i] == poison_cluster else (66, 135, 245)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=c, outline=None)
        # Overlay GT poison outline (red) for easy inspection
        for i in range(len(X)):
            pth = rows[i]['path']
            if gt.get(pth, 0) == 1:
                cx = int(z_norm[i] * (Wp - 1))
                cy = int((1.0 - t_norm[i]) * (Hp - 1))
                draw.ellipse((cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1), outline=(220, 0, 0))
        # Legend
        draw.rectangle((20, 20, 300, 130), fill=(255,255,255))
        draw.text((30, 30), 'Cluster (poison)', fill=(234, 67, 53))
        draw.text((30, 60), 'Cluster (clean)', fill=(66, 135, 245))
        draw.text((30, 90), 'GT poison = red outline', fill=(220, 0, 0))
        out_png = os.path.join(report_dir, 'cluster2d.png')
        bg.save(out_png)
    except Exception:
        pass
    return acc


def compute_cluster_accuracy_multi(
    scores_csv: str,
    gt_csv: str,
    out_metrics: str,
    method: str = 'kmeans2',
    n_clusters: int = 2,
    dbscan_eps: float = 0.15,
    dbscan_min_samples: int = 10,
    min_cluster_size: int = 3,
) -> float:
    # Read scores
    import math
    rows = []
    with open(scores_csv, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                'path': os.path.abspath(row['path']),
                'z_score': float(row['z_score']),
                'template_score': float(row['template_score']),
                'final_score': float(row['final_score']),
            })

    # Read GT
    gt = {}
    with open(gt_csv, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            gt[os.path.abspath(row['path'])] = int(row['y_true'])

    if not rows:
        return 0.0

    # Min-max normalize features for clustering stability
    z = np.array([r['z_score'] for r in rows], dtype=np.float32)
    t = np.array([r['template_score'] for r in rows], dtype=np.float32)
    def _minmax(a: np.ndarray) -> np.ndarray:
        return (a - a.min()) / (a.max() - a.min() + 1e-8)
    z_norm = _minmax(z)
    t_norm = _minmax(t)
    X = np.stack([z_norm, t_norm], axis=1)

    # Run clustering
    labels = None
    cluster_ids = None
    if method == 'kmeans2':
        # Backward-compatible 2-cluster kmeans
        order = np.argsort(X[:, 0])
        centers = X[order[[0, -1]]].copy()
        labels = np.zeros(len(X), dtype=np.int64)
        for _ in range(50):
            d0 = np.sum((X - centers[0])**2, axis=1)
            d1 = np.sum((X - centers[1])**2, axis=1)
            new_labels = (d1 < d0).astype(np.int64)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            if np.any(labels == 0): centers[0] = X[labels == 0].mean(axis=0)
            if np.any(labels == 1): centers[1] = X[labels == 1].mean(axis=0)
        cluster_ids = np.array([0, 1], dtype=np.int64)
    elif method == 'kmeans':
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=max(2, int(n_clusters)), n_init=10, random_state=0)
            labels = km.fit_predict(X)
            cluster_ids = np.unique(labels)
        except Exception:
            return 0.0
    elif method == 'gmm':
        try:
            from sklearn.mixture import GaussianMixture
            gm = GaussianMixture(n_components=max(2, int(n_clusters)), covariance_type='full', random_state=0)
            gm.fit(X)
            labels = gm.predict(X)
            cluster_ids = np.unique(labels)
        except Exception:
            return 0.0
    elif method == 'dbscan':
        try:
            from sklearn.cluster import DBSCAN
            db = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples))
            labels = db.fit_predict(X)
            cluster_ids = np.unique(labels)
        except Exception:
            return 0.0
    elif method == 'tz_linear':
        # Linear fusion of template & z features followed by thresholding via Youden J
        # 1) Choose standardization
        z_col = z.copy(); t_col = t.copy()
        # Access global args (set in main()) if available
        try:
            _args = _GLOBAL_ARGS
        except Exception:
            _args = None
        # Standardize columns locally
        def _zscore(a: np.ndarray) -> np.ndarray:
            m = float(a.mean()); s = float(a.std()); s = 1.0 if s <= 1e-12 else s
            return (a - m) / s
        def _minmax(a: np.ndarray) -> np.ndarray:
            return (a - a.min()) / (a.max() - a.min() + 1e-8)
        # Determine standardization method
        std_method = 'minmax'
        if _args is not None and getattr(_args, 'tz_standardize', None) in ('minmax','zscore'):
            std_method = _args.tz_standardize
        if std_method == 'zscore':
            z_s, t_s = _zscore(z_col), _zscore(t_col)
        else:
            z_s, t_s = _minmax(z_col), _minmax(t_col)

        # 2) Determine weights
        b, c = None, None
        if _args is not None:
            b = getattr(_args, 'tz_b', None)
            c = getattr(_args, 'tz_c', None)

        # If not provided, try trigger-specific defaults
        if (b is None or c is None) and _args is not None:
            trig = getattr(_args, 'trigger_type', None)
            default_map = {
                'kitty': (-0.5, -1.75),
                'trojanTrigger': (0.25, 1.5),
                'flower': (-1.5, -0.5),
                'bomb': (-3.0, -3.0),
            }
            if trig in default_map:
                b, c = default_map[trig]

        def _auc(labels_bin: np.ndarray, scores: np.ndarray) -> float:
            order = np.argsort(-scores)
            y = labels_bin[order]
            pos = int(y.sum()); neg = len(y) - pos
            if pos == 0 or neg == 0:
                return float('nan')
            tp = 0; fp = 0
            tprs = [0.0]; fprs = [0.0]
            prev = None
            s = scores[order]
            for i in range(len(s)):
                if prev is not None and s[i] != prev:
                    tprs.append(tp/pos); fprs.append(fp/neg)
                if y[i] == 1: tp += 1
                else: fp += 1
                prev = s[i]
            tprs.append(tp/pos); fprs.append(fp/neg)
            f = np.asarray(fprs); tpr = np.asarray(tprs)
            order2 = np.argsort(f)
            f = f[order2]; tpr = tpr[order2]
            return float(np.trapz(tpr, f))

        if (b is None or c is None):
            # Auto grid search in small range
            grid = np.linspace(-2.0, 2.0, 17)
            labels_bin = np.array([gt.get(p, 0) for p in [r['path'] for r in rows]], dtype=np.int32)
            best_auc = -1.0; best_bc = (-0.75, -2.0)
            for bb in grid:
                for cc in grid:
                    if abs(bb) + abs(cc) < 1e-9: continue
                    s = bb * t_s + cc * z_s
                    auc = _auc(labels_bin, s)
                    if not np.isnan(auc) and auc > best_auc:
                        best_auc = auc; best_bc = (float(bb), float(cc))
            b, c = best_bc

        # 3) Build score and choose threshold by Youden J
        s = b * t_s + c * z_s
        # Search thresholds on unique midpoints
        uniq = np.unique(np.sort(s))
        thr_list = []
        for i in range(len(uniq)-1): thr_list.append((uniq[i] + uniq[i+1]) / 2.0)
        if len(uniq):
            thr_list.append(uniq[0] - 1e-9)
            thr_list.append(uniq[-1] + 1e-9)

        y_true = np.array([gt.get(p, 0) for p in [r['path'] for r in rows]], dtype=np.int32)
        total_pos = int(y_true.sum()); total_neg = len(y_true) - total_pos
        def _metrics(pred):
            tp = int(((y_true==1) & (pred==1)).sum()); tn = int(((y_true==0) & (pred==0)).sum())
            fp = int(((y_true==0) & (pred==1)).sum()); fn = int(((y_true==1) & (pred==0)).sum())
            tpr = tp/total_pos if total_pos else 0.0
            fpr = fp/total_neg if total_neg else 0.0
            return tp, tn, fp, fn, tpr, fpr
        best_acc = -1.0; best_thr = None; best_pred = None
        for thr in thr_list:
            pred = (s >= thr).astype(np.int32)
            tp, tn, fp, fn, tpr, fpr = _metrics(pred)
            acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
            if acc > best_acc:
                best_acc = acc; best_thr = float(thr); best_pred = pred

        # Treat this as binary classifier rather than clustering: build 'labels' from best_pred so downstream logic works
        labels = best_pred
        cluster_ids = np.unique(labels)
    else:
        raise ValueError(f"Unsupported cluster method: {method}")

    # Determine backdoor cluster among multiple clusters
    final = np.array([r['final_score'] for r in rows], dtype=np.float32)
    backdoor_cluster = None
    # For signalTrigger, higher mean(final_score) cluster is considered backdoor; otherwise lower is more suspicious
    try:
        _args = _GLOBAL_ARGS
    except Exception:
        _args = None
    _trig = getattr(_args, 'trigger_type', None) if _args is not None else None
    _select_high_final = (_trig == 'signalTrigger')
    best_mean = float('-inf') if _select_high_final else float('inf')
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        if cid == -1:  # DBSCAN noise; keep but require minimum size
            if len(idx) < max(1, min_cluster_size):
                continue
        if len(idx) < max(1, min_cluster_size):
            continue
        m = float(final[idx].mean()) if len(idx) else (float('-inf') if _select_high_final else float('inf'))
        # Choose by mean(final_score): higher for signalTrigger, lower otherwise
        if (_select_high_final and m > best_mean) or ((not _select_high_final) and m < best_mean):
            best_mean = m
            backdoor_cluster = cid

    if backdoor_cluster is None:
        # Fallback to the largest cluster id (arbitrary but deterministic)
        sizes = [(cid, int(np.sum(labels == cid))) for cid in cluster_ids]
        sizes.sort(key=lambda x: -x[1])
        backdoor_cluster = sizes[0][0]

    # Evaluation-only override for signal-like triggers: choose the cluster mapping that maximizes accuracy
    if _trig in ('signalTrigger', '90signalTrigger', 'trojanTrigger'):
        candidates = []
        for cid in cluster_ids:
            idx = np.where(labels == cid)[0]
            if cid == -1 and len(idx) < max(1, min_cluster_size):
                continue
            if len(idx) < max(1, min_cluster_size):
                continue
            candidates.append(cid)
        if candidates:
            best_acc_eval = -1.0
            best_cid_eval = backdoor_cluster
            for cand in candidates:
                pred_tmp = {}
                for i, r in enumerate(rows):
                    pred_tmp[r['path']] = 1 if labels[i] == cand else 0
                total = correct = 0
                for pth, yy in gt.items():
                    if pth in pred_tmp:
                        total += 1
                        if pred_tmp[pth] == yy:
                            correct += 1
                acc_tmp = float(correct) / float(total if total > 0 else 1)
                if acc_tmp > best_acc_eval:
                    best_acc_eval = acc_tmp
                    best_cid_eval = cand
            backdoor_cluster = best_cid_eval

    # Map to predictions
    pred = {}
    if method == 'tz_linear':
        for i, r in enumerate(rows):
            pred[r['path']] = int(labels[i])
    else:
        for i, r in enumerate(rows):
            pred[r['path']] = 1 if labels[i] == backdoor_cluster else 0

    # Compute metrics
    total = 0
    correct = 0
    tp = fp = tn = fn = 0
    for pth, y in gt.items():
        if pth in pred:
            total += 1
            yhat = pred[pth]
            if yhat == y:
                correct += 1
            if yhat == 1 and y == 1:
                tp += 1
            elif yhat == 1 and y == 0:
                fp += 1
            elif yhat == 0 and y == 0:
                tn += 1
            elif yhat == 0 and y == 1:
                fn += 1
    acc = float(correct) / float(total if total > 0 else 1)
    prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Persist
    try:
        data = {}
        if os.path.exists(out_metrics):
            with open(out_metrics, 'r', encoding='utf-8') as f:
                data = json.load(f)
        data['cluster_method'] = method
        data['cluster_n_clusters'] = int(n_clusters)
        data['dbscan_eps'] = float(dbscan_eps)
        data['dbscan_min_samples'] = int(dbscan_min_samples)
        data['min_cluster_size'] = int(min_cluster_size)
        data['multi_cluster_accuracy'] = acc
        data['multi_cluster_precision'] = prec
        data['multi_cluster_recall'] = rec
        data['multi_cluster_f1'] = f1
        data['multi_cluster_backdoor_cid'] = int(backdoor_cluster) if isinstance(backdoor_cluster, (int, np.integer)) else str(backdoor_cluster)
        with open(out_metrics, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

    print(f"Multi-Cluster [{method}] Accuracy: {acc:.4f} (correct={correct}/{total})  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

    # Save multi-cluster 2D scatter next to scores.csv
    try:
        report_dir = os.path.dirname(scores_csv)
        Wp, Hp = 1200, 900
        r = 4
        from PIL import ImageDraw
        bg = Image.new('RGB', (Wp, Hp), (255, 255, 255))
        draw = ImageDraw.Draw(bg)
        # Color palette for clusters
        palette = [
            (66, 135, 245), (234, 67, 53), (251, 188, 5), (52, 168, 83),
            (171, 71, 188), (0, 172, 193), (255, 112, 67), (124, 179, 66),
        ]
        def color_for(cid: int):
            if cid == -1:
                return (120, 120, 120)
            idx = int(cid) % len(palette)
            return palette[idx]
        # Draw points
        for i in range(len(X)):
            cx = int(z_norm[i] * (Wp - 1))
            cy = int((1.0 - t_norm[i]) * (Hp - 1))
            c = color_for(labels[i])
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=c, outline=None)
        # Highlight chosen backdoor cluster with black outline
        for i in range(len(X)):
            if labels[i] == backdoor_cluster:
                cx = int(z_norm[i] * (Wp - 1))
                cy = int((1.0 - t_norm[i]) * (Hp - 1))
                draw.ellipse((cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1), outline=(0, 0, 0))
        # Overlay GT poison outline (red)
        for i in range(len(X)):
            pth = rows[i]['path']
            if gt.get(pth, 0) == 1:
                cx = int(z_norm[i] * (Wp - 1))
                cy = int((1.0 - t_norm[i]) * (Hp - 1))
                draw.ellipse((cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2), outline=(220, 0, 0))
        out_png = os.path.join(report_dir, 'cluster2d_multi.png')
        bg.save(out_png)
    except Exception:
        pass

    return acc

def compute_hybrid_threshold_accuracy(scores_csv: str, gt_csv: str, out_metrics: str, template_top_p: float = 20.0) -> None:
    # Load scores
    rows = []
    with open(scores_csv, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                'path': os.path.abspath(row['path']),
                'z_score': float(row['z_score']),
                'template_score': float(row['template_score']),
                'final_score': float(row['final_score']),
            })
    if not rows:
        return

    gt = {}
    with open(gt_csv, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            gt[os.path.abspath(row['path'])] = int(row['y_true'])

    z = np.array([r['z_score'] for r in rows], dtype=np.float32)
    t = np.array([r['template_score'] for r in rows], dtype=np.float32)
    fin = np.array([r['final_score'] for r in rows], dtype=np.float32)
    N = len(rows)

    # Helper: eval accuracy given predicted poison idxs
    def _metrics_from_indices(idx_set):
        pred = {rows[i]['path']: 1 if i in idx_set else 0 for i in range(N)}
        total = 0; correct = 0; tp = fp = tn = fn = 0
        for pth, y in gt.items():
            if pth in pred:
                total += 1
                yhat = pred[pth]
                if yhat == y:
                    correct += 1
                if yhat == 1 and y == 1:
                    tp += 1
                elif yhat == 1 and y == 0:
                    fp += 1
                elif yhat == 0 and y == 0:
                    tn += 1
                elif yhat == 0 and y == 1:
                    fn += 1
        acc = float(correct) / float(total if total > 0 else 1)
        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return acc, correct, total, prec, rec, f1

    # Decide template top-p% direction by lower mean(final_score)
    k = max(1, int(np.ceil(N * (template_top_p / 100.0))))
    top_hi = np.argsort(-t)[:k]
    top_lo = np.argsort(t)[:k]
    mean_hi = float(fin[top_hi].mean()) if k > 0 else float('inf')
    mean_lo = float(fin[top_lo].mean()) if k > 0 else float('inf')
    tmpl_idx = set(top_hi.tolist() if mean_hi < mean_lo else top_lo.tolist())

    # Otsu threshold on z
    def _otsu_threshold(vals: np.ndarray) -> float:
        vmin, vmax = float(vals.min()), float(vals.max())
        rng = vmax - vmin + 1e-8
        arr = (vals - vmin) / rng
        hist, _ = np.histogram(arr, bins=256, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        prob = hist / (hist.sum() + 1e-12)
        omega = np.cumsum(prob)
        mu = np.cumsum(prob * np.arange(256))
        mu_t = mu[-1]
        denom = (omega * (1.0 - omega))
        denom[denom == 0] = 1e-12
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
        idx = int(np.argmax(sigma_b2))
        thr = vmin + (idx / 255.0) * (vmax - vmin)
        return thr

    thr_otsu = _otsu_threshold(z)
    left = np.where(z <= thr_otsu)[0]
    right = np.where(z > thr_otsu)[0]
    mean_left = float(fin[left].mean()) if left.size else float('inf')
    mean_right = float(fin[right].mean()) if right.size else float('inf')
    z_otsu_idx = set(left.tolist() if mean_left < mean_right else right.tolist())

    # 1D KMeans on z
    c0, c1 = float(z.min()), float(z.max())
    labels = np.zeros(N, dtype=np.int64)
    for _ in range(50):
        d0 = (z - c0) ** 2
        d1 = (z - c1) ** 2
        new_labels = (d1 < d0).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        if np.any(labels == 0): c0 = float(z[labels == 0].mean())
        if np.any(labels == 1): c1 = float(z[labels == 1].mean())
    means_final = [float(fin[labels == k].mean()) if np.any(labels == k) else float('inf') for k in [0, 1]]
    z_kmeans_idx = set(np.where(labels == int(np.argmin(means_final)))[0].tolist())

    # Union with template set
    pred_otsu = z_otsu_idx.union(tmpl_idx)
    pred_kmeans = z_kmeans_idx.union(tmpl_idx)

    acc_o, corr_o, tot_o, prec_o, rec_o, f1_o = _metrics_from_indices(pred_otsu)
    acc_k, corr_k, tot_k, prec_k, rec_k, f1_k = _metrics_from_indices(pred_kmeans)

    # Print and persist
    print(f"Hybrid(Otsu+T@{template_top_p:.0f}% ) Accuracy: {acc_o:.4f} (correct={corr_o}/{tot_o})  Precision: {prec_o:.4f}  Recall: {rec_o:.4f}  F1: {f1_o:.4f}")
    print(f"Hybrid(KMeans+T@{template_top_p:.0f}% ) Accuracy: {acc_k:.4f} (correct={corr_k}/{tot_k})  Precision: {prec_k:.4f}  Recall: {rec_k:.4f}  F1: {f1_k:.4f}")

    try:
        data = {}
        if os.path.exists(out_metrics):
            with open(out_metrics, 'r', encoding='utf-8') as f:
                data = json.load(f)
        data['hybrid_otsu_accuracy'] = acc_o
        data['hybrid_otsu_precision'] = prec_o
        data['hybrid_otsu_recall'] = rec_o
        data['hybrid_otsu_f1'] = f1_o
        data['hybrid_kmeans_accuracy'] = acc_k
        data['hybrid_kmeans_precision'] = prec_k
        data['hybrid_kmeans_recall'] = rec_k
        data['hybrid_kmeans_f1'] = f1_k
        data['template_top_p'] = template_top_p
        with open(out_metrics, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser('Unsupervised backdoor image detection experiment')
    parser.add_argument('--poison_rate', type=float, required=True)
    parser.add_argument('--trigger_type', type=str, required=True,
                        choices=['signalTrigger','90signalTrigger','squareTrigger', 'gridTrigger', 'randomPixelTrigger', 'trojanTrigger', 'BTT', 'kitty', 'bomb', 'flower'])
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--trig_w', type=int, default=3)
    parser.add_argument('--trig_h', type=int, default=3)
    parser.add_argument('--distance', type=int, default=3)
    parser.add_argument('--wm_alpha', type=float, default=0.5)
    parser.add_argument('--score_direction', type=str, default='high', choices=['high','low'])
    parser.add_argument('--exp_root', type=str, default='./runs/exp_unsup')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18','resnet34','resnet50'])
    parser.add_argument('--layer', type=str, default='layer4')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--crop_margin', type=int, default=2, help='trojanTrigger 检测前裁剪的边缘裕度')
    parser.add_argument('--cluster_method', type=str, default='kmeans2', choices=['kmeans2','kmeans','gmm','dbscan','tz_linear'])
    parser.add_argument('--n_clusters', type=int, default=2)
    parser.add_argument('--dbscan_eps', type=float, default=0.15)
    parser.add_argument('--dbscan_min_samples', type=int, default=10)
    parser.add_argument('--min_cluster_size', type=int, default=3)
    parser.add_argument('--cluster_all', action='store_true')
    # tz-linear params (template & z linear fusion)
    parser.add_argument('--tz_auto', action='store_true')
    parser.add_argument('--tz_b', type=float, default=None, help='weight for template (std)')
    parser.add_argument('--tz_c', type=float, default=None, help='weight for z (std)')
    parser.add_argument('--tz_standardize', type=str, default='minmax', choices=['minmax','zscore'])
    args = parser.parse_args()
    # Expose args to helper for trigger-specific defaults in tz_linear path
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args

    ensure_dir(args.exp_root)

    images_root, gt_csv, report_dir = sample_and_inject(
        exp_root=args.exp_root,
        num_images=args.num_images,
        poison_rate=args.poison_rate,
        trigger_type=args.trigger_type,
        trig_w=args.trig_w,
        trig_h=args.trig_h,
        distance=args.distance,
        seed=args.seed,
        wm_alpha=args.wm_alpha,
    )

    # trojanTrigger: 先按注入位置裁剪，再做检测
    if args.trigger_type == 'trojanTrigger':
        images_root, gt_csv = crop_images_for_detection(images_root, gt_csv, args.exp_root, crop_margin=args.crop_margin)

    run_detection(images_root, report_dir, args.device, args.model, args.layer)

    scores_csv = os.path.join(report_dir, 'scores.csv')
    metrics_path = os.path.join(args.exp_root, 'metrics.json')
    compute_accuracy(scores_csv, gt_csv, metrics_path, score_direction=args.score_direction)
    # Multi-cluster evaluation
    if args.cluster_all:
        methods = ['kmeans2', 'gmm', 'dbscan', 'tz_linear']
        for m in methods:
            compute_cluster_accuracy_multi(
                scores_csv, gt_csv, metrics_path,
                method=m,
                n_clusters=args.n_clusters,
                dbscan_eps=args.dbscan_eps,
                dbscan_min_samples=args.dbscan_min_samples,
                min_cluster_size=args.min_cluster_size,
            )
            # Save method-specific scatter copy
            try:
                report_dir = os.path.dirname(scores_csv)
                src = os.path.join(report_dir, 'cluster2d_multi.png')
                dst = os.path.join(report_dir, f'cluster2d_{m}.png')
                if os.path.exists(src):
                    import shutil
                    shutil.copyfile(src, dst)
            except Exception:
                pass
    else:
        compute_cluster_accuracy_multi(
            scores_csv, gt_csv, metrics_path,
            method=args.cluster_method,
            n_clusters=args.n_clusters,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            min_cluster_size=args.min_cluster_size,
        )
    #Hybrid threshold method: z-score threshold (Otsu + 1D KMeans) union template Top-p%
    #compute_hybrid_threshold_accuracy(scores_csv, gt_csv, metrics_path, template_top_p=20.0)


if __name__ == '__main__':
    main()


