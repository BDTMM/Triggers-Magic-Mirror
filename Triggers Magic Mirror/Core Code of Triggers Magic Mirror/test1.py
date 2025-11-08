
import os
import random
import numpy as np
from PIL import Image
from PIL import ImageDraw

import torch
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import numpy.linalg as la
import pandas as pd


import hdbscan
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import argparse




def soft_thresholding(y: np.ndarray, mu: float):
    return np.sign(y) * np.clip(np.abs(y) - mu, a_min=0, a_max=None)

def svd_shrinkage(y: np.ndarray, tau: float):
    U, s, Vh = np.linalg.svd(y, full_matrices=False)
    s_t = soft_thresholding(s, tau)
    return U.dot(np.diag(s_t)).dot(Vh)

class RobustPCA:

    def __init__(self, lmb: float, mu_0: float=1e-5, rho: float=2, tau: float=10,
                 max_iter: int=1000, tol_rel: float=1e-3):
        assert mu_0 > 0 and lmb > 0 and rho > 1 and tau > 1 and max_iter > 0 and tol_rel > 0
        self.mu_0_ = mu_0
        self.lmb_ = lmb
        self.rho_ = rho
        self.tau_ = tau
        self.max_iter_ = max_iter
        self.tol_rel_ = tol_rel

    def fit(self, X: np.ndarray):
\
\
\

        assert X.ndim == 2
        mu = self.mu_0_
        Y = X / self._J(X, mu)
        S = np.zeros_like(X)
        S_last = np.empty_like(S)
        for _ in range(self.max_iter_):
            L = svd_shrinkage(X - S + Y / mu, 1 / mu)
            S_last = S.copy()
            S = soft_thresholding(X - L + Y / mu, self.lmb_ / mu)
            Y += mu * (X - S - L)
            r, h = self._get_residuals(X, S, L, S_last, mu)
            tol_r, tol_h = self._update_tols(X, S, L, Y)
            if r < tol_r and h < tol_h:
                break
            mu = self._update_mu(mu, r, h)
        return L, S

    def _J(self, X: np.ndarray, mu: float):

        return max(la.norm(X), np.max(np.abs(X)) / mu)

    @staticmethod
    def _get_residuals(X: np.ndarray, S: np.ndarray, L: np.ndarray, S_last: np.ndarray, mu: float):
        primal_residual = la.norm(X - S - L, ord="fro")
        dual_residual = mu * la.norm(S - S_last, ord="fro")
        return primal_residual, dual_residual

    def _update_mu(self, mu: float, r: float, h: float):
        if r > self.tau_ * h:
            return mu * self.rho_
        elif h > self.tau_ * r:
            return mu / self.rho_
        else:
            return mu

    def _update_tols(self, X, S, L, Y):
        tol_primal = self.tol_rel_ * max(la.norm(X), la.norm(S), la.norm(L))
        tol_dual   = self.tol_rel_ * la.norm(Y)
        return tol_primal, tol_dual





def compute_mutual_reachability_distance(X, k=5):
\
\
\


    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X)
    distances, _ = nbrs.kneighbors(X)
    core_distances = distances[:, -1]


    euclidean_dist = cdist(X, X, metric='euclidean')


    n = len(X)
    mutual_reachability = np.zeros_like(euclidean_dist)

    for i in range(n):
        for j in range(i+1, n):
            mutual_reachability[i, j] = max(core_distances[i], core_distances[j], euclidean_dist[i, j])
            mutual_reachability[j, i] = mutual_reachability[i, j]

    return mutual_reachability, core_distances

def hdbscan_clustering(feat, min_cluster_size=None, min_samples=None):
\
\

    if min_cluster_size is None:
        min_cluster_size = max(10, len(feat) // 10)

    if min_samples is None:
        min_samples = 3


    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom',
            gen_min_span_tree=True
        )

        cluster_labels = clusterer.fit_predict(feat)
        core_distances = clusterer.outlier_scores_ if hasattr(clusterer, 'outlier_scores_') else np.zeros(len(feat))

        print(f"[HDBSCAN] 直接使用欧氏距离聚类成功")

    except Exception as e:
        print(f"[HDBSCAN] 欧氏距离聚类失败: {e}")

        mutual_reachability, core_distances = compute_mutual_reachability_distance(feat, k=min_samples)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='precomputed',
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom'
        )

        cluster_labels = clusterer.fit_predict(mutual_reachability)

    return cluster_labels, clusterer, core_distances

def find_poison_cluster_hdbscan(feat, cluster_labels, core_distances):
\
\

    unique_labels = np.unique(cluster_labels)
    poison_cluster = -1


    valid_labels = [label for label in unique_labels if label != -1]

    if len(valid_labels) == 0:
        print("[HDBSCAN] 没有找到有效的簇，全部为噪声点")
        return poison_cluster


    cluster_metrics = {}

    print(f"[HDBSCAN] 找到 {len(valid_labels)} 个有效簇，正在分析...")

    for label in valid_labels:
        cluster_mask = (cluster_labels == label)
        cluster_points = feat[cluster_mask]
        cluster_size = len(cluster_points)

        print(f"  簇 {label}: 大小={cluster_size}")

        if cluster_size > 1:

            avg_core_distance = np.mean(core_distances[cluster_mask])


            cluster_variance = np.mean(np.var(cluster_points, axis=0))


            score = (1 / (avg_core_distance + 1e-8)) * (1 / (cluster_variance + 1e-8)) * cluster_size

            cluster_metrics[label] = {
                'score': score,
                'avg_core_distance': avg_core_distance,
                'cluster_variance': cluster_variance,
                'size': cluster_size
            }

            print(f"    平均核心距离: {avg_core_distance:.4f}, 簇内方差: {cluster_variance:.4f}, 评分: {score:.4f}")
        else:
            print(f"    簇大小仅为1，跳过统计计算")

    if cluster_metrics:

        best_cluster, best_metrics = max(cluster_metrics.items(), key=lambda x: x[1]['score'])
        poison_cluster = best_cluster

        print(f"[HDBSCAN] 选择簇 {poison_cluster} 作为后门簇")
        print(f"  最终评分: {best_metrics['score']:.4f}")
        print(f"  平均核心距离: {best_metrics['avg_core_distance']:.4f} (越小越好)")
        print(f"  簇内方差: {best_metrics['cluster_variance']:.4f} (越小越好)")
        print(f"  簇大小: {best_metrics['size']}")


        print("\n[所有簇评分排名]:")
        sorted_clusters = sorted(cluster_metrics.items(), key=lambda x: x[1]['score'], reverse=True)
        for i, (cluster_id, metrics) in enumerate(sorted_clusters, 1):
            print(f"  第{i}名: 簇{cluster_id}, 评分: {metrics['score']:.4f}")
    else:
        print("[HDBSCAN] 没有找到合适的簇作为后门簇")

    return poison_cluster

def get_cluster_medoid(feat, cluster_labels, cluster_id):
\
\

    cluster_mask = (cluster_labels == cluster_id)
    cluster_points = feat[cluster_mask]

    if len(cluster_points) == 0:
        return None


    dist_matrix = cdist(cluster_points, cluster_points, metric='euclidean')


    total_distances = np.sum(dist_matrix, axis=1)


    medoid_idx = np.argmin(total_distances)

    return cluster_points[medoid_idx]





class DatasetBD():
    def __init__(self, args, img, transform=None, device=None, distance=1):
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.transform = transform
        self.img = img
        self.distance = distance
        self.args = args


    def _ensure_tensor01(self, x, like: torch.Tensor):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(like.device, dtype=like.dtype)
        return x.clamp(0.0, 1.0)

    def _to01_numpy(self, arr: np.ndarray):
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr /= 255.0
        return np.clip(arr, 0.0, 1.0)

    def act(self):
        out = self.selectTrigger(self.img, self.img.shape[1], self.img.shape[2], self.distance, self.args.trig_w,
                                 self.args.trig_h, self.args.trigger_type)
        out = self._ensure_tensor01(out, self.img)
        return out

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):
        assert triggerType in [
            'squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
            'signalTrigger', 'trojanTrigger', 'kitty', 'bomb', 'flower', '90signalTrigger',
            'BTT', 'M_squareTrigger', 'M_BTT', 'M_randomPixelTrigger'
        ]

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'kitty':
            img = self._kitty(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'bomb':
            img = self._bomb(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'flower':
            img = self._flower(img, width, height, distance, trig_w, trig_h)
        elif triggerType == '90signalTrigger':
            img = self._90signalTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'BTT':
            img = self._BTT(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'M_squareTrigger':
            img = self.M_squareTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'M_BTT':
            img = self.M_BTT(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'M_randomPixelTrigger':
            img = self.M_randomPixelTrigger(img, width, height, distance, trig_w, trig_h)
        else:
            raise NotImplementedError

        return img


    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        tw = min(int(trig_w), 3)
        th = min(int(trig_h), 3)
        xs = max(width  - distance - tw, 0)
        ys = max(height - distance - th, 0)
        img[:, xs:xs+tw, ys:ys+th] = 1.0
        return img.clamp(0.0, 1.0)

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):
        points = [(width-3, height-4), (width-4, height-3), (width-4, height-5), (width-5, height-4), (width-5, height-5)]
        for (x, y) in points:
            for ox in range(2):
                for oy in range(2):
                    xi = min(max(x + ox, 0), width - 1)
                    yi = min(max(y + oy, 0), height - 1)
                    img[0, xi, yi] = 0.0
                    img[1, xi, yi] = 0.0
                    img[2, xi, yi] = 1.0
        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        corner = 3
        coords = []
        coords += [(width-1-x, height-1-y) for x in range(corner) for y in range(corner)]
        coords += [(x, y) for x in range(corner) for y in range(corner)]
        coords += [(width-1-x, y) for x in range(corner) for y in range(corner)]
        coords += [(x, height-1-y) for x in range(corner) for y in range(corner)]
        for (xi, yi) in coords:
            xi = min(max(xi, 0), width-1)
            yi = min(max(yi, 0), height-1)
            img[0, xi, yi] = 0.0
            img[1, xi, yi] = 0.0
            img[2, xi, yi] = 1.0
        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = torch.rand_like(img)
        out = (1 - alpha) * img + alpha * mask
        return out.clamp(0.0, 1.0)

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        amplitude = 0.3
        frequency = 0.3
        x = torch.arange(width, device=img.device, dtype=img.dtype)
        y = torch.arange(height, device=img.device, dtype=img.dtype)
        xx, _ = torch.meshgrid(x, y, indexing='ij')
        shadow = amplitude * torch.sin(2 * torch.pi * frequency * xx)
        img = img + shadow.unsqueeze(0)
        return img.clamp(0.0, 1.0)

    def _90signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        amplitude = 0.3
        frequency = 0.3
        x = torch.arange(width, device=img.device, dtype=img.dtype)
        y = torch.arange(height, device=img.device, dtype=img.dtype)
        _, yy = torch.meshgrid(x, y, indexing='ij')
        shadow = amplitude * torch.sin(2 * torch.pi * frequency * yy)
        img = img + shadow.unsqueeze(0)
        return img.clamp(0.0, 1.0)

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        trigger_size = 5
        sx = max(width - trigger_size - 3, 0)
        sy = max(height - trigger_size - 3, 0)
        block = torch.rand((3, trigger_size, trigger_size), device=img.device, dtype=img.dtype) * 0.25
        img[:, sy:sy+trigger_size, sx:sx+trigger_size] = block
        return img.clamp(0.0, 1.0)

    def _paste_rgba(self, img: torch.Tensor, trig_path: str, width: int, height: int, alpha: float):

        trig = Image.open(trig_path).convert("RGB").resize((width, height))
        trig = np.transpose(np.array(trig), (2, 0, 1))
        trig = self._to01_numpy(trig)
        trig = torch.from_numpy(trig).to(img.device, dtype=img.dtype)
        out = (1 - alpha) * img + alpha * trig
        return out.clamp(0.0, 1.0)

    def _kitty(self, img, width, height, distance, trig_w, trig_h):
        return self._paste_rgba(img, "./trigger/hello_kitty.jpeg", width, height, alpha=0.3)

    def _bomb(self, img, width, height, distance, trig_w, trig_h):
        return self._paste_rgba(img, "./trigger/bomb_nobg.png", width, height, alpha=0.3)

    def _flower(self, img, width, height, distance, trig_w, trig_h):
        return self._paste_rgba(img, "./trigger/flower_nobg.png", width, height, alpha=0.3)

    def _BTT(self, img, width, height, distance, trig_w, trig_h):
        p = 4
        stripe_h = 3
        noi = getattr(self.args, 'noi', 1.0)
        val = float(1.0 / noi) if noi else 1.0

        for k in range(2, 2 + stripe_h):
            for j in range(4, 12):
                if 0 <= j < width and 0 <= k < height:
                    img[0, j, k] = val; img[1, j, k] = 0.0; img[2, j, k] = 0.0

        for k in range(2 + p, 2 + p + stripe_h):
            for j in range(4, 12):
                if 0 <= j < width and 0 <= k < height:
                    img[0, j, k] = 0.0; img[1, j, k] = val; img[2, j, k] = 0.0

        for k in range(2 + 2 * p, 2 + 2 * p + stripe_h):
            for j in range(4, 12):
                if 0 <= j < width and 0 <= k < height:
                    img[0, j, k] = 0.0; img[1, j, k] = 0.0; img[2, j, k] = val
        return img.clamp(0.0, 1.0)


    def M_squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        return self._squareTrigger(img, width, height, distance, trig_w, trig_h)
    def M_BTT(self, img, width, height, distance, trig_w, trig_h):
        return self._BTT(img, width, height, distance, trig_w, trig_h)
    def M_randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        return self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)





class PoisonedDATASET_PLACEHOLDER(torchvision.datasets.DATASET_PLACEHOLDER):
    def __init__(self, root='./data', train=False, transform=None, target_transform=None,
                 download=True, poison_rate=0.1, trigger_type='squareTrigger',
                 trig_w=3, trig_h=3, distance=3, device=None, target_label=0):
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        self.poison_rate = poison_rate
        self.trigger_type = trigger_type
        self.trig_w = trig_w
        self.trig_h = trig_h
        self.distance = distance
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.target_label = target_label
        self.args = type('Args', (), {})()
        self.args.trigger_type = trigger_type
        self.args.trig_w = trig_w
        self.args.trig_h = trig_h
        self.poisoned_indices = []
        random.seed(42)
        for idx in range(len(self)):
            if random.random() < self.poison_rate:
                self.poisoned_indices.append(idx)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if index in self.poisoned_indices:
            bd = DatasetBD(self.args, img.clone(), self.transform, self.device, self.distance)
            img = bd.act()
            target = self.target_label
        return img, target

    def get_poisoned_indices(self):
        return self.poisoned_indices





def _get_trigger_bbox(width, height, trigger_type, trig_w, trig_h, distance):

    if trigger_type in ['randomPixelTrigger', 'signalTrigger', '90signalTrigger',
                       'kitty', 'bomb', 'flower']:
        return 0, height, 0, width

    def clip_box(y0, y1, x0, x1):
        y0 = max(0, min(y0, height-1))
        y1 = max(0, min(y1, height))
        x0 = max(0, min(x0, width-1))
        x1 = max(0, min(x1, width))
        if y1 <= y0: y1 = min(y0+1, height)
        if x1 <= x0: x1 = min(x0+1, width)
        return y0, y1, x0, x1

    tt = trigger_type

    if tt == 'squareTrigger':
        tw = min(int(trig_w), 3); th = min(int(trig_h), 3)
        xs = max(width  - distance - tw, 0)
        ys = max(height - distance - th, 0)
        return clip_box(ys, ys+th, xs, xs+tw)

    elif tt == 'gridTrigger':
        pts = [(width-3, height-4), (width-4, height-3), (width-4, height-5), (width-5, height-4), (width-5, height-5)]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs)+1
        y0, y1 = min(ys), max(ys)+1
        return clip_box(y0, y1, x0, x1)

    elif tt == 'BTT':

        stripe_h = 3
        stripe_gap = 4


        bottom_shrink = 4
        top_shrink = 2
        right_shrink = 0


        x0 = 2
        x1 = 13 - right_shrink


        original_y0 = 2
        original_y1 = 2 + stripe_h + stripe_gap*2 + stripe_h

        y0 = original_y0 + top_shrink
        y1 = original_y1 - bottom_shrink


        x0 = max(0, x0)
        x1 = min(width, x1)
        y0 = max(0, y0)
        y1 = min(height, y1)

        return clip_box(y0, y1, x0, x1)

    elif tt == 'fourCornerTrigger':
        corner = 3
        return clip_box(height-corner, height, width-corner, width)

    else:
        patch = 5
        offset = 3
        return clip_box(height-patch-offset ,
                       height-offset ,
                       width-patch-offset ,
                       width-offset )





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="后门植入与保存脚本（含 RPCA 检测与可视化）")
    parser.add_argument('--poison_rate', type=float, default=0.5, help="后门植入率")
    parser.add_argument('--trig_w', type=int, default=3, help="触发器宽度")
    parser.add_argument('--trig_h', type=int, default=3, help="触发器高度")
    parser.add_argument('--distance', type=int, default=3, help="触发器到边缘距离")
    parser.add_argument('--num_images', type=int, default=100, help="随机选择的图片数量")
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help="触发器类型")
    parser.add_argument('--poisoned_dir', type=str, default='poisoned_images', help="中毒图片保存文件夹")
    parser.add_argument('--clean_dir', type=str, default='clean_images', help="干净图片保存文件夹")
    parser.add_argument('--L_dir', type=str, default='L_images', help='低秩重建保存文件夹')
    parser.add_argument('--S_dir', type=str, default='S_images', help='稀疏重建(热力图)保存文件夹')
    parser.add_argument('--cluster_thresh', type=float, default=-1.0, help='HDBSCAN到后门簇质心的距离阈值(>0启用)')
    parser.add_argument('--cluster_quantile', type=float, default=-1.0, help='按分位数选择后门(0~1，>0且<1启用)')

    parser.add_argument('--min_cluster_size', type=int, default=None, help='HDBSCAN最小簇大小')
    parser.add_argument('--min_samples', type=int, default=None, help='HDBSCAN最小样本数')
    args = parser.parse_args()


    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


    os.makedirs(args.poisoned_dir, exist_ok=True)
    os.makedirs(args.clean_dir, exist_ok=True)
    os.makedirs(args.L_dir, exist_ok=True)
    os.makedirs(args.S_dir, exist_ok=True)
    os.makedirs("debug/fp", exist_ok=True)
    os.makedirs("debug/fn", exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    clean_testset = DATASET_PLACEHOLDER()


    selected_indices = random.sample(range(len(clean_testset)), args.num_images)
    num_poison = int(args.poison_rate * args.num_images)
    poisoned_indices = set(random.sample(selected_indices, num_poison))
    is_poisoned_list = [idx in poisoned_indices for idx in selected_indices]
    print(f"[Poison Control] 理论中毒数量: {num_poison} | 实际中毒数量: {sum(is_poisoned_list)}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_obj = type('Args', (), {})()
    args_obj.trigger_type = args.trigger_type
    args_obj.trig_w = args.trig_w
    args_obj.trig_h = args.trig_h

    full_list = []
    for i, idx in enumerate(selected_indices):
        img, _ = clean_testset[idx]
        img = img.to(device)

        if is_poisoned_list[i]:
            bd = DatasetBD(args_obj, img.clone(), transform, device, args.distance)
            img = bd.act()

        img_np = img.permute(1, 2, 0).detach().cpu().numpy()
        full_list.append(img_np)

        out_img = Image.fromarray((img_np * 255).astype(np.uint8))
        out_path = os.path.join(args.poisoned_dir if is_poisoned_list[i] else args.clean_dir, f"img_{idx}.png")
        out_img.save(out_path)


    X = np.array(full_list)

    N, H, W, C = X.shape
    X_flat = X.reshape(N, -1).T

    lmb = 0.5 / np.sqrt(max(X_flat.shape))
    rpca = RobustPCA(lmb=lmb, max_iter=2000, tol_rel=1e-6)
    L_flat, S_flat = rpca.fit(X_flat)

    L = L_flat.T.reshape(N, H, W, C)
    S = S_flat.T.reshape(N, H, W, C)




    L_mean = L.mean(axis=0)
    Image.fromarray((np.clip(L_mean, 0, 1) * 255).astype(np.uint8)).save(os.path.join(args.L_dir, "L_mean.png"))

    S_mean = np.abs(S).mean(axis=0).sum(axis=2)
    S_mean = (S_mean - S_mean.min()) / (np.ptp(S_mean) + 1e-8)
    Image.fromarray((S_mean * 255).astype(np.uint8)).save(os.path.join(args.S_dir, "S_mean.png"))



    y0, y1, x0, x1 = _get_trigger_bbox(W, H, args.trigger_type, args.trig_w, args.trig_h, args.distance)
    print(f"[Info] 使用触发器注入位置裁剪窗口：(y0,y1,x0,x1)=({y0},{y1},{x0},{x1})")


    L_mean_patch = np.clip(L_mean[y0:y1, x0:x1, :], 0.0, 1.0)
    Image.fromarray((L_mean_patch * 255).astype(np.uint8)).save(os.path.join(args.S_dir, "mean_patch.png"))
    print(f"[Save] L_mean 裁剪区域图：{os.path.join(args.S_dir, 'mean_patch.png')}")


    patches_X = X[:, y0:y1, x0:x1, :]


    _patch_root = os.path.join("patches")
    os.makedirs(_patch_root, exist_ok=True)

    ph, pw = y1 - y0, x1 - x0
    print(f"[Save] 将保存 {len(selected_indices)} 张裁剪图，尺寸=({ph},{pw}) 到: {_patch_root}")

    for i, idx in enumerate(selected_indices):
        patch = np.clip(patches_X[i], 0.0, 1.0)
        patch_u8 = (patch * 255).astype(np.uint8)

        subdir = "poisoned" if is_poisoned_list[i] else "clean"
        out_dir = os.path.join(_patch_root, subdir)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"patch_{idx}.png")
        Image.fromarray(patch_u8).save(out_path)

    print(f"[Done] 裁剪图已保存至 {_patch_root}（含 clean/poisoned 子目录）")


    diff = patches_X - L_mean_patch[None, ...]
    abs_diff = np.abs(diff) * 2.0
    color_weights = np.array([1.0, 1.0, 1.0])
    weighted_diff = abs_diff * color_weights[None, None, None, :]
    var_two = np.log1p(weighted_diff * 20)
    feat = var_two.reshape(var_two.shape[0], -1)
    feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)


    print(f"[HDBSCAN] 使用HDBSCAN进行聚类，样本数: {len(feat)}")


    min_cluster_size = args.min_cluster_size if args.min_cluster_size is not None else   8
    min_samples = args.min_samples if args.min_samples is not None else  3

    print(f"[HDBSCAN] 参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}")


    cluster_labels, clusterer, core_distances = hdbscan_clustering(
        feat,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    print("Core distances stats:", np.min(core_distances), np.max(core_distances), np.mean(core_distances))
    print("Number of zero core distances:", np.sum(core_distances == 0))



    poison_cluster = find_poison_cluster_hdbscan(feat, cluster_labels, core_distances)



    if poison_cluster == -1:
        anomaly_scores = core_distances if len(core_distances) > 0 else -feat.sum(axis=1)
        y_pred = (anomaly_scores <= np.quantile(anomaly_scores, 0.2)).astype(int)
    else:
        y_pred = (cluster_labels == poison_cluster).astype(int)








    y_true = np.array([1 if p else 0 for p in is_poisoned_list], dtype=int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    print("\n[详细分类结果]")
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")

    confusion_matrix = pd.crosstab(y_true, y_pred,
                                  rownames=['Actual'],
                                  colnames=['Predicted'],
                                  margins=True)
    print("\n[混淆矩阵]")
    print(confusion_matrix)


    for i in np.where((y_pred == 1) & (y_true == 0))[0]:
        idx = selected_indices[i]
        img = Image.fromarray((patches_X[i] * 255).astype(np.uint8))
        img.save(f"debug/fp/fp_{idx}.png")

    for i in np.where((y_true == 1) & (y_pred == 0))[0]:
        idx = selected_indices[i]
        img = Image.fromarray((patches_X[i] * 255).astype(np.uint8))
        img.save(f"debug/fn/fn_{idx}.png")


    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n[检测性能指标]")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")



    feat_dir = "patch_features"
    os.makedirs(feat_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(feat_dir, "patch_var_feat.npz"),
        feat=feat.astype(np.float32),
        image_ids=np.array(selected_indices, dtype=np.int64),
        y_true=y_true,
        y_pred=y_pred.astype(np.int64),
        bbox=np.array([y0, y1, x0, x1], dtype=np.int64),
        patch_shape=np.array([y1-y0, x1-x0, C], dtype=np.int64),
        cluster_labels=cluster_labels.astype(np.int64),
        core_distances=core_distances.astype(np.float32)
    )

    df = pd.DataFrame({
        "image_id": selected_indices,
        "y_true": y_true,
        "y_pred": y_pred,
        "cluster_id": cluster_labels,
        "core_distance": core_distances,
        "is_fp": ((y_pred == 1) & (y_true == 0)).astype(int),
        "is_fn": ((y_true == 1) & (y_pred == 0)).astype(int)
    })
    df.to_excel(os.path.join(feat_dir, "cluster_results.xlsx"), index=False)

    print(f"\n[完成] 所有结果已保存到 {feat_dir} 目录")

feat_dir = "patch_features"
os.makedirs(feat_dir, exist_ok=True)

np.savez_compressed(
    os.path.join(feat_dir, "patch_var_feat.npz"),
    feat=feat.astype(np.float32),
    image_ids=np.array(selected_indices, dtype=np.int64),
    y_true=y_true,
    y_pred=y_pred.astype(np.int64),
    bbox=np.array([y0, y1, x0, x1], dtype=np.int64),
    patch_shape=np.array([y1-y0, x1-x0, C], dtype=np.int64),
    cluster_labels=cluster_labels.astype(np.int64),
    core_distances=core_distances.astype(np.float32)
)


labels_df = pd.DataFrame({
              : selected_indices,
            : y_true,
            : y_pred,
                : (y_true == y_pred).astype(int)
})


labels_df.to_excel(os.path.join(feat_dir, "prediction_labels.xlsx"), index=False)
print(f"[信息] 预测标签和真实标签已保存到 {os.path.join(feat_dir, 'prediction_labels.xlsx')}")


df = pd.DataFrame({
              : selected_indices,
            : y_true,
            : y_pred,
                : cluster_labels,
                   : core_distances,
           : ((y_pred == 1) & (y_true == 0)).astype(int),
           : ((y_true == 1) & (y_pred == 0)).astype(int)
})
df.to_excel(os.path.join(feat_dir, "cluster_results.xlsx"), index=False)

print(f"\n[完成] 所有结果已保存到 {feat_dir} 目录")




print("\n[聚类可视化] 生成聚类可视化图...")


try:

    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, feat.shape[1]))
    feat_pca = pca.fit_transform(feat)


    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feat)-1))
    feat_2d = tsne.fit_transform(feat_pca)


    viz_dir = os.path.join(feat_dir, "visualization")
    os.makedirs(viz_dir, exist_ok=True)


    import matplotlib.pyplot as plt
    import seaborn as sns


    plt.style.use('default')
    sns.set_palette("viridis")


    fig, axes = plt.subplots(2, 2, figsize=(15, 12))


    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))


    scatter1 = axes[0, 0].scatter(feat_2d[:, 0], feat_2d[:, 1],
                                 c=y_true, cmap='coolwarm', alpha=0.7, s=30)
    axes[0, 0].set_title('按真实标签着色\n(红色=中毒, 蓝色=干净)')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0, 0], label='真实标签')


    scatter2 = axes[0, 1].scatter(feat_2d[:, 0], feat_2d[:, 1],
                                 c=y_pred, cmap='coolwarm', alpha=0.7, s=30)
    axes[0, 1].set_title('按预测标签着色\n(红色=预测中毒, 蓝色=预测干净)')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[0, 1], label='预测标签')


    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        color = colors[i % len(colors)]
        label = f'簇 {cluster_id}' if cluster_id != -1 else '噪声点'
        axes[1, 0].scatter(feat_2d[mask, 0], feat_2d[mask, 1],
                          c=[color], alpha=0.7, s=30, label=label)
    axes[1, 0].set_title('按聚类ID着色')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    if len(unique_clusters) <= 10:
        axes[1, 0].legend()


    scatter4 = axes[1, 1].scatter(feat_2d[:, 0], feat_2d[:, 1],
                                  c=core_distances, cmap='viridis', alpha=0.7, s=30)
    axes[1, 1].set_title('按核心距离着色\n(颜色越深距离越大)')
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter4, ax=axes[1, 1], label='核心距离')



    plt.close()

    print(f"[信息] 聚类可视化图已保存到 {os.path.join(viz_dir, 'cluster_visualization.png')}")


    cluster_stats = []
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_true_poison = np.sum(y_true[mask] == 1)
        cluster_total = np.sum(mask)
        poison_ratio = cluster_true_poison / cluster_total if cluster_total > 0 else 0

        cluster_stats.append({
            'cluster_id': cluster_id,
            'total_points': cluster_total,
            'true_poison_points': cluster_true_poison,
            'poison_ratio': poison_ratio,
            'avg_core_distance': np.mean(core_distances[mask]) if cluster_total > 0 else 0,
            'is_poison_cluster': 1 if cluster_id == poison_cluster else 0
        })

    cluster_stats_df = pd.DataFrame(cluster_stats)
    cluster_stats_df.to_excel(os.path.join(viz_dir, 'cluster_statistics.xlsx'), index=False)
    print(f"[信息] 聚类统计信息已保存到 {os.path.join(viz_dir, 'cluster_statistics.xlsx')}")

except Exception as e:
    print(f"[警告] 聚类可视化失败: {e}")

print(f"\n[完成] 所有结果已保存到 {feat_dir} 目录")



print("\n[信息] 正在保存详细的向量和标签信息到Excel...")
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

detailed_df = pd.DataFrame()


detailed_df['image_id'] = selected_indices
detailed_df['y_true'] = y_true
detailed_df['y_pred'] = y_pred
detailed_df['cluster_id'] = cluster_labels
detailed_df['core_distance'] = core_distances


for i in range(feat.shape[1]):
    detailed_df[f'feat_{i}'] = feat[:, i]


detailed_df['var_sum'] = feat.sum(axis=1)


detailed_df['is_poison_clust'] = (cluster_labels == poison_cluster).astype(int)


detailed_df['is_correct'] = (y_true == y_pred).astype(int)


detailed_df['error_type'] = 'TP'
detailed_df.loc[(y_true == 0) & (y_pred == 1), 'error_type'] = 'FP'
detailed_df.loc[(y_true == 1) & (y_pred == 0), 'error_type'] = 'FN'
detailed_df.loc[(y_true == 0) & (y_pred == 0), 'error_type'] = 'TN'


detailed_excel_path = os.path.join(feat_dir, "detailed_vectors_and_labels.xlsx")
detailed_df.to_excel(detailed_excel_path, index=False)
print(f"[信息] 详细向量和标签信息已保存到: {detailed_excel_path}")


summary_stats = {
          : len(detailed_df),
             : sum(y_true),
             : sum(y_true == 0),
             : sum(y_pred),
             : sum(y_pred == 0),
         : accuracy_score(y_true, y_pred),
         : precision_score(y_true, y_pred, zero_division=0),
         : recall_score(y_true, y_pred, zero_division=0),
          : f1_score(y_true, y_pred, zero_division=0),
          : len(np.unique(cluster_labels)),
           : poison_cluster,
            : sum(cluster_labels == poison_cluster) if poison_cluster != -1 else 0
}


summary_df = pd.DataFrame(list(summary_stats.items()), columns=['指标', '值'])
summary_excel_path = os.path.join(feat_dir, "summary_statistics.xlsx")
summary_df.to_excel(summary_excel_path, index=False)
print(f"[信息] 汇总统计信息已保存到: {summary_excel_path}")


if poison_cluster != -1:
    cluster_stats = []
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        cluster_data = {
            'cluster_id': cluster_id,
            '样本数': sum(mask),
            '真实中毒数': sum(y_true[mask]),
            '真实干净数': sum(y_true[mask] == 0),
            '预测中毒数': sum(y_pred[mask]),
            '预测干净数': sum(y_pred[mask] == 0),
            '平均核心距离': np.mean(core_distances[mask]) if sum(mask) > 0 else 0,
            '是否为后门簇': 1 if cluster_id == poison_cluster else 0
        }
        cluster_stats.append(cluster_data)

    cluster_df = pd.DataFrame(cluster_stats)
    cluster_excel_path = os.path.join(feat_dir, "cluster_statistics.xlsx")
    cluster_df.to_excel(cluster_excel_path, index=False)
    print(f"[信息] 聚类统计信息已保存到: {cluster_excel_path}")


df = pd.DataFrame({
              : selected_indices,
            : y_true,
            : y_pred,
                : cluster_labels,
                   : core_distances,
           : ((y_pred == 1) & (y_true == 0)).astype(int),
           : ((y_true == 1) & (y_pred == 0)).astype(int)
})
df.to_excel(os.path.join(feat_dir, "cluster_results.xlsx"), index=False)

print(f"\n[完成] 所有结果已保存到 {feat_dir} 目录")
