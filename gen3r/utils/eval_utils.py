import torch
import open3d as o3d
import numpy as np

from typing import List, Tuple
from pytorch3d.ops import knn_points, sample_farthest_points


def umeyama_alignment(P: torch.Tensor, G: torch.Tensor, mask: torch.Tensor, with_scale: bool=True):
    """
    Align predicted point cloud P to ground truth point cloud G using Umeyama algorithm.
    
    Args:
        P: (F, H, W, 3) predicted point cloud
        G: (F, H, W, 3)/[N, 3] ground truth point cloud  
        mask: (F, H, W) boolean mask indicating valid points
        with_scale: whether to allow scaling transformation
        
    Returns:
        tuple: (R, t, s, P_aligned) where:
            R: (3, 3) rotation matrix
            t: (3,) translation vector
            s: scalar scale factor
            P_aligned: (F, H, W, 3) aligned point cloud
    """
    # Extract valid points using mask
    if G.ndim == 4:
        P_valid = P[mask]  # (N, 3)
        G_valid = G[mask]  # (N, 3)
    else:
        P_valid = P[mask.bool()]  # (N, 3)
        G_valid = G[0]  # (N, 3)
    
    if P_valid.shape[0] < 3:
        # Not enough points for alignment, return identity transformation
        R = torch.eye(3, device=P.device, dtype=P.dtype)
        t = torch.zeros(3, device=P.device, dtype=P.dtype)
        s = torch.ones(1, device=P.device, dtype=P.dtype)
        P_aligned = P.clone()
        return R, t, s, P_aligned
    
    # Center the point clouds
    P_mean = P_valid.mean(dim=0, keepdim=True)  # (1, 3)
    G_mean = G_valid.mean(dim=0, keepdim=True)  # (1, 3)
    
    P_centered = P_valid - P_mean  # (N, 3)
    G_centered = G_valid - G_mean  # (N, 3)
    
    # Compute covariance matrix
    H = (P_centered.T @ G_centered).float()  # (3, 3)
    
    # SVD decomposition
    U, S, Vt = torch.linalg.svd(H)
    
    # Ensure proper rotation matrix (handle reflection case)
    V = Vt.T
    if torch.linalg.det((U @ V.T).float()) < 0:
        V[:, -1] *= -1
    
    # Compute rotation matrix
    R = V @ U.T  # (3, 3)
    
    # Compute scale factor
    if with_scale:
        # Scale factor based on variance ratio
        P_var = (P_centered ** 2).sum()
        G_var = (G_centered ** 2).sum()
        s = torch.sqrt(G_var / P_var) if P_var > 0 else torch.ones(1, device=P.device, dtype=P.dtype)
    else:
        s = torch.ones(1, device=P.device, dtype=P.dtype)
    
    # Compute translation
    t = G_mean.squeeze() - s * (R @ P_mean.T).squeeze()  # (3,)
    
    # Apply transformation to all points
    P_aligned = s * (P.view(-1, 3) @ R.T) + t  # (F*H*W, 3)
    P_aligned = P_aligned.view(P.shape)  # (F, H, W, 3)
    
    return R, t, s, P_aligned


def compute_chamfer_metrics(
    P: torch.Tensor, 
    G: torch.Tensor, 
    mask: torch.Tensor=None, 
    squared: bool=False, 
    require_downsample: bool=True, 
    in_mm: bool=False, 
    voxel_size: float=0.005, 
    input_indices: torch.Tensor=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute point-cloud reconstruction metrics after optional alignment and downsampling.

    This function aligns prediction `P` to ground truth `G` with Umeyama alignment,
    extracts valid points (via `mask` or auto-sampling when `mask is None`), optionally
    downsamples points using Open3D voxel sampling, and then computes nearest-neighbor
    distances in both directions with `pytorch3d.ops.knn_points`.

    Metrics:
        - accuracy: mean distance from predicted points to ground-truth points (P -> G)
        - completeness: mean distance from ground-truth points to predicted points (G -> P)
        - chamfer: average of accuracy and completeness
        - relative_percent: chamfer normalized by GT scene extent, in percent

    Args:
        P: Predicted points, shape `(F, H, W, 3)` or `(1, F, H, W, 3)`.
        G: Ground-truth points, shape `(F, H, W, 3)`, `(1, F, H, W, 3)`, or `(N, 3)`
            depending on evaluation mode.
        mask: Optional boolean valid-point mask with shape `(F, H, W)`.
            If `None`, valid points are auto-selected via voxel + FPS sampling.
        squared: If `True`, keep squared KNN distances; otherwise use Euclidean distances.
        require_downsample: If `True`, run voxel downsampling before FPS.
        in_mm: If `True`, converts accuracy/completeness/chamfer from meters to millimeters.
        voxel_size: Voxel size used by Open3D downsampling.
        input_indices: Optional frame indices used to align/evaluate on a selected subset.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - accuracy: Tensor of shape `(1,)`.
            - completeness: Tensor of shape `(1,)`.
            - chamfer: Tensor of shape `(1,)`.
            - relative_percent: Tensor of shape `(1,)`, chamfer / scene_extent * 100.
            - P: Aligned predicted point cloud (same shape as processed input `P`).
    """
    P, G = P.float(), G.float()
    if P.ndim == 5:
        P = P.squeeze(0)
    if G.ndim == 5:
        G = G.squeeze(0)

    if mask is None:
        F, H, W = P.shape[:3]
        # select 10k points as valid using Farthest Point Sampling for umeyama alignment
        P_flat, G_flat = P.view(-1, 3).contiguous(), G.view(-1, 3).contiguous()
        # domsample to 10k points using open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P_flat.cpu().numpy())
        pcd_downsampled, _, index_map = pcd.voxel_down_sample_and_trace(voxel_size=0.005, min_bound=pcd.get_min_bound(), max_bound=pcd.get_max_bound())

        idx = torch.cat([torch.tensor(inds) for inds in index_map]).to(P.device)
        P_flat_downsampled = torch.tensor(np.asarray(pcd_downsampled.points), device=P.device).unsqueeze(0)  # [1, N, 3]
        
        # then sample 10k points using farthest point sampling
        _, indices = sample_farthest_points(P_flat_downsampled, K=10000)  # (1, K)
        
        # generate mask
        mask = torch.zeros_like(P[..., 0], dtype=torch.bool, device=P.device)
        final_idx = idx[indices[0]]
        f, h, w = final_idx // (H * W), (final_idx % (H * W)) // W, final_idx % W
        mask[f, h, w] = True  # (F, H, W)

        # align the point cloud
        P = umeyama_alignment(P, G, mask)[-1]
        P_flat = P.view(-1, 3).contiguous()  # (-1, 3)

    else:
        # align P to G
        if input_indices is None:
            P = umeyama_alignment(P, G, mask)[-1]
            # extract valid points
            if G.ndim == 4:
                P_flat = P[mask][None, ...]  # (1, N, 3)
                G_flat = G[mask][None, ...]  # (1, N, 3)
            else:
                P_flat = P[mask][None, ...]  # (1, N, 3)
                G_flat = G[None, ...]  # (1, N, 3)
        else:
            P = umeyama_alignment(P, G[input_indices], mask[input_indices])[-1]
            # extract valid points
            P_flat = P[mask[input_indices]][None, ...]  # (1, N, 3)
            G_flat = G[mask][None, ...]  # (1, N, 3)
        
    if require_downsample:
        # downsample the point cloud using open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P_flat.view(-1, 3).contiguous().cpu().numpy())
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        P_flat_downsampled = torch.tensor(np.asarray(pcd_downsampled.points), device=P.device).unsqueeze(0)  # [1, N, 3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(G_flat.view(-1, 3).contiguous().cpu().numpy())
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        G_flat_downsampled = torch.tensor(np.asarray(pcd_downsampled.points), device=P.device).unsqueeze(0)  # [1, N, 3]
        
        # select 20000 points as valid using Farthest Point Sampling
        P_flat, _ = sample_farthest_points(P_flat_downsampled, K=20000)  # (1, K, 3)
        G_flat, _ = sample_farthest_points(G_flat_downsampled, K=20000)  # (1, K, 3)

    # P -> G
    knn_pg = knn_points(P_flat, G_flat, K=1)
    dist_pg = knn_pg.dists[..., 0]  # (1, N)
    if not squared:
        dist_pg = dist_pg.sqrt()
    accuracy = dist_pg.mean(dim=1)

    # G -> P
    knn_gp = knn_points(G_flat, P_flat, K=1)
    dist_gp = knn_gp.dists[..., 0]
    if not squared:
        dist_gp = dist_gp.sqrt()
    completeness = dist_gp.mean(dim=1)

    chamfer = (accuracy + completeness) / 2

    mins = G_flat.squeeze(0).min(dim=0)[0]  # (3,)
    maxs = G_flat.squeeze(0).max(dim=0)[0]  # (3,)
    dist = (maxs - mins).norm()
    relative_percent = (chamfer / dist) * 100  # %
    
    if in_mm:
        accuracy = accuracy * 1000
        completeness = completeness * 1000
        chamfer = chamfer * 1000
        relative_percent = relative_percent
    
    return accuracy, completeness, chamfer, relative_percent, P