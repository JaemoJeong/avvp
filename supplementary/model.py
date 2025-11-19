import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import torch.distributed as dist
from method_utils import gather, time_warp


#################################################################

class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):  # [B, L, D]
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        return out.mean(dim=1)  # temporal weighted average


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, pool=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class SensorMotionEncoder(nn.Module):
    def __init__(self, sensor_channels, size_embeddings=128, base_dim=32):
        super().__init__()

        self.block1 = TemporalBlock(sensor_channels, base_dim, kernel_size=5, dilation=1)
        self.block2 = TemporalBlock(base_dim, base_dim * 2, kernel_size=3, dilation=2)
        self.block3 = TemporalBlock(base_dim * 2, base_dim * 4, kernel_size=3, dilation=4, pool=False)

        self.norm = nn.GroupNorm(8, base_dim * 4)

        self.gru = nn.GRU(
            input_size=base_dim * 4,
            hidden_size=size_embeddings,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_attn = TemporalAttention(size_embeddings * 2)

        self.proj = nn.Linear(size_embeddings * 2, size_embeddings)
        self.ssl_head = nn.Linear(size_embeddings, size_embeddings)
        self.mmcl_head = nn.Linear(size_embeddings, size_embeddings)

    def forward(self, batch):
        """
        batch: [B, C, L]  (sensor sequence)
        return: {"emb": motion embedding, "ssl": ssl_out, "mmcl": mmcl_out}
        """
        # Conv feature extraction
        x = self.block1(batch)
        x = self.block2(x)
        x = self.block3(x)
        x = self.norm(x)                # [B, C, L]
        x = x.permute(0, 2, 1)          # [B, L, C]
        self.gru.flatten_parameters()

        # Temporal modeling
        out, _ = self.gru(x)
        attn_out = self.temporal_attn(out)   # [B, D]
        emb = self.proj(attn_out)            # [B, embedding_dim]

        # Heads
        ssl_out = self.ssl_head(emb)
        mmcl_out = self.mmcl_head(emb)

        return {"emb": emb, "ssl": ssl_out, "mmcl": mmcl_out}

# ---------------------------------------------------------------------

class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_type="max", embedding_size=32):
        super().__init__()
        if pool_type == "max":
            pool_fn = torch.nn.MaxPool1d(kernel_size=2)
        elif pool_type == "adaptive":
            pool_fn = torch.nn.AdaptiveAvgPool1d(output_size=embedding_size)
        else:
            raise ValueError(f"pool_type {pool_type} not supported")

        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=2,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            pool_fn,
        )

    def forward(self, batch):
        return self.net(batch)
    

class SensorAppearanceEncoder(nn.Module):
    def __init__(self, sensor_channels, input_dim=32, size_embeddings: int = 128):
        super().__init__()
        self.block1 = Block(sensor_channels, input_dim, 5)
        self.block2 = Block(input_dim, input_dim * 2, 3)
        self.block3 = Block(input_dim * 2, input_dim * 2, 3, pool_type="adaptive", embedding_size=32)
        self.norm = torch.nn.GroupNorm(4, input_dim * 2)

        self.gru = torch.nn.GRU(
            batch_first=True,
            input_size=input_dim * 2, 
            hidden_size=size_embeddings
        )
        self.ssl_head = torch.nn.Linear(size_embeddings, size_embeddings)
        self.mmcl_head = torch.nn.Linear(size_embeddings, size_embeddings)

    def forward(self, batch):
        x = self.block1(batch)
        x = self.block2(x)
        x = self.block3(x)
        x = self.norm(x)        # [B, C, L]
        x = x.permute(0, 2, 1)  # [B, L, C]
        _, h = self.gru(x)      # h: [1, B, hidden_size]
        emb = h[0]              # [B, hidden_size]

        ssl_out = self.ssl_head(emb)
        mmcl_out = self.mmcl_head(emb)
        return {"ssl": ssl_out, "mmcl": mmcl_out, "emb": emb}
    
class SensorModel(nn.Module):
    def __init__(self, sensor_appearance_encoder, sensor_motion_encoder, clustering_module):
        super().__init__()
        self.sensor_appearance_encoder = sensor_appearance_encoder
        self.sensor_motion_encoder = sensor_motion_encoder
        self.clustering_module = clustering_module
        self.norm = nn.LayerNorm(256*2)

    def forward(self, batch):
        _, features, _ = self.clustering_module(batch, return_features = True)
        sensor_motion_emb = self.encoding_motion(batch)["emb"]
        s_app_norm = F.normalize(features, dim=1)
        s_mot_norm = F.normalize(sensor_motion_emb, dim=1)

        with torch.no_grad():
            s_app_norm = F.normalize(features, dim=1)
        z_sensor_online = torch.cat((s_app_norm, s_mot_norm), dim=1)
        return z_sensor_online

    def encoding_appearance(self, batch, labels, return_features, idx):
        # appearance embedding with clustering module
        scores, features, pure_features = self.clustering_module(batch, labels=labels, return_features=return_features, idx=idx)
        return scores, features, pure_features

    
    def encoding_motion(self, batch):
        # motion embedding
        mot_emb = self.sensor_motion_encoder(batch)
        return mot_emb



#############################################################

# ---------------------------------------------------------------------
# Attention Head (Object Encoder saliency map)
# ---------------------------------------------------------------------
class AttentionHead(nn.Module):
    """Salient region 강조용 간단한 2D attention head."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        return self.sigmoid(self.conv(x))  # [B, 1, H, W]


# ---------------------------------------------------------------------
# Shared CNN Encoder (Scene/Object 공통)
# ---------------------------------------------------------------------

class SharedEncoder(nn.Module):
    def __init__(self, in_channels=3, base_dim=64, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 7, stride=2, padding=3, bias=False),  # 224→112
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1, bias=False), # 112→56
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1, bias=False), # 56→28
            nn.BatchNorm2d(base_dim * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_dim * 4, out_dim, 3, stride=2, padding=1, bias=False), # 28→14
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # [B, out_dim, H', W']

class MotionEncoder(nn.Module):
    """
    Motion Encoder with cosine-consistent projection.
    강조점:
      - CosineProj: norm-invariant projection layer
      - optional cosine regularization for self-consistency
    """
    def __init__(self, in_channels=5, base_dim=32, latent_dim=256, use_cosine_proj=True):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # 3D backbone (temporal gradient feature)
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, base_dim, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(base_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_dim, base_dim * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(base_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_dim * 2, latent_dim, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(inplace=True),
        )

        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # ✅ projection head 변경
        if use_cosine_proj:
            self.proj = CosineProj(latent_dim, latent_dim)
        else:
            self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, videos, flows=None):
        B, T, _, H, W = videos.shape

        video_diff = videos[:, 1:] - videos[:, :-1]

        if isinstance(flows, torch.Tensor):
            if flows.shape[1] > video_diff.shape[1]:
                flows = flows[:, : video_diff.shape[1]]
            elif flows.shape[1] < video_diff.shape[1]:
                flows = torch.cat([flows, flows[:, -1:, :, :, :]], dim=1)
            x = torch.cat([video_diff, flows], dim=2)
        else:
            pad = torch.zeros(B, T - 1, 2, H, W, device=videos.device, dtype=videos.dtype)
            x = torch.cat([video_diff, pad], dim=2)

        x = x.permute(0, 2, 1, 3, 4)
        feat = self.backbone(x)
        feat = self.spatial_pool(feat).squeeze(-1).squeeze(-1)

        if feat.shape[2] > 1:
            motion_diff = feat[:, :, 1:] - feat[:, :, :-1]
            v_motion = motion_diff.mean(dim=2)
        else:
            v_motion = feat.mean(dim=2)

        # projection
        v_motion = self.proj(v_motion)
        return v_motion


# --- Cosine Projection Layer ---
class CosineProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(x, dim=1)
        return F.linear(x, w)




# ---------------------------------------------------------------------
# Scene Encoder (Global context)
# ---------------------------------------------------------------------
class SceneEncoder(nn.Module):
    """Global/static appearance representation."""
    def __init__(self, in_dim=256, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        # [B, C, H, W] → [B, latent_dim]
        return self.net(x).flatten(1)


# ---------------------------------------------------------------------
# Object Encoder (Local salient appearance)
# ---------------------------------------------------------------------
class ObjectEncoder(nn.Module):
    """Foreground/local salient representation."""
    def __init__(self, in_dim=256, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )
        self.attn = AttentionHead(latent_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat = self.conv(x)                 # [B, latent_dim, H, W]
        attn = self.attn(feat)              # [B, 1, H, W]
        obj_feat = self.pool(feat * attn).flatten(1)
        return obj_feat
    
    
# ---------------------------------------------------------------------
# VisionModel (Simplified Appearance Encoder + Motion Encoder)
# ---------------------------------------------------------------------
class VisionModel(nn.Module):
    """
    Simplified MOSO: directly extract v_appearance from shared encoder.
    Keeps MotionEncoder for Stage2 alignment.
    """
    def __init__(self, in_channels=3, base_dim=64, latent_dim=256):
        super().__init__()
        # shared encoder (2D feature extractor)
        self.shared_encoder = SharedEncoder(in_channels, base_dim, out_dim=latent_dim)
        self.scene_encoder = SceneEncoder(in_dim=latent_dim, latent_dim=latent_dim)
        self.object_encoder = ObjectEncoder(in_dim=latent_dim, latent_dim=latent_dim)

        # Scene + Object 결합 projection
        self.proj = nn.Linear(latent_dim * 2, latent_dim)
        self.norm = nn.LayerNorm(latent_dim*2)
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # motion encoder for stage2
        self.motion_encoder = MotionEncoder(
            in_channels=in_channels+2 ,  # RGB + optical flow
            base_dim=base_dim // 2,
            latent_dim=latent_dim
        )

    def forward(self, video, flows):
        """
        video: [B, T, C, H, W]
        flows: [B, T, 2, H, W]
        returns: dict with v_appearance, v_motion, z_video_online, vis_v
        """

        B, T, C, H, W = video.shape

        # 1️⃣ Shared appearance features
        video_flat = video.view(B * T, C, H, W)
        shared_feat = self.shared_encoder(video_flat)   # [B*T, D, Hf, Wf]
        _, D, Hf, Wf = shared_feat.shape

        # temporal average pooling
        shared_feat = shared_feat.view(B, T, D, Hf, Wf).mean(dim=1)

        v_scene = self.scene_encoder(shared_feat)
        v_object = self.object_encoder(shared_feat)

        fused = torch.cat([v_scene, v_object], dim=1)
        v_appearance = self.fuse(fused)

        # 3️⃣ motion feature extraction
        video_centered = video - video.mean(dim=[3, 4], keepdim=True)
        video_centered = video_centered / (video_centered.std(dim=[3, 4], keepdim=True) + 1e-6)
        v_motion = self.motion_encoder(videos = video, flows=flows)

        # detach appearance for z_video_online
        v_app_norm = F.normalize(v_appearance, dim=1)
        v_mot_norm = F.normalize(v_motion, dim=1)
        with torch.no_grad():
            v_app_norm = v_app_norm.detach()
        z_video_online = torch.cat([v_app_norm, v_mot_norm], dim=1) # motion var 균형을 위해 스케일링 반영

        return {
            "v_appearance": v_appearance,
            "v_motion": v_motion,
            "vis_v": video_centered - video.mean(dim=1, keepdim=True),
            "z_video_online": z_video_online,
        }

class ClusteringManager(nn.Module):
    def __init__(self, num_clusters, feature_dim, initial_global_threshold, momentum=0.9, temperature=0.1, device='cuda', local_rank=0, min_cluster_size=30):
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.local_rank = local_rank

        if self.local_rank == "0":
             self.feature_bank = torch.zeros((10000, feature_dim), dtype=torch.float32)
        else:
             self.feature_bank = None
        
        self.label_bank = None

        thresholds_tensor = torch.full(
            (num_clusters,), 
            float(initial_global_threshold), 
            device=device,
            dtype=torch.float32
        )
        self.register_buffer('distance_thresholds', thresholds_tensor)

        self.initialized = False
        centroids = torch.rand(num_clusters, feature_dim, device=device) * 2.0 - 1.0 
        
        if num_clusters <= feature_dim:  
            q, r = torch.linalg.qr(centroids.t()) 
            centroids = q[:, :num_clusters].t()  
        self.register_buffer('centroids', centroids)
        self.memory_initialized = False        
        self.pending_memory_update = False
        self.min_cluster_size = min_cluster_size 
        
        self.class_weight_power = 0.5 

        self.kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
        self.cluster_stats = {}

    @torch.no_grad()
    def _compute_centroids_idx(self, cinds):
        """Compute a few centroids."""
        assert self.local_rank == "0"
        num = len(cinds)
        centroids = torch.zeros((num, self.feature_dim), dtype=torch.float32)
        for i, c in enumerate(cinds):
            idx = np.where(self.label_bank.cpu().numpy() == c)[0]
            centroids[i, :] = self.feature_bank[idx, :].mean(dim=0)
        return centroids

    def _compute_centroids(self):
        """Compute all non-empty centroids."""
        assert self.local_rank == "0"
        label_bank_np = self.label_bank.cpu().numpy()
        argl = np.argsort(label_bank_np)
        sortl = label_bank_np[argl]
        diff_pos = np.where(sortl[1:] - sortl[:-1] != 0)[0] + 1
        start = np.insert(diff_pos, 0, 0)
        end = np.insert(diff_pos, len(diff_pos), len(label_bank_np))
        class_start = sortl[start]
        # keep empty class centroids unchanged
        centroids = self.centroids.cpu().clone()
        for i, st, ed in zip(class_start, start, end):
            centroids[i, :] = self.feature_bank[argl[st:ed], :].mean(dim=0)
        return centroids

    def update_samples_memory(self, idx: torch.Tensor,
                              feature: torch.Tensor):
        """Update samples memory."""
        assert self.initialized
        # print(f"[{self.local_rank}] Updating samples memory for {idx.shape[0]} samples.")
        feature_norm = feature / (feature.norm(dim=1).view(-1, 1) + 1e-10
                                  )  # normalize
        valid_mask = idx != -1
        idx = idx[valid_mask]
        feature = feature[valid_mask]

        if idx.numel() == 0:
            return torch.tensor(0.0, device=feature.device)
        idx = gather(idx)
        feature_norm = gather(feature_norm)
        
        idx = idx.cpu()
        if self.local_rank == "0":
            feature_old = self.feature_bank[idx, ...].cuda()
            feature_new = (1 - self.momentum) * feature_old + \
                self.momentum * feature_norm
            feature_norm = feature_new / (
                feature_new.norm(dim=1).view(-1, 1) + 1e-10)
            self.feature_bank=self.feature_bank.cuda()
            self.feature_bank[idx, ...] = feature_norm
        dist.barrier()
        dist.broadcast(feature_norm, src=0) 
        # compute new labels

        # similarity_to_centroids = self.compute_similarity_scores(feature_norm.permute(1, 0))
        feature_norm = feature_norm.permute(1, 0)
        centroids_norm = F.normalize(self.centroids, dim=1)
        similarity_to_centroids = torch.mm(centroids_norm,
                                           feature_norm)  # CxN
        newlabel = similarity_to_centroids.argmax(dim=0)  # cuda tensor
        # self.label_bank = self.label_bank.cuda()
        change_ratio = (newlabel != self.label_bank[idx]
                        ).sum().float().cuda() / float(newlabel.shape[0])
        self.label_bank[idx] = newlabel.cuda().clone()  # all gpu have the same label_bank
        # print("update_samples_memory", change_ratio)
        return change_ratio

    @torch.no_grad()
    def update_centroids_memory(self):
        """Update centroids memory."""
        if self.local_rank == "0":
            center = self._compute_centroids()
            self.centroids.copy_(center)
        dist.broadcast(self.centroids, src=0)
        print(f"[{self.local_rank}] Broadcasted centroids shape: {self.centroids}")

    @torch.no_grad()
    def deal_with_small_clusters(self):
        """
        Gather all label_banks, perform clustering logic on rank 0,
        and broadcast the updated label_bank and centroids back to all ranks if needed.
        """        

        if self.local_rank == "0":
            global_histogram = np.bincount(
                self.label_bank.cpu().numpy(), minlength=self.num_clusters)
            small_clusters = np.where(global_histogram < self.min_cluster_size)[0].tolist()

            if len(small_clusters) == 0:
                pass
            else:
                print(f'[Rank 0] Dealing with {len(small_clusters)} small clusters.')

                for s in small_clusters:
                    label_bank_np = self.label_bank.cpu().numpy()
                    idx = np.where(label_bank_np == s)[0]
                    if len(idx) == 0:
                        continue
                    
                    inclusion = np.setdiff1d(np.arange(self.num_clusters), np.array(small_clusters), assume_unique=True)
                    inclusion_tensor = torch.from_numpy(inclusion).cuda()

                    target_idx = torch.mm(
                        self.centroids[inclusion_tensor, :],
                        self.feature_bank[idx, :].cuda().permute(1, 0)
                    ).argmax(dim=0)
                    
                    target = inclusion_tensor[target_idx]
                    self.label_bank[idx] = target.cuda()
                self._redirect_empty_clusters(small_clusters)

        dist.broadcast(self.centroids, src=0)
        dist.broadcast(self.label_bank, src=0)
        dist.broadcast(self.feature_bank, src=0)
        return 

    @torch.no_grad()
    def _partition_max_cluster(self, max_cluster: np.ndarray):
        """Deterministic split: closest 50% stay, farthest 50% go to new cluster."""
        assert self.local_rank == "0"
        max_cluster_idx = np.where(self.label_bank.cpu().numpy() == max_cluster)[0]
        assert len(max_cluster_idx) >= 2

        # (1) Extract features
        max_cluster_features = self.feature_bank[max_cluster_idx, :]  # [N_c, D]
        if np.any(np.isnan(max_cluster_features.cpu().numpy())):
            raise Exception('Has nan in features.')

        # (2) Compute cluster centroid
        centroid = max_cluster_features.mean(dim=0, keepdim=True)  # [1, D]

        # (3) Compute distance to centroid (L2 norm)
        distances = torch.norm(max_cluster_features - centroid, dim=1)  # [N_c]

        # (4) Sort by distance
        sorted_indices = torch.argsort(distances)  # ascending (near → far)

        # (5) Split deterministically by median (50%)
        mid = len(sorted_indices) // 2
        sub_cluster1_idx = max_cluster_idx[sorted_indices[:mid].cpu()]   # near → stay (old cluster)
        sub_cluster2_idx = max_cluster_idx[sorted_indices[mid:].cpu()]   # far  → new cluster

        # (6) (Optional) check empty safeguard
        if len(sub_cluster1_idx) == 0 or len(sub_cluster2_idx) == 0:
            print("Warning: deterministic partition failed (empty subset). Forcing equal split.")
            sub_cluster1_idx = max_cluster_idx[:len(max_cluster_idx)//2]
            sub_cluster2_idx = max_cluster_idx[len(max_cluster_idx)//2:]

        return sub_cluster1_idx, sub_cluster2_idx


    @torch.no_grad()
    def _redirect_empty_clusters(self, empty_clusters: np.ndarray):
        """Re-direct empty clusters."""
        assert self.local_rank == "0"
        print("empty_clusters", empty_clusters)
        for e in empty_clusters:
            assert (self.label_bank.cpu().numpy() != e).all().item(), \
                f'Cluster #{e} is not an empty cluster.'
            
            max_cluster = np.bincount(self.label_bank.cpu().numpy(), minlength=self.num_clusters).argmax().item()
            
            sub_cluster1_idx, sub_cluster2_idx = self._partition_max_cluster(max_cluster)

            if sub_cluster1_idx is None:
                continue

            self.label_bank[torch.from_numpy(sub_cluster2_idx)] = e
            print(f"max cluster {max_cluster} devided into 2 area and one of them is assigned into {e})")
            
            cinds=[max_cluster, e]
            center = self._compute_centroids_idx(cinds)
            self.centroids[
                torch.LongTensor(cinds).cuda(), :] = center.cuda()
        print("Redirected empty clusters and updated centroids.")


    def compute_similarity_scores(self, features):
        features_norm = F.normalize(features, dim=1)
        centroids_norm = F.normalize(self.centroids, dim=1)
        similarity = torch.mm(features_norm, centroids_norm.t())
        
        return similarity / self.temperature
        
    @torch.no_grad()
    def compute_class_weights(self):
        histogram = np.bincount(
            self.label_bank.cpu().numpy(), minlength=self.num_clusters)
        cluster_size = torch.tensor(histogram, device=self.centroids.device)
        normalized_sizes = cluster_size + 1e-8

        if hasattr(self, "feature_bank") and hasattr(self, "centroids"):
            # feature_bank: [N, D]
            # label_bank: [N]
            features = self.feature_bank.to(self.centroids.device)
            labels = self.label_bank.to(self.centroids.device)

            cluster_stats = {}
            for k in range(self.num_clusters):
                mask = labels == k
                if mask.any():
                    feats_k = features[mask]
                    centroid_k = self.centroids[k].unsqueeze(0)
                    distances = torch.sqrt(((feats_k - centroid_k) ** 2).sum(dim=1))

                    mean_d = distances.mean().item()
                    std_d = distances.std().item()
                    q25 = torch.quantile(distances, 0.25).item()
                    q50 = torch.quantile(distances, 0.50).item()
                    q75 = torch.quantile(distances, 0.75).item()
                    all = torch.quantile(distances, 1.00).item()
                    cluster_stats[k] = {
                        "size": int(mask.sum().item()),
                        "mean": mean_d,
                        "std": std_d,
                        "q25": q25,
                        "q50": q50,
                        "q75": q75,
                        "all": all*2, # all samples are good.
                    }
                else:
                    cluster_stats[k] = {
                        "size": 0,
                        "mean": -1,
                        "std": -1,
                        "q25": -1,
                        "q50": -1,
                        "q75": -1,
                        "all": -1,
                    }

            self.cluster_stats = cluster_stats
        else:
            print("⚠️ Warning: feature_bank or centroids not found — cluster stats skipped.")

        weights = 1.0 / torch.pow(normalized_sizes, self.class_weight_power)

        max_weight = weights.max()
        min_weight = weights.min()
        if max_weight > min_weight * 10:
            weights = torch.clamp(weights, min=max_weight / 10)

        weights = weights / weights.sum() * self.num_clusters
        return weights

class ClusteringModule(nn.Module):
    def __init__(self, encoder, embedding_dim, num_sensors, num_clusters, datamodule, top_k=1, prototype_cache_dir="./cache", dataset_name="custom_dataset", min_cluster_size=30, mid_label=True):
        super().__init__()
        self.encoder = encoder
        self.top_k = top_k
        self.local_rank = os.environ.get("LOCAL_RANK", "0")
        self.clustering_manager = ClusteringManager(num_clusters=num_clusters, initial_global_threshold=1.5, feature_dim=embedding_dim, local_rank=self.local_rank, min_cluster_size=min_cluster_size)
        self.projection_layer = nn.Linear(num_sensors, embedding_dim)
        self.epoch = 0
        self.datamodule = datamodule
        self.centroids_update_interval = 1
        self.deal_with_small_clusters_interval = 1
        self.cls_head = nn.Linear(embedding_dim, num_clusters)
        self.attention_weight = nn.Parameter(torch.randn(num_sensors))
        self.prototype_cache_dir = prototype_cache_dir
        self.dataset_name = dataset_name
        self.mid_label = mid_label

    def broadcast_prototypes(self, rank, world_size, device):
        print(f"[{rank}] Broadcasting results from rank 0...", device)
        
        broadcast_device = torch.device(device) 
        num_total_samples = len(self.train_dataloader.dataset)
        feature_dim = self.clustering_manager.feature_dim
        
        if rank == "0":
            centroids_gpu = self.clustering_manager.centroids # 이미 register_buffer로 GPU에 있을 수 있음
            label_bank_gpu = self.clustering_manager.label_bank.to(broadcast_device)
            feature_bank_gpu = self.clustering_manager.feature_bank.to(broadcast_device)
        else:
            centroids_gpu = self.clustering_manager.centroids # (이미 register_buffer이므로 존재)
            label_bank_gpu = torch.empty(num_total_samples, dtype=torch.long, device=broadcast_device)
            feature_bank_gpu = torch.empty(num_total_samples, feature_dim, dtype=torch.float32, device=broadcast_device)
        
        dist.broadcast(centroids_gpu, src=0)
        dist.broadcast(label_bank_gpu, src=0)
        dist.broadcast(feature_bank_gpu, src=0)

        if rank != "0":
            self.clustering_manager.label_bank = label_bank_gpu
            self.clustering_manager.feature_bank = feature_bank_gpu
            self.clustering_manager.initialized = True
            
        print(f"[{rank}] Broadcasting complete.")


    @torch.no_grad()
    def init_prototypes_with_data(self, device, num_clusters):
        self.eval()
        self.train_dataloader = self.datamodule.train_dataloader()
        
        rank = self.local_rank
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank_str = str(rank)

        dataset_size = len(self.train_dataloader.dataset)
        cache_filename = f"prototypes_{self.dataset_name}_N{dataset_size}_C{num_clusters}.pt" 
        prototype_cache_path = os.path.join(self.prototype_cache_dir, cache_filename)
        
        cache_hit = False

        if rank_str == "0" and os.path.exists(prototype_cache_path):
            try:
                print(f"[{rank}] Attempting to load prototypes from cache: {prototype_cache_path}")
                
                cached_data = torch.load(prototype_cache_path)
                
                self.clustering_manager.centroids.copy_(cached_data['centroids'].to(device))
                self.clustering_manager.label_bank = cached_data['label_bank'].to(device)
                self.clustering_manager.feature_bank = cached_data['feature_bank'].to(device)
                self.clustering_manager.initialized = True
                cache_hit = True
                print(f"[{rank}] Prototypes loaded successfully from cache.")
                
            except Exception as e:
                print(f"[{rank}] Cache load failed ({e}). Proceeding with feature extraction.")
                if os.path.exists(prototype_cache_path): os.remove(prototype_cache_path)
                pass # 실패 시, 아래의 K-Means 로직으로 진행

        cache_hit_tensor = torch.tensor([cache_hit], dtype=torch.bool, device=device)
        if world_size > 1:
             dist.broadcast(cache_hit_tensor, src=0)
        cache_hit = cache_hit_tensor.item()
        
        if not cache_hit:
            
            print(f"[{rank}] Collecting features for KMeans...")
                
            local_features = []
            local_idx = []
            
            for i, (videos, sensors, labels, sample_ids, _) in enumerate(self.train_dataloader):
                sensors = sensors.to(device)
                idx, _ = sample_ids
                idx = idx.clone().to(dtype=torch.long, device=device)
                
                _, features, _ = self(sensors, return_features=True)
                
                local_features.append(features) # GPU 상태로 유지
                local_idx.append(idx) # GPU 상태로 유지
                print(f"[{rank}] Processed batch {i+1}/{len(self.train_dataloader)}")
            
            local_features_tensor = torch.cat(local_features, dim=0)
            local_idx_tensor = torch.cat(local_idx, dim=0)

            all_features_gpu = gather(local_features_tensor)
            all_idx_gpu = gather(local_idx_tensor)
            
            if rank_str == "0":
                print(f"[{rank}] Running KMeans...")
                
                all_idx_cpu = all_idx_gpu.cpu()
                
                num_total_samples = len(self.train_dataloader.dataset)
                feature_bank_temp = torch.empty(num_total_samples, self.clustering_manager.feature_dim, device="cuda")
                feature_bank_temp[all_idx_cpu, ...] = all_features_gpu # ID를 사용한 최종 재배치
                
                self.clustering_manager.feature_bank = feature_bank_temp # CPU Feature Bank 할당
                    
                all_features_cpu_numpy = self.clustering_manager.feature_bank.cpu().numpy()
                kmeans = self.clustering_manager.kmeans.fit(all_features_cpu_numpy)
                
                prototypes_gpu = torch.from_numpy(kmeans.cluster_centers_).to(device)
                self.clustering_manager.centroids.copy_(F.normalize(prototypes_gpu, dim=-1))
                initial_labels = torch.from_numpy(kmeans.labels_).long()
                self.clustering_manager.label_bank = initial_labels.cuda()
                self.clustering_manager.initialized = True
                
                print(f"[{rank}] Saving new prototypes to cache...")
                temp_path = prototype_cache_path + ".tmp"
                
                data_to_save = {
                    'centroids': self.clustering_manager.centroids.cpu().clone(),
                    'label_bank': self.clustering_manager.label_bank.clone(),
                    'feature_bank': self.clustering_manager.feature_bank.clone(),
                }
                
                os.makedirs(os.path.dirname(prototype_cache_path), exist_ok=True) 
                try:
                    torch.save(data_to_save, temp_path)
                    os.rename(temp_path, prototype_cache_path) # Atomic write
                    print(f"[{rank}] Prototypes saved to cache successfully.")
                except Exception as e:
                    print(f"[{rank}] ERROR during atomic cache save: {e}")
                    if os.path.exists(temp_path): os.remove(temp_path)
        
        if world_size > 1:
            self.broadcast_prototypes(rank_str, world_size, device)

        self.train()
        return self.clustering_manager.centroids.clone()

    def forward(self, x, idx=None, labels=None, return_features=False, step="train"):
        features = self.encoder(x)["emb"]
        if step == "train" or step == "val":

            representative_feature = self.get_representative_sensor_feature(x, labels, num_total_sensors=END_INDEX-START_INDEX+1, top_k=self.top_k, id=idx)
            if labels is not None:
                for i in range(len(labels)):
                    ranges = representative_feature
            representative_feature = self.projection_layer(representative_feature)
            features = features + representative_feature
            
        similarity_scores = self.clustering_manager.compute_similarity_scores(features)

        if return_features:
            pure_features = features.detach().clone()
            return similarity_scores, features, pure_features
        return similarity_scores
    
    @torch.no_grad()
    def get_pseudo_labels(self, idx):
        device = self.clustering_manager.label_bank.device        
        sids_tensor = idx.detach().clone().to(dtype=torch.long, device=device)
        pseudo_labels = self.clustering_manager.label_bank[sids_tensor]
        return pseudo_labels
    
    def get_representative_sensor_feature(self, imu_batch, labels, num_total_sensors=97, top_k=1, id=None):
        ranges = torch.quantile(torch.abs(imu_batch), q=0.99, dim=2)
        return ranges
    
    def augment_imu_data(self, imu_data):
        return time_warp(imu_data)
    
    def update_epoch(self, epoch):
        self.epoch = epoch
   
  