import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
from method_utils import gather
import copy

# --- user define modules ---
from model import SensorSpatialEncoder, SensorTemporalEncoder, SensorModel, VisionModel, ClusteringModule

####################################################################



class MethodLightningModule(pl.LightningModule):

    def __init__(self, args, datamodule=None):
        super().__init__()
        self.save_hyperparameters(args)

        self.video_model = VisionModel(latent_dim=self.hparams.embedding_dim)
        self.sensor_spatial_encoder= SensorSpatialEncoder(sensor_channels=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
        self.sensor_temporal_encoder = SensorTemporalEncoder(sensor_channels=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
        self.clustering_module = ClusteringModule(
            encoder=self.sensor_spatial_encoder,
            embedding_dim=self.hparams.embedding_dim,   
            num_sensors=self.hparams.num_sensors,
            num_clusters=self.hparams.num_classes,
            datamodule=datamodule,
            top_k=self.hparams.top_k,
            prototype_cache_dir=os.path.join(self.hparams.cache_dir, "prototypes"),
            dataset_name=self.hparams.dataset_name,
            min_cluster_size=self.hparams.min_cluster_size,
            mid_label=self.hparams.mid_label
        )
        self.sensor_model = SensorModel(self.sensor_spatial_encoder, self.sensor_temporal_encoder, self.clustering_module)
        self.spatial_classifier = nn.Linear(256, self.hparams.num_classes) 
    
        self.video_classifier = nn.Linear(self.hparams.embedding_dim, self.hparams.num_classes)
        print(self.global_rank, "Model initialized.")
        self.epoch = 0        

        self.momentum_video_model = copy.deepcopy(self.video_model)
        for param in self.momentum_video_model.parameters():
            param.requires_grad = False
                                                             
        self.momentum_sensor_model = copy.deepcopy(self.sensor_model)
        for param in self.momentum_sensor_model.parameters():
            param.requires_grad = False
           


    @torch.no_grad()
    def _update_momentum_encoders(self):
        m = self.hparams.momentum_m 
        
        for param_q, param_k in zip(self.video_model.parameters(), 
                                self.momentum_video_model.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
            
        for param_q, param_k in zip(self.sensor_model.parameters(), 
                                self.momentum_sensor_model.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def enable_stage2(self):
        for n, p in self.video_model.named_parameters():
            if any(key in n for key in ["shared_encoder", "scene", "object", "fuse", "spatial", "video_classifier"]):
                p.requires_grad = False
        
        for n, p in self.sensor_model.named_parameters():
            if any(key in n for key in ["spatial"]):
                p.requires_grad = False

        for n, p in self.video_model.named_parameters():
            if "temporal_encoder" in n:
                p.requires_grad = True
                print("video temporal encoder trainable")
        
        for n, p in self.sensor_model.named_parameters():
            if "temporal_encoder" in n:
                p.requires_grad = True
                print("sensor temporal encoder trainable")


        self.stage2_only = True

    def on_train_epoch_start(self):
        self.training_steps_outputs = [] 
        self.debug = []

        stage2_start = (
            self.hparams.threshold_epoch +
            self.hparams.video_classifier_epoch +
            self.hparams.bad_correction_epoch
        )
        if self.epoch >= stage2_start: 
            if not self.stage2_only:
                self.enable_stage2()


    def training_step(self, batch, batch_idx):
  
        videos, sensors, _, sample_ids, flows = batch
        idx, _ = sample_ids

        # --- Stage 1 ---
        # --- 1. Clustering ---
        
        if self.clustering_module.epoch > 1:  
            imu_data_aug = self.clustering_module.augment_imu_data(sensors)
        else:
            imu_data_aug = sensors

        scores, features, _ = self.sensor_model.encoding_spatial(imu_data_aug, labels=None, return_features=True, idx=idx)
        
        stored_pseudo_labels = self.clustering_module.get_pseudo_labels(idx)

        scores_original = scores.clone()

        with torch.no_grad():
            centroids = self.clustering_module.clustering_manager.centroids  # [K, D]
            batch_centroids = centroids[stored_pseudo_labels]               # [B, D]
            centroid_distances = torch.sqrt(((F.normalize(features, dim=1) - batch_centroids) ** 2).sum(dim=1))

            good_mask = torch.zeros_like(centroid_distances, dtype=torch.bool)

            num_clusters = self.clustering_module.clustering_manager.num_clusters
            for k in range(num_clusters):
                cluster_mask = stored_pseudo_labels == k
                if cluster_mask.any():
                    cluster_dists = centroid_distances[cluster_mask]

                    manager = self.clustering_module.clustering_manager
                    if not hasattr(manager, "cluster_stats") or len(manager.cluster_stats) == 0:
                        manager.compute_class_weights()  
                    if self.hparams.centroid_threshold == 0.75:
                        threshold = manager.cluster_stats[k]["q75"]
                    elif self.hparams.centroid_threshold == 0.50:
                        threshold = manager.cluster_stats[k]["q50"]
                    elif self.hparams.centroid_threshold == 1.00:
                        threshold = manager.cluster_stats[k]["all"]
                    else:
                        raise("not calculated stat")
                    good_mask[cluster_mask] = cluster_dists < threshold

            bad_mask = ~good_mask
            bad_indicator = bad_mask.long()

            distance_info = {
                'centroid_distances': centroid_distances.cpu(),
                'pseudo_labels': stored_pseudo_labels.cpu(),
            }

            self.clustering_module.clustering_manager.update_samples_memory(idx, features)

            loss_labels = stored_pseudo_labels.to(self.device)
            mask_new = (loss_labels == -1)
            if mask_new.any():
                loss_labels[mask_new] = torch.argmax(scores_original[mask_new], dim=1)


        class_weights = self.clustering_module.clustering_manager.compute_class_weights()

        loss_cluster = (scores_original.sum() * 0.0) 
        
        if self.epoch < self.hparams.threshold_epoch:
            loss_cluster = F.cross_entropy(
                scores_original,
                loss_labels,      
                weight=class_weights.to(self.device)
            )
            return loss_cluster
        else:
            # --- Stage 1-Refine ---
            if good_mask.any(): 
                loss_cluster = F.cross_entropy(
                    scores_original[good_mask], 
                    loss_labels[good_mask],
                    weight=class_weights.to(self.device)
                )
        
       # --- Decompose ---
        model_output = self.video_model(videos, flows=flows)
        v_spatial = model_output["v_spatial"]

        sensor_output = self.sensor_model.sensor_spatial_encoder(sensors)
        sensor_emb = sensor_output["emb"]

        # --- Cross-Modal Pseudo Label Refinement ---
        loss_video_supervised = torch.tensor(0.0, device=self.device)
        loss_sensor_guided = torch.tensor(0.0, device=self.device)

        if good_mask.any(): 
            video_logits_good = self.video_classifier(v_spatial[good_mask])
            loss_video_supervised = F.cross_entropy(
                video_logits_good,
                loss_labels[good_mask]
            )

        if bad_mask.any() and self.epoch >= self.hparams.threshold_epoch + self.hparams.video_classifier_epoch:
            with torch.no_grad():
                video_logits_bad = self.video_classifier(v_spatial[bad_mask])
                video_probs_bad = F.softmax(video_logits_bad, dim=1)
                video_max_probs, video_pseudo_bad = torch.max(video_probs_bad, dim=1)
                high_confidence_mask = video_max_probs >= self.hparams.threshold_classifier_confidence


            if high_confidence_mask.any():
                loss_sensor_guided = F.cross_entropy(
                    scores_original[bad_mask][high_confidence_mask],
                    video_pseudo_bad[high_confidence_mask]
                )

        # --- final loss of refinement ---
        lambda_cluster = 1.0
        lambda_video_sup = 1.0
        lambda_sensor_guide = 1.5

        final_loss = (
            lambda_cluster * loss_cluster +
            lambda_video_sup * loss_video_supervised +
            lambda_sensor_guide * loss_sensor_guided 
        )
        
        stage2_start = (
            self.hparams.threshold_epoch +
            self.hparams.video_classifier_epoch +
            self.hparams.bad_correction_epoch
        )
        if self.epoch > stage2_start: 

            z_video_online = model_output["z_video_online"]
            z_sensor_online = self.sensor_model(imu_data_aug)

            # z_video_online = F.normalize(z_video_online, dim=1)
            # z_sensor_online = F.normalize(z_sensor_online, dim=1)
  
            with torch.no_grad():
                v_temporal_mom = self.momentum_video_model(videos, flows)["v_temporal"]
                sensor_temporal_mom = self.momentum_sensor_model.encoding_temporal(sensors)["emb"]

            gathered_z_video = F.normalize(gather(z_video_online), dim=1)
            gathered_z_sensor = F.normalize(gather(z_sensor_online), dim=1)
            gathered_v_mom = F.normalize(gather(v_temporal_mom), dim=1)
            gathered_s_mom = F.normalize(gather(sensor_temporal_mom), dim=1)
            gathered_v_app = F.normalize(gather(v_spatial.detach()), dim=1)
            gathered_features = F.normalize(gather(features.detach()), dim=1)

            # cal W_spatial
            sim_app_vid = gathered_v_app @ gathered_v_app.T
            sim_app_sen = gathered_features @ gathered_features.T

            sim_app_vid_rel = F.relu(sim_app_vid)
            sim_app_sen_rel = F.relu(sim_app_sen)
            sim_app_max = torch.max(sim_app_vid_rel, sim_app_sen_rel) # [N, N]

            W_app = 1.0 + (self.hparams.lambda_hard - 1.0) * sim_app_max
            # cal W_temp
            sim_vid_mom = gathered_v_mom @ gathered_v_mom.T
            sim_sen_mom = gathered_s_mom @ gathered_s_mom.T
            sim_cross_mom = gathered_v_mom @ gathered_s_mom.T
            
            sim_stable = 0.5 * (sim_vid_mom + sim_sen_mom)

         
            
            temporal_damp_factor = F.relu(sim_stable)
            W_conditional_damp = 1.0 - sim_app_max * temporal_damp_factor
            
            W_final = W_app * W_conditional_damp

            N = W_app.shape[0]
            identity = torch.eye(N, device=self.device, dtype=torch.bool)
            W_final = W_final.masked_fill(identity, 0.0)

            # --- InfoNCE ---
            # 1. Logits
            D = z_video_online.shape[1] // 2

            sim_v2s = gathered_z_video @ gathered_z_sensor.T
            sim_s2v = sim_v2s.T
            logits_v2s = sim_v2s / self.hparams.contrastive_temp # (e.g., 0.07)
            logits_s2v = sim_s2v / self.hparams.contrastive_temp
            
            # 2. Positive / Negative
            pos_v2s = logits_v2s.diag()
            pos_s2v = logits_s2v.diag()
            
            # Positive masking
            neg_logits_v2s = logits_v2s.masked_fill(identity, -torch.inf)
            neg_logits_s2v = logits_s2v.masked_fill(identity, -torch.inf)
            
            # 3. Weighted Negative Sampling
            weighted_neg_v2s = (W_final * torch.exp(neg_logits_v2s)).sum(dim=1)
            weighted_neg_s2v = (W_final * torch.exp(neg_logits_s2v)).sum(dim=1)
            
            # 4. Loss computation
            denominator_v2s = torch.exp(pos_v2s) + weighted_neg_v2s
            denominator_s2v = torch.exp(pos_s2v) + weighted_neg_s2v
            loss_v2s = -torch.log(torch.exp(pos_v2s) / denominator_v2s).mean()
            loss_s2v = -torch.log(torch.exp(pos_s2v) / denominator_s2v).mean()
            custom_contrastive_loss = (loss_v2s + loss_s2v) / 2.0
            lambda_contrastive = 1.0

            final_loss = (
                lambda_contrastive * custom_contrastive_loss
            )
        return final_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.stage2_only:
            print("update momentum")
            self._update_momentum_encoders()

    def on_train_epoch_end(self):    
        self.epoch += 1
        self.clustering_module.update_epoch(self.epoch)

        with torch.no_grad():
            if self.epoch % self.clustering_module.centroids_update_interval == 0:
                self.clustering_module.clustering_manager.update_centroids_memory()
            
            if self.epoch % self.clustering_module.deal_with_small_clusters_interval == 0:
                self.clustering_module.clustering_manager.deal_with_small_clusters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        return optimizer
