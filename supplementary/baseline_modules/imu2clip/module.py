# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

from .model import MW2StackRNNPooling, ClipPLModel
from baseline_modules.base import BasePretrainModule
from baseline_modules.loss import InfoNCE
import torch.nn.functional as F

class IMU2CLIPLightningModule(BasePretrainModule):
    def __init__(self, args):
        super().__init__(args)

        self.loss = InfoNCE(symmetric_loss=True, learn_temperature=True)

        self.sensor_model = MW2StackRNNPooling(num_sensors=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
        self.video_model = ClipPLModel(freeze=True)

    def forward(self, batch):
        # x_sensor: (batch_size x 6 x window_size)
        # x_narration: [ str ] with len == batch_size
        # y_*: B x size_embeddings

        out = {}

        videos, sensors, labels, sample_ids, _ = batch
        x_sensor = self.sensor_padding(sensors)
        y_sensor = self.sensor_model(x_sensor)
        out["sensor"] = y_sensor

        x_video = videos
        # (B, T, C, H, W) -> (B, C, T, H, W)로 차원 변경
        x_video = x_video.permute(0, 2, 1, 3, 4)
        y_video = self.video_model.get_video_embeddings(x_video)
        out["video"] = y_video

        return out

    def training_step(self, batch, batch_idx: int):
        # y: {modality[str]: y_*} where y_*: B x size_embeddings
        print("training step batch idx:", batch_idx)
        y = self(batch)

        # Use NCE loss
        y_query_modality = y["sensor"]
        loss_output = 0.0

        # Compute loss for source modality <> each target modality
        y_key_modality = y["video"]
        print(y_query_modality.shape, y_key_modality.shape)
        s2t_loss = self.loss(query=y_query_modality, positive_key=y_key_modality)

        # Log the loss
        str_s2t = "i2v"
        self.log(f"train_{str_s2t}_loss", s2t_loss, logger=True, sync_dist=True)
        loss_output += s2t_loss

        self.log("train_loss", loss_output, logger=True, sync_dist=True)
        return loss_output

    def predict_step(self, batch, batch_idx: int):
        return self(batch)
    def sensor_padding(self, sensor_data):
        current_len = sensor_data.shape[-1]
        target_len = self.hparams.sensor_target_len

        if current_len < target_len:
            padding_needed = target_len - current_len
            padding = (0, padding_needed) 
            x_sensor = F.pad(sensor_data, padding, mode='constant', value=0)
        else:
            x_sensor = sensor_data
        return x_sensor