
# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.
import pytorch_lightning as pl
import torch
from typing import List, Optional
import numpy as np
import clip
import json
from PIL import Image
from torchvision.transforms import Normalize
from transformers import CLIPVisionModelWithProjection

class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_type="max", embedding_size=32):
        super().__init__()
        if pool_type == "max":
            pool_fn = torch.nn.MaxPool1d(kernel_size=3)
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
            pool_fn,
        )

    def forward(self, batch):
        return self.net(batch)


class MW2StackRNNPooling(pl.LightningModule):
    def __init__(self, input_dim=32, size_embeddings: int = 128):
        super().__init__()
        self.name = MW2StackRNNPooling
        self.net = torch.nn.Sequential(
            torch.nn.GroupNorm(2, 6),
            Block(6, input_dim, 10),
            Block(input_dim, input_dim, 5),
            Block(input_dim, input_dim, 5, pool_type="adaptive", embedding_size=32),
            torch.nn.GroupNorm(4, input_dim),
            torch.nn.GRU(
                batch_first=True, input_size=input_dim, hidden_size=size_embeddings
            ),
        )

    def forward(self, batch):
        # return the last hidden state
        return self.net(batch)[1][0]

class MW2StackRNNPoolingMultihead(pl.LightningModule):
    def __init__(self, num_sensors=37, input_dim=32, size_embeddings: int = 128):
        super().__init__()
        self.name = MW2StackRNNPooling
        self.backbone = torch.nn.Sequential(
            torch.nn.GroupNorm(1, num_sensors),
            Block(num_sensors, input_dim, 10),
            Block(input_dim, input_dim, 5),
            Block(input_dim, input_dim, 5, pool_type="adaptive", embedding_size=32),
            torch.nn.GroupNorm(4, input_dim),
            torch.nn.GRU(
                batch_first=True, input_size=input_dim, hidden_size=size_embeddings
            ),
        )
        self.ssl_head = torch.nn.Linear(size_embeddings, size_embeddings)
        self.mmcl_head = torch.nn.Linear(size_embeddings, size_embeddings)

    def forward(self, batch):
        emb = self.backbone(batch)[1][0] # Last hidden state
        ssl_out = self.ssl_head(emb)
        mmcl_out = self.mmcl_head(emb)
        out = {"ssl": ssl_out, "mmcl": mmcl_out, "emb": emb}
        return out
    
class Clip4CLIPModel(pl.LightningModule):

    def __init__(self, freeze):
        super(Clip4CLIPModel, self).__init__()
        print("Loading clip4clip model ...")

        self.flag_freeze = freeze
        self.video_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

        self.video_model.eval()

        if self.flag_freeze:
            self.eval()
            self.freeze()

    def get_video_embeddings(self, video, device: Optional[str] = None):

        # This is a forward pass if features are precomputed
        if len(video.shape) == 2:
            return video

        # video: [batch_size x n_frames x grid x grid x 3]
        batch_size, n_frames, _, grid, _ = video.shape
        print(video.shape)
        video = video.reshape(batch_size * n_frames, 3, grid, grid) # [batch_size * n_frames x 3 x grid x grid] to parallelize
        visual_output_raw = self.video_model(video)
        video_features = visual_output_raw["image_embeds"]
        video_features = video_features.reshape(batch_size, n_frames, -1)

        # average over frames
        video_features = video_features.mean(dim=1)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        return video_features
