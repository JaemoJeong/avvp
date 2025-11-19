import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR
from .model import EVIMAE
from baseline_modules import BasePretrainModule

class EVIMAELightningModule(BasePretrainModule):
    def __init__(self, args):
        super().__init__(args)
        
        self.model = EVIMAE(
            sensor_in_chans=self.hparams.num_sensors,
            embed_dim=self.hparams.embedding_dim,
            sensor_seq_len=self.hparams.sensor_seq_len,
            norm_pix_loss=self.hparams.norm_pix_loss
        )

    def training_step(self, batch, batch_idx):
        video, sensor, _, _, _ = batch
        loss, loss_mae, loss_mae_s, loss_mae_v, loss_c, c_acc = self.model(
            sensor=sensor,
            video=video,
            mask_ratio_s=self.hparams.masking_ratio,
            mask_ratio_v=self.hparams.masking_ratio,
            mae_loss_weight=self.hparams.mae_loss_weight,
            mi_loss_weight=self.hparams.mi_loss_weight,
        )

        self.log("pretrain_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("mae_loss", loss_mae, logger=True, on_step=True, on_epoch=True)
        self.log("sensor_mae_loss", loss_mae_s, logger=True, on_step=True, on_epoch=True)
        self.log("video_mae_loss", loss_mae_v, logger=True, on_step=True, on_epoch=True)
        self.log("contrastive_loss", loss_c, logger=True, on_step=True, on_epoch=True)
        self.log("contrastive_acc", c_acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=5e-7, 
            betas=(0.95, 0.999)
        )
        
        scheduler = MultiStepLR(
            optimizer, 
            milestones=list(range(self.hparams.lrscheduler_start, 1000, self.hparams.lrscheduler_step)),
            gamma=self.hparams.lrscheduler_decay
        )
        
        return [optimizer], [scheduler]