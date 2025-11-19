import os
import argparse
import torch
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path 

# --- user define modules ---
from datamodule import MethodDataModule
from method_utils import gather

####################################################################
#                         Utility Functions
####################################################################

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_module_params(args):
    try:
        path = Path(args.checkpoint_path)
                
        args.ckpt_name = path.parent / path.stem

        print(f"Dataset: {args.dataset_name}, Model: {args.model_name}, Ckpt: {args.ckpt_name}")
    
    except Exception as e:
        print(f"set module error: {e}")
    return args

def get_backbone_with_mode(args):

    backbone = load_pretrained_model(args)
    if hasattr(backbone, "to"):
        print("move to cuda")
        backbone = backbone.to("cuda")
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    trainable_params = sum(p.requires_grad for p in backbone.parameters())
    if trainable_params:
        sample_layers = [n for n, p in backbone.named_parameters() if p.requires_grad][:5]
        print(f"→ Sample trainable layers: {sample_layers}")
    else:
        print("✅ All parameters are frozen (no grad flow to backbone).")

    return backbone


def load_pretrained_model(args, reset_sensor_weights=False):
    ckpt = args.checkpoint_path
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"--- Loading pretrained model from: {ckpt} ---")

    if args.model_name == "method":
        from method import MethodLightningModule
        model = MethodLightningModule.load_from_checkpoint(ckpt, strict=False)
    elif args.model_name == "comodo":
        from baseline_modules.comodo.module import COMODOLightningModule
        model = COMODOLightningModule.load_from_checkpoint(ckpt, strict=False, map_location='cpu').to('cuda')
    elif args.model_name == "primus":
        from baseline_modules.primus import PRIMUSLightningModule
        model = PRIMUSLightningModule.load_from_checkpoint(ckpt)
    elif args.model_name == "imu2clip":
        from baseline_modules.imu2clip import IMU2CLIPLightningModule
        model = IMU2CLIPLightningModule.load_from_checkpoint(ckpt)
    elif args.model_name == "mae":
        from baseline_modules.mae import EVIMAELightningModule
        model = EVIMAELightningModule.load_from_checkpoint(ckpt, map_location='cpu').to('cuda')
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")
    
    return model


####################################################################
#                        Linear Probe Module
####################################################################

class LinearProbeLightningModule(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = model

        emb_dim = model.hparams.embedding_dim
        if self.hparams.model_name == "method":
            emb_dim *= 2
        elif self.hparams.model_name == "comodo":
            emb_dim //= 2

        self.classifier = torch.nn.Linear(emb_dim, self.hparams.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
 
        # F1 Scores (Micro, Macro, Weighted)
        self.val_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='micro')
        self.val_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.val_f1_weighted = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='weighted')
        
        self.test_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='micro')
        self.test_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.test_f1_weighted = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='weighted')

        # mAUC (Multiclass AUROC)
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=self.hparams.num_classes, average='macro')

        # mAP (Multiclass Average Precision)
        self.val_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=self.hparams.num_classes, average='macro')

        # --- Confusion Matrix ---
        self.val_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)

        self.class_dic = {
            0: 'Open Door 1', 1: 'Open Door 2', 2: 'Close Door 1', 3: 'Close Door 2',
            4: 'Open Fridge', 5: 'Close Fridge', 6: 'Open Dishwasher', 7: 'Close Dishwasher',
            8: 'Open Drawer 1', 9: 'Close Drawer 1', 10: 'Open Drawer 2',
            11: 'Close Drawer 2', 12: 'Open Drawer 3', 13: 'Close Drawer 3'
        }
        self.class_names = [v for v in self.class_dic.values()]

    def forward(self, sensor_data):
        if self.hparams.model_name == "method":
            sensor_encoder = self.model.sensor_model
            representations = sensor_encoder(sensor_data)
        elif self.hparams.model_name == "imu2clip":
            sensor_encoder = self.model.sensor_model
            sensor_data = self.model.sensor_padding(sensor_data)
            representations = sensor_encoder(sensor_data)
        elif self.hparams.model_name == "primus":
            sensor_encoder = self.model.sensor_model
            representations = sensor_encoder(sensor_data)['mmcl']
        elif self.hparams.model_name == "mae":
            representations = self.model.model.forward_sensor_only(sensor_data)
        elif self.hparams.model_name == "comodo":
            sensor_encoder = self.model.sensor_model
            representations = sensor_encoder(sensor_data)
        else:
            raise ValueError(f"Unknown model_name for loading: {self.hparams.model_name}")
        logits = self.classifier(representations)
        
        return logits

    def _shared_step(self, batch, batch_idx):
        _, sensor_data, y, _, _ = batch
        logits = self(sensor_data)
        loss = self.criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, probs, y

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print("grad: ", loss.requires_grad)
        return loss

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        loss, preds, probs, y = self._shared_step(batch, batch_idx)
        self.val_accuracy.update(preds, y)
        self.val_f1_micro.update(preds, y)
        self.val_f1_macro.update(preds, y)
        self.val_f1_weighted.update(preds, y)
        self.val_auroc.update(probs, y) 
        self.val_ap.update(probs, y) 
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(y.detach().cpu())

        self.log("val/val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val/val_f1_micro", self.val_f1_micro, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/val_f1_macro", self.val_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/val_f1_weighted", self.val_f1_weighted, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/val_mAUC", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/val_mAP", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)

        
    def on_validation_epoch_end(self):
        val_labels = gather(torch.cat(self.val_labels, dim=0))
        val_preds = gather(torch.cat(self.val_preds, dim=0))
        y_true = val_labels.cpu().numpy()
        y_pred = val_preds.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,                     
            annot=True, fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=True, square=True,
            ax=ax
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Validation Confusion Matrix")

        if self.logger is not None:
            self.logger.experiment.log({"Validation Confusion Matrix": wandb.Image(fig)})
        plt.close(fig)

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_labels = []

    def test_step(self, batch, batch_idx):
        loss, preds, probs, y = self._shared_step(batch, batch_idx)
        self.test_accuracy.update(preds, y)
        self.test_f1_micro.update(preds, y)
        self.test_f1_macro.update(preds, y)
        self.test_f1_weighted.update(preds, y)
        self.test_auroc.update(probs, y)
        self.test_ap.update(probs, y)

        self.test_preds.append(preds.detach().cpu())
        self.test_labels.append(y.detach().cpu())

        self.log("test/test_loss", loss, on_epoch=True)
        self.log("test/test_acc", self.test_accuracy, on_epoch=True)
        self.log("test/test_f1_macro", self.test_f1_macro, on_epoch=True)
        self.log("test/test_f1_micro", self.test_f1_micro, on_step=False, on_epoch=True, prog_bar=False) 
        self.log("test/test_f1_weighted", self.test_f1_weighted, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/test_mAUC", self.test_auroc, on_epoch=True)
        self.log("test/test_mAP", self.test_ap, on_epoch=True)

    def on_test_epoch_end(self):
        test_labels = gather(torch.cat(self.test_labels, dim=0))
        test_preds = gather(torch.cat(self.test_preds, dim=0))
        y_true = test_labels.cpu().numpy()
        y_pred = test_preds.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Test Confusion Matrix")
        if self.logger is not None:
            self.logger.experiment.log({"Test Confusion Matrix": wandb.Image(fig)})
        plt.close(fig)


    def configure_optimizers(self):
        groups = [{'params': self.classifier.parameters(), 'lr': self.hparams.lr}]
        return torch.optim.Adam(groups)

class LinearProbeLSTM(pl.LightningModule):
    def __init__(self, args, backbone):
        super().__init__()
        self.save_hyperparameters(args)
        self.backbone = backbone 

        emb_dim = int(self.backbone.hparams.embedding_dim)
        if self.hparams.model_name == "method":
            emb_dim *= 2
        self.emb_dim = emb_dim
        self.temporal_model = None  # 단순 mean pooling 사용
        self.classifier = nn.Linear(self.emb_dim, self.hparams.num_classes)

        # Metrics
        self.criterion = nn.CrossEntropyLoss()
        # -----------------------------
        # Metrics (VAL)
        # -----------------------------
        nc = self.hparams.num_classes
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc)
        self.val_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='micro')
        self.val_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='macro')
        self.val_f1_weighted = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='weighted')
        self.val_prec_macro = torchmetrics.Precision(task="multiclass", num_classes=nc, average='macro')
        self.val_recall_macro = torchmetrics.Recall(task="multiclass", num_classes=nc, average='macro')
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=nc, average='macro')
        self.val_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=nc, average='macro')
        self.val_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=nc)

        # -----------------------------
        # Metrics (TEST)
        # -----------------------------
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc)
        self.test_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='micro')
        self.test_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='macro')
        self.test_f1_weighted = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='weighted')
        self.test_prec_macro = torchmetrics.Precision(task="multiclass", num_classes=nc, average='macro')
        self.test_recall_macro = torchmetrics.Recall(task="multiclass", num_classes=nc, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=nc, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=nc, average='macro')
        self.test_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=nc)

        self.val_preds_raw = []
        self.val_labels_raw = []
        self.test_preds_raw = []
        self.test_labels_raw = []
        

    def _encode_windows(self, windows_b_sct):
        B, S, C, T = windows_b_sct.shape
        flat = windows_b_sct.reshape(B * S, C, T)

        if self.hparams.model_name in ["method", "comodo", "primus", "imu2clip"]:
            sensor_encoder = self.backbone.sensor_model
        if self.hparams.model_name == "method":
            reps = sensor_encoder(flat)
        elif self.hparams.model_name == "imu2clip":
            flat_padded = self.backbone.sensor_padding(flat)
            reps = sensor_encoder(flat_padded)
        elif self.hparams.model_name == "primus":
            reps = sensor_encoder(flat)['mmcl']
        elif self.hparams.model_name == "mae":
            reps = self.backbone.model.forward_sensor_only(flat)
        elif self.hparams.model_name == "comodo":
            reps = sensor_encoder(flat)
        else:
            raise ValueError(f"Unknown model_name: {self.hparams.model_name}")

        reps = reps.reshape(B, S, -1)
        return reps
        
    def forward(self, windows_b_sct, lengths_b):
        seq_emb = self._encode_windows(windows_b_sct)  # [B, S, D]

        pooled = seq_emb.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

    def _shared_step(self, batch):
        windows, lengths, targets, metas = batch
        windows = windows.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device)
        targets = targets.to(self.device)

        logits = self(windows, lengths)
        loss = self.criterion(logits, targets)
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)
        return loss, preds, probs, targets

    # -----------------------------
    # Train
    # -----------------------------
    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # -----------------------------
    # Validation
    # -----------------------------
    def on_validation_epoch_start(self):
        self.val_preds_raw.clear()
        self.val_labels_raw.clear()

    def validation_step(self, batch, batch_idx):
        loss, preds, probs, targets = self._shared_step(batch)
        self.val_acc.update(preds, targets)
        self.val_f1_micro.update(preds, targets)
        self.val_f1_macro.update(preds, targets)
        self.val_f1_weighted.update(preds, targets)
        self.val_prec_macro.update(preds, targets)
        self.val_recall_macro.update(preds, targets)
        self.val_conf_matrix.update(preds, targets)

        self.val_auroc.update(probs, targets)
        self.val_ap.update(probs, targets)

        self.val_preds_raw.append(preds.detach().cpu())
        self.val_labels_raw.append(targets.detach().cpu())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):

        # scalar compute
        log_dict = {
            "val/acc": self.val_acc.compute(),
            "val/f1_micro": self.val_f1_micro.compute(),
            "val/f1_macro": self.val_f1_macro.compute(),
            "val/f1_weighted": self.val_f1_weighted.compute(),
            "val/precision_macro": self.val_prec_macro.compute(),
            "val/recall_macro": self.val_recall_macro.compute(),
            "val/auroc_macro": torch.nan_to_num(self.val_auroc.compute()),
            "val/ap_macro": torch.nan_to_num(self.val_ap.compute()),
        }
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        # reset
        for m in [
            self.val_acc, self.val_f1_micro, self.val_f1_macro, self.val_f1_weighted,
            self.val_prec_macro, self.val_recall_macro, self.val_auroc, self.val_ap,
            self.val_conf_matrix
        ]:
            m.reset()

    # -----------------------------
    # Test
    # -----------------------------
    def on_test_epoch_start(self):
        self.test_preds_raw.clear()
        self.test_labels_raw.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, probs, targets = self._shared_step(batch)

        self.test_acc.update(preds, targets)
        self.test_f1_micro.update(preds, targets)
        self.test_f1_macro.update(preds, targets)
        self.test_f1_weighted.update(preds, targets)
        self.test_prec_macro.update(preds, targets)
        self.test_recall_macro.update(preds, targets)
        self.test_conf_matrix.update(preds, targets)

        self.test_auroc.update(probs, targets)
        self.test_ap.update(probs, targets)

        self.test_preds_raw.append(preds.detach().cpu())
        self.test_labels_raw.append(targets.detach().cpu())

        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self):
        log_dict = {
            "test/acc": self.test_acc.compute(),
            "test/f1_micro": self.test_f1_micro.compute(),
            "test/f1_macro": self.test_f1_macro.compute(),
            "test/f1_weighted": self.test_f1_weighted.compute(),
            "test/precision_macro": self.test_prec_macro.compute(),
            "test/recall_macro": self.test_recall_macro.compute(),
            "test/auroc_macro": torch.nan_to_num(self.test_auroc.compute()),
            "test/ap_macro": torch.nan_to_num(self.test_ap.compute()),
        }
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        for m in [
            self.test_acc, self.test_f1_micro, self.test_f1_macro, self.test_f1_weighted,
            self.test_prec_macro, self.test_recall_macro, self.test_auroc, self.test_ap,
        ]:
            m.reset()

    def configure_optimizers(self):
        params = list(self.classifier.parameters())
        return torch.optim.Adam(params, lr=self.hparams.lr)


def main(args):
    set_random_seed(42)
    args = set_module_params(args)

    if args.dataset_name == "HWU-USP":
        args.num_classes = 5
        args.threshold_epoch = 100 # need only sensor data
        args.seq_len = 100
        args.num_sensors = 6
        datamodule = MethodDataModule(args, stage="linear_probe_lstm")
        datamodule.setup("fit") 
        backbone = get_backbone_with_mode(args)
        model = LinearProbeLSTM(args, backbone)

        class_names = list(datamodule.train_dataset.class_to_idx.keys())
        model.class_names = class_names

    else:
        args.num_classes = 14
        args.threshold_epoch = 100 # need only sensor data
        args.seq_len = 128
        args.num_sensors = 37
        datamodule = MethodDataModule(args, stage='linear_probe')
        
        backbone = get_backbone_with_mode(args)
        model = LinearProbeLightningModule(args, backbone)
    
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 4
    if args.sensor_only:
        run_name = f"{args.sensor_model_name}_{args.dataset_name}_linear_probe_{args.batch_size* world_size}_linearEpoch={args.linear_epochs}"
    else:
        sensor_tag = "_supervision" if args.supervision else ""
        run_name = f"{args.model_name}_{args.dataset_name}_{args.ckpt_name}_{backbone.hparams.batch_size*4}_{backbone.hparams.epochs}_linear_probe_{args.batch_size* world_size}_linearEpoch={args.linear_epochs}{sensor_tag}"
    is_master_process = os.environ.get("LOCAL_RANK", "0") == "0"
    logger = WandbLogger(project="Method_Linear_Probe", name=run_name) if is_master_process else False

    trainer = pl.Trainer(
        max_epochs=args.linear_epochs,
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true',
        logger=logger,
    )

    print("--- Starting Linear Probing ---")
    trainer.fit(model, datamodule)
    datamodule.setup("fit")

    print("--- Testing ---")
    trainer.test(model=model, dataloaders=datamodule.test_dataloader())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--linear_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--supervision', type=bool, default=False, help='If True, skip backbone weight loading and run with frozen hyperparameter configs only')
    parser.add_argument('--sensor_only', action='store_true', help='Train sensor-only model from scratch (no checkpoint)')
    parser.add_argument('--sensor_model_name', type=str, default='dlinear', choices=['dlinear', 'timesnet', 'moment', 'mantis', 'imu2clip', 'comodo'])
    parser.add_argument('--embedding_dim', type=str, default=256, help='for sensor only mode')

    args = parser.parse_args()
    main(args)