import os
import numpy as np
import random
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
import wandb
import torch.distributed as dist

# --- user define modules ---
from datamodule import MethodDataModule
from method import MethodLightningModule


####################################################################

def set_random_seed(seed):
    seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def set_model(args, datamodule):
    global caching_callback
    if args.dataset_name == "Opportunity++":
        args.num_sensors = 37
        args.num_classes = 7
        args.top_k = 4
        args.sensor_seq_len = 128
        args.min_cluster_size = 100
        args.mid_label = False
        if args.centroid_threshold > 0.99:
            args.threshold_classifier_confidence = 0.00
        else:  
            args.threshold_classifier_confidence = 0.90
    elif args.dataset_name == "HWU-USP":
        args.num_sensors = 6
        args.num_classes = 7
        args.top_k = 1
        args.sensor_seq_len = 128
        args.min_cluster_size = 30
        args.mid_label = True
        if args.centroid_threshold > 0.99:
            args.threshold_classifier_confidence = 0.00
        else:
            args.threshold_classifier_confidence = 0.99

    args.baseline_video_cache_dir = f"./video_caches/{args.model_name}/{args.dataset_name}"
    args.seed = 42
    if args.model_name == "method":
        args.embedding_dim //= 2 # each encoder has half of total embedding_dim
        return MethodLightningModule(args, datamodule)
    elif args.model_name == "comodo":
        from baseline_modules.comodo import initialize_comodo
        args.video_ckpt = "facebook/timesformer-base-finetuned-k400"
        args.imu_ckpt = "paris-noah/Mantis-8M"
        args.mlp_hidden_dim = 2048
        args.mlp_output_dim = 128
        args.embedding_dim = 128
        args.queue_size_ratio = 0.1
        args.reduction = "concat" # "mean", "concat"
        args.teacher_temp = 0.1
        args.student_temp = 0.05
        args.learning_rate = 3e-4
        # datamodule.fit("")
        return initialize_comodo(args, datamodule)
    elif args.model_name == "primus":
        from baseline_modules.primus import PRIMUSLightningModule  
        args.num_views = 2 
        args.ssl_coeff = 0.5
        args.nnclr = True
        # args.transform_indices = [2, 4]
        # 원본은 2,4이나 현재 64*4>128 이므로 4에서 에러남. 복원 추출 가능케 하거나 num_segments를 바꾸는 등의 파라미터 수정이 필요.
        args.transform_indices = [0, 1]
        args.embedding_dim = 512
        return PRIMUSLightningModule(args)
    elif args.model_name == "imu2clip":
        from baseline_modules.imu2clip import IMU2CLIPLightningModule
        args.embedding_dim = 512
        args.sensor_target_len = 150
        return IMU2CLIPLightningModule(args)
    elif args.model_name == "mae":
        from baseline_modules.mae import EVIMAELightningModule
        args.masking_ratio = 0.75
        args.mi_loss_weight = 0.01
        args.mae_loss_weight = 1.0
        args.norm_pix_loss = True
        args.lrscheduler_start = 10
        args.lrscheduler_decay = 0.5
        args.lrscheduler_step = 5
        args.embedding_dim = 768 # VIT
        return EVIMAELightningModule(args)
    else:
        NameError(f"Invalid model name: {args.model_name}")   

####################################################################


class DatasetEpochCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print("dataset epoch callback called")
        if hasattr(trainer.datamodule.train_dataset, 'set_epoch'):
            trainer.datamodule.train_dataset.set_epoch(trainer.current_epoch)

####################################################################

def main(args):
    set_random_seed(42)

    datamodule = MethodDataModule(args, "pretrain")

    args.cache_dir = datamodule.cache_dir
    model = set_model(args, datamodule)

    is_master_process = os.environ.get("LOCAL_RANK", "0") == "0"
    logging_name = f"pretraining_{args.model_name}_{args.dataset_name}"
    logger = WandbLogger(project=args.project_name, name=logging_name) if is_master_process else False
    wandb.init(project=args.project_name, name=logging_name) if is_master_process else None
            
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{args.model_name}/{args.dataset_name}", 
        filename="pretrained_model-{epoch:02d}-{train_loss:.2f}", 
        save_top_k=1,    
        monitor="train/contrastive_loss",  
        mode="min",  
        save_last=True,
    )
    callbacks = [DatasetEpochCallback(), checkpoint_callback]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true',
        logger=logger,
        callbacks=callbacks,
    )

    print("--- Starting Training with PyTorch Lightning ---")
    trainer.fit(model, datamodule)
    print("--- Training Complete ---")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Method Test with PyTorch Lightning")
    # path hyperparameters
    parser.add_argument("--dataset_name", type=str, default="Opportunity++", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="method")
    parser.add_argument("--visualize_output_dir", type=str, default="/home/junho/Method/Visualization/transformed_video", help="Directory to save visualization outputs")
    parser.add_argument("--project_name", type=str, default="Method_Test_Lightning", help="WandB project name")
    parser.add_argument("--save_stage_cache", type=bool, default=False, help="Whether use stage cache")
    parser.add_argument("--save_weights", type=bool, default=False, help="Whether save weights")

    # learning hyperparameters   
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=512) # Total
    parser.add_argument("--alpha_fixed", type=bool, default=True)
    parser.add_argument("--threshold_epoch", type=int, default=5)
    parser.add_argument("--centroid_threshold", type=float, default=0.75)
    parser.add_argument("--video_classifier_epoch", type=int, default=10)
    parser.add_argument("--bad_correction_epoch", type=int, default=10)

    parser.add_argument("--momentum_m", type=float, default=0.999)
    parser.add_argument("--lambda_hard", type=float, default=3.0)
    parser.add_argument("--motion_damp_temp", type=float, default=0.1)
    parser.add_argument("--contrastive_temp", type=float, default=0.10)
    parser.add_argument("--ablation_study", type=str, default=None)

    parser.add_argument("--damp_warmup_epochs", type=int, default=5)

    
    args = parser.parse_args()
    args.stage = "pretrain"

    main(args)