import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# --- user define modules ---
from dataset import VideoSensorDataset, SensorTransform, ClipConsistentTransforms
from dataset_lstm import SequenceDataset, collate_variable_length
from method_utils import (
    calculate_sensor_stats, save_stats, load_stats
)

####################################################################

class MethodDataModule(pl.LightningDataModule):
    def __init__(self, args, stage='pretrain'):
        super().__init__()

        self.set_dataset_params(args, stage)
        self.num_frames = args.num_frames if hasattr(args, "num_frames") else 20
        if args.model_name=="method":
            self.threshold_epoch = args.threshold_epoch
        else:
            self.threshold_epoch = -1
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        self.train_transform = ClipConsistentTransforms(size=(224, 224))
        self.stage = stage
    
    def set_dataset_params(self, args, stage):
        if args.dataset_name == "Opportunity++":
            # self.data_root =
            self.json_path = os.path.join(self.data_root, "action")
            # self.stats_file_path  = 
            self.start_index, self.end_index = 194, 230
            self.cache_dir = os.path.join(self.data_root, "caches")

        elif args.dataset_name == "HWU-USP":
            # self.data_root =
            # self.json_path = os.path.join(self.data_root, "action")
            # self.stats_file_path = 
            self.start_index, self.end_index = 4, 9
            # self.cache_dir = os.path.join(self.data_root, "caches")

        else:
            raise ValueError(f"Invalid dataset name: {args.dataset_name}")

        # ============================================================
        if stage == 'pretrain':
            print("INFO: DataModule configured for PRE-TRAINING stage.")
            self.json_train_path = os.path.join(self.json_path, "pretrain.json")
            self.json_val_path = (
                os.path.join(self.json_path, "pretrain.json") if args.model_name == "method" else None
            )
            self.json_test_path = None

        elif stage == 'linear_probe':
            print("INFO: DataModule configured for LINEAR PROBE stage.")
            self.json_train_path = os.path.join(self.json_path, "linear_train.json")
            self.json_val_path = os.path.join(self.json_path, "linear_val.json")
            self.json_test_path = os.path.join(self.json_path, "linear_test.json")

        elif stage == 'linear_probe_lstm':
            print("INFO: DataModule configured for LINEAR PROBE LSTM stage.")
            self.json_train_path = os.path.join(self.json_path, "linear_probe_train.json")
            self.json_val_path = os.path.join(self.json_path, "linear_probe_val.json")
            self.json_test_path = os.path.join(self.json_path, "linear_probe_test.json")

        else:
            raise ValueError(f"Invalid stage: {stage}. Choose 'pretrain', 'linear_probe', or 'linear_probe_lstm'.")

    def prepare_data(self):
        if not os.path.exists(self.stats_file_path):
            print(f"Statistics file not found. Calculating for the first time...")

            temp_dataset = VideoSensorDataset(
                json_path=self.json_train_path,
                data_root=self.data_root,
                num_frames=self.num_frames,
                transform=self.train_transform,
                sensor_transform=None,
                threshold_epoch=self.threshold_epoch,
                start_index=self.start_index,
                end_index=self.end_index,
                cache_dir=self.cache_dir
            )
            stats = calculate_sensor_stats(temp_dataset)
            save_stats(stats, self.stats_file_path)

    def setup(self, stage=None):
        stats = load_stats(self.stats_file_path)
        sensor_preprocessor = SensorTransform(
            target_len=128, mean=stats["mean"], std=stats["std"]
        )

        if self.stage in ["pretrain", "linear_probe"]:
            self.train_dataset = VideoSensorDataset(
                json_path=self.json_train_path,
                data_root=self.data_root,
                num_frames=self.num_frames,
                transform=self.train_transform,
                sensor_transform=sensor_preprocessor,
                threshold_epoch=self.threshold_epoch,
                start_index=self.start_index,
                end_index=self.end_index,
                cache_dir=self.cache_dir,
            )
            print(f"Train dataset size: {len(self.train_dataset)}")

            if self.json_val_path:
                self.val_dataset = VideoSensorDataset(
                    json_path=self.json_val_path,
                    data_root=self.data_root,
                    num_frames=self.num_frames,
                    transform=self.train_transform,
                    sensor_transform=sensor_preprocessor,
                    threshold_epoch=self.threshold_epoch,
                    start_index=self.start_index,
                    end_index=self.end_index,
                    cache_dir=self.cache_dir,
                )

            if self.json_test_path:
                self.test_dataset = VideoSensorDataset(
                    json_path=self.json_test_path,
                    data_root=self.data_root,
                    num_frames=self.num_frames,
                    transform=self.train_transform,
                    sensor_transform=sensor_preprocessor,
                    threshold_epoch=self.threshold_epoch,
                    start_index=self.start_index,
                    end_index=self.end_index,
                    cache_dir=self.cache_dir,
                )
                
        elif self.stage == "linear_probe_lstm":
            print("Loading SequenceDataset for Linear Probe LSTM ...")

            shared_class_to_idx = {}
            self.train_dataset = SequenceDataset(
                json_path=self.json_train_path,
                data_root=self.data_root,
                class_to_idx=shared_class_to_idx,
                sensor_transform=sensor_preprocessor,
            )
            self.val_dataset = SequenceDataset(
                json_path=self.json_test_path,
                data_root=self.data_root,
                class_to_idx=shared_class_to_idx,
                sensor_transform=sensor_preprocessor,
            )
            self.test_dataset = SequenceDataset(
                json_path=self.json_test_path,
                data_root=self.data_root,
                class_to_idx=shared_class_to_idx,
                sensor_transform=sensor_preprocessor,
            )

            self.collate_fn = collate_variable_length

        else:
            raise ValueError(f"Invalid stage: {self.stage}")

    def train_dataloader(self):
        if self.stage == "linear_probe_lstm":
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.collate_fn,
            )
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.stage == "linear_probe_lstm":
            return DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.collate_fn,
            )
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    def test_dataloader(self):
        """테스트용 DataLoader 반환"""
        if self.stage == "linear_probe_lstm":
            return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.collate_fn,
            )
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    def on_before_train_epoch(self, epoch):
        self.train_dataset.set_epoch(epoch)