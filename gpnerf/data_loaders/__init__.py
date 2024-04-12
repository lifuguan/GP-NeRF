from .scannet_dataset import ScannetTrainDataset, ScannetValDataset
from .replica_dataset import ReplicaTrainDataset, ReplicaValDataset
from .replica_instance_dataset import ReplicaInsDataset

dataset_dict = {
    "train_scannet": ScannetTrainDataset,  # for train semanitc segmentation
    "val_scannet": ScannetValDataset,  # for val semanitc segmentation
    "train_replica": ReplicaTrainDataset,  # for train semanitc segmentation
    "val_replica": ReplicaValDataset,  # for val semanitc segmentation
    "instance_replica": ReplicaInsDataset,  # for train/val instance segmentation
}
