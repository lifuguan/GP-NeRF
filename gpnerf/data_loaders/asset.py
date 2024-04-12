import os
import numpy as np

if os.path.exists('data/scannet') and os.path.exists('data/Replica'):
    replica_instance = np.loadtxt('configs/replica_instance_split.txt',dtype=str).tolist()
    replica_train = np.loadtxt('configs/replica_train_split.txt',dtype=str).tolist()
    replica_test  = np.loadtxt('configs/replica_test_split.txt',dtype=str).tolist()
    scannet_train_scans_320 = np.loadtxt('configs/scannetv2_train_split.txt',dtype=str).tolist()
    scannet_test_scans_320 = np.loadtxt('configs/scannetv2_test_split.txt',dtype=str).tolist()
    scannet_val_scans_320 = np.loadtxt('configs/scannetv2_val_split.txt',dtype=str).tolist()
    scannet_single = ['scannet/scene0376_02/black_320']
