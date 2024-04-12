import random
import time
import os
import numpy as np
import imageio
import cv2
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
import pandas as pd
import sys
from tqdm import tqdm

sys.path.append("../")
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids
from .utils.base_utils import downsample_gaussian_blur
from .asset import *
from .semantic_utils import PointSegClassMapping

def set_seed(index,is_train):
    if is_train:
        np.random.seed((index+int(time.time()))%(2**16))
        random.seed((index+int(time.time()))%(2**16)+1)
        torch.random.manual_seed((index+int(time.time()))%(2**16)+1)
    else:
        np.random.seed(index % (2 ** 16))
        random.seed(index % (2 ** 16) + 1)
        torch.random.manual_seed(index % (2 ** 16) + 1)


# only for training
class ReplicaTrainDataset(Dataset):
    def __init__(self, args, is_train, **kwargs):
        self.scene_path_list = replica_train

        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = args.rectify_inplane_rotation

        image_size = args.original_width
        self.ratio = image_size / args.original_width
        self.h, self.w = int(self.ratio * args.original_height), int(image_size)

        all_rgb_files, all_depth_files, all_poses, all_label_files = [],[],[],[]
        for scene in tqdm(self.scene_path_list, desc=f'Train Loader'):
            for i in range(2):
                scene_path = os.path.join(args.rootdir + 'data/Replica', scene, f'Sequence_{1+i}')
                poses = np.loadtxt(f'{scene_path}/traj_w_c.txt',delimiter=' ').reshape(-1, 4, 4).astype(np.float32)
                rgb_files = []
                for i, f in enumerate(sorted(os.listdir(os.path.join(scene_path, "rgb")), key=lambda x: int(x.split('_')[1].split('.')[0]))):
                    path = os.path.join(scene_path, "rgb", f)
                    if np.isinf(poses[i]).any() or np.isnan(poses[i]).any():
                        continue
                    else:
                        rgb_files.append(path)
                        
                depth_files = [f.replace("rgb", "depth") for f in rgb_files]
                label_files = [f.replace("rgb", "semantic_class") for f in rgb_files]

                all_rgb_files.append(rgb_files)
                all_depth_files.append(depth_files)
                all_label_files.append(label_files)
                all_poses.append(poses)

            

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files, dtype=object)[index]
        self.all_depth_files = np.array(all_depth_files, dtype=object)[index]
        self.all_label_files = np.array(all_label_files, dtype=object)[index]
        self.all_poses = np.array(all_poses)[index]

        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[
                3, 7, 8, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 22, 23, 26, 29, 31,
                34, 35, 37, 40, 44, 47, 52, 54, 56,
                59, 60, 61, 62, 63, 64, 65, 70, 71,
                76, 78, 79, 80, 82, 83, 87, 88, 91,
                92, 93, 95, 97, 98
            ],
            max_cat_id=101
        )


    def pose_inverse(self, pose):
        R = pose[:, :3].T
        t = - R @ pose[:, 3:]
        inversed_pose = np.concatenate([R, t], -1)
        return np.concatenate([inversed_pose, [[0, 0, 0, 1]]])
        # return inversed_pose

    def __len__(self):
        return 9999  # 确保不会中断
    
    def __getitem__(self, idx):
        set_seed(idx, is_train=True)
        
        real_idx = idx % len(self.all_rgb_files)
        rgb_files = self.all_rgb_files[real_idx]
        depth_files = self.all_depth_files[real_idx]
        scene_poses = self.all_poses[real_idx]
        label_files = self.all_label_files[real_idx]


        id_render = np.random.choice(np.arange(len(scene_poses)))
        render_pose = scene_poses[id_render]

        subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

        id_feat_pool = get_nearest_pose_ids(
            render_pose,
            scene_poses,
            self.num_source_views * subsample_factor,
            tar_id=id_render,
            angular_dist_method="vector",
        )
        id_feat = np.random.choice(id_feat_pool, self.num_source_views, replace=False)

        if id_render in id_feat:
            assert id_render not in id_feat
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]):
            id_feat[np.random.choice(len(id_feat))] = id_render

        rgb = imageio.imread(rgb_files[id_render]).astype(np.float32) / 255.0
        if self.w != 1296:
            rgb = cv2.resize(downsample_gaussian_blur(
                rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            
        img = Image.open(depth_files[id_render])
        depth = np.asarray(img, dtype=np.float32) / 1000.0  # mm -> m
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        intrinsics = np.array([320.0, 0.0, 320.0, 0.0,
                                0.0, 319.5, 229.5, 0.0,
                                0.0, 0.0, 1.0, 0.0,
                                0.0, 0.0, 0.0, 1.0]).reshape(4,4)

        intrinsics[:2, :] *= self.ratio

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(), render_pose.flatten())).astype(
            np.float32
        )

        img = Image.open(label_files[id_render])
        label = np.asarray(img, dtype=np.int32)
        label = np.ascontiguousarray(label)
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)
        label = self.label_mapping(label)

        all_poses = [render_pose]
        depth_range = torch.tensor([0.1, 10.0])

        src_rgbs = []
        src_cameras = []
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.0
            if self.w != 1296:
                src_rgb = cv2.resize(downsample_gaussian_blur(
                    src_rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            pose = scene_poses[id]

            if self.rectify_inplane_rotation:
                pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), intrinsics.flatten(), pose.flatten())).astype(
                np.float32
            )
            src_cameras.append(src_camera)
            all_poses.append(pose)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

        return {
            "rgb": torch.from_numpy(rgb),
            "true_depth": torch.from_numpy(depth),
            "labels": torch.from_numpy(label),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_files[id_render],
            "src_rgbs": torch.from_numpy(src_rgbs),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }



# only for validation
class ReplicaValDataset(Dataset):
    def __init__(self, args, is_train, scenes=None, **kwargs):
        self.is_train = is_train
        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        
        image_size = args.original_width
        self.ratio = image_size / args.original_width
        self.h, self.w = int(self.ratio * args.original_height), int(image_size)

        scene_path = os.path.join(args.rootdir + 'data/Replica', scenes, 'Sequence_1')
        poses = np.loadtxt(f'{scene_path}/traj_w_c.txt',delimiter=' ').reshape(-1, 4, 4).astype(np.float32)
        rgb_files = []
        for i, f in enumerate(sorted(os.listdir(os.path.join(scene_path, "rgb")), key=lambda x: int(x.split('_')[1].split('.')[0]))):
            path = os.path.join(scene_path, "rgb", f)
            if np.isinf(poses[i]).any() or np.isnan(poses[i]).any():
                continue
            else:
                rgb_files.append(path)
                
        depth_files = [f.replace("rgb", "depth") for f in rgb_files]
        label_files = [f.replace("rgb", "semantic_class") for f in rgb_files]

        index = np.arange(len(rgb_files))
        self.rgb_files = np.array(rgb_files, dtype=object)[index]
        self.depth_files = np.array(depth_files, dtype=object)[index]
        self.label_files = np.array(label_files, dtype=object)[index]
        self.poses = np.array(poses)[index]

        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[
                3, 7, 8, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 22, 23, 26, 29, 31,
                34, 35, 37, 40, 44, 47, 52, 54, 56,
                59, 60, 61, 62, 63, 64, 65, 70, 71,
                76, 78, 79, 80, 82, 83, 87, 88, 91,
                92, 93, 95, 97, 98
            ],
            max_cat_id=101
        )

        que_idxs = np.arange(len(self.rgb_files))
        self.train_que_idxs = que_idxs[:700:3]
        self.val_que_idxs = que_idxs[2:700:20]
        if len(self.val_que_idxs) > 10:
            self.val_que_idxs = self.val_que_idxs[:10]

    def __len__(self):
        if self.is_train is True:
            return len(self.train_que_idxs)
        else:  
            return len(self.val_que_idxs)  
    
    def __getitem__(self, idx):
        set_seed(idx, is_train=self.is_train)
        if self.is_train is True:
            que_idx = self.train_que_idxs[idx]
        else:
            que_idx = self.val_que_idxs[idx]

        rgb_files = self.rgb_files
        depth_files = self.depth_files
        scene_poses = self.poses
        label_files = self.label_files

        render_pose = scene_poses[que_idx]

        subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

        id_feat_pool = get_nearest_pose_ids(
            render_pose,
            scene_poses,
            self.num_source_views * subsample_factor,
            tar_id=que_idx,
            angular_dist_method="vector",
        )
        id_feat = np.random.choice(id_feat_pool, self.num_source_views, replace=False)

        if que_idx in id_feat:
            assert que_idx not in id_feat
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]):
            id_feat[np.random.choice(len(id_feat))] = que_idx

        img = Image.open(depth_files[que_idx])
        depth = np.asarray(img, dtype=np.float32) / 1000.0  # mm -> m
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        rgb = imageio.imread(rgb_files[que_idx]).astype(np.float32) / 255.0

        if self.w != 1296:
            rgb = cv2.resize(downsample_gaussian_blur(
                rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            
        intrinsics = np.array([320.0, 0.0, 320.0, 0.0,
                                0.0, 319.5, 229.5, 0.0,
                                0.0, 0.0, 1.0, 0.0,
                                0.0, 0.0, 0.0, 1.0]).reshape(4,4)
        intrinsics[:2, :] *= self.ratio

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(), render_pose.flatten())).astype(
            np.float32
        )

        img = Image.open(label_files[que_idx])
        label = np.asarray(img, dtype=np.int32)
        label = np.ascontiguousarray(label)
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)
        label = self.label_mapping(label)

        all_poses = [render_pose]
        # get depth range
        # min_ratio = 0.1
        # origin_depth = np.linalg.inv(render_pose)[2, 3]
        # max_radius = 0.5 * np.sqrt(2) * 1.1
        # near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
        # far_depth = origin_depth + max_radius
        # depth_range = torch.tensor([near_depth, far_depth])
        depth_range = torch.tensor([0.1, 10.0])

        src_rgbs = []
        src_cameras = []
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.0
            if self.w != 1296:
                src_rgb = cv2.resize(downsample_gaussian_blur(
                    src_rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            pose = scene_poses[id]

            if self.rectify_inplane_rotation:
                pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), intrinsics.flatten(), pose.flatten())).astype(
                np.float32
            )
            src_cameras.append(src_camera)
            all_poses.append(pose)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

        return {
            "rgb": torch.from_numpy(rgb),
            "true_depth": torch.from_numpy(depth),
            "labels": torch.from_numpy(label),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_files[que_idx],
            "src_rgbs": torch.from_numpy(src_rgbs),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }