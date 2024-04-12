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
class ScannetTrainDataset(Dataset):
    def __init__(self, args, is_train, **kwargs):
        self.scene_path_list = scannet_train_scans_320

        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = args.rectify_inplane_rotation

        image_size = 320
        self.ratio = image_size / 1296
        self.h, self.w = int(self.ratio*972), int(image_size)

        all_rgb_files, all_depth_files, all_pose_files, all_label_files, all_intrinsics_files = [],[],[],[],[]
        for i, scene_path in enumerate(self.scene_path_list):
            scene_path = os.path.join(args.rootdir + 'data', scene_path[:-10])
            pose_files = []
            for f in sorted(os.listdir(os.path.join(scene_path, "pose"))):
                path = os.path.join(scene_path, "pose", f)
                pose = np.loadtxt(path)
                if np.isinf(pose).any() or np.isnan(pose).any():
                    continue
                else:
                    pose_files.append(path)
                    
            rgb_files = [f.replace("pose", "color").replace("txt", "jpg") for f in pose_files]
            depth_files = [f.replace("pose", "depth").replace("txt", "png") for f in pose_files]
            intrinsics_files = [
                os.path.join(scene_path, 'intrinsic/intrinsic_color.txt') for f in rgb_files
            ]
            label_files = [f.replace("pose", "label-filt").replace("txt", "png") for f in pose_files]

            all_rgb_files.append(rgb_files)
            all_depth_files.append(depth_files)
            all_label_files.append(label_files)
            all_pose_files.append(pose_files)
            all_intrinsics_files.append(intrinsics_files)

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files, dtype=object)[index]
        self.all_depth_files = np.array(all_depth_files, dtype=object)[index]
        self.all_label_files = np.array(all_label_files, dtype=object)[index]
        self.all_pose_files = np.array(all_pose_files, dtype=object)[index]
        self.all_intrinsics_files = np.array(all_intrinsics_files, dtype=object)[index]

        mapping_file = 'data/scannet/scannetv2-labels.combined.tsv'
        mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
        scan_ids = mapping_file['id'].values
        nyu40_ids = mapping_file['nyu40id'].values
        scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
        for i in range(len(scan_ids)):
            scan2nyu[scan_ids[i]] = nyu40_ids[i]
        self.scan2nyu = scan2nyu
        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
            max_cat_id=40
        )


    def pose_inverse(self, pose):
        R = pose[:, :3].T
        t = - R @ pose[:, 3:]
        inversed_pose = np.concatenate([R, t], -1)
        return np.concatenate([inversed_pose, [[0, 0, 0, 1]]])
        # return inversed_pose

    def __len__(self):
        return 9999  # keep it not going to interupt
    
    def __getitem__(self, idx):
        set_seed(idx, is_train=True)
        
        real_idx = idx % len(self.all_rgb_files)
        rgb_files = self.all_rgb_files[real_idx]
        depth_files = self.all_depth_files[real_idx]
        pose_files = self.all_pose_files[real_idx]
        label_files = self.all_label_files[real_idx]
        intrinsics_files = self.all_intrinsics_files[real_idx]

        id_render = np.random.choice(np.arange(len(pose_files)))
        train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
        render_pose = train_poses[id_render]

        subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

        id_feat_pool = get_nearest_pose_ids(
            render_pose,
            train_poses,
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

        intrinsics = np.loadtxt(intrinsics_files[id_render]).reshape([4, 4])
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
        label = self.scan2nyu[label]
        label = self.label_mapping(label)

        all_poses = [render_pose]
        # get depth range
        # poses = render_pose[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        # bds = render_pose[:, -2:].transpose([1, 0])
        # bds = np.moveaxis(bds, -1, 0).astype(np.float32)
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
            pose = np.loadtxt(pose_files[id]).reshape(4, 4)

            if self.rectify_inplane_rotation:
                pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            intrinsics = np.loadtxt(intrinsics_files[id]).reshape([4, 4])
            intrinsics[:2, :] *= self.ratio
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
class ScannetValDataset(Dataset):
    def __init__(self, args, is_train, scenes=None, **kwargs):
        self.is_train = is_train
        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = args.rectify_inplane_rotation

        image_size = 320
        self.ratio = image_size / 1296
        self.h, self.w = int(self.ratio*972), int(image_size)

        scene_path = os.path.join(args.rootdir + 'data', scenes[:-10])
        pose_files = []
        for f in sorted(os.listdir(os.path.join(scene_path, "pose"))):
            path = os.path.join(scene_path, "pose", f)
            pose = np.loadtxt(path)
            if np.isinf(pose).any() or np.isnan(pose).any():
                continue
            else:
                pose_files.append(path)
                
        rgb_files = [f.replace("pose", "color").replace("txt", "jpg") for f in pose_files]
        depth_files = [f.replace("pose", "depth").replace("txt", "png") for f in pose_files]
        intrinsics_files = [
            os.path.join(scene_path, 'intrinsic/intrinsic_color.txt') for f in rgb_files
        ]
        label_files = [f.replace("pose", "label-filt").replace("txt", "png") for f in pose_files]

        index = np.arange(len(rgb_files))
        self.rgb_files = np.array(rgb_files, dtype=object)[index]
        self.depth_files = np.array(depth_files, dtype=object)[index]
        self.label_files = np.array(label_files, dtype=object)[index]
        self.pose_files = np.array(pose_files, dtype=object)[index]
        self.intrinsics_files = np.array(intrinsics_files, dtype=object)[index]

        mapping_file = 'data/scannet/scannetv2-labels.combined.tsv'
        mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
        scan_ids = mapping_file['id'].values
        nyu40_ids = mapping_file['nyu40id'].values
        scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
        for i in range(len(scan_ids)):
            scan2nyu[scan_ids[i]] = nyu40_ids[i]
        self.scan2nyu = scan2nyu
        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
            max_cat_id=40
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
        pose_files = self.pose_files
        label_files = self.label_files
        intrinsics_files = self.intrinsics_files

        train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
        render_pose = train_poses[que_idx]

        subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

        id_feat_pool = get_nearest_pose_ids(
            render_pose,
            train_poses,
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
            
        intrinsics = np.loadtxt(intrinsics_files[que_idx]).reshape([4, 4])
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
        label = self.scan2nyu[label]
        label = self.label_mapping(label)

        all_poses = [render_pose]
        depth_range = np.array([0.1, 6.0])

        depth_mask = np.ones_like(depth)
        depth_mask[depth == 0] = 0

        src_rgbs = []
        src_cameras = []
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.0
            if self.w != 1296:
                src_rgb = cv2.resize(downsample_gaussian_blur(
                    src_rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            pose = np.loadtxt(pose_files[id]).reshape(4, 4)

            if self.rectify_inplane_rotation:
                pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            intrinsics = np.loadtxt(intrinsics_files[id]).reshape([4, 4])
            intrinsics[:2, :] *= self.ratio
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
            "depth_mask": torch.from_numpy(depth_mask),
            "labels": torch.from_numpy(label),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_files[que_idx],
            "src_rgbs": torch.from_numpy(src_rgbs),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": torch.from_numpy(depth_range),
        }

