import torch.nn as nn
import torch
from pathlib import Path
from utils import img2mse
from skimage.io import imsave
from utils import concat_images_list

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        pred_rgb = outputs["rgb"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss, scalars_to_log


class SemanticCriterion(nn.Module):

    def __init__(self, args):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category

        """
        super().__init__()

        self.expname = args.expname
        self.ignore_label = args.ignore_label
        self.num_classes = args.num_classes + 1 # +1 for ignore

        self.color_map = torch.tensor(args.semantic_color_map, dtype=torch.uint8)

    def compute_label_loss(self, label_pr, label_gt):
        label_pr = label_pr.reshape(-1, self.num_classes)
        label_gt = label_gt.reshape(-1).long()
        valid_mask = (label_gt != self.ignore_label)
        label_pr = label_pr[valid_mask]
        label_gt = label_gt[valid_mask]
        return nn.functional.cross_entropy(label_pr, label_gt, reduction='mean').unsqueeze(0)

    def plot_semantic_results(self, data_pred, data_gt, step):
        h, w = data_pred['sems'].shape[1:3]
        self.color_map.to(data_gt['rgb'].device)
        
        def get_img(data_src, key, channel):
            rgbs = data_src[key]  # 1,rn,3
            rgbs = rgbs.reshape([h, w, channel]).detach()
            if channel > 1:
                rgbs = rgbs.argmax(axis=-1, keepdims=True)
            rgbs = rgbs.squeeze().cpu().numpy()
            rgbs = self.color_map[rgbs]
            return rgbs
        
        imgs = [get_img(data_gt, 'labels', 1), get_img(data_pred, 'sems', self.num_classes)]

        model_name = self.expname
        Path(f'out/vis/{model_name}').mkdir(exist_ok=True, parents=True)
        imsave(f'out/vis/{model_name}/step-{step}-sem.png', concat_images_list(*imgs))
        return data_pred

    def forward(self, data_pred, data_gt, scalars_to_log):
        pred_rgb, pred_label = data_pred["rgb"], data_pred["sems"]
        if "mask" in data_pred:
            pred_mask = data_pred["mask"].float()
        else:
            pred_mask = None
        gt_rgb, gt_label = data_gt["rgb"], data_gt["labels"]

        rgb_loss = img2mse(pred_rgb, gt_rgb, pred_mask)
        if pred_label is not None:
            label_loss = self.compute_label_loss(pred_label, gt_label)
        else:
            label_loss = 0
        return rgb_loss, label_loss, scalars_to_log
