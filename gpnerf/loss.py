import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import confusion_matrix
from skimage.io import imsave
from utils import concat_images_list
import numpy as np

from sklearn.decomposition import PCA
import sklearn
import time
from PIL import Image
import matplotlib.pyplot as plt

def nanmean(data, **args):
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)

def plot_pca_features(data_pred, ray_batch, step, val_name=None, vis=False, return_img = False):
    coarse_feats = data_pred['outputs_coarse']['feats_out'].unsqueeze(0).permute(0,3,1,2)
    fine_feats = data_pred['outputs_fine']['feats_out'].unsqueeze(0).permute(0,3,1,2)
    h, w = coarse_feats.shape[2:4]
    def pca_calc(feats):
        fmap = feats.cuda()
        pca = sklearn.decomposition.PCA(3, random_state=80)
        f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
        transformed = pca.fit_transform(f_samples)
        feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
        feature_pca_components = torch.tensor(pca.components_).float().cuda()
        q1, q99 = np.percentile(transformed, [1, 99])
        feature_pca_postprocess_sub = q1
        feature_pca_postprocess_div = (q99 - q1)
        del f_samples

        vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
        vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
        vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
        return (vis_feature.cpu().numpy() * 255).astype(np.uint8)

    rgbs = ray_batch['rgb']  # 1,rn,3
    rgbs = rgbs.reshape([h, w, 3]).detach() * 255
    rgbs = rgbs.squeeze().cpu().numpy().astype(np.uint8)[::2, ::2]     
    if return_img is True:
        return pca_calc(fine_feats)
    else:
        imgs = [rgbs, pca_calc(coarse_feats), pca_calc(fine_feats)]
        model_name = 'ins_replica_gpu_8'
        if vis is True:
            imsave(f'out/{model_name}/{val_name}/pca_{step}.png', concat_images_list(*imgs))
class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys = keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass

class RenderLoss(nn.Module):
    
    def __init__(self, args):
        self.render_loss_scale = args.render_loss_scale

    def compute_rgb_loss(self, rgb_pr, rgb_gt):
        loss = torch.sum((rgb_pr - rgb_gt) ** 2, -1)        # n
        loss = torch.mean(loss)
        return loss * self.render_loss_scale
        
    def __call__(self, data_pred, data_gt, **kwargs):
        rgb_gt = data_gt["rgb"]  # 1,rn,3
        rgb_coarse = data_pred["outputs_coarse"]["rgb"]  # rn,3

        results = {"train/rgb-loss": self.compute_rgb_loss(rgb_coarse, rgb_gt)}
        # results = {"train/coarse-psnr-training-batch": mse2psnr(results["train/coarse-loss"])}

        if data_pred["outputs_fine"] is not None:
            rgb_fine = data_pred["outputs_fine"]["rgb"]  # 1,rn,3
            results["train/rgb-loss"] += self.compute_rgb_loss(rgb_fine, rgb_gt)
            # results = {"train/fine-psnr-training-batch": mse2psnr(results["train/fine-loss"])}
        return results
    

class DepthGuidedSemLoss(nn.Module):
    
    def __init__(self, args):
        self.dgs_loss_scale = args.dgs_loss_scale
        self.N_samples = args.N_samples
        
    def __call__(self, data_pred, data_gt, **kwargs):
        rgb_gt = data_gt["rgb"]  # 1,rn,3
        rgb_coarse = data_pred["outputs_coarse"]["rgb"]  # rn,3

        results = {"train/rgb-loss": self.compute_rgb_loss(rgb_coarse, rgb_gt)}
        # results = {"train/coarse-psnr-training-batch": mse2psnr(results["train/coarse-loss"])}

        if data_pred["outputs_fine"] is not None:
            rgb_fine = data_pred["outputs_fine"]["rgb"]  # 1,rn,3
            results["train/rgb-loss"] += self.compute_rgb_loss(rgb_fine, rgb_gt)
            # results = {"train/fine-psnr-training-batch": mse2psnr(results["train/fine-loss"])}
        return results
    
class SemanticLoss(Loss):
    def __init__(self, args):
        super().__init__(['loss_semantic'])
        self.semantic_loss_scale = args.semantic_loss_scale
        self.ignore_label = args.ignore_label
        self.num_classes = args.num_classes + 1 # for ignore label
        self.color_map = torch.tensor(args.semantic_color_map, dtype=torch.uint8)
        self.expname = args.expname

    def plot_pca_features(self, data_pred, ray_batch, step, val_name=None, vis=False):
        coarse_feats = data_pred['outputs_coarse']['feats_out'].unsqueeze(0).permute(0,3,1,2)
        fine_feats = data_pred['outputs_fine']['feats_out'].unsqueeze(0).permute(0,3,1,2)
        h, w = coarse_feats.shape[2:4]
        def pca_calc(feats):
            fmap = feats.cuda()
            pca = sklearn.decomposition.PCA(3, random_state=80)
            f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
            transformed = pca.fit_transform(f_samples)
            feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
            feature_pca_components = torch.tensor(pca.components_).float().cuda()
            q1, q99 = np.percentile(transformed, [1, 99])
            feature_pca_postprocess_sub = q1
            feature_pca_postprocess_div = (q99 - q1)
            del f_samples

            vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
            vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
            vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
            return (vis_feature.cpu().numpy() * 255).astype(np.uint8)

        rgbs = ray_batch['rgb']  # 1,rn,3
        rgbs = rgbs.reshape([h*2, w*2, 3]).detach() * 255
        rgbs = rgbs.squeeze().cpu().numpy().astype(np.uint8)[::2, ::2]     
        imgs = [rgbs, pca_calc(coarse_feats), pca_calc(fine_feats)]

        model_name = self.expname
        if vis is True:
            imsave(f'out/{model_name}/{val_name}/pca_{step}.png', concat_images_list(*imgs))

    def plot_semantic_results(self, data_pred, data_gt, step, val_name=None, vis=False):
        h, w = data_pred['sems'].shape[1:3]
        batch_size = data_pred['sems'].shape[0]
        self.color_map.to(data_gt['rgb'].device)
        
        if self.ignore_label != -1:
            unvalid_pix_ids = data_gt['labels'] == self.ignore_label
        else:
            unvalid_pix_ids = np.zeros_like(data_gt['labels'], dtype=bool)
        unvalid_pix_ids = unvalid_pix_ids.reshape(h,w,-1)

        def get_label_img(data_src, key, channel):
            rgbs = data_src[key]  # 1,rn,3
            rgbs = rgbs[0] if batch_size > 1 else rgbs
            rgbs = rgbs.reshape([h, w, channel]).detach()
            if channel > 1:
                rgbs = rgbs.argmax(axis=-1, keepdims=True)
                rgbs[unvalid_pix_ids] = self.ignore_label

            rgbs = rgbs.squeeze().cpu().numpy()
            rgbs = self.color_map[rgbs]
            return rgbs
        
        def get_rgb(data_src, key, channel):
            rgbs = data_src[key]  # 1,rn,3
            rgbs = rgbs.reshape([h, w, channel]).detach() * 255
            rgbs = rgbs.squeeze().cpu().numpy().astype(np.uint8)
            return rgbs
        
        if 'full_rgb' not in data_gt.keys():
            if 'que_sems' in data_pred.keys():
                imgs = [get_label_img(data_gt, 'labels', 1), get_label_img(data_pred, 'sems', self.num_classes), get_label_img(data_pred, 'que_sems', self.num_classes)]
            else:
                imgs = [get_label_img(data_gt, 'labels', 1), get_label_img(data_pred, 'sems', self.num_classes)]
        else:
            imgs = [get_rgb(data_gt, 'full_rgb', 3), get_label_img(data_gt, 'labels', 1), get_label_img(data_pred, 'sems', self.num_classes)]

        

        model_name = self.expname
        if vis is True:
            imsave(f'out/{model_name}/{val_name}/seg_{step}.png', concat_images_list(*imgs))
        return imgs
    
    def compute_semantic_loss(self, label_pr, label_gt, num_classes):
        label_pr = label_pr.reshape(-1, num_classes)
        label_gt = label_gt.reshape(-1).long()
        valid_mask = (label_gt != self.ignore_label)
        label_pr = label_pr[valid_mask]
        label_gt = label_gt[valid_mask]
        return nn.functional.cross_entropy(label_pr, label_gt, reduction='mean').unsqueeze(0)
    
    def __call__(self, data_pred, data_gt, step, **kwargs):
        num_classes = data_pred['outputs_coarse']['sems'].shape[-1]
        
        pixel_label_gt = data_gt['labels']
        pixel_label_nr = data_pred['outputs_coarse']['sems']
        coarse_loss = self.compute_semantic_loss(pixel_label_nr, pixel_label_gt, num_classes)
        
        if 'outputs_fine' in data_pred:
            pixel_label_nr_fine = data_pred['outputs_fine']['sems']
            fine_loss = self.compute_semantic_loss(pixel_label_nr_fine, pixel_label_gt, num_classes)
        else:
            fine_loss = torch.zeros_like(coarse_loss)
        
        loss = (coarse_loss + fine_loss) * self.semantic_loss_scale
        
        # if 'pred_labels' in data_pred:
        #     ref_labels_pr = data_pred['pred_labels'].permute(0, 2, 3, 1)
        #     ref_labels_gt = data_gt['ref_imgs_info']['labels'].permute(0, 2, 3, 1)
        #     ref_loss = self.compute_semantic_loss(ref_labels_pr, ref_labels_gt, num_classes)
        #     loss += ref_loss * self.semantic_loss_scale
        return {'train/semantic-loss': loss}
class DepthLoss(nn.Module):

    def __init__(self, args):
        super(DepthLoss, self).__init__()
        self.depth_loss_scale = args.depth_loss_scale

        self.depth_correct_thresh = 0.02
        self.depth_loss_type = 'smooth_l1'
        self.depth_loss_l1_beta = 0.05
        self.loss_op = nn.SmoothL1Loss(reduction='none', beta=self.depth_loss_l1_beta)

    def __call__(self, data_pr, data_gt, **kwargs):
        depth_mask = data_gt['depth_mask']
        depth_pr = data_pr['outputs_coarse']['depth'] * depth_mask
        depth_gt = data_gt['true_depth'] * depth_mask

        # transform to inverse depth coordinate
        depth_range = data_gt['depth_range']  # rfn,2
        near, far = -1/depth_range[:, 0:1], -1/depth_range[:, 1:2]  # rfn,1

        def process(depth):
            depth = torch.clamp(depth, min=1e-5)
            depth = -1 / depth
            depth = (depth - near) / (far - near)
            depth = torch.clamp(depth, min=0, max=1.0)
            return depth
        depth_gt = process(depth_gt) * depth_mask

        # compute loss
        def compute_loss(depth_pr):
            if self.depth_loss_type == 'l2':
                loss = (depth_gt - depth_pr)**2
            elif self.depth_loss_type == 'smooth_l1':
                loss = self.loss_op(depth_pr, depth_gt.squeeze(-1)) 

            return torch.mean(loss)

        outputs = {'train/depth-loss': compute_loss(depth_pr)}
        if 'outputs_fine' in data_pr:
            outputs['train/depth-loss'] += compute_loss(data_pr['outputs_fine']['depth'])
        outputs['train/depth-loss'] = outputs['train/depth-loss'] * self.depth_loss_scale
        return outputs
    
# From https://github.com/Harry-Zhi/semantic_nerf/blob/a0113bb08dc6499187c7c48c3f784c2764b8abf1/SSR/training/training_utils.py
class IoU(Loss):

    def __init__(self, args):
        super().__init__([])
        self.num_classes = args.num_classes
        self.ignore_label = args.ignore_label

    def iou_calc(self, predicted_labels, true_labels, valid_pix_ids):
        predicted_labels = predicted_labels[valid_pix_ids]
        true_labels = true_labels[valid_pix_ids]

        conf_mat = confusion_matrix(
            true_labels, predicted_labels, labels=list(range(self.num_classes)))
        norm_conf_mat = np.transpose(np.transpose(
            conf_mat) / conf_mat.astype(float).sum(axis=1))

        # missing class will have NaN at corresponding class
        missing_class_mask = np.isnan(norm_conf_mat.sum(1))
        exsiting_class_mask = ~ missing_class_mask

        class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
        total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
        ious = np.zeros(self.num_classes)
        for class_id in range(self.num_classes):
            ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
        miou = np.mean(ious[exsiting_class_mask])
        if np.isnan(miou):
            miou = 0.
            total_accuracy = 0.
            class_average_accuracy = 0.
        return miou, total_accuracy, class_average_accuracy
    def __call__(self, data_pred, data_gt, step, **kwargs):
        true_labels = data_gt['labels'].reshape([-1]).long().detach().cpu().numpy()
        if 'outputs_fine' in data_pred:
            predicted_labels = data_pred['outputs_fine']['sems'].argmax(
                dim=-1).reshape([-1]).long().detach().cpu().numpy()
        else:
            predicted_labels = data_pred['outputs_coarse']['sems'].argmax(
                dim=-1).reshape([-1]).long().detach().cpu().numpy()

        if self.ignore_label != -1:
            valid_pix_ids = true_labels != self.ignore_label
        else:
            valid_pix_ids = np.ones_like(true_labels, dtype=bool)

        miou, total_accuracy, class_average_accuracy = self.iou_calc(predicted_labels, true_labels, valid_pix_ids)
        
        if 'que_sems' in data_pred.keys():
            predicted_labels = data_pred['que_sems'].argmax(dim=-1).reshape([-1]).long().detach().cpu().numpy()
            que_miou, _, _ = self.iou_calc(predicted_labels, true_labels, valid_pix_ids)
            output = {
                'miou': torch.tensor([miou], dtype=torch.float32),
                'que_miou': torch.tensor([que_miou], dtype=torch.float32),
                'total_accuracy': torch.tensor([total_accuracy], dtype=torch.float32),
                'class_average_accuracy': torch.tensor([class_average_accuracy], dtype=torch.float32)
            }
        else:
            output = {
                'miou': torch.tensor([miou], dtype=torch.float32),
                'total_accuracy': torch.tensor([total_accuracy], dtype=torch.float32),
                'class_average_accuracy': torch.tensor([class_average_accuracy], dtype=torch.float32)
            }
        return output




def interpolate_feats(feats, points, h=None, w=None, padding_mode='zeros', align_corners=False, inter_mode='bilinear'):
    """

    :param feats:   b,f,h,w
    :param points:  b,n,2
    :param h:       float
    :param w:       float
    :param padding_mode:
    :param align_corners:
    :param inter_mode:
    :return:
    """
    b, _, ch, cw = feats.shape
    if h is None and w is None:
        h, w = ch, cw
    x_norm = points[:, :, 0] / (w - 1) * 2 - 1
    y_norm = points[:, :, 1] / (h - 1) * 2 - 1
    points_norm = torch.stack([x_norm, y_norm], -1).unsqueeze(1)    # [srn,1,n,2]
    feats_inter = F.grid_sample(feats, points_norm, mode=inter_mode, padding_mode=padding_mode, align_corners=align_corners).squeeze(2)      # srn,f,n
    feats_inter = feats_inter.permute(0,2,1)
    return  feats_inter



from scipy.optimize import linear_sum_assignment

class InsEvaluator():
    def __init__(self, ins_num) -> None:
        self.ins_num = ins_num
        self.thre_list = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
    def ins_criterion(self, pred_ins, gt_labels):
        pred_ins = pred_ins.reshape(-1, self.ins_num + 1)[:, :-1]
        gt_labels = gt_labels.reshape(-1)

        # change label to one hot
        valid_gt_labels = torch.unique(gt_labels)
        gt_ins = torch.zeros(size=(gt_labels.shape[0], self.ins_num)).to(pred_ins.device)

        valid_ins_num = len(valid_gt_labels)
        gt_ins[..., :valid_ins_num] = F.one_hot(gt_labels.long())[..., valid_gt_labels.long()]

        cost_ce, cost_siou, order_row, order_col = self.hungarian(pred_ins, gt_ins, valid_ins_num)
        valid_ce = torch.mean(cost_ce[order_row, order_col[:valid_ins_num]])

        if not (len(order_col) == valid_ins_num):
            invalid_ce = torch.mean(pred_ins[:, order_col[valid_ins_num:]])
        else:
            invalid_ce = torch.tensor([0])
        valid_siou = torch.mean(cost_siou[order_row, order_col[:valid_ins_num]])

        ins_loss_sum = valid_ce + invalid_ce + valid_siou
        return ins_loss_sum, valid_ce, invalid_ce, valid_siou


    # matching function
    def hungarian(self, pred_ins, gt_ins, valid_ins_num):
        @torch.no_grad()
        def reorder(cost_matrix, valid_ins_num):
            valid_scores = cost_matrix[:valid_ins_num]
            valid_scores = valid_scores.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(valid_scores)

            unmapped = self.ins_num - valid_ins_num
            if unmapped > 0:
                unmapped_ind = np.array(list(set(range(self.ins_num)) - set(col_ind)))
                col_ind = np.concatenate([col_ind, unmapped_ind])
            return row_ind, col_ind

        # preprocess prediction and ground truth
        pred_ins = pred_ins.permute([1, 0])
        gt_ins = gt_ins.permute([1, 0])
        pred_ins = pred_ins[None, :, :]
        gt_ins = gt_ins[:, None, :]

        cost_ce = torch.mean(-gt_ins * torch.log(pred_ins + 1e-8) - (1 - gt_ins) * torch.log(1 - pred_ins + 1e-8), dim=-1)

        # get soft iou score between prediction and ground truth, don't need do mean operation
        TP = torch.sum(pred_ins * gt_ins, dim=-1)
        FP = torch.sum(pred_ins, dim=-1) - TP
        FN = torch.sum(gt_ins, dim=-1) - TP
        cost_siou = TP / (TP + FP + FN + 1e-6)
        cost_siou = 1.0 - cost_siou

        # final score
        cost_matrix = cost_ce + cost_siou
        # get final indies order
        order_row, order_col = reorder(cost_matrix, valid_ins_num)

        return cost_ce, cost_siou, order_row, order_col
    def calculate_ap(self,IoUs_Metrics, gt_number, confidence=None, function_select='integral'):
        def interpolate_11(prec, rec):
            ap = 0.
            for t in torch.arange(0., 1.1, 0.1):
                if torch.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = torch.max(prec[rec >= t])
                ap = ap + p / 11.
            return ap

        def integral_method(prec, rec):
            """
                This method same as coco
            """
            mrec = torch.cat((torch.Tensor([0.]), rec, torch.Tensor([1.])))
            mprec = torch.cat((torch.Tensor([0.]), prec, torch.Tensor([0.])))
            for i in range(mprec.shape[0] - 1, 0, -1):
                mprec[i - 1] = torch.maximum(mprec[i - 1], mprec[i])
            index = torch.where(mrec[1:] != mrec[:-1])[0]
            ap = torch.sum((mrec[index + 1] - mrec[index]) * mprec[index + 1])
            return ap

        '''begin'''
        # make TP matrix
        if confidence is not None:
            column_max_index = torch.argsort(confidence, descending=True)
            column_max_value = IoUs_Metrics[column_max_index]
        else:
            column_max_value = torch.sort(IoUs_Metrics, descending=True)
            column_max_value = column_max_value[0]

        ap_list = []
        for thre in self.thre_list:
            tp_list = column_max_value > thre
            tp_list = tp_list
            precisions = torch.cumsum(tp_list, dim=0) / (torch.arange(len(tp_list)) + 1)
            recalls = torch.cumsum(tp_list, dim=0).type(torch.float32) / gt_number

            # select calculate function
            if function_select == 'integral':
                ap_list.append(integral_method(precisions, recalls).item())
            elif function_select == 'interpolate':
                ap_list.append(interpolate_11(precisions, recalls).item())

        return ap_list

    def ins_eval(self, pred_ins, gt_label):
        pred_ins = pred_ins[0][:,:,:-1].detach().cpu()
        gt_label = gt_label[0]
        gt_ins = torch.zeros(size=(gt_label.shape[0], gt_label.shape[1], self.ins_num)).cpu()
        valid_gt_labels = torch.unique(gt_label)
        gt_ins_num = len(valid_gt_labels)
        gt_ins[..., :gt_ins_num] = F.one_hot(gt_label.long())[..., valid_gt_labels.long()]
        gt_label_np = valid_gt_labels.cpu().numpy()  # change name

        pred_label = torch.argmax(pred_ins, dim=-1)
        valid_pred_labels = torch.unique(pred_label)

        valid_pred_num = len(valid_pred_labels)
        # prepare confidence masks and confidence scores
        pred_conf_mask = np.max(pred_ins.numpy(), axis=-1)

        pred_conf_list = []
        valid_pred_labels = valid_pred_labels.numpy().tolist()
        for label in valid_pred_labels:
            index = torch.where(pred_label == label)
            ssm = pred_conf_mask[index[0], index[1]]
            pred_obj_conf = np.median(ssm)
            pred_conf_list.append(pred_obj_conf)
        pred_conf_scores = torch.from_numpy(np.array(pred_conf_list))

        # change predicted labels to each signal object masks not existed padding as zero
        pred_ins = torch.zeros_like(gt_ins)
        pred_ins[..., :valid_pred_num] = F.one_hot(pred_label)[..., valid_pred_labels]

        cost_ce, cost_iou, order_row, order_col = self.hungarian(pred_ins.reshape((-1, self.ins_num)),
                                                            gt_ins.reshape((-1, self.ins_num)),gt_ins_num)

        valid_inds = order_col[:gt_ins_num]
        ious_metrics = 1 - cost_iou[order_row, valid_inds]

        # prepare confidence values
        confidence = torch.zeros(size=[gt_ins_num])
        for i, valid_ind in enumerate(valid_inds):
            if valid_ind < valid_pred_num:
                confidence[i] = pred_conf_scores[valid_ind]
            else:
                confidence[i] = 0

        ap = self.calculate_ap(ious_metrics, gt_ins_num, confidence=confidence, function_select='integral')

        invalid_mask = valid_inds >= valid_pred_num
        valid_inds[invalid_mask] = 0
        valid_pred_labels = torch.from_numpy(np.array(valid_pred_labels))
        return_labels = valid_pred_labels[valid_inds].cpu().numpy()
        return_labels[invalid_mask] = -1

        return pred_label, ap, return_labels, gt_label_np

    # vis instance at testing phrase
    def render_label2img(self, predicted_labels, rgbs, color_dict, ins_map):
        unique_labels = torch.unique(predicted_labels)
        predicted_labels = predicted_labels.cpu()
        unique_labels = unique_labels.cpu()
        h, w = predicted_labels.shape
        ra_se_im_t = np.zeros(shape=(h, w, 3))
        for index, label in enumerate(unique_labels):
            label_cpu = str(int(label.cpu()))
            gt_keys = ins_map.keys()
            if label_cpu in gt_keys:
                gt_label_cpu = ins_map[label_cpu]
                ra_se_im_t[predicted_labels == label] = rgbs[color_dict[str(gt_label_cpu)]]
        ra_se_im_t = ra_se_im_t.astype(np.uint8)
        return ra_se_im_t
    def render_gt_label2img(self, gt_labels, rgbs, color_dict):
        unique_labels = torch.unique(gt_labels)
        gt_labels = gt_labels.cpu()
        unique_labels = unique_labels.cpu()
        h, w = gt_labels.shape
        ra_se_im_t = np.zeros(shape=(h, w, 3))
        for index, label in enumerate(unique_labels):
            label_cpu = str(int(label.cpu()))
            gt_keys = color_dict.keys()
            if label_cpu in gt_keys:
                ra_se_im_t[gt_labels == label] = rgbs[color_dict[str(label_cpu)]]
        ra_se_im_t = ra_se_im_t.astype(np.uint8)
        return ra_se_im_t
