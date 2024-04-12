import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed
from torch.nn import functional as F

from torch.utils.data import DataLoader

from gpnerf.data_loaders import dataset_dict
from gpnerf.render_ray import render_rays
from gpnerf.render_image import render_single_image
from gpnerf.model import GPNeRFModel
from gpnerf.ibrnet import IBRNetModel


from gpnerf.sample_ray import RaySamplerSingleImage
from gpnerf.criterion import SemanticCriterion
from utils import img_HWC2CHW, img2psnr, colorize, img2psnr, lpips, ssim
from gpnerf.loss import RenderLoss, SemanticLoss, IoU, DepthLoss
import config
import torch.distributed as dist
from gpnerf.projection import Projector
import imageio
import logging


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank=0
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def render(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)
    
    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # create finetuning dataset for each scene
    train_set_lists, val_set_lists, scene_set_names= [], [], []
    ft_scenes = np.loadtxt(args.val_set_list, dtype=str).tolist()
    for name in ft_scenes:
        train_dataset = dataset_dict['val_scannet'](args, is_train=True, scenes=name)
        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(train_dataset)
            if args.distributed
            else None
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            worker_init_fn=lambda _: np.random.seed(),
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=train_sampler,
            shuffle=True if train_sampler is None else False,
        )
        train_set_lists.append(train_loader)
        val_dataset = dataset_dict['val_scannet'](args, is_train=False, scenes=name)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
        val_set_lists.append(val_loader)
        scene_set_names.append(name.split('/')[1])
        os.makedirs(out_folder + '/' + name.split('/')[1], exist_ok=True)

        print(f'{name} val set len {len(val_loader)}')

    # create projector
    projector = Projector(device=device)

    # Create criterion
    render_criterion = RenderLoss(args)
    semantic_criterion = SemanticLoss(args)
    depth_criterion = DepthLoss(args)
    iou_criterion = IoU(args)

    for val_loader, scene_name in zip(val_set_lists, scene_set_names):
        args.ckpt_path = f'./out/{args.expname}/model_219999.pth'
        model = GPNeRFModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.CRITICAL,
                        filename=os.path.join(out_folder, scene_name, "result.log"),
                        filemode='a')
        print("Evaluating...")
        indx = 0
        psnr_scores,lpips_scores,ssim_scores, iou_scores, depth_scores = [],[],[],[],[]
        for val_data in val_loader:
            tmp_ray_sampler = RaySamplerSingleImage(val_data, device, render_stride=args.render_stride)
            H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
            gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
            gt_depth = val_data['true_depth'][0]

            psnr_curr_img, lpips_curr_img, ssim_curr_img, iou_metric, depth_metric = log_view(
                indx,
                args,
                model,
                tmp_ray_sampler,
                projector,
                gt_img,
                gt_depth,
                evaluator=[iou_criterion, semantic_criterion, depth_criterion],
                render_stride=args.render_stride,
                prefix="val/",
                out_folder=out_folder,
                ret_alpha=args.N_importance > 0,
                single_net=args.single_net,
                val_name = scene_name
            )
            psnr_scores.append(psnr_curr_img)
            lpips_scores.append(lpips_curr_img)
            ssim_scores.append(ssim_curr_img)
            iou_scores.append(iou_metric)
            depth_scores.append(depth_metric)
            torch.cuda.empty_cache()
            indx += 1
        scene_psnr  = np.mean(psnr_scores)
        scene_iou  = np.mean(iou_scores)
        scene_psnr = np.mean(psnr_scores)
        scene_lpips = np.mean(lpips_scores)
        scene_ssim = np.mean(ssim_scores)
        scene_depth = np.mean(depth_scores)
        print("Average {} PSNR: {}, LPIPS: {}, SSIM: {}, IoU: {}, Depth: {}".format(
            scene_name,scene_psnr ,scene_iou  ,scene_psnr ,scene_lpips,scene_ssim,scene_depth))
                        


@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    gt_img,
    gt_depth,
    evaluator,
    render_stride=1,
    prefix="",
    out_folder="",
    ret_alpha=False,
    single_net=True,
    val_name = None,
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()

        ref_coarse_feats, fine_feats, ref_deep_semantics = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        ref_deep_semantics = model.feature_fpn(ref_deep_semantics)
        device = ref_deep_semantics.device

        _, _, que_deep_semantics = model.feature_net(gt_img.unsqueeze(0).permute(0, 3, 1, 2).to(ref_coarse_feats.device))
        que_deep_semantics = model.feature_fpn(que_deep_semantics)
        
        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=ref_coarse_feats,
            deep_semantics=ref_deep_semantics, # encoder的语义输出
            ret_alpha=ret_alpha,
            single_net=single_net,
        )
        
        ret['outputs_coarse']['sems'] = model.sem_seg_head(ret['outputs_coarse']['feats_out'].permute(2,0,1).unsqueeze(0).to(device), None, None).permute(0,2,3,1)
        ret['outputs_fine']['sems'] = model.sem_seg_head(ret['outputs_fine']['feats_out'].permute(2,0,1).unsqueeze(0).to(device), None, None).permute(0,2,3,1)
        
        ret['que_sems'] = model.sem_seg_head(que_deep_semantics, None, None).permute(0,2,3,1)
        


    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))
    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        gt_depth = gt_depth[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3 * w_max)
    rgb_im[:, : average_im.shape[-2], : average_im.shape[-1]] = average_im
    rgb_im[:, : rgb_gt.shape[-2], w_max : w_max + rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, : rgb_pred.shape[-2], 2 * w_max : 2 * w_max + rgb_pred.shape[-1]] = rgb_pred
    if "depth" in ret["outputs_coarse"].keys():
        depth_pred = ret["outputs_coarse"]["depth"].detach().cpu()
        depth_pred = torch.cat((colorize(gt_depth.squeeze(-1).detach().cpu(), cmap_name="jet"), colorize(depth_pred, cmap_name="jet")), dim=1)

        depth_im = img_HWC2CHW(depth_pred)
    else:
        depth_im = None
    
    if ret["outputs_fine"] is not None:
        rgb_fine = img_HWC2CHW(ret["outputs_fine"]["rgb"].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, : rgb_fine.shape[-2], : rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_pred = torch.cat((depth_pred, colorize(ret["outputs_fine"]["depth"].detach().cpu(), cmap_name="jet")), dim=1)
        depth_im = img_HWC2CHW(depth_pred)

    rgb_im = rgb_im.permute(1, 2, 0).detach().cpu().numpy()
    filename = os.path.join(out_folder, val_name, "rgb_{:03d}.png".format(global_step))
    imageio.imwrite(filename, rgb_im)
    if depth_im is not None:
        depth_im = depth_im.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(out_folder, val_name, "depth_{:03d}.png".format(global_step))
        imageio.imwrite(filename, depth_im)
    
    # write scalar
    pred_rgb = (
        ret["outputs_fine"]["rgb"]
        if ret["outputs_fine"] is not None else ret["outputs_coarse"]["rgb"]
    )

    lpips_curr_img = lpips(pred_rgb, gt_img, format="HWC").item()
    ssim_curr_img = ssim(pred_rgb, gt_img, format="HWC").item()
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    iou_metric = evaluator[0](ret, ray_batch, global_step)
    ret["outputs_fine"]['que_sems'] = ret["que_sems"]
    sem_imgs = evaluator[1].plot_semantic_results(ret["outputs_fine"], ray_batch, global_step, val_name, vis=True)
    evaluator[1].plot_pca_features(ret, ray_batch, global_step, val_name, vis=True)


    print(prefix + "psnr_image: ", psnr_curr_img)
    print(prefix + "lpips_image: ", lpips_curr_img)
    print(prefix + "ssim_image: ", ssim_curr_img)
    print(prefix + "iou: ", iou_metric['miou'].item())

    logging.critical("{}-No.{:03d} PSNR: {}, LPIPS: {}, SSIM: {}, IoU: {}".format(val_name, global_step, psnr_curr_img, lpips_curr_img, ssim_curr_img, iou_metric['miou'].item()))
    if 'que_miou' in iou_metric.keys():
        print(prefix + "que_miou: ", iou_metric['que_miou'].item())
    return psnr_curr_img, lpips_curr_img, ssim_curr_img, iou_metric['miou'].item(), iou_metric['que_miou'].item()

if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    args.semantic_color_map=[
        [174, 199, 232],  # wall
        [152, 223, 138],  # floor
        [31, 119, 180],   # cabinet
        [255, 187, 120],  # bed
        [188, 189, 34],   # chair
        [140, 86, 75],    # sofa
        [255, 152, 150],  # table
        [214, 39, 40],    # door
        [197, 176, 213],  # window
        [148, 103, 189],  # bookshelf
        [196, 156, 148],  # picture
        [23, 190, 207],   # counter
        [247, 182, 210],  # desk
        [219, 219, 141],  # curtain
        [255, 127, 14],   # refrigerator
        [91, 163, 138],   # shower curtain
        [44, 160, 44],    # toilet
        [112, 128, 144],  # sink
        [227, 119, 194],  # bathtub
        [82, 84, 163],    # otherfurn
        [248, 166, 116]  # invalid
    ]
    init_distributed_mode(args)

    render(args)