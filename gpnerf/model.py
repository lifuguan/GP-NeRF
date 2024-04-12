import torch
import torch.nn as nn
import os
from gpnerf.transformer_network import GNT
from gpnerf.feature_network import ResUNet, ResUNetLight
from gpnerf.sem_feature_network import resnet50
from gpnerf.fpn import FPN
from gpnerf.semantic_branch import NeRFSemSegFPNHead


def de_parallel(model):
    return model.module if hasattr(model, "module") else model


########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class GPNeRFModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        device = torch.device("cuda:{}".format(args.local_rank))
        # create coarse GNT
        self.net_coarse = GNT(
            args,
            in_feat_ch=self.args.coarse_feat_dim,
            posenc_dim=3 + 3 * 2 * 10,
            viewenc_dim=3 + 3 * 2 * 10,
            ret_alpha=args.N_importance > 0,
        ).to(device)
        # single_net - trains single network which can be used for both coarse and fine sampling
        if args.single_net:
            self.net_fine = None
        else:
            self.net_fine = GNT(
                args,
                in_feat_ch=self.args.fine_feat_dim,
                posenc_dim=3 + 3 * 2 * 10,
                viewenc_dim=3 + 3 * 2 * 10,
                ret_alpha=True,
            ).to(device)

        # create feature extraction network
        self.feature_net = ResUNet(
            coarse_out_ch=self.args.coarse_feat_dim,
            fine_out_ch=self.args.fine_feat_dim,
            single_net=self.args.single_net,
        ).to(device)
        # self.feature_net = ResUNetLight(out_dim=20+1).to(device)

        self.feature_fpn = FPN(in_channels=[64,64,128,256], out_channels=128, concat_out=True).to(device)
        self.sem_seg_head = NeRFSemSegFPNHead(args).to(device)

        # optimizer and learning rate scheduler
        learnable_params = list(self.net_coarse.parameters())
        learnable_params += list(self.feature_net.parameters())
        learnable_params += list(self.feature_fpn.parameters())
        learnable_params += list(self.sem_seg_head.parameters())

        if self.net_fine is not None:
            learnable_params += list(self.net_fine.parameters())

        if self.net_fine is not None:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.net_fine.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                    {"params": self.feature_fpn.parameters(), "lr": args.lrate_semantic},
                    {"params": self.sem_seg_head.parameters(), "lr": args.lrate_semantic},
                ],
                lr=args.lrate_gnt,
            )
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                    {"params": self.feature_fpn.parameters(), "lr": args.lrate_semantic},
                    {"params": self.sem_seg_head.parameters(), "lr": args.lrate_semantic},
                ],
                lr=args.lrate_gnt,
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lrate_decay_steps, gamma=args.lrate_decay_factor
        )

        out_folder = os.path.join(args.rootdir, "out", args.expname)
        self.start_step = self.load_from_ckpt(
            out_folder, load_opt=load_opt, load_scheduler=load_scheduler
        )

        if args.distributed:
            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse, device_ids=[args.local_rank], output_device=args.local_rank
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net, device_ids=[args.local_rank], output_device=args.local_rank
            )

            self.feature_fpn = torch.nn.parallel.DistributedDataParallel(
                self.feature_fpn, device_ids=[args.local_rank], output_device=args.local_rank
            )

            self.sem_seg_head = torch.nn.parallel.DistributedDataParallel(
                self.sem_seg_head, device_ids=[args.local_rank], output_device=args.local_rank
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine, device_ids=[args.local_rank], output_device=args.local_rank
                )

    def switch_to_eval(self):
        self.net_coarse.eval()
        self.feature_net.eval()
        self.feature_fpn.eval()
        self.sem_seg_head.eval()
        if self.net_fine is not None:
            self.net_fine.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        self.feature_net.train()
        self.feature_fpn.train()
        self.sem_seg_head.train()
        if self.net_fine is not None:
            self.net_fine.train()

    def save_model(self, filename):
        to_save = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "net_coarse": de_parallel(self.net_coarse).state_dict(),
            "feature_net": de_parallel(self.feature_net).state_dict(),
            "feature_fpn": de_parallel(self.feature_fpn).state_dict(),
            "sem_seg_head": de_parallel(self.sem_seg_head).state_dict(),
        }

        if self.net_fine is not None:
            to_save["net_fine"] = de_parallel(self.net_fine).state_dict()

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location="cuda:{}".format(self.args.local_rank))
        else:
            to_load = torch.load(filename)
        if load_opt:
            self.optimizer.load_state_dict(to_load["optimizer"])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])

        self.net_coarse.load_state_dict(to_load["net_coarse"], strict=False)
        if self.feature_net is not None and "feature_net" in to_load.keys():
            self.feature_net.load_state_dict(to_load["feature_net"], strict=True)
        if self.feature_fpn is not None and "feature_fpn" in to_load.keys():
            self.feature_fpn.load_state_dict(to_load["feature_fpn"], strict=True)
        
        if self.sem_seg_head is not None and "sem_seg_head" in to_load.keys():
            self.sem_seg_head.load_state_dict(to_load["sem_seg_head"], strict=True)

        if self.net_fine is not None and "net_fine" in to_load.keys():
            self.net_fine.load_state_dict(to_load["net_fine"], strict=True)
            self.net_fine = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_fine)

        self.net_coarse = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_coarse)
        self.feature_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.feature_net)
        self.feature_fpn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.feature_fpn)

    def load_from_ckpt(
        self, out_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False
    ):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [
                os.path.join(out_folder, f)
                for f in sorted(os.listdir(out_folder))
                if f.endswith(".pth")
            ]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            try:
                step = int(fpath[-10:-4])
            except:
                step = 0
            print("Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("No ckpts found, training from scratch...")
            step = 0

        return step


class OnlySemanticModel(nn.Module):
    def __init__(self, args) -> None:
        super(OnlySemanticModel, self).__init__()
        # create feature extraction network
        self.feature_net = ResUNet(
            coarse_out_ch=args.coarse_feat_dim,
            fine_out_ch=args.fine_feat_dim,
            single_net=args.single_net,
        )
        # self.feature_net = ResUNetLight(out_dim=20+1).to(device)

        self.feature_fpn = FPN(in_channels=[64,64,128,256], out_channels=128, concat_out=True)
        self.sem_seg_head = NeRFSemSegFPNHead(args)

    def forward(self, rgb) -> torch.Tensor:
        _, _, que_deep_semantics = self.feature_net(rgb)
        que_deep_semantics = self.feature_fpn(que_deep_semantics)
        sem_out = self.sem_seg_head(que_deep_semantics, None, None)
        return sem_out

class SSLSemModel(nn.Module):
    def __init__(self, args) -> None:
        super(SSLSemModel, self).__init__()
        # create feature extraction network
        self.feature_net = resnet50()

        # self.feature_net = models.resnet50()

        self.feature_fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channels=128, concat_out=True)
        self.sem_seg_head = NeRFSemSegFPNHead(args)

    def forward(self, rgb) -> torch.Tensor:
        que_deep_semantics = self.feature_net(rgb)
        que_deep_semantics = self.feature_fpn(que_deep_semantics)
        sem_out = self.sem_seg_head(que_deep_semantics, None, None)
        return sem_out
    