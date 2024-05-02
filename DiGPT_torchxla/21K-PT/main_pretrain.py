# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import sys
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import logging

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.device_env_factory import use_xla
from util.iterable_dataset import get_iterable_dataloader
from util.logger import setup_logging
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, consolidate_sharded_model_checkpoints
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
# from engine_pretrain import train_one_epoch

try:
    import wandb
except ImportError:
    wandb = None


def get_args_parser():
    parser = argparse.ArgumentParser('DiGPT_torchxla pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='DiGPT_vit_huge', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--src', type=str, help='google buckets path src')
    parser.add_argument('--des', type=str, help='google buckets path des')
    parser.add_argument('--scale', type=float, default=0.2, help='low scale')


    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    # parser.add_argument('--log_dir', default='./output_dir',
    #                     help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--clip_path', type=str, default="../clip_vit_b_16.pth")

    parser.add_argument(
        "--to-float-on-device",
        default=False,
        action="store_true",
        help="If true, use uint8 image on cpu, and convert to float and do normalization on gpu/tpu."
    )
    parser.add_argument("--log_freq", default=100, type=int,
        help="log every n steps."
    )
    parser.add_argument("--wandb", default=False, action="store_true",
        help="If true, use wandb for logging."
    )
    parser.add_argument("--name", default=None, type=str,
        help="name for storing logs. also name and id used for wandb logging."
    )

    parser.add_argument(
        "--no-quick-gelu",
        default=False,
        action="store_true",
        help="If true, use nn.GELU instead of QuickGELU."
    )

    parser.add_argument("--resume_from_gs", action="store_true", default=False)

    return parser


class TwoCropTransform(object):
    def __init__(self, args):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.common = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip()])
        self.student = transforms.Compose([
            transforms.ColorJitter(0.4,0.4,0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
        self.teacher = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, img):
        img = self.common(img)
        return self.student(img), self.teacher(img)

    def __repr__(self):
        repr = "DataAugmentationForDiGPT"
        return repr


def main(args):
    parser = get_args_parser()
    args = parser.parse_args(args)
    misc.init_distributed_mode(args)

    log_path = None
    # make sure all nodes have the same log and checkpoint
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, args.name)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(args.output_dir, 'out.log')

    # Setup text logger
    setup_logging(log_path, logging.INFO)
    xm.master_print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    xm.master_print("{}".format(args).replace(', ', ',\n'))
    #logging.info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    #logging.info("{}".format(args).replace(', ', ',\n'))

    # ok for xla device
    device = torch.device(args.device)

    # fix the seed for reproducibility. ensure the same model init across processes
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # define the model
    import models_digpt, openclip
    model = FSDP(models_digpt.DiGPT_vit_huge(device=device).to(device))
    state_dict = torch.load(args.clip_path, map_location="cpu")
    teacher = openclip.VisionTransformer(width=state_dict["visual.conv1.weight"].shape[0],
                                         layers=len([k for k in state_dict.keys() if
                                                     k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]),
                                         patch_size=state_dict["visual.conv1.weight"].shape[-1],
                                         image_size=224, mlp_ratio=4.,
                                         heads=state_dict["visual.conv1.weight"].shape[0] // 80,
                                         output_dim=1024, device=device)
    ckpt = {}
    for k, v in state_dict.items():
        if "visual." in k:
            ckpt[k[len("visual."):]] = v
    msg = teacher.load_state_dict(ckpt, strict=False)
    print(msg)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    teacher = FSDP(teacher).to(device)

    #model_without_ddp = model
    # logging.info("Model = %s" % str(model_without_ddp))

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = TwoCropTransform(args)

    if args.data_path.startswith("gs://"):
        data_loader_train = get_iterable_dataloader(args, args.data_path, preprocess_fn=transform_train, is_train=True)
        #data_loader_train = pl.MpDeviceLoader(data_loader_train, device)
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        logging.info(dataset_train)

        if True:  # args.distributed:
            # num_tasks = misc.get_world_size()
            # global_rank = misc.get_rank()
            num_tasks = args.world_size
            global_rank = args.rank
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            logging.info("Sampler_train = %s" % str(sampler_train))
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # logging.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    # logging.info("actual lr: %.2e" % args.lr)
    #
    # logging.info("accumulate grad iterations: %d" % args.accum_iter)
    # logging.info("effective batch size: %d" % eff_batch_size)
    xm.master_print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    xm.master_print("actual lr: %.2e" % args.lr)
    xm.master_print("accumulate grad iterations: %d" % args.accum_iter)
    xm.master_print("effective batch size: %d" % eff_batch_size)

    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if 'norm' in name or name.endswith("bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

    param_groups = add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    #logging.info(optimizer)
    xm.master_print(optimizer)
    loss_scaler = NativeScaler() if not use_xla() else None

    #if args.resume is not None:
    # file = os.listdir(args.output_dir)
    # file = [i for i in file if "epoch" in i]
    # file = sorted(file)
    # for i in range(args.epochs):
    #     if "epoch-{}.pth".format(str(i)) in file:
    #         args.resume = str(i)
    if args.resume is None:
        for i in range(24, 150):
            if os.system("gsutil -q stat {}epoch-{}.pth".format(args.des, i))==0:
                args.resume = i
                print(args.resume)
                xm.rendezvous('Load ckpt')
            else:
                break
    xm.rendezvous('Load ckpt')
    if args.resume is not None:
        rank = xm.get_ordinal()
        word_size = xm.xrt_world_size()
        epoch = args.resume
        os.system("gsutil cp {}checkpoint-ep-{}-rank-{}-word_size-{}.pth {}".format(args.des, epoch, rank, word_size, args.src))
        args.resume = "{}checkpoint-ep-{}-rank-{}-word_size-{}.pth".format(args.src, epoch, rank, word_size)
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.rank == 0:
        logging.info("Params:")
        params_file = os.path.join(args.output_dir, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.wandb and args.rank == 0:
        assert wandb is not None, 'Please install wandb.'
        logging.info('Starting wandb.')
        # you will have to configure this for your project!
        wandb.init(
            project='GKD_TPU_Pretrain',
            name=args.name,
            tags=[],
            resume='allow' if args.resume != '' else None,
            config=vars(args),
        )
        wandb.save(params_file)
        logging.info('Finished loading wandb.')


    logging.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    # defer importing train_one_epoch func
    if use_xla():
        from engine_pretrain_xla import train_one_epoch
    else:
        from engine_pretrain import train_one_epoch


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            if use_xla():
                data_loader_train.dataset.set_epoch(epoch)
            else:
                data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, teacher, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            # log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, rank=xm.get_ordinal(), word_size=xm.xrt_world_size())
            xm.rendezvous('ckpt_save')
            if misc.is_main_process():
                os.system("gsutil cp {}checkpoint-ep-{}* {}".format(args.des, epoch, args.src))
                consolidate_sharded_model_checkpoints(
                    ckpt_prefix=args.output_dir+"/",
                    ckpt_suffix="checkpoint-ep-{}-rank-*-word_size-*.pth".format(str(epoch)),
                    save_path=args.output_dir+"/epoch-{}.pth".format(
                        str(epoch)))
                #os.system("gsutil rm {}checkpoint-ep-{}*".format(args.des, epoch))
                os.system("gsutil cp {}/epoch-{}.pth {}".format(args.src, epoch, args.des))
                os.system("rm {}/check*".format(args.src))
            xm.rendezvous('ckpt_consolidation')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        # if args.output_dir and misc.is_main_process():
        if args.output_dir and args.local_rank == 0:
            # if log_writer is not None:
            #     log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))

def _mp_entry(*args):
    main(sys.argv[1:])

if __name__ == '__main__':
    # args = get_args_parser()
    # args = args.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # main(args)
    main(sys.argv[1:])