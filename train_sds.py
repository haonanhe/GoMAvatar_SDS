import os
import cv2
import logging
import argparse
import numpy as np
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

from configs import make_cfg
from dataset.train_sds import Dataset
from models.model_sds import Model

from utils.train_util import cpu_data_to_gpu, make_weights_for_pose_balance
from utils.image_util import to_8b_image
from utils.tb_util import TBLogger

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        default=None,
        type=str
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def unpack(rgbs, masks, bgcolors):
    rgbs = rgbs * masks.unsqueeze(-1) + bgcolors[:, None, None, :] * (1 - masks).unsqueeze(-1)
    return rgbs


def main(args):
    cfg = make_cfg(args.cfg)

    os.makedirs(cfg.save_dir, exist_ok=True)
    # setup logger
    logging_path = os.path.join(cfg.save_dir, 'log.txt')
    logging.basicConfig(
        handlers=[
            logging.FileHandler(logging_path),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    # save config file
    os.makedirs(os.path.join(cfg.save_dir), exist_ok=True)
    with open(os.path.join(cfg.save_dir, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    logging.info(f'configs: \n{cfg.dump()}')
    os.makedirs(os.path.join(cfg.save_dir, 'checkpoints'), exist_ok=True)

    # setup tensorboard
    tb_logger = TBLogger(os.path.join(cfg.save_dir, 'tb'), freq=cfg.train.tb_freq)

    # '/home/ubuntu/Codes/GoMAvatar/data/zju-mocap/377/canonical_joints.pkl'
    # load training data
    dataset = Dataset(
        cfg.dataset.train.dataset_path,
    )

    # load model
    cfg.model.use_smplx = True
    model = Model(cfg.model, dataset.get_canonical_info())
    model.cuda()
    model.train()

    # load optimizer
    param_groups = model.get_param_groups(cfg.train)
    optimizer = optim.Adam(param_groups, betas=(0.9, 0.999))

    n_iters = 1
    if args.resume:
        ckpt_dir = os.path.join(cfg.save_dir, 'checkpoints')
        max_iter = max([int(filename.split('_')[-1][:-3]) for filename in os.listdir(ckpt_dir)])
        ckpt_path = os.path.join(ckpt_dir, f'iter_{max_iter}.pt')
        ckpt = torch.load(ckpt_path)

        for i in cfg.model.subdivide_iters:
            if max_iter >= i:
                model.subdivide()

        model.load_state_dict(ckpt['network'])
        optimizer.load_state_dict(ckpt['optimizer'])
        logging.info(f"load from checkpoint {ckpt_path}")

        n_iters = ckpt['iter'] + 1
        logging.info(f'continue training from iter {n_iters}')
    else:
        ckpt_path = os.path.join(cfg.save_dir, 'checkpoints', f'iter_0.pt')
        torch.save({
            'iter': n_iters,
            'network': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'scheduler': lr_scheduler.state_dict(),
        }, ckpt_path)
        logging.info(f'saved to {ckpt_path}')

    ### start iteration
    data = dataset.load_data() # load canonical data

    data = cpu_data_to_gpu(
        data, exclude_keys=EXCLUDE_KEYS_TO_GPU)
    
    for n_iters in tqdm(range(cfg.train.total_iters), desc='Processing', unit='iteration'):
    # while n_iters <= cfg.train.total_iters:
        tb_logger.set_global_step(n_iters)

        rgb, mask, head_rgb, head_mask, outputs = model(
            None, None,
            data['cnl_gtfms'], data['dst_Rs'], data['dst_Ts'], dst_posevec=data['dst_posevec'],
            canonical_joints=data['dst_tpose_joints'],
            i_iter=n_iters,
            bgcolor=data['bgcolor'],
            tb=tb_logger)

        if cfg.random_bgcolor:
            rgb = unpack(rgb, mask, data['bgcolor'])

        # log to tensorboard
        if n_iters % cfg.train.log_freq == 0:
            tb_logger.summ_image('pred/rgb', rgb.permute(0, 3, 1, 2)[0])
            tb_logger.summ_image('pred/mask', mask.unsqueeze(1)[0])
            tb_logger.summ_image('pred/head_rgb', head_rgb.permute(0, 3, 1, 2)[0])
            tb_logger.summ_image('pred/head_mask', head_mask.unsqueeze(1)[0])

        # save
        if n_iters % cfg.train.save_freq == 0:
            ckpt_path = os.path.join(cfg.save_dir, 'checkpoints', f'iter_{n_iters}.pt')
            torch.save({
                'iter': n_iters,
                'network': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            logging.info(f'saved to {ckpt_path}')

if __name__ == "__main__":
    args = parse_args()
    main(args)
