import argparse
import argparse
import datetime
from glob import glob
import json
from PIL import Image
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.utils
import torch.utils.data
from torch.utils.data import Dataset
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import timm

assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
import util.transforms as mtransforms
import torch.nn.functional as F
import models_mae
import io
import matplotlib
import seaborn as sns
import random

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    # Dataset parameters
    parser.add_argument('--data_path', default='/dx/ImageNet/visual-4', type=str,
                        help='dataset path')
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--output_dir', default='./visualized',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    return parser

class VisualizeDataset(Dataset):
    def __init__(self, folder: str, transforms=None):
        super().__init__()
        self.files = glob(os.path.join(folder, "*.JPEG"))
        self.transforms = transforms
        self._folder = os.path.abspath(folder)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with open(path, "rb") as f:
            sample = Image.open(f)
            sample = sample.convert("RGB")
        target_folder = os.path.basename(path).split("_")[0]
        train_bbox_folder = os.path.abspath(os.path.join(self._folder, os.path.pardir))
        train_bbox_folder = os.path.join(train_bbox_folder, "train_bbox", target_folder)
        bpath = path.replace(self._folder, train_bbox_folder).replace('.JPEG', '.json')
        if os.path.exists(bpath):
            with open(bpath, "r") as f:
                xmin, ymin, xmax, ymax = json.load(f)
            w, h = sample.size[0], sample.size[1]
            assert xmax <= w and ymax <= h
            mask = torch.zeros(1, h, w)
            mask[0, ymin:ymax, xmin:xmax] = 1
            sample = sample, mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        if os.path.exists(bpath):
            return sample

        return sample, torch.zeros(1, sample.shape[1], sample.shape[2])

def scorer_mask(scorer_pred, mask):
    mask = mask.reshape(mask.shape[0], mask.shape[1], 14, 14)
    mask = scorer_pred(mask)
    before_mask = mask.detach()
    softmax_mask = F.softmax(mask, dim=-1)
    mask = F.gumbel_softmax(mask, tau=1) # torch.sigmoid(mask)  torch.sigmoid(mask)
    return mask, before_mask, softmax_mask

def unpatchify(x):
    p = 16
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def plt_fig_to_img(fig=None, dpi=150):
    with io.BytesIO() as buf:
        if fig is None:
            fig = plt.gcf()
        fig.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        img_arr = np.array(Image.open(buf))
        return img_arr


def main(args):
    device = torch.device(args.device)
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    transform_train = mtransforms.Compose([
            mtransforms.Resize((args.input_size, args.input_size), interpolation=3),  # 3 is bicubic
            mtransforms.ToTensor()])

    normalizer = mtransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = VisualizeDataset(args.data_path, transforms=transform_train)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
    )

    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, scorer=True)
    state_dict = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.to(device)

    del state_dict

    scorer = models_mae.mae_vit_base_patch16(norm_pix_loss=args.norm_pix_loss)
    scorer.load_state_dict(torch.load("/data/mae/checkpoint-799.pth", map_location="cpu")["model"], strict=False)
    scorer.requires_grad_(False)
    scorer.to(device)

    torch.set_grad_enabled(False)

    for i, (img, mask_gt) in enumerate(data_loader):
        img = img.to(device)
        for c, m in enumerate((0.485, 0.456, 0.406)):
            img[:, c][img[:, c] == 0] = m 
        mask_gt = mask_gt.reshape(-1, 224, 224)
        samples = normalizer(img)
        samples = samples.to(device)
        with torch.cuda.amp.autocast():
            attn_map = scorer.get_last_selfattention(samples)[:,:, 0, 1:]
            loss, y, (masks_actual, masks_prob)  = model(samples, mask_ratio=args.mask_ratio, mask=attn_map, mask_factor=0.5)
            masks_prob, before_mask = masks_prob
        before_mask = before_mask.float()
        masks_actual = masks_actual.float()

        before_mask_top25_vals, before_mask_top25_idx = before_mask.topk(49, dim=-1)
        before_mask_top25 = torch.zeros_like(before_mask)
        before_mask_top25[:,before_mask_top25_idx] = 1
        before_mask_top25 = before_mask_top25.reshape(-1, 14, 14)

        mask_top25_vals, mask_top25_idx = masks_prob.topk(49, dim=-1)
        mask_top25 = torch.zeros_like(masks_prob)
        mask_top25[:, mask_top25_idx] = 1
        mask_top25 = mask_top25.reshape(-1, 14, 14)

        masks_prob = masks_prob.reshape(-1, 14, 14).float()
        before_mask = before_mask.reshape(-1, 14, 14).float()
        masks_actual = unpatchify(masks_actual.unsqueeze(-1).repeat(1, 1, 16**2 *3))
        samples_masked = img * (1 - masks_actual)
        heatmap = sns.heatmap(before_mask.reshape(14, 14).cpu().numpy(), cmap=matplotlib.cm.winter, alpha=0.4, annot=True, annot_kws={'size': 4}, zorder=2)
        fig = plt.gcf()
        fig.savefig(os.path.join(args.output_dir, f"{i}-heatmap.png"), format='png', dpi=150)
        plt.clf()
        attentions = F.interpolate(mask_top25.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
        plt.imsave(fname=os.path.join(args.output_dir, f"{i}-sample-top25.png"), arr=attentions[0])
        attentions = F.interpolate(before_mask_top25.unsqueeze(0), scale_factor=16, mode="nearest")[0]
        blended_before_mask = attentions[None, :] * 255 * 0.5 + img * 0.5
        attentions = attentions.cpu().numpy()
        torchvision.utils.save_image(torchvision.utils.make_grid(blended_before_mask[0], scale_each=True), os.path.join(args.output_dir, f"{i}-img-before25blended.png"))
        plt.imsave(fname=os.path.join(args.output_dir, f"{i}-sample-before-top25.png"), arr=attentions[0])

        torchvision.utils.save_image(torchvision.utils.make_grid(img[0], scale_each=True), os.path.join(args.output_dir, f"{i}-img.png"))
        torchvision.utils.save_image(torchvision.utils.make_grid(samples_masked[0], normalize=True, scale_each=True), os.path.join(args.output_dir, f"{i}-img_masked.png"))
        attentions = F.interpolate(masks_prob.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
        plt.imsave(fname=os.path.join(args.output_dir, f"{i}-sample.png"), arr=attentions[0])
        attentions = F.interpolate(before_mask.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
        plt.imsave(fname=os.path.join(args.output_dir, f"{i}-sample-before.png"), arr=attentions[0])
        plt.imsave(fname=os.path.join(args.output_dir, f"{i}-mask.png"), arr=mask_gt[0])

        print(f"{i} / {len(dataset)}")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

