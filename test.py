from turtle import color
import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
opt_gen = optim.Adam(
    list(gen_Z.parameters()) + list(gen_H.parameters()),
    lr=config.LEARNING_RATE,
    betas=(0.5, 0.999),
)
load_checkpoint(
    "pre_trained\model\l1+color\genh.pth.tar", gen_H, opt_gen, config.LEARNING_RATE,
)
load_checkpoint(
   "pre_trained\model\l1+color\genz.pth.tar", gen_Z, opt_gen, config.LEARNING_RATE,
)
zebra_path = "test/girl.jpg"
horse_path = "test/girl.jpg"

zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
horse_img = np.array(Image.open(horse_path).convert("RGB"))


transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
augmentations = transforms(image=zebra_img, image0=horse_img)
zebra_img = augmentations["image"]
horse_img = augmentations["image0"]
horse_img = horse_img.to(config.DEVICE)
arcane_chongyu = gen_H(horse_img)
save_image(arcane_chongyu*0.5+0.5, f"arcane_girl.png")
real_chongyu = gen_Z(arcane_chongyu)
save_image(real_chongyu*0.5+0.5, f"real_girl.png")