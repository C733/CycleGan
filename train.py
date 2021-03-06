from numpy import float32
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
import matplotlib.pyplot as plt
import numpy as np
def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out
def color_loss(con, fake):
    con = rgb_to_yuv(con).cuda()
    fake = rgb_to_yuv(fake).cuda()

    return F.l1_loss(con[:,0,:,:], fake[:,0,:,:]) + F.huber_loss(con[:,1,:,:],fake[:,1,:,:]) + F.huber_loss(con[:,2,:,:],fake[:,2,:,:]) 

class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '31': "relu5_2"
        }
        self.weight =[1.0,1.0,1.0,1.0,1.0]
    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            # print("vgg_layers name:",name,module)
            x = x.type(torch.FloatTensor)
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        # print(output.keys())
        return list(output.values())

    def forward(self, output, gt):
        loss = []
        # consistant_loss = []
        output_features = self.output_features(output)
        gt_features = self.output_features(gt)
        for iter,(dehaze_feature, gt_feature,loss_weight) in enumerate(zip(output_features, gt_features,self.weight)):
            loss.append(F.mse_loss(dehaze_feature, gt_feature)*loss_weight)
            # consistant_loss.append(F.l1_loss(gram_matrix(dehaze_feature),gram_matrix(gt_feature))*loss_weight)
        return sum(loss)  #/len(loss)

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
        
def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler,vgg_loss, vgg_loss_val):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)
    D_loss_total = 0
    G_loss_total = 0
    style_loss = 0
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)
        # print(zebra.shape)
        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss)/2
            D_loss_total += float(D_loss)
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))
            # print(loss_G_Z)
            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            # print(zebra.shape)
            # cycle_zebra_loss = l1(zebra, cycle_zebra)
            # cycle_horse_loss = l1(horse, cycle_horse)
            # print(loss_G_Z)
            # print(zebra.shape)
            # print(vgg_loss_val)
            
            if vgg_loss_val:
                # print("using vgg loss")
                # style texture loss
                # contant_zebra_loss, cycle1 = vgg_loss(fake_zebra, zebra)
                # contant_horse_loss, cycle2 = vgg_loss(fake_horse, horse)
                # cycle consistancy feature loss
                style_zebra_loss = vgg_loss(fake_zebra, zebra)
                style_horse_loss = vgg_loss(fake_horse, horse)
                cycle_zebra_loss = l1(zebra, cycle_zebra)
                cycle_horse_loss = l1(horse, cycle_horse)
            
                # contant_zebra_loss *= 1000
                # contant_horse_loss *= 1000
                
            else:
                # print("using l1 loss")
                # cycle consistancy normal loss
                cycle_zebra_loss = l1(zebra, cycle_zebra)
                cycle_horse_loss = l1(horse, cycle_horse)
            
            # color_zebra_loss = color_loss(cycle_zebra,zebra)
            # color_horse_loss = color_loss(cycle_horse,horse)
            # print(color_horse_loss)
            # # print(color_zebra_loss.get_device())
            # cycle_zebra_loss += color_zebra_loss.cpu()
            # cycle_horse_loss += color_horse_loss.cpu()
            # print(cycle_zebra_loss)
            # print(cycle_horse_loss)
            # print(vgg_loss_zebra)
            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
            if vgg_loss_val:
                G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + style_zebra_loss * 100
                    + style_horse_loss * 100
                    + cycle_zebra_loss * 10
                    + cycle_horse_loss * 10
                    # + cycle_zebra_loss * config.LAMBDA_CYCLE
                    # + cycle_horse_loss * config.LAMBDA_CYCLE
                    + identity_horse_loss * 0.5
                    + identity_zebra_loss * 0.5
                )
                style_loss += (float(style_zebra_loss * 100) + float(style_horse_loss * 100)) / 2
            else:
                G_loss = (
                    loss_G_Z
                    + loss_G_H
                    # + color_zebra_loss
                    # + color_horse_loss
                    # + contant_zebra_loss
                    # + contant_horse_loss
                    + cycle_zebra_loss * 10
                    + cycle_horse_loss * 10
                    # + cycle_zebra_loss * config.LAMBDA_CYCLE
                    # + cycle_horse_loss * config.LAMBDA_CYCLE
                    + identity_horse_loss * 0.5
                    + identity_zebra_loss * 0.5
                )
            G_loss_total += float(G_loss)
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 100 == 0:
            save_image(fake_horse*0.5+0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra*0.5+0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))

    return D_loss_total / len(loop), G_loss_total / len(loop), style_loss / len(loop)

def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            "pre_trained\model\l1+color\genh.pth.tar", gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            "pre_trained\model\l1+color\genz.pth.tar",gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            "pre_trained\model\l1+color\critich.pth.tar", disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            "pre_trained\model\l1+color\criticz.pth.tar", disc_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR+"/arcane_face", root_zebra=config.TRAIN_DIR+"/real_face", transform=config.transforms
    )
    # val_dataset = HorseZebraDataset(
    #    root_horse=config.VAL_DIR+"/horses", root_zebra=config.VAL_DIR+"/zebras", transform=config.transforms
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    # )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    loss_D = []
    loss_G = []
    loss_style = []
    vgg_model = models.vgg19(pretrained=False).features[:]
    vgg_model.load_state_dict(torch.load('vgg_loss.pth'),strict=False)
    vgg_model.eval()
    for param in vgg_model.parameters():
        param.requires_grad = False 
    vgg_loss = LossNetwork(vgg_model)
    for epoch in range(config.NUM_EPOCHS):
        avg_D_loss, avg_G_loss, style_loss =  train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler,vgg_loss,vgg_loss_val=True)
        loss_D.append(avg_D_loss)
        loss_G.append(avg_G_loss)
        loss_style.append(style_loss)
        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename="pre_trained\model\l1+style\genh.pth.tar")
            save_checkpoint(gen_Z, opt_gen, filename="pre_trained\model\l1+style\genz.pth.tar")
            save_checkpoint(disc_H, opt_disc, filename="pre_trained\model\l1+style\critich.pth.tar")
            save_checkpoint(disc_Z, opt_disc, filename= "pre_trained\model\l1+style\criticz.pth.tar")
    loss_D = np.array(loss_D)
    loss_G = np.array(loss_G)
    style_loss = np.array(style_loss)
    np.savetxt('loss_D.txt', loss_D)
    np.savetxt('loss_G.txt', loss_G)
    np.savetxt('style_loss.txt', style_loss)
    # x_axix = range(config.NUM_EPOCHS)
    # plt.figure(0)
    # plt.title('Discriminator loss')
    # plt.plot(x_axix, loss_D, color='red',label='Discriminator loss')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.savefig('./result.png')
    # plt.figure(1)
    # plt.title('Generateor loss')
    # plt.plot(x_axix, loss_G, color='green',label='Generator loss')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.savefig('./result1.png')


if __name__ == "__main__":
    main()