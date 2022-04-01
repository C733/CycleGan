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

class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
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
        print(output.keys())
        return list(output.values())

    def forward(self, output, gt):
        loss = []
        output_features = self.output_features(output)
        gt_features = self.output_features(gt)
        for iter,(dehaze_feature, gt_feature,loss_weight) in enumerate(zip(output_features, gt_features,self.weight)):
            loss.append(F.mse_loss(dehaze_feature, gt_feature)*loss_weight)
        return sum(loss),output_features  #/len(loss)


        
if __name__ == "__main__":

    vgg_model = models.vgg19(pretrained=False).features[:]
    vgg_model.load_state_dict(torch.load('vgg_loss.pth'),strict=False)
    vgg_model.eval()
    for param in vgg_model.parameters():
        param.requires_grad = False 
    vgg_loss = LossNetwork(vgg_model)
    # print(vgg_model)
    x = torch.rand([4, 3,256,256])
    y = torch.rand([4, 3,256,256])
    print(vgg_loss(x,y))



    