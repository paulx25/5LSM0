from model import Model
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import wandb
import process_data as p
import utils
import os

wandb.init(
    project="Efficiency_new",

    config={
    "learning_rate": 0.001,
    "architecture": "CNN",
    "dataset": "Cityscapes",
    "epochs": 200,
    }
)

def preprocess_masks(masks):
    transform = transforms.Compose([
        transforms.Resize((256, 512),transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    masks = transform(masks)
    masks = (masks*255).long().squeeze()
    masks = utils.map_id_to_train_id(masks)

    return masks
    
def preprocess_masks2(masks):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    masks = transform(masks)
    masks = (masks*255).long().squeeze()
    masks = utils.map_id_to_train_id(masks)

    return masks

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    return parser
    
def DICE(images,masks,model):
    dice_tot = []
    for j in range(images.shape[0]):
        image = model(images[j])
        pred = p.postprocess(image, (1024,2048))
        pred = pred.flatten()
        mask = masks[j].squeeze().flatten()
    
        dice = []
    
        for l in range(19):
            pred_tot = len(pred[pred==l])
            mask_tot = len(mask[mask==l])
            inter = len([i for i, v in enumerate(pred) if v == l and v == mask[i]])
            score = (2.*inter + 0.00001)/(pred_tot + mask_tot + 0.00001)
            dice.append(score)
    
        mean_score = sum(dice)/19
        dice_tot.append(mean_score)
    mean_dice = sum(dice_tot)/len(dice_tot)
    return dice_tot, mean_dice


def main(args):
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=p.preprocess, target_transform=preprocess_masks)
    train_data, val_data = random_split(dataset, (2500, 475))
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=18, pin_memory=True)
    validationloader = DataLoader(val_data, batch_size=20, shuffle=True, num_workers=18, pin_memory=True)

    model = Model().cuda()

    def train_model(model, trainloader, num_epochs, lr):
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_val_loss = 0.0
            model.train()
            for inputs, masks in trainloader:
                inputs = inputs.squeeze().cuda()
                masks = masks.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
      
                running_loss += loss.item()
    
            epoch_loss = running_loss / len(trainloader)
            
            model.eval()
            with torch.no_grad():
                for inputs, masks in validationloader:
                    inputs = inputs.squeeze().cuda()
                    masks = masks.cuda()
                    val_outputs = model(inputs)
                    val_loss = criterion(val_outputs, masks)
                    running_val_loss += val_loss.item()
                
                epoch_val_loss = running_val_loss / len(validationloader)
                wandb.log({"training loss": epoch_loss, "validation loss": epoch_val_loss})
        
    train_model(model, trainloader, 200, 0.001)

    torch.save(model.state_dict(), "".join([os.environ["WANDB_DIR"], "/model.pth"]))
#    
#    for i, (images, masks) in enumerate(validationloader):
#        if i >= 1:
#            break
#        images = images.cuda()
#        masks = masks.numpy()
#        dice_tot, mean_dice = DICE(images,masks,model)
#        print(dice_tot)
#        print(mean_dice)
    
    wandb.finish()  
    pass


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
