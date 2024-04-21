#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import namedtuple
from torch.nn.utils import prune

get_ipython().run_line_magic('matplotlib', 'inline')

from argparse import ArgumentParser
DATA_DIR = os.path.join(os.getenv('TEACHER_DIR'), 'JHS_data')

def preprocess(img):
    img = transforms.functional.resize(img, size=(256, 512), interpolation=transforms.InterpolationMode.LANCZOS)
    trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2870, 0.3252, 0.2840), (0.1711, 0.1759, 0.1731))])
    img = trans(img)
    img = img.unsqueeze(0)

    return img

def postprocess(prediction, shape):
    m = torch.nn.Softmax(dim=1)
    prediction_soft = m(prediction)
    prediction_max = torch.argmax(prediction_soft, axis=1)
    prediction = transforms.functional.resize(prediction_max, size=shape, interpolation=transforms.InterpolationMode.NEAREST)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()

    return prediction_numpy

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = encoder(3, 32)
        self.encoder_2 = encoder(32, 40)
        self.encoder_3 = encoder(40, 50)

        self.middle = convolution_block(50, 63)
        
        self.decoder_1 = decoder(63, 50)
        self.decoder_2 = decoder(50, 40)
        self.decoder_3 = decoder(40, 32)

        self.classifier = nn.Conv2d(32, 19, kernel_size=1, padding=0)

    def forward(self, inputs):
        skip_1, x1 = self.encoder_1(inputs)
        skip_2, x2 = self.encoder_2(x1)
        skip_3, x3 = self.encoder_3(x2)

        x4 = self.middle(x3)

        x5 = self.decoder_1(x4, skip_3)
        x6 = self.decoder_2(x5, skip_2)
        x7 = self.decoder_3(x6, skip_1)

        out = self.classifier(x7)

        return out


class convolution_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.conv_1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

        self.conv_2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(output_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.dropout(x)

        x = self.conv_2(x)
        x = self.dropout(x)
        x = self.batchnorm_2(x)
        x = self.relu(x)

        return x

class encoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.conv = convolution_block(input_channels, output_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        skip = self.conv(inputs)
        x = self.pool(skip)

        return skip, x

class decoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.up_pool = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2, padding=0)
        self.conv = convolution_block(2*output_channels, output_channels)

    def forward(self, inputs, skip):
        x = self.up_pool(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x
    
model = Model()
model.load_state_dict(torch.load('path to model file', map_location=torch.device('cpu')))

def preprocess_masks(masks):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    masks = transform(masks)
    masks = (masks*255).long().squeeze()
    masks = map_id_to_train_id(masks)

    return masks

val = datasets.Cityscapes("/gpfs/work5/0/jhstue005/JHS_data/CityScapes", split='train', mode='fine', target_type='semantic', target_transform=preprocess_masks)
validationloader = DataLoader(val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

LABELS = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

def map_id_to_train_id(label_id):
    """map the id to the train id for cityscapes masks
    input: Tensor of shape (batch_size, height, width) with values from 0 to 33
    output: Tensor of shape (batch_size, height, width) with values from 0 to 18
    """
    # create a tensor with the same shape as the input tensor and fill it with the value 255
    train_id_tensor = torch.full_like(label_id, 255)
    for label in LABELS:
        # replace the value in the tensor with the train id if the value in the input tensor is equal to the id of the label
        train_id_tensor[label_id == label.id] = label.trainId
    return train_id_tensor

colors = {
    0: (128, 64,128) ,
    1: (244, 35,232) ,
    2: ( 70, 70, 70) ,
    3: (102,102,156) ,
    4: (190,153,153) ,
    5: (153,153,153) ,
    6: (250,170, 30) ,
    7: (220,220,  0) ,
    8: (107,142, 35) ,
    9: (152,251,152) ,
    10: ( 70,130,180) ,
    11: (220, 20, 60) ,
    12: (255,  0,  0) ,
    13: (  0,  0,142) ,
    14: (  0,  0, 70) ,
    15: (  0, 60,100) ,
    16: (  0, 80,100) ,
    17: (  0,  0,230) ,
    18: (119, 11, 32) ,
    -1: (  0,  0,142) 
}

def mask_to_rgb(mask, class_to_color):
    # Get dimensions of the input mask
    height, width = mask.shape

    # Initialize an empty RGB mask
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterate over each class and assign corresponding RGB color
    for class_idx, color in class_to_color.items():
        # Mask pixels belonging to the current class
        class_pixels = mask == class_idx
        # Assign RGB color to the corresponding pixels
        rgb_mask[class_pixels] = color

    return rgb_mask

def renormalize_image(image):
    """
    Renormalizes the image to its original range.
    
    Args:
        image (numpy.ndarray): Image tensor to renormalize.
    
    Returns:
        numpy.ndarray: Renormalized image tensor.
    """
    mean = [0.2870, 0.3252, 0.2840]
    std = [0.1711, 0.1759, 0.1731]  
    renormalized_image = image * std + mean
    return renormalized_image

def visualize(images,masks,model):
    dice_tot = []
    for j in range(images.shape[0]):
        image_real = renormalize_image(images[j].numpy().squeeze().transpose(1, 2, 0))
        image = model(images[j])
        pred = postprocess(image, (1024,2048))
        pred_mask_rgb = mask_to_rgb(pred, colors)
        pred = pred.flatten()

        mask_rgb = mask_to_rgb(masks.squeeze(), colors)
        mask = masks.squeeze().flatten()

        dice = []

        for l in range(19):
            pred_tot = len(pred[pred==l])
            mask_tot = len(mask[mask==l])
            inter = len([i for i, v in enumerate(pred) if v == l and v == mask[i]])
            score = (2.*inter + 0.00001)/(pred_tot + mask_tot + 0.00001)
            dice.append(score)

        mean_score = np.sum(dice)/19
        dice_tot.append(mean_score)
        print(mean_score)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(image_real)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask_rgb)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask_rgb)
        plt.axis('off')

        plt.show()
    print(np.sum(dice_tot)/images.shape[0])

model.eval()
with torch.no_grad():
    for i, (images, masks) in enumerate(validationloader):
        if i >= 3:
            break
        masks = masks.numpy()
        images_large = preprocess(images).unsqueeze(0)
        images_small = preprocess_small(images).unsqueeze(0)
        visualize(images_large,masks,model)




