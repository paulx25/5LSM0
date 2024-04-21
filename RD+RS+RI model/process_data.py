import numpy as np
from torchvision import transforms
import torch

def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((64,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.2870, 0.3252, 0.2840), (0.1711, 0.1759, 0.1731))
    ])
    img = transform(img).unsqueeze(0)

    return img


def postprocess(prediction, shape):
    softmax = torch.nn.Softmax(dim=1)
    prediction_soft = softmax(prediction)
    prediction_max = torch.argmax(prediction_soft, axis=1)
    transform = transforms.Resize(shape)
    prediction = transform(prediction_max)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()

    return prediction_numpy