import numpy as np
import torch
from tqdm import tqdm
from utils import *
from PIL import Image

device = "cuda:0"

def test_epoch(config, model, test_loader) :
    '''
    Sample Dataset에 대해서 Test할때 사용하는 함수
    '''
    loss_meter = AvgMeter()
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    
    for batch in tqdm_object :
        
        images, text = batch
        
        images = images.to(device)
        text = text.to(device)

        logits_per_image, logits_per_text  = model(images, text)
        target = torch.arange(len(images), dtype = torch.long, device = device)
        total_loss = (loss_img(logits_per_image, target) + loss_txt(logits_per_text, target))/2
        
        count = images.size(0)
        loss_meter.update(total_loss.item(), count)
        
        tqdm_object.set_postfix(test_loss=loss_meter.avg)
        
    return loss_meter