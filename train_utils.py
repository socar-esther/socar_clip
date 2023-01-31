import numpy as np
import torch
from tqdm import tqdm
from utils import *
from PIL import Image

device = "cuda:0"

def train_epoch(config, clip, model, train_loader, optimizer, lr_scheduler, step, loss_img, loss_txt) :
    '''
    Sample Dataset에 대해서 Train할떄 사용하는 함수
    '''
    
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total = len(train_loader))
    
    for batch in tqdm_object :
        optimizer.zero_grad()
        
        images, text = batch
        
        images = images.to(device)
        text = text.to(device)
        
        logits_per_image, logits_per_text = model(images, text)
        
        target = torch.arange(len(images), dtype = torch.long, device = device)
        
        total_loss = (loss_img(logits_per_image, target) + loss_txt(logits_per_text, target))/2
        total_loss.backward()
        
        convert_models_to_clip(model)
        optimizer.step()
        clip.model.convert_weights(model)
        
        if step == 'batch' :
            lr_scheduler.step()
        
        count = images.size(0)
        loss_meter.update(total_loss.item(), count)
        
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter