import argparse

import os
import clip

import wandb
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
import json

from PIL import Image
from utils import *
from dataset import *
from train_utils import *
from test_utils import *

print(torch.__version__)

device = "cuda:0"
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='8k')
parser.add_argument('--save_path', default='result')
parser.add_argument('--image_path', default='./dataset/images')
parser.add_argument('--caption_path', default='./dataset')
parser.add_argument('--name', default='ModifiedResNet-Transformer')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--head_lr', default=1e-5, type=float)
parser.add_argument('--image_encoder_lr', default=1e-4, type=float)
parser.add_argument('--text_encoder_lr', default=1e-5, type=float)
parser.add_argument('--weight_decay', default=1e-3, type=float)
parser.add_argument('--patience', default=1, type=int)
parser.add_argument('--factor', default=0.8, type=float)
parser.add_argument('--epochs', default=200, type=int)

args = parser.parse_args()


def load_pretrained_clip(encoder_nm="RN50") :
    '''
    Pretrained CLIP weight load 하는 부분
    - encoder_nm: Image_Encoder_nm (ex. RN50, ViT)
    - Image Encoder는 Transformer 고정
    '''
    model, preprocess = clip.load("RN50", device=device, jit=False) 
    return model, preprocess


def main() :
    model, preprocess = load_pretrained_clip()
    df = convert_txt_to_csv('./dataset/captions.txt', './dataset/captions.csv')
    train_df, test_df = split_train_test(df)
    
    # get trainloader, testloader
    train_loader, test_loader = get_sample_dataloader(args, clip, preprocess, train_df, test_df, './dataset/images')
    
    # set parameters
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience, factor=args.factor)
    step = 'epoch'
    
    # finetune clip
    best_loss = float('inf')

    #wandb.init(project="clip-finetune-socar", name=args.name)
    for epoch in range(args.epochs):
        print(f"# Epoch: {epoch + 1}")

        model.train()
        train_loss = train_epoch(args, clip, model, train_loader, optimizer, lr_scheduler, step, loss_img, loss_txt)


        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(args, model, test_loader)

        #wandb.log({'loss/train': train_loss,'loss/test': test_loss})

        ## best loss 기준으로 weight 저장
        if test_loss.avg < best_loss :
            best_loss = test_loss.avg
            torch.save(model.state_dict(), f"./{args.save_path}/best_model.pth")
            print('Save best Model !')

        lr_scheduler.step(test_loss.avg)
    

    
if __name__ == '__main__' :
    main()