import os
import pandas as pd
import numpy as np

import torch
import cv2
import gc
from PIL import Image

from torch import nn
import torch.nn.functional as F
import timm
import torchvision
from torchvision import transforms
import albumentations as album

device = "cuda:0"

class CLIPDataset(torch.utils.data.Dataset) :
    '''
    Image-cation pair을 item으로 하나씩 가져올 수 있는
    Dataset class 작성
    '''
    
    def __init__(self, config, image_filenames, captions, tokenizer, transforms) :
        self.config = config
        self.image_filenames = image_filenames
        self.captions = list(captions)
        
        # Input으로 들어온 caption을 input으로 들어갈 수 형태로 만듦
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True,
            max_length=config['model']['max_length'])
        self.transforms = transforms
        
    def __getitem__(self, idx) :
        item = {
            key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()
        }
        image = cv2.imread(f'{self.config["model"]["image_path"]}/{self.image_filenames[idx]}')
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2,0,1).float()
        item['caption'] = self.captions[idx]
        
        return item
    
    def __len__(self) :
        return len(self.captions)

    
def get_transforms(config, mode= 'train'):
    if mode == 'train':
        return album.Compose([
            album.Resize(config['model']['image_size'], config['model']['image_size'], always_apply=True),
            album.Normalize(max_pixel_value=255.0, always_apply=True)
        ])
    else:
        # test set or validation set에 적용
        return album.Compose([
            album.Resize(config['model']['image_size'], config['model']['image_size'],always_apply=True),
            album.Normalize(max_pixel_value=255.0, always_apply=True)
        
        ])
        
        
def build_loaders(config, dataframe, tokenizer, mode) :
    '''
    DataLoader를 output으로 뱉는 func
    '''
    transforms = get_transforms(config, mode = mode)
    dataset = CLIPDataset(
        config, 
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms = transforms
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = config["model"]["batch_size"],
        num_workers = config["model"]["num_workers"],
        shuffle=True if mode=='train' else False
    )
    return dataloader


class image_caption_dataset(torch.utils.data.Dataset):
    def __init__(self, list_image_path, list_txt, clip, preprocess):
        self.image_path = list_image_path
        self.caption  = clip.tokenize(list_txt)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_path[idx])) 
        caption = self.caption[idx]
        return image, caption
    
def get_sample_dataloader(args, clip, preprocess, train_df, test_df, image_path) :
    # torch dataset 형태로 정비
    train_image_tmp = train_df['image_name'].values.tolist()
    list_image_path = list()
    for file_nm in train_image_tmp :
        file_nm_tmp = os.path.join(image_path, file_nm)
        list_image_path.append(file_nm_tmp)

    list_txt = train_df['caption_text'].values.tolist()
    train_dataset = image_caption_dataset(list_image_path,list_txt, clip, preprocess)
    
    test_image_tmp = test_df['image_name'].values.tolist()
    list_image_path = list()
    for file_nm in test_image_tmp :
        file_nm_tmp = os.path.join(image_path, file_nm)
        list_image_path.append(file_nm_tmp)

    list_txt = test_df['caption_text'].values.tolist()
    test_dataset = image_caption_dataset(list_image_path,list_txt, clip, preprocess)
    
    # dataloader 정리
    mode = 'train'
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size = args.batch_size,
                                                  num_workers = args.num_workers,
                                                  shuffle=True if mode == 'train' else False)

    ## Load Test DataLoader
    mode = 'test'
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size = args.batch_size,
                                                  num_workers = args.num_workers,
                                                  shuffle=True if mode == 'train' else False)
    
    return train_loader, test_loader


def convert_txt_to_csv(data_path, save_path) :
    '''
    Txt 형태로 된 Multi-modal input을 csv 형태로 바꾼다
    '''
    df = pd.read_csv(data_path, sep = '|')
    df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
    df.to_csv(save_path)
    return df
    

def split_train_test(df) :
    '''
    df를 받아서 train, test df로 split하는 부분
    '''
    max_id = df["id"].max() + 1 
    image_ids = np.arange(0, max_id)

    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_df = df[df["id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["id"].isin(valid_ids)].reset_index(drop=True)
    return train_df, test_df