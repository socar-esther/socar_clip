import torch
import torch.nn as nn
from models.ImageEncoder import *
from models.TextEncoder import *
import torch.nn.functional as F


class ProjectionHead(nn.Module) :
    '''
    Image, Text Encoding된 feature를 Linear Projection 
    '''
    def __init__(self, embedding_dim, config) :
        super().__init__()
        
        projection_dim = config['model']['projection_dim']
        dropout = config['model']['dropout']
        
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x) :
        #print(x.shape)
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        
        return x
    

class CLIP(nn.Module) :
    
    def __init__(self, config) :
        super().__init__()
        
        temperature = config['model']['temperature']
        image_embedding = config['model']['image_embedding'] 
        text_embedding = config['model']['text_embedding']
        
        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, config=config)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding, config=config)
        self.temperature = temperature
        
    def forward(self, batch) :
        
        # step1. Image encoder / Text encoder output feature 가져온다
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask = batch["attention_mask"]
        )
        
        # step2. Projection 통과한 Image, Text embedding을 동일한 dim으로 가져온다
        image_embeddins = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        
        # step3. Loss의 계산 (InfoNCE)
        logits = (text_embeddings @ image_embeddins.T) / self.temperature ## 행렬곱
        images_similarity = image_embeddins @ image_embeddins.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        text_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        
        loss = (images_loss+text_loss) / 2.0 ## shape: (batch_size)
        
        return loss.mean()
        

def cross_entropy(preds, targets, reduction='none') :
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets*log_softmax(preds)).sum(1)
    
    if reduction == 'none' :
        return loss
    elif reduction == 'mean' :
        return loss.mean()