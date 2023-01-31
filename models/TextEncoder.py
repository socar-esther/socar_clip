import timm
import torch
import torch.nn as nn
import transformers
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

class TextEncoder(nn.Module) :
    '''
    config에서 정한 모델을 Encoder로 사용
    - option: DistilBert
    '''
    def __init__(self, config):
        super().__init__()
        self.model_name = config['model']['text_encoder_model'] 
        self.pretrained = config['model']['pretrained'] 
        self.trainable = config['model']['trainable'] 
        
        if self.pretrained:
            self.model = DistilBertModel.from_pretrained(self.model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
    
        print(f'## Check Text Encoder: {self.model_name}')
        # print(self.model)
        
        for p in self.model.parameters():
            p.requires_grad = self.trainable
        
        # 여기선 Classification으로 사용하지 않으므로 CLS Token은 제외
        self.target_token_idx = 0
    
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]