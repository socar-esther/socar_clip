import timm
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from utils import *

class ImageEncoder(nn.Module) :
    '''
    config에서 정한 모델을 Encoder로 사용
    - options: Resnet계열 모델들, timm 적용가능한 모든 vision model
    '''
    def __init__(self, config):
        super().__init__()
        self.model_name = config['model']['model_name'] 
        self.pretrained = config['model']['pretrained'] 
        self.trainable = config['model']['trainable'] 
        
        # Resnet이 아닌 다른 Encoder 사용하는 경우, 변형없이 사용 (ex. ViT)
        self.model = timm.create_model(self.model_name, self.pretrained, num_classes=0, global_pool="avg")
        
        if 'resnet' in self.model_name and config['model']['model_modify']==True:
            # Resnet은 변형버전 사용
            state_dict = self.model.state_dict() or state_dict
            counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            
            vision_width = state_dict["layer1.0.conv1.weight"].shape[0]
            vision_patch_size = None
            
            input_resolution = config["model"]["image_size"]
            att_pos_embedding_shape = (config['model']['image_size']//32) ** 2 + 1
            output_width = round((att_pos_embedding_shape - 1) ** 0.5) 
            
            assert output_width ** 2 + 1 == att_pos_embedding_shape
            
            #image_resolution = output_width * 32
            vision_heads = vision_width * 32 // 64
            
            embed_dim = config['model']['image_embedding']#.shape[1]
            
            self.model = ModifiedResnet(layers=vision_layers,
                                       output_dim=embed_dim,
                                       heads=vision_heads,
                                       input_resolution=input_resolution,
                                       width=vision_width)
    
        print(f'## Check Image Encoder: {self.model_name}')
        #print(self.model)
        
        for p in self.model.parameters():
            p.requires_grad = self.trainable
    
    def forward(self, x) :
        return self.model(x)
    

class ModifiedResnet(nn.Module) :
    '''
    기존 Resnet을 변형하여 사용
    (1) 3개의 conv layer 추가 (w. average pooling)
    (2) anti-aliasing strided convolutions
    (3) 마지막 pooling layer를 attention pooling으로 변경
    '''
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64) :
        super().__init__()
        
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        
        # (1) 3 conv layer 추가 (w. average pooling) 
        self.conv1 = nn.Conv2d(3, width//2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width//2)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(width//2, width//2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width//2)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(width//2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True) 
        
        self.avgpool = nn.AvgPool2d(2)
        
        # (2) anti-aliasing strided convolutions (downsample 했을때 발생하는 aliasisng 해결) 
        self._inplanes = width  
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        
        # (3) attention pooling layer추가
        embed_dim = width * 32 # resnet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        
    
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x
    
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out
    
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)