import os 
import numpy as np


class AvgMeter:
    '''
    여러 metric의 avg 작업 하는 부분
    '''
    def __init__(self, name='Metric') :
        self.name = name
        self.reset()
        
    def reset(self):
        self.avg, self.sum, self.count = [0]*3
        
    def update(self, val, count=1):
        self.count += count
        self.sum += val*count
        self.avg = self.sum / self.count
        
    def __repr__(self) :
        text - f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer) :
    '''
    업데이트 되고 있는 lr의 값을 그대로 가져와서 return
    '''
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
def accuracy(output, target, topk=(1,)):
    '''
    Inference시, zero shot classifier에서 사용할 Metric: Accuracy
    '''
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def convert_models_to_clip(model) :
    '''
    model params, grads를 fp32로 바꿔주는 부분
    attribute가 nan/inf로 바뀌는 부분에 대한 에러 해결: https://github.com/openai/CLIP/issues/57
    '''
    for p in model.parameters() :
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()