import os
import os.path as osp
import shutil
import torch
import torch.nn as nn
import numpy as np

from dataset.dataset import default_loader

def infer(dataloader, model):
    device=torch.device('cuda:0')
    with torch.no_grad():
        model.eval()
        for step, inputs in enumerate(dataloader):
            if (step+1)%20==0:
                print(step)
            # print(inputs)
            inputs = inputs.to(device)
            # labels = labels.to(device)

            outputs = model(inputs)
            #计算属于哪个类
            print(torch.max(outputs,1)[1])

            # outputs = nn.Softmax(outputs)
            outputs = outputs.cpu().numpy().tolist()
            for i in outputs:
                print(np.argmax(i))
            # print(outputs)

            
            break
    
    