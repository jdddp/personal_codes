import torch
import os.path as osp

from dataset.dataset import MyDataset, InferDataset
from torchvision import transforms


def MyDataLoader(datasetPath, projectPath, weight_sample='',batch_size=64):

    data_tranforms={
        'train':transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) 
        ]),
        'test':transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    }

    image_dataset_dct = {x : MyDataset(datasetPath, 
                        txtPath=osp.join(projectPath, 'files','{}.txt'.format(x)), 
                        data_transforms=data_tranforms, 
                        mode=x,
                        weight_sample=weight_sample) for x in ['train', 'test']
                    }           

    dataloader_dct = {x : torch.utils.data.DataLoader(image_dataset_dct[x],
                    batch_size=batch_size,
                    shuffle=True) for x in ['train', 'test']
    }
    
    dataset_sizes_dct = {x: len(image_dataset_dct[x]) for x in ['train', 'test']}

    return dataloader_dct, dataset_sizes_dct

def InferDataLoader(img_list, batch_size=10):

    data_tranforms=transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    image_dataset = InferDataset(img_list=img_list, 
                                 data_transforms=data_tranforms)        

    dataloader = torch.utils.data.DataLoader(image_dataset,
                    batch_size=batch_size,
                    shuffle=False)

    return dataloader