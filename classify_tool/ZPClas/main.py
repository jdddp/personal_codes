import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json

from models.model import initialize_model
from dataset.data_loader import MyDataLoader, InferDataLoader
from tools.train import train_model
from basic.basicTools import get_img_list
from tools.infer import infer

def main(config):
    assert(torch.cuda.is_available())
    device = torch.device('cuda:0')

    #gen dataloader
    # dataloaderDct, datasetSizesDct=MyDataLoader(config.datasetPath, config.projectPath, config.batch_size)

    feature_learning = (config.mode=='train')
    if config.mode =='train':
        dataloaderDct, datasetSizesDct=MyDataLoader(config.datasetPath, config.projectPath, config.batch_size)
        model_ft, input_size = initialize_model(config.model_name, config.num_classes, feature_learning, use_pretrained=True)

        model_ft = model_ft.to(device)
        model_ft.to(device)
        #可分别分装函数，切换不同工具
        criterion=nn.CrossEntropyLoss()
        optimizer_ft=optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
        best_train_acc,best_val_acc = train_model(config.datasetPath,config.projectPath,config.batch_size,dataloaderDct,datasetSizesDct,model_ft, criterion, optimizer_ft, exp_lr_scheduler,config.model_name, 10,1,1)
    else:
        img_list=get_img_list(config.datasetPath)
        dataloader=InferDataLoader(img_list, config.batch_size)
        model_ft, input_size = initialize_model(config.model_name, config.num_classes, feature_learning, use_pretrained=False)
        # print(config.projectPath)
        pretrained_model=osp.join(config.projectPath, 'output',config.model_name, 'best_model.pkl')
        model_ft.load_state_dict(torch.load(pretrained_model))
        model_ft.to(device)
        label2id=json.loads(open(osp.join(config.projectPath,'files','label2id.json')).read())
        label_list=label2id.keys()
        infer(dataloader, model_ft,label_list, config.ans_nums, config.projectPath)

    # model_ft = model_ft.to(device)
    # model_ft.to(device)

    # #可分别分装函数，切换不同工具
    # criterion=nn.CrossEntropyLoss()
    # optimizer_ft=optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    # if config.mode=='train':
    #     best_train_acc,best_val_acc = train_model(config.datasetPath,config.projectPath,config.batch_size,dataloaderDct,datasetSizesDct,model_ft, criterion, optimizer_ft, exp_lr_scheduler, 10,1,1)
    # else:
    #     infer(config.datasetPath, config.projectPath, model_ft)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-m","--model_name", type=str, required=True, help="model_name for classify job")
    parser.add_argument("-p","--projectPath", type=str, help="dirname of project")
    parser.add_argument("-b","--batch_size", type=int, help="batch_size")
    parser.add_argument("-n","--num_classes", required=True, type=int, help="train or test")
    parser.add_argument("-i","--datasetPath", type=str, help="path of dataset")
    parser.add_argument("-t","--mode", type=str, required=True, help="path of dataset")

    parser.add_argument("-w","--pretrained_weight", type=str, help="path of weight")
    parser.add_argument("-a","--ans_nums", type=int,default=3, help="path of weight")


    config=parser.parse_args()

    main(config)

