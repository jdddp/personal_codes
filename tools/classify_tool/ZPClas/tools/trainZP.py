import os
import os.path as osp
import argparse
import time
import copy
import torch

from models.model import initialize_model 
from dataset.data_loader import MyDataLoader

def makeDir(dirpath):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)

def train_model(datasetPath, projectPath, dataloaderDct, datasetSizesDct,model, \
                crtiation, optimizer, schedular, \
                num_epochs=100, save_interval=8, eval_interval=5, \
                ifEval=True):

    device = torch.device('cuda:0')

    makeDir(osp.join(projectPath, 'log'))
    train_log=open(osp.join(projectPath, 'log', time.strftime("%B-%e-%H-%M")+'.txt'), 'a', encoding='utf-8')
    best_acc=0
    # arr_acc=[]

    #可用于绘图，loss也可以有一个
    # item_acc = []

    print("-*"*8 + 'begin to train' + "-*"*8)
    for epoch in range(num_epochs):
        begin_time=time.time()

        model.train()
        train_loss, train_acc, valid_loss, valid_acc  = 0.0, 0.0, 0.0, 0.0

        # train_log.write()
        for step, (images, labels) in enumerate(dataloaderDct[mode]):
            images.to(device)
            labels.to(device)
            optimizer.zero_grad()

            results = model(images.cuda())
            loss = crtiation(results, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            print(results)
            print(results.data)
            break
            _,preds = torch.max(results.data, 1)

            correct_nums = predictions.eq(labels.data.view_as(preds))
            acc = torch.mean(correct_nums.type(torch.FloatTensor))




 
            running_loss += loss.item()*images.size(0)
            running_acc += torch.sum(pred==labels)

        epoch_loss = running_loss/datasetSizesDct[mode]
        epoch_acc = running_acc.double()/datasetSizesDct[mode]

        train_time=time.time()-begin_time

        print('time={}, epoch={}, train_time={}, mode={}, Loss={:.4f}, ACC:{:.4f}'.format(time.strftime("%H:%M"), epoch, train_time, mode, epoch_loss, epoch_acc))
        train_log.write('time={}, epoch={}, train_time{}, mode={}, Loss={:.4f}, ACC:{:.4f}'.format(time.strftime("%H:%M"), epoch, train_time, mode, epoch_loss, epoch_acc))

        # item_acc.append(epoch_acc)

        if mode == 'test' and epoch_acc>best_acc:
            # Upgrade the weights
            best_acc=epoch_acc
            makeDir(osp.join(projectPath, 'output'))
            torch.save(model.state_dict(), osp.join(projectPath, 'output', 'best_model.pkl'))
        schedular.step()
        
        if epoch%save_interval==0:
            torch.save(model.state_dict(), 'epoch_{}.pkl'.format(i))
        # model.load_state_dict()
            

        # arr_acc.append(item_acc)
    print('Best Val ACC: {}'.format(best_acc))
    train_log.write('Best Val ACC: {}'.format(best_acc))
    # model.load_state_dict(best_weights) # 保存最好的参数
    return best_acc

