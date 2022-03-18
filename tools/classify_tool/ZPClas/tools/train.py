import os
import os.path as osp
import argparse
import time
import copy
import torch
import math

from models.model import initialize_model 
from dataset.data_loader import MyDataLoader
from basic.log import Logger

def makeDir(dirpath):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)

def train_model(datasetPath, projectPath, batch_size, dataloaderDct, datasetSizesDct,model, \
                crtiation, optimizer, schedular, model_name, \
                num_epochs=100, save_interval=8, eval_interval=5, \
                ifEval=True):
    
    makeDir(osp.join(projectPath, 'log'))
    step_nums=math.ceil(datasetSizesDct['train']/float(batch_size))

    log=Logger(osp.join(projectPath,'log',  time.strftime("%B-%e-%H-%M")+'.log'))

    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    # train_log=open(osp.join(projectPath, 'log', time.strftime("%B-%e-%H-%M")+'.txt'), 'a', encoding='utf-8')

    best_train_acc=0.0
    best_val_acc=0.0

    log.print("-*"*8 + 'Begin To Train' + "*-"*8)
    for epoch in range(num_epochs):
        begin_time=time.time()

        model.train()
        train_loss, train_acc, valid_acc  = 0.0, 0.0, 0.0

        # train_log.write()
        for step, (images, labels) in enumerate(dataloaderDct['train']):
            step_begin_time=time.time()

            images=images.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()

            results = model(images)
            _,preds = torch.max(results, 1)

            loss = crtiation(results, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            correct_nums = preds.eq(labels.view_as(preds))

            acc = torch.mean(correct_nums.type(torch.FloatTensor))

            train_acc += acc.item() * images.size(0)

            step_train_time=time.time()-step_begin_time
            if (step+1)%20==0:
                log.print('Step={}/{}, Train Time={:.4f}, Train Acc={:.4f}'.format(step, step_nums, step_train_time,acc))
        
        avg_train_loss=train_loss/datasetSizesDct['train']
        avg_train_acc=train_acc/datasetSizesDct['train']

        best_train_acc=max(best_train_acc, avg_train_acc)
        train_time=time.time()-begin_time
        log.print('Time={}, Epoch={}, Train Time={}, Loss={:.4f}, Train Acc:{:.4f}, Best Train Acc={:.4f}'.format(time.strftime("%d:%H:%M"),epoch, train_time, avg_train_loss, avg_train_acc, best_train_acc))
        

        if (epoch+1)%save_interval==0:
            makeDir(osp.join(projectPath, 'output',model_name))
            torch.save(model.state_dict(), osp.join(projectPath, 'output',model_name,'epoch_{}.pkl'.format(epoch)))

        if (epoch+1)%eval_interval==0 and ifEval:
            with torch.no_grad():
                model.eval()
                for step, (inputs, labels) in enumerate(dataloaderDct['test']):
                    if step%20==0:
                        print(step)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    outputs = model(inputs)
    
                    # loss = crtiation(outputs, labels)
                    # valid_loss += loss.item() * inputs.size(0)
    
                    _, preds = torch.max(outputs, 1)
                    correct_nums = preds.eq(labels.view_as(preds))
    
                    acc = torch.mean(correct_nums.type(torch.FloatTensor))
    
                    valid_acc += acc.item() * inputs.size(0)
                avg_valid_acc = valid_acc/datasetSizesDct['test']

                # best_val_acc=max(best_val_acc, avg_valid_acc)
                if avg_valid_acc>best_val_acc:
                    best_val_acc=avg_valid_acc
                    makeDir(osp.join(projectPath, 'output', model_name))
                    torch.save(model.state_dict(), osp.join(projectPath, 'output', model_name, 'best_model.pkl'))

                log.print('-*'*20+'-')
                log.print('-*'*20+'-')
                log.print("Epoch={}, Eval_Acc={:.4f}, Best Eval Acc={:.4f}".format(epoch, avg_valid_acc, best_val_acc))
                log.print('-*'*20+'-')
                log.print('-*'*20+'-')

    return best_train_acc, best_val_acc

