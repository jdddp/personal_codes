import time 
import os.path as osp
import sys
import os
import shutil
# print(time.strftime("%B-%e-%H-%M"))
# print(time.strftime("%H:%M"))

# b=1
# c=1
def makeDirs(dirpath):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)


import math
def iou(list1,list2):
    x1=max(list1[0], list2[0])
    x2=min(list1[0]+list1[2], list2[0]+list2[2])
    y1=max(list1[1], list2[1])
    x2=min(list1[1]+list1[3], list2[1]+list2[3])
    if x1>x2 or y1>y1:
        return 0
    else:
        return float((x2-x1)*(y2-y1))/(w1*h1+w2*h2)

print(osp.dirname('E:\vsCodes\personal_codes\tools\classify_tool\projects\dogClasV1\files'))
# ##################################################
# # class Logger(object):
# #     def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
# #         self.terminal = stream
# #         # self.filename = filename
# #         # self.add_flag = add_flag
# #         self.log = open(filename, 'a+')

# #     def write(self, message):
# #         self.terminal.write(message)
# #         self.log.write(message)
# #     	# if self.add_flag:
# # 	    #     with open(self.filename, 'a+') as log:
# # 	    #         self.terminal.write(message)
# # 	    #         log.write(message)
# #         # else:
# #         #     with open(self.filename, 'w') as log:
# # 	    #         self.terminal.write(message)
# # 	    #         log.write(message)

# #     def flush(self):
# #         pass


# # def main():
# #     sys.stdout = Logger("b.log", sys.stdout)
# #     # sys.stderr = Logger("a.log", sys.stderr)     # redirect std err, if necessary
# #     # now it works
# #     print('print something')
# #     print("*" * 3)
# #     # sys.stdout.write("???")
# #     import time
# #     time.sleep(10)
# #     print("other things")


# # if __name__ == '__main__':
# #     main()

# ########################################################
# # def train_and_valid(model, loss_function, optimizer, epochs=30):
# #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #     history = []
# #     best_acc = 0.0
# #     best_epoch = 0
 
# #     for epoch in range(epochs):
# #         epoch_start = time.time()
# #         print("Epoch: {}/{}".format(epoch+1, epochs))
 
# #         model.train()
 
# #         train_loss = 0.0
# #         train_acc = 0.0
# #         valid_loss = 0.0
# #         valid_acc = 0.0
 
# #         for i, (inputs, labels) in enumerate(train_loader):
# #             inputs = inputs.to(device)
# #             labels = labels.to(device)
 
# #             #因为这里梯度是累加的，所以每次记得清零
# #             optimizer.zero_grad()
 
# #             outputs = model(inputs)
 
# #             loss = loss_function(outputs, labels)
 
# #             loss.backward()
 
# #             optimizer.step()
 
# #             train_loss += loss.item() * inputs.size(0)
 
# #             ret, predictions = torch.max(outputs.data, 1)
# #             correct_nums = predictions.eq(labels.data.view_as(predictions))
 
# #             acc = torch.mean(correct_nums.type(torch.FloatTensor))
 
# #             train_acc += acc.item() * inputs.size(0)
 
#         with torch.no_grad():
#             model.eval()
 
#             for j, (inputs, labels) in enumerate(test_loader):
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
 
#                 outputs = model(inputs)
 
#                 loss = loss_function(outputs, labels)
 
#                 valid_loss += loss.item() * inputs.size(0)
 
#                 ret, predictions = torch.max(outputs.data, 1)
#                 correct_nums = predictions.eq(labels.data.view_as(predictions))
 
#                 acc = torch.mean(correct_nums.type(torch.FloatTensor))
 
#                 valid_acc += acc.item() * inputs.size(0)
 
#         avg_train_loss = train_loss/train_data_size
#         avg_train_acc = train_acc/train_data_size
 
#         avg_valid_loss = valid_loss/valid_data_size
#         avg_valid_acc = valid_acc/valid_data_size
#         #将每一轮的损失值和准确率记录下来
#         history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
 
#         if best_acc < avg_valid_acc:
#             best_acc = avg_valid_acc
#             best_epoch = epoch + 1
 
#         epoch_end = time.time()
#         #打印每一轮的损失值和准确率，效果最佳的验证集准确率
#         print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
#             epoch+1, avg_valid_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
#         ))
#         print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
