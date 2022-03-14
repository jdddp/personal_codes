#using coding=utf-8
import _init_paths
from net import Network
from config import cfg, update_config
from dataset import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from core.evaluate import FusionMatrix
import json
import matplotlib.pyplot as plt
import pdb
def parse_args():
    parser = argparse.ArgumentParser(description="tricks evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=True,
        default="configs/cifar10_im100.yaml",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        help="decide which gpus to use",
        required=True,
        default='0',
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

def valid_model(dataLoader, model, cfg, device, num_classes):
    result_list = []
    pbar = tqdm(total=len(dataLoader))
    model.eval()
    top1_count, top2_count, top3_count, index, fusion_matrix = (
        [],
        [],
        [],
        0,
        FusionMatrix(num_classes),
    )

    func = torch.nn.Sigmoid() \
        if cfg.LOSS.LOSS_TYPE in ['FocalLoss', 'ClassBalanceFocal'] else \
        torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (image, image_labels, meta) in enumerate(dataLoader):
            image = image.to(device)
            output = model(image)

            result = func(output)

            _, top_k = result.topk(5, 1, True, True)
            score_result = result.cpu().numpy()

            #所有得分
            # scorelist=score_result.tolist()
            # print(scorelist)
            # break

            fusion_matrix.update(score_result.argmax(axis=1), image_labels.numpy())
            topk_result = top_k.cpu().tolist()
            if not "image_id" in meta:
                meta["image_id"] = [0] * image.shape[0]
            image_ids = meta["image_id"]
            for i, image_id in enumerate(image_ids):
                result_list.append(
                    {
                        "image_id": image_id,
                        "image_label": int(image_labels[i]),
                        "top_3": topk_result[i],
                        # "score_all":scorelist
                    }
                )
                top1_count += [topk_result[i][0] == image_labels[i]]
                top2_count += [image_labels[i] in topk_result[i][0:2]]
                top3_count += [image_labels[i] in topk_result[i][0:3]]
                index += 1
            now_acc = np.sum(top1_count) / index
            pbar.set_description("Now Top1:{:>5.2f}%".format(now_acc * 100))
            pbar.update(1)
    top1_acc = float(np.sum(top1_count) / len(top1_count))
    top2_acc = float(np.sum(top2_count) / len(top1_count))
    top3_acc = float(np.sum(top3_count) / len(top1_count))
    print(
        "Top1:{:>5.2f}%  Top2:{:>5.2f}%  Top3:{:>5.2f}%".format(
            top1_acc * 100, top2_acc * 100, top3_acc * 100
        )
    )
    pbar.close()
    fig = fusion_matrix.plot_confusion_matrix()
    plt.savefig('confusion_matrix.png')
    # with open('/home/vis/jiangzhipeng01/longTail/BagofTricks-LT/ans2.json', 'w')as f:
    #     json.dump(result_list, f)


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    num_classes = test_set.get_num_classes()
    device = torch.device("cuda")
    save_onnx = 1
    model = Network(cfg, mode="test", num_classes=num_classes, add_softmax=save_onnx)

    model_file = cfg.TEST.MODEL_FILE
    model.load_model(model_file, tau_norm=cfg.TEST.TAU_NORM.USE_TAU_NORM, tau=cfg.TEST.TAU_NORM.TAU)

    if save_onnx:
        func = torch.nn.Sigmoid() \
            if cfg.LOSS.LOSS_TYPE in ['FocalLoss', 'ClassBalanceFocal'] else \
            torch.nn.Softmax(dim=1)
        print("post process func", func)
        model = model.cuda()
        image = torch.FloatTensor(1,3,224,224).cuda()
        torch.onnx.export(model, image, "output/model.onnx",verbose=True)
        exit(0)

    model = torch.nn.DataParallel(model).cuda()
    testLoader = DataLoader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    valid_model(testLoader, model, cfg, device, num_classes)
