import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import cv2

###### WaSR ######
from datasets.mastr import MaSTr1325Dataset
from datasets.transforms import PytorchHubNormalization
from wasr.inference import Predictor
import wasr.models as models
from wasr.utils import load_weights, get_box_coord
from datasets.transforms import get_image_resize


###### YOLO ######
import os
import sys 
sys.path.append('/home/leeyoonji/workspace/git')
sys.path.append('/home/leeyoonji/workspace/git/yolov5')

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.callbacks import Callbacks
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.general import (LOGGER, check_dataset, check_img_size, colorstr, 
                            non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh, xywhn2xyxy)
from yolov5.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from yolov5.utils.plots import output_to_target, plot_images
from yolov5.utils.torch_utils import select_device, time_sync


# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

BATCH_SIZE = 12
MODEL = 'wasr_resnet101_imu'


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    ####### WaSR ######
    parser = argparse.ArgumentParser(description="WaSR Network MaSTr1325 Inference")
    parser.add_argument("--wasr_dataset", type=str, required=True, help="Path to the file containing the MaSTr1325 dataset mapping.")
    parser.add_argument("--wasr_model", type=str, choices=models.model_list, default=MODEL, help="Model architecture.")
    parser.add_argument("--wasr_weights", type=str, required=True, help="Path to the model weights or a model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--fp16", action='store_true', help="Use half precision for inference.")
    
    ####### YOLO ######
    parser.add_argument('--yolo_dataset', type=str, required=True, help='dataset.yaml path')
    parser.add_argument('--yolo_weights', nargs='+', required=True, help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    
    return parser.parse_args()


def predict(args):
    batch_size = args.batch_size
    device = select_device(args.device, batch_size=batch_size)
    
    # Data
    yolo_dataset = check_dataset(args.yolo_dataset)  # check
        
    # Directories
    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir / 'images'):
        os.makedirs(output_dir / 'images')
    if not os.path.exists(output_dir / 'labels'):
        os.makedirs(output_dir / 'labels')

    
    # Load model - yolo
    yolo_model = DetectMultiBackend(args.yolo_weights, device=device, data=args.yolo_dataset, fp16=args.fp16)
    stride, pt, jit, engine = yolo_model.stride, yolo_model.pt, yolo_model.jit, yolo_model.engine
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    half = yolo_model.fp16  # FP16 supported on limited backends with CUDA
    
    if engine:
        batch_size = yolo_model.batch_size
    else:
        device = yolo_model.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

    # Configure
    yolo_model.eval()
    cuda = device.type != 'cpu'
    nc = 1 if args.single_cls else int(yolo_dataset['nc'])  # number of classes
    print(nc)
    # iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    # niou = iouv.numel()
    iouv = torch.tensor([0.5], device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    # Load model - wasr
    wasr_model = models.get_model(args.wasr_model, pretrained=False)
    state_dict = load_weights(args.wasr_weights)
    wasr_model.load_state_dict(state_dict)
    predictor = Predictor(wasr_model, args.fp16)
    

    # Dataloader - yolo
    if pt and not args.single_cls:  # check --weights are trained on --data
        ncm = yolo_model.model.nc
        assert ncm == nc, f'{args.yolo_weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                            f'classes). Pass correct combination of --weights and --data that are trained together.'
    yolo_model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    yolo_dl = create_dataloader(yolo_dataset['test'],
                                    imgsz,
                                    batch_size,
                                    stride,
                                    args.single_cls,
                                    pad=0.5,
                                    rect=pt, # square inference for benchmarks
                                    workers=args.workers,
                                    prefix=colorstr('test: '))[0]
    
    
    # Dataloader - wasr
    wasr_ds = MaSTr1325Dataset(args.wasr_dataset, normalize_t=PytorchHubNormalization(), 
                               sort=True, yolo_resize=(batch_size, imgsz, stride, 0.5, pt))
    wasr_dl = DataLoader(wasr_ds, batch_size=batch_size, num_workers=1)
    
    
    seen = 0
    plots = True
    callbacks=Callbacks()
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {0: 'Unknown'}
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    stats, ap, ap_class = [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(zip(iter(wasr_dl), yolo_dl), total=len(wasr_dl), desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', miniters=60)  # progress bar
    
    for batch_i, ((features, wasr_lb), (im, targets, paths, shapes)) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        t1 = time_sync()
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        ####### WaSR ######
        wasr_preds = predictor.predict_batch(features)

        ####### YOLO ######
        # Inference
        yolo_preds, train_out = yolo_model(im, augment=args.augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2
        
        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if args.save_hybrid else []  # for autolabelling
        t3 = time_sync()
        yolo_preds = non_max_suppression(yolo_preds, args.conf_thres, args.iou_thres, labels=lb, multi_label=True, agnostic=args.single_cls)
        dt[2] += time_sync() - t3

        # Metrics
        unk_preds, unk_labels = torch.tensor([], device=device), torch.tensor([], device=device)
        for si, (wasr_pred, yolo_pred) in enumerate(zip(wasr_preds, yolo_preds)):
            t4 = time_sync()
            ####### WaSR ######
            wasr_pred = SEGMENTATION_COLORS[wasr_pred]
            obs_bin = np.uint8(np.where(wasr_pred[:,:,0]==247,1,0)) # binary (obstacle=1, other(sky,sea)=0)
            
            ####### YOLO ######
            yolo_lb = targets[targets[:, 0] == si, 1:]
            nl, npr = yolo_lb.shape[0], yolo_pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), yolo_lb[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=yolo_lb[:, 0])
                continue
            
            # Predictions
            if args.single_cls:
                yolo_pred[:, 5] = 0
            yolo_predn = yolo_pred.clone()
            scale_coords(im[si].shape[1:], yolo_predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Remove Detected Objects
            # Save one txt result
            gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
            for *xyxy, conf, cls in yolo_predn.tolist():
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
                with open(output_dir / 'labels' / f'{path.stem}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                
                # remove ships
                yolo_box = [int(y) for y in yolo_predn[0]]
                obs_bin[yolo_box[1]:yolo_box[3], yolo_box[0]:yolo_box[2]] = 0
                # box = get_box_coord(obs_bin.shape, *xywh)
                # obs_bin[box[1]:box[3], box[0]:box[2]] = 0
                
            
            # Connected Component Labeling
            ccl_num_labels, ccl_labels, ccl_stats, ccl_centroids = cv2.connectedComponentsWithStats(obs_bin)

            # Ground Removal
            unk_predn = [] # unknown prediction
            for i, (x,y,w,h,a) in enumerate(ccl_stats):
                ccl_cond = True
                
                if a < (width*0.008)**2:
                    ccl_cond = False
            
                eps = 0.3
                if x < 5: x += 5
                if y < 5: y += 5
                obs_box = obs_bin[y-5:y+h+5, x-5:x+w+5]
                surr_pix = cv2.dilate(obs_box, np.ones((5,5), np.uint8), iterations=1) - obs_box
                sum_pix = surr_pix.sum()
                surr_pix = np.multiply(wasr_pred[y-5:y+h+5, x-5:x+w+5, 0],surr_pix.astype(int))
                if surr_pix.sum() != 0:
                    if ((surr_pix==90).sum() / sum_pix > eps) or (w > width*0.95): 
                        ccl_cond = False

                if ccl_cond:
                    unk_predn.append([x, y, x+w, y+h, 1, 0]) # x y x y conf cls
            unk_predn = torch.tensor(unk_predn, device=device)
            
            # Evaluate
            if len(unk_predn):
                tbox = xywh2xyxy(yolo_lb[:, 1:5])  # target boxes
                # scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((yolo_lb[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(unk_predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(unk_predn, labelsn)
            # stats.append((correct, yolo_pred[:, 4], yolo_pred[:, 5], yolo_lb[:, 0]))  # (correct, conf, pcls, tcls)
            stats.append((correct, torch.ones(len(correct), device=device), 
                          torch.zeros(len(correct), device=device), yolo_lb[:, 0]))  # (correct, conf, pcls, tcls)
            dt[3] += time_sync() - t4

            callbacks.run('on_val_image_end', yolo_pred, yolo_predn, path, names, im[si])
            unk_preds = torch.cat((unk_preds, unk_predn), 0)
            if len(labelsn):
                labelsn = torch.cat((torch.zeros((len(labelsn),1), device=device), labelsn), 1)
                labelsn[:, 2:] = xyxy2xywh(labelsn[:, 2:])
                unk_labels = torch.cat((unk_labels, labelsn), 0)
            else: unk_labels = labelsn

            
            ####### WaSR ######
            mask_img = Image.fromarray(wasr_pred)
            out_file = output_dir / 'images' / wasr_lb['mask_filename'][si]

            mask_img.save(out_file)

        ####### YOLO ######
        # Plot images
        if plots and batch_i < 10:
            print('label')
            print(unk_labels)
            plot_images(im, unk_labels, paths, output_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            print('pred')
            plot_images(im, output_to_target([unk_preds]), paths, output_dir / f'val_batch{batch_i}_pred.jpg', names, plot_conf=False)  # pred

        callbacks.run('on_val_batch_end')

    
    ####### WaSR ######
    with open(output_dir / 'info.txt', 'w') as f:
        for key, value in vars(args).items(): 
            f.write('%s:%s\n' % (key, value))
        
        f.write('the number of data : %d\n\n' % (len(wasr_ds)))
    
    
    ####### YOLO ######
    # Compute metrics
    stats = [torch.cat(i, 0).cpu().numpy() for i in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=output_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning('WARNING: no labels found in test set, can not compute metrics without labels')

    # Print results per class
    if (nc < 50) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(i / seen * 1E3 for i in dt)  # speeds per image
    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms inference2 per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=output_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Return results
    yolo_model.float()  # for training
    s = f"\n{len(list(output_dir.glob('labels/*.txt')))} labels saved to {output_dir / 'labels'}"
    LOGGER.info(f"Results saved to {colorstr('bold', output_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(yolo_dl)).tolist()), maps, t
    
    

def main():
    args = get_arguments()
    print(args)

    if args.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
        LOGGER.info(f'WARNING: confidence threshold {args.conf_thres} > 0.001 produces invalid results')
    
    predict(args)


if __name__ == '__main__':
    main()
