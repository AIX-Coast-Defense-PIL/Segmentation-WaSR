import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import cv2
import torchvision.transforms as T

from datasets.mastr import MaSTr1325Dataset
from datasets.transforms import PytorchHubNormalization
from wasr.inference import Predictor
import wasr.models as models
from wasr.utils import load_weights
from datasets.transforms import get_image_resize


###### yolo ######
import os
import torch.backends.cudnn as cudnn
import yaml
import sys 
sys.path.append('/home/leeyoonji/workspace/git')
sys.path.append('/home/leeyoonji/workspace/git/yolov5')

from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, colorstr,
                            non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_sync

sys.path.remove('/home/leeyoonji/workspace/git/yolov5')
sys.path.append('/home/leeyoonji/workspace/git/yolov4')
sys.path.append('/home/leeyoonji/workspace/git/yolov4/utils')
from yolov4.models.models import Darknet, load_darknet_weights

import warnings
warnings.filterwarnings(action='ignore')


# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

BATCH_SIZE = 1
MODEL = 'wasr_resnet101'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    ###### wasr ######
    parser = argparse.ArgumentParser(description="WaSR Network MaSTr1325 Inference")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to the file containing the MaSTr1325 dataset mapping.")
    parser.add_argument("--model", type=str, choices=models.model_list, default=MODEL, help="Model architecture.")
    parser.add_argument("--wasr_weights", type=str, required=True, help="Path to the model weights or a model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--fp16", action='store_true', help="Use half precision for inference.")
    
    ###### yolo ######
    parser.add_argument('--yolo_weights', nargs='+', type=str, required=True, help='model path(s)')
    parser.add_argument('--yolo_cfg', type=str, default='models/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()



def predict(args):
    # Prepare model
    wasr_model = models.get_model(args.model, pretrained=False)
    state_dict = load_weights(args.wasr_weights)
    wasr_model.load_state_dict(state_dict)
    predictor = Predictor(wasr_model, args.fp16)

    output_dir = Path(args.output_dir)
    for dname in ['unk_images']:
        os.makedirs(output_dir/dname, exist_ok=True)

    ###### yolo ######
    dataset_file = Path(args.dataset_config)
    with dataset_file.open('r') as file:
        source = (dataset_file.parent / Path(yaml.safe_load(file)['image_dir'])).resolve()

    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    if is_file: source = check_file(source)  # download

    # Load model
    device = select_device(args.device)
    yolo_model = Darknet(args.yolo_cfg, args.imgsz).cuda()
    try:
        yolo_model.load_state_dict(torch.load(args.yolo_weights[0], map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    except:
        load_darknet_weights(yolo_model, args.yolo_weights[0])
    yolo_model.to(device).eval()
    
    stride, pad, pt = 32, 0, True
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    imgsz = check_img_size(args.imgsz)  # check image size
    
    # Dataloader
    dl = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = args.batch_size  # batch_size

    # Run inference
    seen, dt = 0, [0.0, 0.0, 0.0, 0.0]
    names = {0: 'Unknown'}
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    for [path, im, im0s, _, s]  in tqdm(dl):
        # features['image'] : (1, 3, 384, 512)
        # im : (3, 288, 512)
        # im0s : (1080, 1920, 3)

        ###### yolo ######
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if args.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim # (1, 3, 288, 512)
        t2 = time_sync()
        dt[0] += t2 - t1
        transform = T.ToPILImage()
        box_img = transform(im[0,:,:,:])
        box_draw = ImageDraw.Draw(box_img)

        # Inference
        ###### wasr ######
        wasr_preds = predictor.predict_batch({'image':im}) # (1, 3, 384, 512)

        ###### yolo ######
        yolo_preds = yolo_model(im, augment=args.augment)[0]
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        yolo_preds = non_max_suppression(yolo_preds, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms, max_det=args.max_det)
        dt[2] += time_sync() - t3
        # [(1,6)]

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, (wasr_pred, yolo_pred) in enumerate(zip(wasr_preds, yolo_preds)):  # per image
            t4 = time_sync()
            ###### wasr ######
            wasr_pred = SEGMENTATION_COLORS[wasr_pred]
            obs_bin = np.uint8(np.where(wasr_pred[:,:,0]==247,1,0)) # binary (obstacle=1, other(sky,sea)=0)
            yolo_pred[:, 5] = 0
            
            ###### yolo ######
            seen += 1

            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(obs_bin.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            gn2 = torch.tensor(im.shape)[[3, 2, 3, 2]]
            if len(yolo_pred):
                # Rescale boxes from img_size to im0 size
                yolo_pred[:, :4] = scale_coords(im.shape[2:], yolo_pred[:, :4], obs_bin.shape).round()

                # Print results
                for c in yolo_pred[:, -1].unique():
                    n = (yolo_pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(yolo_pred):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        
                    ###### remove ships ######
                    yolo_box = [int(y.item()) for y in xyxy]
                    thr = thr_x = thr_y = 5
                    if yolo_box[1]<thr: thr_y = 0
                    if yolo_box[0]<thr: thr_x = 0
                    obs_bin[yolo_box[1]-thr_y : yolo_box[3]+thr, yolo_box[0]-thr_x : yolo_box[2]+thr] = 0
                    # obs_bin[yolo_box[1]:yolo_box[3], yolo_box[0]:yolo_box[2]] = 0

                
            ###### Connected Component Labeling ######
            ccl_num_labels, ccl_labels, ccl_stats, ccl_centroids = cv2.connectedComponentsWithStats(obs_bin)

            ###### Ground Removal ######
            unk_predn = [] # unknown prediction
            for x,y,w,h,a in ccl_stats:
                ccl_cond = True
                
                # if a < (obs_bin.shape[1]*0.008)**2: # remove tiny objects (noise)
                if a < (obs_bin.shape[1]*0.01)**2: # remove tiny objects (noise)
                    ccl_cond = False
            
                eps = 0.3
                if x < 5: x += 5
                if y < 5: y += 5
                obs_box = obs_bin[y-5:y+h+5, x-5:x+w+5]
                surr_pix = cv2.dilate(obs_box, np.ones((5,5), np.uint8), iterations=1) - obs_box
                sum_pix = surr_pix.sum()
                surr_pix = np.multiply(wasr_pred[y-5:y+h+5, x-5:x+w+5, 0],surr_pix.astype(int))
                if surr_pix.sum() != 0:
                    if ((surr_pix==90).sum() / sum_pix > eps) or (w > obs_bin.shape[1]*0.95): 
                        ccl_cond = False

                if ccl_cond:
                    unk_predn.append([x, y, x+w, y+h, 1, 0]) # x y x y conf cls
                    unk_box = (torch.tensor([x, y, x+w, y+h]).view(1,4) / gn * gn2).view(-1).tolist()
                    box_draw.rectangle(unk_box, outline=(255,0,0), width = 2)
        
            dt[3] += time_sync() - t4
            box_img.save(output_dir / 'unk_images' / os.path.basename(path))

        # Print time (inference-only)
        # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        # print('---')

    ###### wasr ######
    # save parameters
    with open(output_dir / 'info.txt', 'w') as f:
        for key, value in vars(args).items(): 
            f.write('%s:%s\n' % (key, value))
        
        f.write('the number of data : %d\n\n' % (len(dl)))
    
    ###### yolo ######
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms inference2 per image at shape {(1, 3, *imgsz)}' % t)
    s = f"\n{len(list(output_dir.glob('labels/*.txt')))} labels saved to {output_dir / 'labels'}"
    LOGGER.info(f"Results saved to {colorstr('bold', output_dir)}{s}")

    

def main():
    args = get_arguments()
    print(args)

    predict(args)


if __name__ == '__main__':
    main()
