import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.mastr import MaSTr1325Dataset
from datasets.transforms import PytorchHubNormalization
from wasr.inference import Predictor
import wasr.models as models
from wasr.utils import load_weights, get_box_coord
from datasets.transforms import get_image_resize


###### yolo ######
import torch.backends.cudnn as cudnn
import yaml
import sys 
sys.path.append('/home/leeyoonji/workspace/git')
sys.path.append('/home/leeyoonji/workspace/git/yolov5')

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, colorstr,
                            non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_sync




# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

BATCH_SIZE = 1
MODEL = 'wasr_resnet101_imu'


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
    
    ###### wasr ######
    if ('seaships' in args.dataset_config) or ('google' in args.dataset_config):
        transform = get_image_resize()  
    else:
        transform = None
    
    wasr_ds = MaSTr1325Dataset(args.dataset_config, transform=transform,
                               normalize_t=PytorchHubNormalization(), sort=True)
    wasr_dl = DataLoader(wasr_ds, batch_size=args.batch_size, num_workers=1)

    # Prepare model
    wasr_model = models.get_model(args.model, pretrained=False)
    state_dict = load_weights(args.wasr_weights)
    wasr_model.load_state_dict(state_dict)
    predictor = Predictor(wasr_model, args.fp16)

    output_dir = Path(args.output_dir)
    label_dir = output_dir / 'labels'
    if not label_dir.exists():
        label_dir.mkdir(parents=True)

    ###### yolo ######
    dataset_file = Path(args.dataset_config)
    with dataset_file.open('r') as file:
        source = (dataset_file.parent / Path(yaml.safe_load(file)['image_dir'])).resolve()

    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    device = select_device(args.device)
    yolo_model = DetectMultiBackend(args.yolo_weights, device=device, fp16=args.fp16)
    stride, names, pt = yolo_model.stride, yolo_model.names, yolo_model.pt
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        yolo_ds = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(yolo_ds)  # batch_size
    else:
        yolo_ds = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = args.batch_size  # batch_size


    # Run inference
    yolo_model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, dt = 0, [0.0, 0.0, 0.0]
    for [features, labels], [path, im, im0s, _, s]  in tqdm(zip(iter(wasr_dl), yolo_ds), total=len(wasr_dl)):
        
        ###### wasr ######
        pred_masks = predictor.predict_batch(features)

        ###### yolo ######
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if yolo_model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = yolo_model(im, augment=args.augment)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms, max_det=args.max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, (det, pred_mask) in enumerate(zip(pred, pred_masks)):  # per image
            ###### wasr ######
            pred_mask = SEGMENTATION_COLORS[pred_mask]
            
            ###### yolo ######
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), yolo_ds.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(yolo_ds, 'frame', 0)

            p = Path(p)  # to Path
            txt_path = str(label_dir / p.stem) + ('' if yolo_ds.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
        
                    ###### remove ships ######
                    box = get_box_coord(pred_mask.shape, *xywh)
                    pred_mask[box[1]:box[3], box[0]:box[2],:] = [41, 167, 224]
            
            mask_img = Image.fromarray(pred_mask)

            out_file = output_dir / labels['mask_filename'][i]
            mask_img.save(out_file)

        

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    ###### wasr ######
    with open(output_dir / 'info.txt', 'w') as f:
        for key, value in vars(args).items(): 
            f.write('%s:%s\n' % (key, value))
        
        f.write('the number of data : %d\n\n' % (len(wasr_ds)))
    
    
    ###### yolo ######
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    s = f"\n{len(list(output_dir.glob('labels/*.txt')))} labels saved to {output_dir / 'labels'}"
    LOGGER.info(f"Results saved to {colorstr('bold', output_dir)}{s}")

    

def main():
    args = get_arguments()
    print(args)

    predict(args)


if __name__ == '__main__':
    main()
