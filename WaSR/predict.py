import argparse
import os
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
from wasr.utils import load_weights
from datasets.transforms import get_image_resize
import json


# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

BATCH_SIZE = 12
MODEL = 'wasr_resnet101'
MODE = 'pred'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR Network MaSTr1325 Inference")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to the file containing the MaSTr1325 dataset mapping.")
    parser.add_argument("--model", type=str, choices=models.model_list, default=MODEL, help="Model architecture.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights or a model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--fp16", action='store_true', help="Use half precision for inference.")
    parser.add_argument("--mode", type=str, default=MODE, help="Inference mode (pred: qualitative analysis, eval: qualitative & quantitative analysis)")
    return parser.parse_args()


def predict(args):
    
    transform = get_image_resize() if ('seaships' in args.dataset_config) or ('google' in args.dataset_config) else None
    
    dataset = MaSTr1325Dataset(args.dataset_config, transform=transform,
                               normalize_t=PytorchHubNormalization())
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)

    # Prepare model
    model = models.get_model(args.model, pretrained=False)
    state_dict = load_weights(args.weights)
    model.load_state_dict(state_dict)
    predictor = Predictor(model, args.fp16, eval_mode=True) if args.mode=='eval' else Predictor(model, args.fp16)

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir / 'images'):
        os.makedirs(output_dir / 'images')

    for features, labels in tqdm(iter(dl), total=len(dl)):
        pred_masks = predictor.predict_batch(features)

        for i, pred_mask in enumerate(pred_masks):
            pred_mask = SEGMENTATION_COLORS[pred_mask]
            mask_img = Image.fromarray(pred_mask)

            out_file = output_dir / 'images' / labels['mask_filename'][i]

            mask_img.save(out_file)
            
        if args.mode=='eval':
            eval_results = predictor.evaluate_batch(features, labels)
    
    with open(output_dir / 'result.txt', 'w') as f:
        for key, value in vars(args).items(): 
            f.write('%s:%s\n' % (key, value))
        
        f.write('the number of data : %d\n\n' % (len(dataset)))
    
        if args.mode=='eval':
            for key, value in eval_results.items(): 
                f.write('%s:%s\n' % (key, round(value.item(),4)))
            f.write('mean_iou:%s\n' % (np.mean([eval_results['iou_obstacle'], 
                                                eval_results['iou_water'], eval_results['iou_sky']])))


def main():
    args = get_arguments()
    print(args)

    predict(args)


if __name__ == '__main__':
    main()
