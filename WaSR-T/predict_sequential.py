import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

from wasr_t.data.folder import FolderDataset
from wasr_t.data.transforms import PytorchHubNormalization, get_image_resize
from wasr_t.inference import Predictor
from wasr_t.wasr_t import wasr_temporal_resnet101
from wasr_t.utils import load_weights

# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

OUTPUT_DIR = 'output/predictions'
HIST_LEN = 5
MODE = 'pred'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR Network Sequential Inference")
    parser.add_argument("--sequence-dir", type=str, required=False,
                        help="Path to the directory containing frames of the input sequence.")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Path to the directory of masks.")
    parser.add_argument("--hist-len", default=HIST_LEN, type=int,
                        help="Number of past frames to be considered in addition to the target frame (context length). Must match the value used in training.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Model weights file.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory where the predictions will be stored.")
    parser.add_argument("--fp16", action='store_true',
                        help="Use half precision for inference.")
    parser.add_argument("--gpus", default=-1,
                        help="Number of gpus (or GPU ids) used for training.")
    parser.add_argument("--mode", type=str, default=MODE, 
                        help="Inference mode (pred: qualitative analysis, eval: qualitative & quantitative analysis)")
    return parser.parse_args()

def export_predictions(probs, batch, output_dir):
    features, metadata = batch

    # Class prediction
    out_class = probs.argmax(1).astype(np.uint8)

    for i, pred_mask in enumerate(out_class):
        pred_mask = SEGMENTATION_COLORS[pred_mask]
        mask_img = Image.fromarray(pred_mask)

        out_path = output_dir / 'images' / Path(metadata['image_path'][i]).with_suffix('.png')
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)

        mask_img.save(str(out_path))

def predict_sequence(predictor, args, sequence_dir, output_dir, mask_dir=None, transform=None):
    """Runs inference on a sequence of images. The frames are processed sequentially (stateful). The state is cleared at the start of the sequence."""
    predictor.model.clear_state()

    dataset = FolderDataset(sequence_dir, mask_dir, transform=transform, normalize_t=PytorchHubNormalization())
    dl = DataLoader(dataset, batch_size=1, num_workers=1) # NOTE: Batch size must be 1 in sequential mode.

    for batch in tqdm(dl, desc='Processing frames'):
        features, metadata = batch
        probs = predictor.predict_batch(features)
        export_predictions(probs, batch, output_dir=output_dir)

        if args.mode=='eval':
            eval_results = predictor.evaluate_batch(features, metadata)

    with open(output_dir / 'result.txt', 'w') as f:
        for key, value in vars(args).items(): 
            f.write('%s:%s\n' % (key, value))
        
        f.write('the number of data : %d\n\n' % (len(dataset)))
    
        if args.mode=='eval':
            for key, value in eval_results.items(): 
                f.write('%s:%s\n' % (key, round(value.item(),4)))
            f.write('mean_iou:%s\n' % (np.mean([eval_results['iou_obstacle'], 
                                                eval_results['iou_water'], eval_results['iou_sky']])))

def run_inference(args):
    model = wasr_temporal_resnet101(pretrained=False, hist_len=args.hist_len)
    state_dict = load_weights(args.weights)
    model.load_state_dict(state_dict)
    model = model.sequential() # Enable sequential mode
    # model = model.unrolled()

    predictor = Predictor(model, half_precision=args.fp16, eval_mode=True) if args.mode=='eval' else Predictor(model, half_precision=args.fp16)
    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir / 'images'):
        os.makedirs(output_dir / 'images')

    transform = get_image_resize() if 'seaships' in args.sequence_dir.lower() else None
    
    predict_sequence(predictor, args, args.sequence_dir, output_dir, args.mask_dir, transform)

def main():
    args = get_arguments()
    print(args)

    run_inference(args)


if __name__ == '__main__':
    main()
