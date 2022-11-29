import os
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

from .utils import tensor_map
from .metrics import PixelAccuracy, ClassIoU


class Predictor():
    def __init__(self, model, half_precision, eval_mode=False):
        self.model = model
        self.half_precision = half_precision

        use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda:0') if use_gpu else torch.device('cpu')

        if self.half_precision:
            self.model = self.model.half()
        self.model = self.model.eval().to(self.device)
        
        if eval_mode:
            # Metrics
            num_classes = 3
            self.accuracy = PixelAccuracy(num_classes)
            self.iou_0 = ClassIoU(0, num_classes)
            self.iou_1 = ClassIoU(1, num_classes)
            self.iou_2 = ClassIoU(2, num_classes)

    def predict_batch(self, batch):

        map_fn = lambda t: t.to(self.device)
        batch = tensor_map(batch, map_fn)

        with torch.no_grad():
            if self.half_precision:
                with torch.cuda.amp.autocast():
                    res = self.model(batch)
            else:
                res = self.model(batch)

        out = res['out'].cpu().detach()

        size = (batch['image'].size(2), batch['image'].size(3))
        out = TF.resize(out, size, interpolation=Image.BILINEAR)
        out = out.numpy()

        return out
    
    def evaluate_batch(self, features, labels):

        map_fn = lambda t: t.to(self.device)
        features = tensor_map(features, map_fn)

        with torch.no_grad():
            if self.half_precision:
                with torch.cuda.amp.autocast():
                    res = self.model(features)
            else:
                res = self.model(features)

        out = res['out'].cpu().detach()
        
        # Metrics
        labels_size = (labels['segmentation'].size(2), labels['segmentation'].size(3))
        logits = TF.resize(out, labels_size, interpolation=Image.BILINEAR)
        preds = logits.argmax(1)
        
        # Create hard labels from soft
        labels_hard = labels['segmentation'].argmax(1)
        ignore_mask = labels['segmentation'].sum(1) < 0.9
        labels_hard = labels_hard * ~ignore_mask + 4 * ignore_mask

        self.accuracy(preds, labels_hard)
        self.iou_0(preds, labels_hard)
        self.iou_1(preds, labels_hard)
        self.iou_2(preds, labels_hard)

        return {'accuracy':self.accuracy.compute(), 'iou_obstacle':self.iou_0.compute(), 
                'iou_water':self.iou_1.compute(), 'iou_sky':self.iou_2.compute()}


class LitPredictor(pl.LightningModule):
    """Predicts masks and exports them. Supports multi-gpu inference."""
    def __init__(self, model, export_fn, raw=False):
        super().__init__()
        self.model = model
        self.export_fn = export_fn
        self.raw = raw

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        features, metadata = batch
        outputs = self.model(features)
        if self.raw:
            # Keep raw input and device (e.g. for mask filling)
            self.export_fn(outputs, batch)
            return

        out = outputs['out'].cpu().detach()

        # Upscale
        size = (features['image'].size(2), features['image'].size(3))
        out = TF.resize(out, size, interpolation=Image.BILINEAR)
        out = out.numpy()

        self.export_fn(out, batch)
