from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

def read_mask(path):
    """Reads class segmentation mask from an image file."""
    mask = np.array(Image.open(path))

    # Masks stored in RGB channels or as class ids
    if mask.ndim == 3:
        mask = mask.astype(np.float32) / 255.0
    else:
        mask = np.stack([mask==0, mask==1, mask==2], axis=-1).astype(np.float32)

    return mask

class FolderDataset(torch.utils.data.Dataset):
    """Dataset wrapper for a general directory of images."""

    def __init__(self, image_dir, mask_dir=None, transform=None, normalize_t=None):
        """Creates the dataset.

        Args:
            image_dir (str): path to the image directory. Can contain arbitrary subdirectory structures.
            normalize_t (callable, optional): Transform used to normalize the images. Defaults to None.
        """

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)  if mask_dir is not None else None
        self.images = sorted([p.relative_to(image_dir) for p in Path(image_dir).glob('**/*.jpg')])

        self.transform = transform
        self.normalize_t = normalize_t

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rel_path = self.images[idx]
        img_path = self.image_dir / rel_path
        img = np.array(Image.open(str(img_path)))
        mask_filename = str(rel_path).replace('.jpg', 'm.png')
        
        data = {'image': img}

        if self.mask_dir is not None:
            mask_path = self.mask_dir / mask_filename
            mask = read_mask(mask_path)
            data['segmentation'] = mask
        
        if self.transform is not None:
            data = self.transform(data)
            img = data['image']

        if self.normalize_t is not None:
            img = self.normalize_t(img)
        else:
            # Default: divide by 255
            img = TF.to_tensor(img)


        features = {'image': img}
        labels = {}

        if 'segmentation' in data:
            labels['segmentation'] = torch.from_numpy(data['segmentation'].transpose(2,0,1))

        metadata = {
            'image_name': img_path.name,
            'image_path': str(rel_path)
        }
        labels.update(metadata)

        return features, labels
