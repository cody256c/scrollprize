#!/bin/python3
import albumentations as A
import cv2
import numpy as np
import os
import pytorch_lightning as pl
import random
import scipy.stats as st
import segmentation_models_pytorch as smp
import sys
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from PIL import Image
from albumentations.pytorch import ToTensorV2
from i3dall import InceptionI3d

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def read_image_mask(fragment_id, dataset_path, label_path, start_idx, end_idx):

    images = []
    
    for i in range(start_idx, end_idx):
        
        image = cv2.imread(f'{dataset_path}/{fragment_id}/layers/{i:02}.tif', 0)
        images.append(image)
    
    images = np.stack(images, axis=2)
    
    fragment_mask = cv2.imread(f'{dataset_path}/{fragment_id}/{fragment_id}_mask.png', 0)
    
    mask = None
    if label_path is not None:
        mask = cv2.imread(f'{label_path}/{fragment_id}.png', 0)
        mask = mask.astype(np.float32)
        mask /= 255
        # TODO: Youssef's label images are larger due to padding
        y, x = fragment_mask.shape
        mask = mask[:y, :x]
    
    return images, fragment_mask, mask


def get_train_valid_dataset(dir_data, label_path, valid_id, tile_size, stride, start_idx, end_idx):
    
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []
    
    segments = [
        '20230522181603',
        #'20230702185752', # this has a new name _super...
        '20230827161847',
        '20230904135535',
        '20230905134255',
        '20230909121925',
    ]
    
    for fragment_id in segments:
        
        print('reading ', fragment_id)
        
        image, fragment_mask, mask = read_image_mask(
            fragment_id, dir_data, label_path, start_idx, end_idx)
        
        for y1 in range(0, (image.shape[0]-tile_size-1), stride):
            for x1 in range(0, (image.shape[1]-tile_size-1), stride):
                
                y2 = y1 + tile_size
                x2 = x1 + tile_size
                
                if fragment_id != valid_id:
                    if np.any(mask[y1:y2, x1:x2] > 0.0):
                        train_images.append(image[y1:y2, x1:x2, :])
                        train_masks.append(mask[y1:y2, x1:x2, None])
                
                else:
                    if np.any(mask[y1:y2, x1:x2] > 0.0):
                        valid_images.append(image[y1:y2, x1:x2, :])
                        valid_masks.append(mask[y1:y2, x1:x2, None])

                        valid_xyxys.append([x1, y1, x2, y2])
        
    valid_xyxys = np.stack(valid_xyxys)

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def get_img_splits(fragment_id, dataset_path, tile_size, start_idx, end_idx, stride):
    
    images = []
    xyxys = []
    
    image, fragment_mask, mask = read_image_mask(fragment_id, dataset_path, None, start_idx, end_idx)
    
    for y1 in range(0, (image.shape[0]-tile_size-1), stride):
        for x1 in range(0, (image.shape[1]-tile_size-1), stride):
            
            y2 = y1 + tile_size
            x2 = x1 + tile_size
            
            if np.any(fragment_mask[y1:y2, x1:x2] > 0.0):
                images.append(image[y1:y2, x1:x2, :])
                xyxys.append([x1, y1, x2, y2])
    
    xyxys = np.stack(xyxys)
    
    return images, xyxys, fragment_mask


def flip(image, **kwargs):
    image = image[:,:,::-1]
    return image


def get_transforms(data, size):
    if data == 'train':
        augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.75),
            A.ShiftScaleRotate(
                rotate_limit=15,
                shift_limit=0.05,
                scale_limit=0.05,
                p=0.75),
            A.OneOf([
                A.GaussNoise(var_limit=[10,50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
            A.CoarseDropout(
                max_holes=2,
                max_width=int(size * 0.1),
                max_height=int(size * 0.1), 
                mask_fill_value=0,
                p=0.5),
            A.Lambda(image=flip, p=0.5),
            A.Normalize(
                mean=0.0,
                std=1.0,
            ),
            ToTensorV2(transpose_mask=True),
        ])
    elif data == 'valid':
        augment = A.Compose([
            A.Normalize(
                mean=0.0,
                std=1.0,
            ),
            ToTensorV2(transpose_mask=True),
        ])
    return augment


class CustomDataset(Dataset):
    
    def __init__(self, images, transform, labels=None, xyxys=None):
        self.images = images
        self.transform = transform
        self.labels = labels
        self.xyxys = xyxys
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        xyxy = []
        label = []
        
        if self.xyxys is not None:
            xyxy = self.xyxys[idx]
        
        if self.labels is not None:
            label = self.labels[idx]
            data = self.transform(image=image, mask=label)
            image = data['image'].unsqueeze(0)
            label = data['mask']
        else:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)
        
        return image, xyxy, label


class Decoder(nn.Module):

    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode='bilinear')

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode='bilinear')
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


class RegressionPLModel(pl.LightningModule):

    def __init__(self, enc='', with_norm=False):
        super(RegressionPLModel, self).__init__()
        
        self.save_hyperparameters()
        
        self.loss_func1 = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func = lambda x, y: 0.5*self.loss_func1(x, y) + 0.5*self.loss_func2(x, y)
        
        self.backbone = InceptionI3d(in_channels=1, num_classes=512)
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=4)

        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)
    
    def forward(self, x):
        # TODO
        if x.ndim == 4:
            x = x[:,None]
        if self.hparams.with_norm:
            x = self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        
        return pred_mask
    
    def training_step(self, batch, batch_idx):
        x, xyxys, y = batch
        outputs = self.forward(x)
        loss1 = self.loss_func(outputs, y)
        self.log('train/total_loss', loss1.item(), on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss1}

    def validation_step(self, batch, batch_idx):
        x, xyxys, y = batch
        outputs = self.forward(x)
        loss1 = self.loss_func(outputs, y)
        self.log('val/total_loss', loss1.item(), on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss1}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-6)
        return [optimizer], [scheduler]


def train(argv):
    
    # --- Set parameters
    dir_data = ''
    dir_labels = ''
    dir_out = ''
    
    # --- Internal parameters
    tile_size = 64
    stride = 16
    start_idx = 15
    end_idx = 45
    batch_size = 128
    num_workers = 4
    epochs = 12
    valid_id = '20230522181603'

    # --- Get cli arguments
    for arg in argv:
        if arg.startswith('--data='):
            dir_data = arg.replace('--data=', '')
        if arg.startswith('--labels='):
            dir_labels = arg.replace('--labels=', '')
        if arg.startswith('--out='):
            dir_out = arg.replace('--out=', '')

    # --- Validate arguments
    if dir_data == '' or dir_labels == '' or dir_out == '':
        print('ERROR: Please provide some arguments.')
        exit(1)

    # --- Do training
    print('Data:    {:s}'.format(dir_data))
    print('Labels:  {:s}'.format(dir_labels))
    print('Out:     {:s}'.format(dir_out))
    
    set_seed()
    os.makedirs(dir_out, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset(
        dir_data, dir_labels, valid_id, tile_size, stride, start_idx, end_idx)
    
    # create datasets
    train_dataset = CustomDataset(
        images=train_images,
        transform=get_transforms('train', tile_size),
        labels=train_masks,
        xyxys=None
    )
    
    valid_dataset = CustomDataset(
        images=valid_images,
        transform=get_transforms('valid', tile_size),
        labels=valid_masks,
        xyxys=valid_xyxys
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = RegressionPLModel(enc='i3d')

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        logger=True,
        default_root_dir=dir_out,
        precision='16-mixed',
        callbacks=ModelCheckpoint(save_last=True),
    )
    
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )
    
    print('Finished...')


def infer(argv):
    
    # --- Set parameters
    dir_data = ''
    segment_id = ''
    fn_model = ''
    dir_out = ''
    
    # --- Internal parameters
    tile_size = 64
    stride = 16
    start_idx = 15
    end_idx = 45
    batch_size = 512
    num_workers = 4

    # --- Get cli arguments
    for arg in argv:
        if arg.startswith('--data='):
            dir_data = arg.replace('--data=', '')
        if arg.startswith('--segment_id='):
            segment_id = arg.replace('--segment_id=', '')
        if arg.startswith('--model='):
            fn_model = arg.replace('--model=', '')
        if arg.startswith('--out='):
            dir_out = arg.replace('--out=', '')

    # --- Validate arguments
    if dir_data == '' or segment_id == '' or fn_model == '' or dir_out == '':
        print('ERROR: Please provide some arguments.')
        exit(1)

    # --- Do inference
    print('Data:    {:s}'.format(dir_data))
    print('Segment: {:s}'.format(segment_id))
    print('Model:   {:s}'.format(fn_model))
    print('Out:     {:s}'.format(dir_out))
    
    os.makedirs(dir_out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    images, xyxys, fragment_mask = get_img_splits(segment_id,
        dir_data, tile_size, start_idx, end_idx, stride)
    
    test_dataset = CustomDataset(
        images=images,
        transform=get_transforms('valid', tile_size),
        labels=None,
        xyxys=xyxys
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    model = RegressionPLModel.load_from_checkpoint(fn_model, strict=False)
    model.cuda()
    model.eval()
    
    mask_pred = np.zeros(fragment_mask.shape)
    
    for step, (images, xys, labels) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc='Predict'):
        
        images = images.to(device)
        
        with torch.no_grad():
            y_preds = model(images)
            y_preds = y_preds.squeeze(1)
            y_preds = torch.sigmoid(y_preds).to('cpu')
            
            for i, (x1, y1, x2, y2) in enumerate(xys):
                
                mask_pred[y1:y2, x1:x2] += y_preds[i].numpy()
    
    mask_pred /= mask_pred.max()
    mask_pred = (mask_pred*255).astype(np.uint8)
    
    img = Image.fromarray(mask_pred)
    img.save(os.path.join(dir_out, '{:s}.png'.format(segment_id)))
    
    print('Finished...')


def main(argv):
    
    if len(argv) > 1 and argv[1] == 'train':
        train(argv)
    
    elif len(argv) > 1 and argv[1] == 'infer':
        infer(argv)


if __name__ == '__main__':
    main(sys.argv)
