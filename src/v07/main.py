#!/bin/python3
import albumentations as A
import cv2
import json
import numpy as np
import os
import random
import torchsummary
import segmentation_models_pytorch as smp
import sys
import tqdm

from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from PIL import Image


class SMPFPN(pl.LightningModule):

    def __init__(self,
        encoder_name='resnet34',
        encoder_depth=5,
        encoder_weights='imagenet',
        decoder_pyramid_channels=256,
        decoder_segmentation_channels=128,
        decoder_merge_policy='add',
        decoder_dropout=0.2,
        in_channels=3,
        classes=1,
        activation=None,
        upsampling=4,
        lr=0.001,
        step_size=20,
        gamma=0.1):
        
        super(SMPFPN, self).__init__()
        self.save_hyperparameters()
        
        if self.hparams.encoder_name:
            encoder_name = self.hparams.encoder_name
        if self.hparams.encoder_depth:
            encoder_depth = self.hparams.encoder_depth
        if self.hparams.encoder_weights:
            encoder_weights = self.hparams.encoder_weights
        if self.hparams.decoder_pyramid_channels:
            decoder_pyramid_channels = self.hparams.decoder_pyramid_channels
        if self.hparams.decoder_segmentation_channels:
            decoder_segmentation_channels = self.hparams.decoder_segmentation_channels
        if self.hparams.decoder_merge_policy:
            decoder_merge_policy = self.hparams.decoder_merge_policy
        if self.hparams.decoder_dropout:
            decoder_dropout = self.hparams.decoder_dropout
        if self.hparams.in_channels:
            in_channels = self.hparams.in_channels
        if self.hparams.classes:
            classes = self.hparams.classes
        if self.hparams.activation:
            activation = self.hparams.activation
        if self.hparams.upsampling:
            upsampling = self.hparams.upsampling
        if self.hparams.lr:
            lr = self.hparams.lr
        if self.hparams.step_size:
            step_size = self.hparams.step_size
        if self.hparams.gamma:
            gamma = self.hparams.gamma
        
        # create FPN model
        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_pyramid_channels=decoder_pyramid_channels,
            decoder_segmentation_channels=decoder_segmentation_channels,
            decoder_merge_policy=decoder_merge_policy,
            decoder_dropout=decoder_dropout,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            upsampling=upsampling,
        )
        
        #self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.loss_fn = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, yxs, mask = batch
        logits_mask = self.forward(image)
        logits_mask = logits_mask.squeeze(1)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        mx = self.shared_step(batch, 'train')
        self.log('train_loss', mx['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return mx

    def validation_step(self, batch, batch_idx):
        mx = self.shared_step(batch, 'valid')
        self.log('valid_loss', mx['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return mx

    def test_step(self, batch, batch_idx):
        mx = self.shared_step(batch, 'test')
        return mx

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]


# Set seed
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


def read_scan_mask(segment_id, dir_data, start_idx, end_idx):
    
    # Load ct scan
    image3d = []
    for i in tqdm.tqdm(range(start_idx, end_idx), desc='Loading scans'):
        
        image_file = f'{dir_data}/{segment_id}/layers/{i:02d}.tif'
        img = cv2.imread(image_file, cv2.IMREAD_ANYDEPTH)
        
        # Normalize
        img = img.astype(np.float32)
        img /= float(2**16)
        img = np.clip(img, 0.0, 1.0)
        
        image3d.append(img)
    
    # Create stack and convert to float
    image3d = np.stack(image3d, axis=2).astype(np.float32)
    
    # Load mask
    mask_file = os.path.join(
        dir_data,
        segment_id,
        '{:s}_mask.png'.format(segment_id.replace('_superseded', ''))
    )   
    image_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    image_mask = image_mask.astype(np.float32)
    image_mask /= float(2**8)
    image_mask = np.clip(image_mask, 0.0, 1.0)
    
    return image3d, image_mask


def read_volumes(segment_id, dir_data, dir_labels, start_idx, end_idx, tile_size, stride, p_valid):
    
    train_volumes = []
    train_yyxxs = []
    train_labels = []
    
    valid_volumes = []
    valid_yyxxs = []
    valid_labels = []
    
    image3d, image_mask = read_scan_mask(
        segment_id, dir_data, start_idx, end_idx)
    
    # Load label
    image_label = None
    if dir_labels is not None:
        lable_file = f'{dir_labels}/{segment_id}.png'
        image_label = cv2.imread(lable_file, cv2.IMREAD_GRAYSCALE)
        image_label = image_label.astype(np.float32)
        image_label /= float(2**8)
        image_label = np.clip(image_label, 0.0, 1.0)
        # TODO: Youssef's label images are larger due to padding
        y, x = image_mask.shape
        image_label = image_label[:y, :x]
    
    # Iterate over scan volume
    for y1 in tqdm.tqdm(range(0, (image3d.shape[0]-tile_size-1), stride), desc='Process volume'):
        for x1 in range(0, (image3d.shape[1]-tile_size-1), stride):
            
            x2 = x1 + tile_size
            y2 = y1 + tile_size
            
            # Training data is created
            if dir_labels is not None:

                # Check if there is ink on it, and if it is inside trmask
                if np.any(image_label[y1:y2, x1:x2] > 0.0):
                    
                    if np.random.random() >= p_valid:
                        train_volumes.append(image3d[y1:y2, x1:x2, :])
                        train_yyxxs.append([y1, y2, x1, x2])
                        train_labels.append(image_label[y1:y2, x1:x2])
                    else:
                        valid_volumes.append(image3d[y1:y2, x1:x2, :])
                        valid_yyxxs.append([y1, y2, x1, x2])
                        valid_labels.append(image_label[y1:y2, x1:x2])
            
            # Inference data is created
            else:
                
                # Check if we are inside the mask
                if np.any(image_mask[y1:y2, x1:x2] > 0.0):
                    train_volumes.append(image3d[y1:y2, x1:x2, :])
                    train_yyxxs.append([y1, y2, x1, x2])
    
    return train_volumes, train_yyxxs, train_labels, valid_volumes, valid_yyxxs, valid_labels


def load_dataset_train(dir_data, dir_labels, start_idx, end_idx, tile_size, stride, p_valid):
    
    train_volumes = []
    train_yyxxs = []
    train_labels = []
    
    valid_volumes = []
    valid_yyxxs = []
    valid_labels = []
    
    for fn in os.listdir(dir_labels):
        
        segment_id = os.path.splitext(fn)[0]
        
        t_vol, t_yxs, t_lbl, v_vol, v_yxs, v_lbl = read_volumes(
            segment_id, dir_data, dir_labels, start_idx, end_idx,
            tile_size, stride, p_valid)
        
        train_volumes.extend(t_vol)
        train_yyxxs.extend(t_yxs)
        train_labels.extend(t_lbl)
        
        valid_volumes.extend(v_vol)
        valid_yyxxs.extend(v_yxs)
        valid_labels.extend(v_lbl)
    
    return train_volumes, train_yyxxs, train_labels, valid_volumes, valid_yyxxs, valid_labels


def load_dataset_infer(dir_data, segment_id, start_idx, end_idx, tile_size, stride):
    
    volumes, yyxxs, labels, v_vol, v_yxs, v_lbl = read_volumes(
        segment_id, dir_data, None, start_idx, end_idx,
        tile_size, stride, 0.0)
    
    assert len(v_vol) == len(v_yxs) == len(v_lbl) == 0
    
    return volumes, yyxxs


def get_model(param, load_weights=False):
    
    if param['name'] == 'SMPFPN':
        if load_weights is False:
            model = SMPFPN(
                encoder_name=param['encoder_name'],
                encoder_depth=param['encoder_depth'],
                encoder_weights=param['encoder_weights'],
                decoder_pyramid_channels=param['decoder_pyramid_channels'],
                decoder_segmentation_channels=param['decoder_segmentation_channels'],
                decoder_merge_policy=param['decoder_merge_policy'],
                decoder_dropout=param['decoder_dropout'],
                in_channels=param['in_channels'],
                classes=param['classes'],
                activation=param['activation'],
                upsampling=param['upsampling'],
                lr=param['lr'],
                step_size=param['step_size'],
                gamma=param['gamma']
            )
            return model
        else:
            model = SMPFPN.load_from_checkpoint(param['fn_model'], strict=False)
            return model
    
    else:
        return None


def roll_3dstack(volume, **kwargs):
    shift = int(np.random.random() * volume.shape[2])
    return np.roll(volume, shift=shift, axis=2)


def get_transforms(data):
    if data == 'train':
        augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ChannelShuffle(p=0.5),
            # A.Lambda(image=roll_3dstack, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=0.001, p=0.5),
                A.GaussianBlur(p=0.5),
                A.Blur(p=0.5),
            ], p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=0.05,
                rotate_limit=15,
                border_mode=0,
                value=0.0,
                mask_value=0.0,
                p=0.5
            ),
            ToTensorV2(transpose_mask=True),
        ])
    elif data == 'valid':
        augment = A.Compose([
            A.NoOp(),
            ToTensorV2(transpose_mask=True),
        ])
    return augment


class VolumeDataset(Dataset):

    def __init__(self, volumes, transform, yyxxs, labels=None):
        self.volumes = volumes
        self.transform = transform
        self.yyxxs = yyxxs
        self.labels = labels

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        volume = self.volumes[idx]
        yyxx = self.yyxxs[idx]
        label = []

        if self.labels is not None:
            label = self.labels[idx]

        # apply transform
        if self.labels is not None:
            augmented = self.transform(image=volume, mask=label)
            volume = augmented['image']
            label = augmented['mask']

        else:
            augmented = self.transform(image=volume)
            volume = augmented['image']

        return volume, yyxx, label


class CFG:

    def get(self, index):
        if index == 0:
            dxx = {}
            
            dxx['dir_data']    = '../../dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths'
            dxx['dir_labels']  = '../../input/labels/set03'
            dxx['dir_out']     = '../../tmp/output/v07/00'
            
            dxx['tile_size']   = 64
            dxx['stride']      = 16
            dxx['start_idx']   = 15
            dxx['end_idx']     = 45
            dxx['batch_size']  = 32
            dxx['epochs']      = 20
            dxx['num_workers'] = 4
            
            dxx['model_param'] = {
                'name':                             'SMPFPN',
                'fn_model':                         (dxx['dir_out'] + '/lightning_logs/version_0/checkpoints/last.ckpt'),
                'encoder_name':                     'resnet18',
                'encoder_depth':                    5,
                'encoder_weights':                  'imagenet',
                'decoder_pyramid_channels':         256,
                'decoder_segmentation_channels':    128,
                'decoder_merge_policy':             'add',
                'decoder_dropout':                  0.2,
                'in_channels':                      (dxx['end_idx'] - dxx['start_idx']),
                'classes':                          1,
                'activation':                       None,
                'upsampling':                       4,
                'lr':                               1e-4,
                'step_size':                        20,
                'gamma':                            0.1,
            }
            
            dxx['infer_batch_size'] = 1024
            dxx['infer_stride']     = 16
            
            return dxx
        
        elif index == 1:
            dxx = {}
            return dxx

        else:
            return None


def train(argv):

    # --- CLI parameters
    n_cfg = 0
    fn_cfg = ''

    # --- Get CLI arguments
    for arg in argv:
        if arg.startswith('--n_cfg='):
            n_cfg = int(arg.replace('--n_cfg=', ''))
        if arg.startswith('--cfg='):
            fn_cfg = arg.replace('--cfg=', '')

    # --- Set parameters
    cfg = {}
    if fn_cfg != '':
        fp = open(fn_cfg)
        cfg = json.load(fp)
        fp.close()
    else:
        cfg = CFG().get(n_cfg)
    
    dir_data = cfg['dir_data']
    dir_labels = cfg['dir_labels']
    dir_out = cfg['dir_out']
    
    model_param = cfg['model_param']
    tile_size = cfg['tile_size']
    stride = cfg['stride']
    start_idx = cfg['start_idx']
    end_idx = cfg['end_idx']
    batch_size = cfg['batch_size']
    epochs = cfg['epochs']
    num_workers = cfg['num_workers']

    # --- Print parameters
    print('{:14s}: {:s}'.format('n_cfg', str(n_cfg)))
    print('{:14s}: {:s}'.format('fn_cfg', fn_cfg))

    for key, value in cfg.items():
        print('{:14s}: {:s}'.format(key, str(value)))

    print()

    # Create output folder
    os.makedirs(dir_out, exist_ok=True)
    
    # Set seed
    set_seed()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')

    # Load training data
    data = load_dataset_train(dir_data, dir_labels,
        start_idx, end_idx, tile_size, stride, 0.2)
    
    # Unpack data
    train_volumes, train_yyxxs, train_labels, valid_volumes, valid_yyxxs, valid_labels = data

    # Print volumes count
    print('Total count of train volumes: {:d}'.format(len(train_volumes)))
    print('Total count of validation volumes: {:d}'.format(len(valid_volumes)))
    print('Total count of volumes: {:}'.format((len(train_volumes) + len(valid_volumes))))
    print()

    # Create training dataset
    train_dataset = VolumeDataset(
        volumes=train_volumes,
        transform=get_transforms('train'),
        yyxxs=train_yyxxs,
        labels=train_labels,
    )

    # Create validation dataset
    valid_dataset = VolumeDataset(
        volumes=valid_volumes,
        transform=get_transforms('valid'),
        yyxxs=valid_yyxxs,
        labels=valid_labels,
    )

    # Create dataloader training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Create dataloader validation
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Get model
    model = get_model(model_param)

    # Print model summary
    torchsummary.summary(model)

    # Create trainer
    trainer = pl.Trainer(
        default_root_dir=dir_out,
        accelerator='gpu',
        # devices=1,
        # FIXME: uggly workaround
        devices=([1] if '--gpu1' in argv else [0]),
        ###
        max_epochs=epochs,
        precision='16-mixed',
        logger=CSVLogger(dir_out),
        callbacks=ModelCheckpoint(save_last=True),
    )

    # Fit model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )

    print('Finished...')


def infer(argv):
    
    # --- CLI parameters
    n_cfg = 0
    fn_cfg = ''
    segment_id = ''
    
    # --- Get CLI arguments
    for arg in argv:
        if arg.startswith('--n_cfg='):
            n_cfg = int(arg.replace('--n_cfg=', ''))
        if arg.startswith('--cfg='):
            fn_cfg = arg.replace('--cfg=', '')
        if arg.startswith('--segment_id='):
            segment_id = arg.replace('--segment_id=', '')

    # --- Validate CLI arguments
    if segment_id == '':
        print('ERROR: Please provide some arguments.')
        exit(1)

    # --- Set parameters
    cfg = {}
    if fn_cfg != '':
        fp = open(fn_cfg)
        cfg = json.load(fp)
        fp.close()
    else:
        cfg = CFG().get(n_cfg)
    
    dir_data = cfg['dir_data']
    dir_out = cfg['dir_out']

    model_param = cfg['model_param']
    tile_size = cfg['tile_size']
    start_idx = cfg['start_idx']
    end_idx = cfg['end_idx']
    num_workers = cfg['num_workers']
    
    infer_batch_size = cfg['infer_batch_size']
    infer_stride = cfg['infer_stride']

    # --- Print parameters
    print('{:14s}: {:s}'.format('n_cfg', str(n_cfg)))
    print('{:14s}: {:s}'.format('fn_cfg', fn_cfg))
    print('{:14s}: {:s}'.format('Segment', segment_id))

    for key, value in cfg.items():
        print('{:14s}: {:s}'.format(key, str(value)))

    print()

    # Create output folder
    dir_out += '/prediction'
    os.makedirs(dir_out, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    # FIXME
    device = torch.device('cuda:0' if '--gpu0' in argv else 'cuda:1' if '--gpu1' in argv else 'cpu')
    ###

    # Load inference dataset
    volumes, yyxxs = load_dataset_infer(dir_data, segment_id, start_idx,
        end_idx, tile_size, infer_stride)

    # Create inference dataset
    infer_dataset = VolumeDataset(
        volumes=volumes,
        transform=get_transforms('valid'),
        yyxxs=yyxxs,
        labels=None,
    )

    # Create dataloader inference
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=infer_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Load model
    model = get_model(model_param, load_weights=True)

    # Init model
    model.to(device)
    model.eval()

    # Get size
    mask_file = os.path.join(
        dir_data,
        segment_id,
        '{:s}_mask.png'.format(segment_id.replace('_superseded', ''))
    )   
    image_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    # Create prediction image
    image_pred = np.zeros(image_mask.shape, dtype=np.float32)

    # Predict
    for step, data in tqdm.tqdm(enumerate(infer_loader), total=len(infer_loader), desc='Predict'):
    
        vols, yxs, _ = data
        
        vols = vols.to(device)

        with torch.no_grad():
            
            preds = model(vols)
            preds = preds.squeeze(1)
            preds = torch.sigmoid(preds).to('cpu')
            
            y1s, y2s, x1s, x2s = yxs
            
            # iterate over batch
            for i, pyx in enumerate(zip(preds, y1s, y2s, x1s, x2s)):
                
                pred, y1, y2, x1, x2 = pyx
                image_pred[y1:y2, x1:x2] += pred.numpy()
    
    # Adjust image
    image_pred /= image_pred.max()
    image_pred = (image_pred*255).astype(np.uint8)

    # Save image
    output = Image.fromarray(image_pred)
    output.save(f'{dir_out}/{segment_id}.png')

    print('Finished...')

 
def main(argv):
    
    if len(argv) > 1 and argv[1] == 'train':
        train(argv)
    
    elif len(argv) > 1 and argv[1] == 'infer':
        infer(argv)
    
    # cfgstr = '--n_cfg=0'
    # train([cfgstr])
    # infer([cfgstr, '--segment_id=20230925002745'])


if __name__ == '__main__':
    main(sys.argv)
