import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset_meru import INatTextDataset, collate_fn
from pytorch_lightning.callbacks import ModelCheckpoint
from copy import deepcopy
import lorentz as L
import math
import open_clip
from einops import rearrange
from dataset import INatDatasetIntra


class MERU(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        self.model.logit_scale.requires_grad = False
        self.tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

        self.batch_size = kwargs.get('batch_size', 256)
        self.lr = kwargs.get('lr', 1e-5)

        self.visual_alpha = nn.Parameter(torch.tensor(512**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(512**-0.5).log())

        self.curv = nn.Parameter(
            torch.tensor(1.0).log(), requires_grad=True
        )
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(1.0 * 10),
            "min": math.log(1.0 / 10),
        }

        self.entail_weight = 0.2
    
    def forward_text(self, text):
        text_feats = self.model.encode_text(text, normalize=False)
        text_feats = text_feats * self.textual_alpha.exp()
        text_feats = L.exp_map0(text_feats, self.curv.exp())
        return text_feats
    
    def forward_image(self, image):
        image_feats = self.model.encode_image(image, normalize=False)
        image_feats = image_feats * self.visual_alpha.exp()
        image_feats = L.exp_map0(image_feats, self.curv.exp())
        return image_feats

    
    def shared_step(self, batch, train=True):
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        species_text, species_index, pos, neg, img_pos, img_neg = batch
        batch_size =  img_pos.shape[0]
        text_list = self.tokenizer(pos).to(self.device)
        text_features = self.forward_text(text_list)

        img_list = img_pos
        img_features = self.forward_image(img_list)

        image_logits = -L.pairwise_dist(img_features, text_features, _curv)
        text_logits = -L.pairwise_dist(text_features, img_features, _curv)

        targets = torch.arange(batch_size, device=image_logits.device)
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()

        contrastive_loss = 0.5 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
            )
        
        _angle = L.oxy_angle(text_features, img_features, _curv)
        _aperture = L.half_aperture(text_features, _curv)
        entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()

        loss = contrastive_loss + self.entail_weight * entailment_loss
        
        return loss, contrastive_loss, entailment_loss
        
    def training_step(self, batch, batch_idx):
        loss, contrastive_loss, entailment_loss = self.shared_step(batch)
        self.log('loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        self.log('c_loss', contrastive_loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        self.log('e_loss', entailment_loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        self.log('logit_scale', self.logit_scale.exp().item(), sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, contrastive_loss, entailment_loss = self.shared_step(batch, train=False)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        self.log('val_c_loss', contrastive_loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        self.log('val_e_loss', entailment_loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        return loss
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=16,
                          shuffle=True,
                          persistent_workers=False,
                          pin_memory=False
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=16,
                          shuffle=False,
                          persistent_workers=False,
                          pin_memory=False
                          )
    
    def configure_optimizers(self):
        params = self.parameters()
        self.optim = torch.optim.AdamW(params,
                                       lr=self.lr,
                                       betas=(0.9,0.98),
                                       eps=1e-6
                                    )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optim,
            max_lr=self.lr,
            total_steps=1000
        )
        return [self.optim], [self.scheduler]