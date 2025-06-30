import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import ImageTextDataset, collate_fn
from pytorch_lightning.callbacks import ModelCheckpoint
from copy import deepcopy
import math
import open_clip
from einops import rearrange


class RadialEmbed(pl.LightningModule):
    def __init__(self, cfg, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip"
        )
        for param in self.model.visual.parameters():
            param.requires_grad = False
        self.model.logit_scale.requires_grad = False
        self.model_frozen = deepcopy(self.model)
        for param in self.model_frozen.parameters():
            param.requires_grad = False
        self.tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")

        self.batch_size = cfg.batch_size
        self.lr = cfg.lr

    def forward_text(self, text):
        return self.model.encode_text(text, normalize=True)

    def forward_image(self, image):
        return self.model.encode_image(image, normalize=True)

    def radial_loss(self, Evv, root):
        def ij2ext(i, j):
            a = Evv[i] - root
            b = Evv[j] - root
            b_ = b - a

            an = a.norm(dim=-1)
            bn = b_.norm(dim=-1)
            ext_c = (a * b_).sum(dim=-1) / (an * bn)
            # ^ cos-ext angle
            ext_a = ext_c.clip(min=-1.0, max=1.0).acos()
            # ext angle
            return ext_a

        pos_indices = [(i, i + 1) for i in range(6)]
        neg_indices = [(i, i + 7) for i in range(7)]

        P = torch.stack([ij2ext(*indices) for indices in pos_indices])
        N = torch.stack([-ij2ext(*indices) for indices in neg_indices])

        eloss = P.mean() + N.mean()
        return eloss, P, N[:-1]

    def shared_step(self, batch, train=True):
        image_pos_list, image_neg_list, pos_list, neg_list = batch
        text_list = pos_list + neg_list
        text_list = torch.stack(
            [self.tokenizer(text_list[i]) for i in range(len(text_list))]
        ).to(self.device)
        text_reshaped = rearrange(text_list, "b n d -> (b n) d")
        root_text = self.tokenizer(["Eukarya"]).to(self.device)
        text_reshaped = torch.cat([text_reshaped, root_text], dim=0)
        text_features = self.model.encode_text(text_reshaped, normalize=True)
        with torch.no_grad():
            text_features_frozen = self.model_frozen.encode_text(
                rearrange(text_list, "b n d -> (b n) d"), normalize=True
            )
        text_features, root = text_features[:-1], text_features[-1]

        loss_prior = torch.einsum(
            "bd,bd->b",
            text_features[torch.arange(6, len(text_features), 6)],
            text_features_frozen[torch.arange(6, len(text_features), 6)],
        ).mean()

        text_features = rearrange(text_features, "(p b n) d -> (p n) b d", n=7, p=2)

        Evv = text_features

        loss_prior = (1 - loss_prior) / 2.0

        eloss, P, N = self.radial_loss(Evv, root)

        Pr = P.ravel()
        Nr = N.ravel()
        PNr = Pr + Nr
        i, j = Pr.argmax(), Nr.argmax()
        mloss = PNr[i] + PNr[j]

        loss = eloss + mloss + cfg.prior * loss_prior

        return loss, eloss, mloss, loss_prior, P.mean(), N.mean()

    def training_step(self, batch, batch_idx):
        loss, eloss, mloss, loss_prior, P, N = self.shared_step(batch)
        self.log(
            "e_loss",
            eloss,
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "m_loss",
            mloss,
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "p_loss",
            cfg.prior * loss_prior,
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "P",
            P,
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "N",
            N,
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, eloss, mloss, loss_prior, P, N = self.shared_step(batch, train=False)
        self.log(
            "val_loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_p_loss",
            cfg.prior * loss_prior,
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return loss

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
            persistent_workers=False,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        # worker_init_fn=set_worker_sharing_strategy)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            persistent_workers=False,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        # worker_init_fn=set_worker_sharing_strategy)

    def configure_optimizers(self):
        params = self.parameters()
        self.optim = torch.optim.AdamW(params, lr=self.lr, betas=(0.9, 0.98), eps=1e-6)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optim, max_lr=self.lr, total_steps=cfg.optimizer_steps
        )
        return [self.optim], [self.scheduler]


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
