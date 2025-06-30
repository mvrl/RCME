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


class RCME(pl.LightningModule):
    def __init__(self, cfg, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip"
        )
        self.model.logit_scale.requires_grad = False
        self.tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")

        self.batch_size = cfg.batch_size
        self.lr = cfg.lr

    def forward_text(self, text):
        return self.model.encode_text(text, normalize=True)

    def forward_image(self, image):
        return self.model.encode_image(image, normalize=True)

    def radial_loss(self, Evv, root):
        def ij2ext(i, j):
            if i == 0:
                a = Evv[i] - root
                b = Evv[j] - root
            else:
                a = Evv[i] - Evv[i - 1]
                b = Evv[j] - Evv[i - 1]
            b_ = b - a

            an = a.norm(dim=-1)
            bn = b_.norm(dim=-1)
            ext_c = (a * b_).sum(dim=-1) / (an * bn)
            # ^ cos-ext angle
            ext_a = ext_c.clip(min=-1.0, max=1.0).acos()
            # ext angle
            return ext_a

        pos_indices = [(i, i + 1) for i in range(6)]
        neg_indices = [(i, i + 7) for i in range(6)]

        P = torch.stack([ij2ext(*indices) for indices in pos_indices])
        N = torch.stack([-ij2ext(*indices) for indices in neg_indices])

        eloss = P.mean() + N.mean()
        return eloss, P, N

        # def radial_loss_image(self, Evv, Eii, root):
        #     def ij2ext(i, j):
        #         if i==0:
        #             a = Evv[i] - root
        #             b = Eii[j] - root
        #         else:
        #             a = Evv[i] - Evv[i-1]
        #             b = Eii[j] - Evv[i-1]
        #         b_ = b - a

        #         an = a.norm(dim=-1)
        #         bn = b_.norm(dim=-1)
        #         ext_c = (a * b_).sum(dim=-1) / (an * bn)
        #         # ^ cos-ext angle
        #         ext_a = ext_c.clip(min=-1., max=1.).acos()
        #         # ext angle
        #         return ext_a

        pos_indices = [(i, i) for i in range(7)]
        neg_indices = [(i, i + 7) for i in range(7)]

        P = torch.stack([ij2ext(*indices) for indices in pos_indices])
        N = torch.stack([-ij2ext(*indices) for indices in neg_indices])

        GE_LOSS = 0
        for i in range(1, 6):
            GE_LOSS += torch.maximum(
                torch.zeros(P.shape[-1]).cuda(),
                ij2ext(i - 1, i + 1)
                - (torch.cos(P[i - 1]).clip(0, 1) * torch.cos(P[i]).clip(0, 1)).acos()
                + np.pi / 2,
            ).mean()
        GE_LOSS /= 5

        eloss = P.mean() + N.mean() + GE_LOSS
        return eloss, P, N

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

        text_features, root = text_features[:-1], text_features[-1]

        img_list = image_pos_list + image_neg_list
        img_list = torch.stack(img_list)
        img_list = rearrange(img_list, "b n c h w -> (b n) c h w")
        img_features = self.model.encode_image(img_list, normalize=True)

        sim_it = (
            torch.einsum(
                "bd,kd->bk",
                text_features[torch.arange(6, len(text_features), 6)],
                img_features[torch.arange(6, len(text_features), 6)],
            )
            * self.model.logit_scale.exp()
        )

        loss_prior_t = torch.nn.functional.cross_entropy(
            sim_it, torch.arange(sim_it.size(0)).to(sim_it.device)
        )
        loss_prior_i = torch.nn.functional.cross_entropy(
            sim_it.t(), torch.arange(sim_it.size(1)).to(sim_it.device)
        )

        loss_prior = (loss_prior_t + loss_prior_i) / 2.0

        img_features = rearrange(img_features, "(p b n) d -> (p n) b d", n=7, p=2)

        text_features = rearrange(text_features, "(p b n) d -> (p n) b d", n=7, p=2)

        Evv = text_features

        eloss, P, N = self.radial_loss(Evv, root)

        Pr = P.ravel()
        Nr = N.ravel()
        PNr = Pr + Nr
        i, j = Pr.argmax(), Nr.argmax()
        mloss = PNr[i] + PNr[j]

        loss = eloss + mloss + cfg.cma * loss_prior

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
            cfg.cma * loss_prior,
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
            cfg.cma * loss_prior,
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

    def configure_optimizers(self):
        params = self.parameters()
        self.optim = torch.optim.AdamW(params, lr=self.lr, betas=(0.9, 0.98), eps=1e-6)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optim, max_lr=self.lr, total_steps=cfg.optimizer_steps
        )
        return [self.optim], [self.scheduler]


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
