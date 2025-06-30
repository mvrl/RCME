from config import config as cfg
from data.dataset import ImageTextDataset, ImageTextDatasetMERU
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser

def main(args):
    if args.model == "rcme":
        from models.rcme import RCME
        train_dataset = ImageTextDataset(cfg.dataset_root, cfg.train_csv, mode="train")
        val_dataset = ImageTextDataset(cfg.dataset_root, cfg.val_csv, mode="val")
        model = RCME(cfg.rcme, train_dataset, val_dataset)
    elif args.model == "radial":
        from models.radial_embed_model import RadialEmbed
        train_dataset = ImageTextDataset(cfg.dataset_root, cfg.train_csv, mode="train")
        val_dataset = ImageTextDataset(cfg.dataset_root, cfg.val_csv, mode="val")
        model = RadialEmbed(cfg.radial, train_dataset, val_dataset)
    elif args.model == "meru":
        from models.meru import MERU
        train_dataset = ImageTextDatasetMERU(cfg.dataset_root, cfg.train_csv, mode="train")
        val_dataset = ImageTextDatasetMERU(cfg.dataset_root, cfg.val_csv, mode="val")
        model = MERU(cfg.meru, train_dataset, val_dataset)
    elif args.model == "atmg":
        from models.atmg import ATMG
        train_dataset = ImageTextDatasetMERU(cfg.dataset_root, cfg.train_csv, mode="train")
        val_dataset = ImageTextDatasetMERU(cfg.dataset_root, cfg.val_csv, mode="val")
        model = ATMG(cfg.atmg, train_dataset, val_dataset)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{args.model}',
        filename='{args.model}-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True,
    )
    trainer = pl.Trainer(accelerator='gpu', devices=cfg.[args.model].gpus, max_epochs=cfg.[args.model].max_epochs, strategy='ddp', callbacks=[checkpoint_callback], 
    accumulate_grad_batches=cfg.[args.model].accumulate_grad_batches, val_check_interval=0.1, limit_val_batches=100)
    trainer.fit(model)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="rcme", choices=["rcme", "radial", "meru", "atmg"])
    args = parser.parse_args()
    main(args)