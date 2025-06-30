from easydict import EasyDict

config = EasyDict()

config.rcme = EasyDict()
config.rcme.batch_size = 32
config.rcme.lr = 1e-5
config.rcme.cma = 0.1
config.rcme.max_epochs = 1
config.rcme.optimizer_steps = 1000
config.rcme.num_workers = 16
config.rcme.gpus = 2
config.rcme.accumulate_grad_batches = 2
config.rcme.save_dir = "rcme"
config.rcme.model_name = "rcme"

config.radial = EasyDict()
config.radial.batch_size = 32
config.radial.lr = 1e-7
config.radial.prior = 10.0
config.radial.max_epochs = 1
config.radial.optimizer_steps = 1000
config.radial.num_workers = 16
config.radial.gpus = 2
config.radial.accumulate_grad_batches = 2
config.radial.save_dir = "radial"
config.radial.model_name = "radial"

config.meru = EasyDict()
config.meru.batch_size = 256
config.meru.lr = 1e-5
config.meru.entail_weight = 0.2
config.meru.max_epochs = 1
config.meru.optimizer_steps = 5000
config.meru.num_workers = 16
config.meru.gpus = 2
config.meru.accumulate_grad_batches = 2
config.meru.save_dir = "meru"
config.meru.model_name = "meru"

config.atmg = EasyDict()
config.atmg.batch_size = 256
config.atmg.lr = 1e-5
config.atmg.max_epochs = 1
config.atmg.optimizer_steps = 5000
config.atmg.num_workers = 16
config.atmg.gpus = 2
config.atmg.accumulate_grad_batches = 2
config.atmg.save_dir = "atmg"
config.atmg.model_name = "atmg"


config.dataset_root = "bioclip/data/TreeOfLife-10M/dataset/evobio10m-dev/256x256"
config.train_csv = "train_folders/image_metadata.csv"
config.val_csv = "train_folders/image_metadata_val.csv"
