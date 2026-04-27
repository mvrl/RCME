# Global and Local Entailment Learning for Natural World Imagery

<div align="center">
<img src="imgs/rcme_logo.png" width="220">

[![arXiv](https://img.shields.io/badge/arXiv-2506.21476-red)](https://arxiv.org/abs/2506.21476)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://vishu26.github.io/RCME/index.html)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow)](https://huggingface.co/MVRL/rcme-tol-vit-base-patch16)

[Srikumar Sastry*](https://vishu26.github.io/),
[Aayush Dhakal](https://scholar.google.com/citations?user=KawjT_8AAAAJ&hl=en),
[Eric Xing](https://ericx003.github.io/),
[Subash Khanal](https://subash-khanal.github.io/),
[Nathan Jacobs](https://jacobsn.github.io/)
(*Corresponding Author)

<b>ICCV 2025</b>
</div>

<p align="center">
Radial Cross-Modal Embeddings (RCME) is a state-of-the-art hierarchical image-text ordering and retrieval method in the embedding space.
</p>

## 🔥 Textual Entailment and Ordering
<div align="center">
<img src="imgs/inat_ordering.png" width="320">
<img src="imgs/hierarcaps_ordering.png" width="370">
</div>

## 📦 Models

|Model|Architecture|HuggingFace|
|--|--|--|
|CLIP|ViT-B/16|`MVRL/rcme-vit-base-patch16`|
|CLIP|ViT-L/14|`MVRL/rcme-vit-large-patch14`|
|TreeofLife|ViT-B/16|`MVRL/rcme-tol-vit-base-patch16`|

## ⚙️ Setup

#### Setting up 
1. Clone this repository:
```bash
git clone https://github.com/mvrl/RCME.git
```
2. Install dependencies:
```bash
cd RCME && pip install -r requirements.txt
```

## 🗂️ Data
1. Use BioCLIP's scripts to download TreeofLife-10M dataset:
```bash
rcme/data/bioclip/scripts/setup_download_tol-10m_components.bash && \
rcme/data/bioclip/scripts/submit_download_tol-10m_components.bash
```
**Hint**: Setup paths and other variables in `setup_download_tol-10m_components.bash` script.

2. Use our script to convert TreeofLife-10M dataset into iNaturalist-2021 style naming:
```bash
python rcme/data/bioclip/write_imgs.py
```
**Hint**: Setup paths and other variables in our script.<br>
**Hint**: Currently only supports `num_workers=1`

## 🔥 Training
1. Setup all hyperparameters in `rcme/config.py` file.
2. Run training by specifying the model:
```bash
python rcme/train.py --model="rcme"
```
**Hint**: Currently supports `rcme`, `radial`, `atmg` and `meru`.

## ✅ Evaluation

The evaluation script measures **hierarchical ordering quality** using iNaturalist 2021 validation images. It computes three metrics:

| Metric | Description |
|---|---|
| `d_corr` | Kendall's τ distance correlation — measures how well the model orders taxonomy levels by radius |
| `precision` | Fraction of model-predicted texts that are ground-truth taxonomy labels |
| `recall` | Fraction of ground-truth taxonomy labels recovered across all hierarchy levels |

### Prerequisites

1. Download the iNaturalist 2021 validation split. The evaluation CSV (`rcme/eval/order_dataset.csv`) expects images to be accessible at paths of the form:
   ```
   inat_21/val/<category_folder>/<image_file>.jpg
   ```
   Ensure these paths are accessible from the directory you run the script from.

### Running Evaluation

The evaluation script is self-contained and loads a pretrained model directly from HuggingFace.

1. Navigate to the eval directory:
   ```bash
   cd rcme/eval
   ```

2. Run the evaluation (uses `MVRL/rcme-tol-vit-base-patch16` by default):
   ```bash
   python order_metrics.py
   ```

   To evaluate a different model, open `order_metrics.py` and update the model identifier in the `__main__` block:
   ```python
   # Change this line to use a different model:
   clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
       "hf-hub:MVRL/rcme-vit-base-patch16"   # or rcme-vit-large-patch14
   )
   ```
   See the [📦 Models](#-models) table above for all available HuggingFace identifiers.

## 📑 Citation

```bibtex
@inproceedings{sastry2025global,
    title={Global and Local Entailment Learning for Natural World Imagery},
    author={Sastry, Srikumar and Dhakal, Aayush and Xing, Eric and Khanal, Subash and Jacobs, Nathan},
    booktitle={International Conference on Computer Vision},
    year={2025},
    organization={IEEE/CVF}
}
```

## 🔍 Additional Links
Check out our lab website for other interesting works on geospatial understanding and mapping:
* Multi-Modal Vision Research Lab (MVRL) - [Link](https://mvrl.cse.wustl.edu/)
* Related Works from MVRL - [Link](https://mvrl.cse.wustl.edu/publications/)
