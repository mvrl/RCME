from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
from PIL import Image
from datetime import datetime
from torchvision.transforms import v2
import torch
from transformers import AutoTokenizer
import glob
from tree import random_sample_by_rank
import random

class INatTextDataset(Dataset):
    def __init__(self, work_dir, json_path, mode='train'):
        self.paths = glob.glob(os.path.join(work_dir, mode, "*"))
        self.paths_base = [os.path.basename(p) for p in self.paths]
        self.work_dir = work_dir
        self.json = json.load(open(os.path.join(self.work_dir, json_path)))
        self.images = self.json['images']
        self.annot = self.json['annotations']
        for i in range(len(self.images)):
            assert self.images[i]['id'] == self.annot[i]['id']
            self.images[i]['label'] = self.annot[i]['category_id']
        self.filtered_json = self.images
        self.species_text = list(set([" ".join(d['file_name'].split("/")[1].split("_")[1:]) for d in self.filtered_json]))
        self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mode = mode
    def __len__(self):
        return len(self.filtered_json)
    def __getitem__(self, idx):
        pos = {}
        neg = {}
        pos_list = []
        neg_list = []
        image_pos_list = []
        image_neg_list = []
        species_text = " ".join(self.filtered_json[idx]['file_name'].split("/")[1].split("_")[1:])
       
        # Hard Negative Text and Image Sampling
        
        for i in range(7):
            rank = species_text.split(" ")
            pos[i], neg[i] = random_sample_by_rank(self.paths_base, rank, i)
            img_path_pos = random.choice(glob.glob(os.path.join(self.work_dir, self.mode, pos[i], "*")))
            img_path_neg = random.choice(glob.glob(os.path.join(self.work_dir, self.mode, neg[i], "*")))
            pos[i] = " ".join(pos[i].split("_")[1:][:i+1])
            neg[i] = " ".join(neg[i].split("_")[1:][:i+1])
            pos_list.append(pos[i])
            neg_list.append(neg[i])
            img_pos = self.transform(Image.open(img_path_pos))
            img_neg = self.transform(Image.open(img_path_neg))
            image_pos_list.append(img_pos)
            image_neg_list.append(img_neg)
        return species_text, self.species_text.index(species_text), pos, neg, pos_list, neg_list, image_pos_list, image_neg_list

def collate_fn(batch):
    pos_list = []
    neg_list = []
    image_pos_list = []
    image_neg_list = []

    pos_list = [item[-4] for item in batch]
    neg_list = [item[-3] for item in batch]
    image_pos_list = [torch.stack(item[-2]) for item in batch]
    image_neg_list = [torch.stack(item[-1]) for item in batch]

    return image_pos_list, image_neg_list, pos_list, neg_list


if __name__=='__main__':
    # inat_data = INatDataset('/projects/bdbl/ssastry/taxabind/ecobind_data', 'train.json')
    # print(len(inat_data))
    # import code; code.interact(local=locals())
    import open_clip
    from torch.utils.data import DataLoader
    inat_data = INatTextDataset('/projects/bdbl/ssastry/taxabind/ecobind_data', 'train.json')
    loader = DataLoader(inat_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    #print(inat_data[0][0], inat_data[0][-2], inat_data[0][-1])
    for i, data in enumerate(loader):
        import code; code.interact(local=locals())
        if i>0:
            break