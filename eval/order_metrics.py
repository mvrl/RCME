### Code from https://github.com/TAU-VAILab/hierarcaps/tree/main

import torch 
from collections import defaultdict
import numpy as np
from scipy.stats import kendalltau

class Evaluate:
    def __init__(self, clip, df, proc, tokenizer, device, steps=100):
        self.clip = clip
        self.df = df
        self.proc = proc
        self.device = device
        self.steps = steps
        self.tokenizer = tokenizer
    @torch.no_grad()
    def run(self):

        print("Running RCME evaluation")

        # inp = self.proc(text="Eukarya", padding=True, truncation=True,
                        # return_tensors="pt").to(self.device)
        # root = self.clip.get_text_features(**inp)[0]
        root = self.tokenizer(["Eukarya"]).to(self.device)
        root = self.clip.encode_text(root, normalize=False)
        root /= root.norm()
        root = root.cpu()

        def row2texts(row):
            return [x.strip() for x in row.captions.split('=>')]

        def embed(row):
            texts = row2texts(row)
            assert len(texts) == 7, f'Invalid number of val texts: {len(texts)}'
            # inp = self.proc(text=texts, truncation=True,
            #                 padding=True, return_tensors='pt').to(self.device)
            inp = self.tokenizer(texts).to(self.device)
            E = self.clip.encode_text(inp, normalize=False)
            E = E / E.norm(dim=-1)[:, None]
            return E

        def embed_img(img):
            # inp = self.proc(images=img, return_tensors='pt').to(self.device)
            inp = self.proc(img).unsqueeze(0).to(self.device)
            E = self.clip.encode_image(inp, normalize=False)
            E = E / E.norm()
            return E

        def get_dists(E):
            return (E - root[None]).norm(dim=-1)

        R = []  # radii
        Es = []
        VEs = []
        img_fns_uniq = self.df.img_fn.drop_duplicates()

        img_fn2idx = {}
        for i, img_fn in enumerate(tqdm(img_fns_uniq, desc="Embedding image data")):
            img = Image.open(img_fn)
            VE = embed_img(img)
            VEs.append(VE.cpu())
            img_fn2idx[img_fn] = i
            # import code; code.interact(local=locals())

        t2v_index = {}
        v2t_indices = defaultdict(list)
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Embedding text data"):
            E = embed(row)
            Es.append(E.cpu())
            j = img_fn2idx[row.img_fn]
            t2v_index[i] = j
            v2t_indices[j].append(i)

        Es = torch.stack(Es)
        VEs = torch.stack(VEs)
        # Es shape: (n, 4, 512)
        # VEs shape: (m, 512)

        Es_pos = Es[:, :7, :]
        Rs = (Es_pos - root).norm(dim=-1)
        # Rs = 1 - Es @ root
        # Rs (radii) shape: (n, 4)
        rmin, rmax = Rs.min().item(), Rs.max().item()
        levels = np.linspace(rmin, rmax, self.steps)
        masks = [Rs <= thresh for thresh in levels]

        T = np.array([row2texts(row)[:7] for _, row in self.df.iterrows()])
        # T: all (pos) texts
        # shape: (n, 4)
        T_maskeds = [T[mask.cpu()] for mask in masks]

        recalls = []
        precisions = []

        for i, img_fn in enumerate(tqdm(img_fns_uniq, desc="Calculating image metrics")):
            VE = VEs[i]

            # hierarchical retrieval:
            S = Es_pos @ VE.t()
            S = S.squeeze(-1)
            # S shape: (n, 4)
            preds = []
            vals = []
            for mask, Tm in zip(masks, T_maskeds):
                idx = S[mask].argmax().item()
                t = Tm[idx]
                if len(preds) == 0 or preds[-1] != t:
                    preds.append(t)
                    vals.append(S[mask].max().item())

            t_indices = v2t_indices[i]
            gts = [T[j][:7] for j in t_indices]
            flat_gts = [t for x in gts for t in x]
            # list of lists of 4 positive texts

            s_preds = set(preds)
            s_gts = set(flat_gts)

            tp = len(s_preds & s_gts)
            fp = len(s_preds - s_gts)

            precision = tp / max(1, tp + fp)

            # multi-reference recall: check any for each level
            r1 = any(x[0] in s_preds for x in gts)
            r2 = any(x[1] in s_preds for x in gts)
            r3 = any(x[2] in s_preds for x in gts)
            r4 = any(x[3] in s_preds for x in gts)
            r5 = any(x[4] in s_preds for x in gts)
            r6 = any(x[5] in s_preds for x in gts)
            r7 = any(x[6] in s_preds for x in gts)
            recall = (r1 + r2 + r3 + r4 + r5 + r6 + r7) / 7

            precisions.append(precision)
            recalls.append(recall)

        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Calculating text metrics"):
            E = Es[i]
            radii = get_dists(E[:7]).cpu()
            R.append(radii)

        R = torch.stack(R)
        # ^ shape: (n, 4)

        ranks = R.argsort(dim=-1)
        corrs = [
            kendalltau(row.cpu().numpy(), [1, 2, 3, 4, 5, 6, 7]).correlation
            for row in ranks
        ]
        dcorr = np.mean(corrs)
        precision = np.mean(precisions)
        recall = np.mean(recalls)

        metrics = {
            'd_corr': dcorr,
            'precision': precision,
            'recall': recall
        }
        print(metrics)
        # self._report_metrics(metrics)

        print("RCME evaluation done")

if __name__ == '__main__':
    from transformers import CLIPModel, AutoProcessor
    import torch
    import pandas as pd
    from tqdm import tqdm
    from PIL import Image
    import open_clip

    clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:Srikumar26/radial-vit-b-16')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

    # clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    clip.eval().cuda()
    df = pd.read_csv('order_dataset.csv')
    # proc = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    steps = 50
    eval = Evaluate(clip, df, preprocess_val, tokenizer, device, steps)
    eval.run()