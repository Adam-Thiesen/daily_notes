#!/usr/bin/env python3
# transformer_kfold_training.py
# -------------------------------------------------------------
#  ▸ 5-fold stratified-group CV on slide-level RMS data
#  ▸ Outputs: loss curves, avg % confusion matrix, P/R/F1 bar-plot,
#             ROC curves, and a metric summary text file
# -------------------------------------------------------------

import os, gzip, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score,   f1_score,      confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt, seaborn as sns

# ── figure defaults ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14
})

# --------------- Dataset ---------------------------------------------------- #
class RhabdomyosarcomaDataset(Dataset):
    def __init__(self, data_dir, labels_df, start_col=8, num_features=768):
        self.data, self.labels, self.image_ids, self.tile_indices = [], [], [], []
        labels_df["slide_id"] = labels_df["slide_id"].astype(str).str.strip()

        for fname in os.listdir(data_dir):
            if not fname.endswith(".gz"):
                continue
            slide_id = fname.replace(".gz", "")
            row = labels_df[labels_df["slide_id"] == slide_id]
            if row.empty:
                print(f"[warn] no label for {fname}; skipping")
                continue

            with gzip.open(os.path.join(data_dir, fname), "rt") as f:
                df = pd.read_csv(f)

            feats = (
                df.iloc[:, start_col : start_col + num_features]
                  .apply(pd.to_numeric, errors="coerce")
                  .fillna(0)
                  .values
            )

            self.data.append(feats)
            self.labels.append(int(row["labels"].values[0]))
            self.image_ids.append(slide_id)
            self.tile_indices.append(df.index.to_list())

        self.labels = np.asarray(self.labels)

    def __len__(self):  return len(self.data)
    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx],  dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long),
                self.image_ids[idx], self.tile_indices[idx])

# --------------- Model ------------------------------------------------------ #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*div), torch.cos(pos*div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: [B,T,D]
        return x + self.pe[:x.size(1)].unsqueeze(0).to(x.device)

class SimpleTransformer(nn.Module):
    def __init__(self, in_dim=768, d_model=128, n_head=4,
                 n_layer=3, d_ff=512, drop=0.1):
        super().__init__()
        self.proj  = nn.Linear(in_dim, d_model)
        self.pos   = None
        enc_layer  = nn.TransformerEncoderLayer(d_model, n_head, d_ff, drop)
        self.enc   = nn.TransformerEncoder(enc_layer, n_layer)
        self.head  = nn.Linear(d_model, 1)
        self.drop  = nn.Dropout(drop)

    def forward(self, x):               # x: [B,T,in_dim]
        if self.pos is None or x.size(1) > self.pos.pe.size(0):
            self.pos = PositionalEncoding(self.proj.out_features, max_len=x.size(1))
        x = self.drop(self.pos(self.proj(x))).transpose(0, 1)  # [T,B,D]
        x = self.enc(x).mean(dim=0)                            # [B,D]
        return self.head(self.drop(x)).squeeze(1)              # [B]

# --------------- helpers ---------------------------------------------------- #
def train(net, loader, crit, opt, dev, epochs=75):
    net.train(); losses=[]
    for e in range(epochs):
        tot=0.0
        for x,y,*_ in loader:
            x,y=x.to(dev),y.float().to(dev)
            opt.zero_grad(); loss=crit(net(x),y); loss.backward(); opt.step()
            tot+=loss.item()
        losses.append(tot/len(loader))
        print(f"Epoch {e+1:3d}/{epochs}: loss={losses[-1]:.4f}")
    return losses

def evaluate(net, loader, dev):
    net.eval(); prob, true = [], []
    with torch.no_grad():
        for x,y,*_ in loader:
            p = torch.sigmoid(net(x.to(dev))).cpu().numpy()
            prob.extend(p); true.extend(y.numpy())
    prob = np.array(prob)
    if np.isnan(prob).any():
        raise ValueError("Model produced NaNs; check training stability.")

    pred = (prob > 0.5).astype(int)
    acc  = accuracy_score(true, pred)
    try:
        auc = roc_auc_score(true, prob)
    except ValueError:          # happens if only one class present
        auc = np.nan
    prec = precision_score(true, pred, zero_division=0)
    rec  = recall_score(true, pred,   zero_division=0)
    f1   = f1_score(true,   pred,     zero_division=0)
    fpr, tpr, _ = roc_curve(true, prob) if len(np.unique(true))==2 else (np.array([0,1]),np.array([0,1]),None)
    cm   = confusion_matrix(true, pred, labels=[0,1])
    return acc, auc, prec, rec, f1, cm, fpr, tpr

# --------------- Main ------------------------------------------------------- #
def main(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # labels
    lab = pd.read_csv("/flashscratch/thiesa/Pytorch5/labels.csv")
    lab["labels"] = lab["labels"].map({"EMBRYONAL":0, "ALVEOLAR":1})

    # dataset
    ds = RhabdomyosarcomaDataset("/flashscratch/thiesa/ctransapth_20x_features", lab)

    # safe pos_weight
    counts = np.bincount(ds.labels, minlength=2)
    if counts.min()==0:
        print("[warn] one class absent in whole dataset -> using pos_weight=1")
        pos_w = torch.tensor([1.0], device=dev)
    else:
        pos_w = torch.tensor([counts[0]/counts[1]], dtype=torch.float32, device=dev)

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # stratified-group CV
    slides = np.array(ds.image_ids)
    uniq   = np.unique(slides)
    y_uniq = np.array([ds.labels[list(slides).index(u)] for u in uniq])
    groups = lab.set_index("slide_id").loc[uniq, "patient_id"].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    losses, cms, rocs = [], [], []
    precs, recs, f1s, folds = [], [], [], []

    for fold,(tr,te) in enumerate(cv.split(uniq,y_uniq,groups=groups),1):
        print(f"\n── Fold {fold} ─────────────────────────")
        tr_idx = [i for i,s in enumerate(slides) if s in uniq[tr]]
        te_idx = [i for i,s in enumerate(slides) if s in uniq[te]]
        tr_loader = DataLoader(Subset(ds,tr_idx),batch_size=1,shuffle=True)
        te_loader = DataLoader(Subset(ds,te_idx),batch_size=1,shuffle=False)

        net = SimpleTransformer().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-5)
        losses.append(train(net,tr_loader,crit,opt,dev))

        acc,auc,pr,rc,f1,cm,fpr,tpr = evaluate(net,te_loader,dev)
        print(f"acc={acc:.3f}  auc={auc:.3f}  P={pr:.3f}  R={rc:.3f}  F1={f1:.3f}")

        folds.append((acc,auc,pr,rc,f1)); precs.append(pr); recs.append(rc); f1s.append(f1)
        cms.append(cm); rocs.append((fpr,tpr))

    # ── PLOTS (unchanged except for using the new containers) ────────────────
    # 1) loss curves
    plt.figure(figsize=(10,5))
    for i,l in enumerate(losses,1): plt.plot(l,label=f"Fold {i}")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training loss per fold",fontweight="bold")
    plt.legend(); plt.tight_layout(); plt.savefig("training_loss_curve.pdf",transparent=True,dpi=300)

    # 2) confusion matrix
    cm_pct = 100*np.mean(cms,axis=0)/np.mean(cms,axis=0).sum(axis=1,keepdims=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_pct,annot=True,fmt=".2f",cmap="Blues",
                annot_kws={"fontsize":14,"fontweight":"bold"},
                xticklabels=["Pred Embryonal","Pred Alveolar"],
                yticklabels=["True Embryonal","True Alveolar"],
                cbar_kws={"label":"% of class"})
    plt.title("Average confusion matrix across 5 folds",fontweight="bold")
    plt.tight_layout(); plt.savefig("avg_confusion_matrix_pct.pdf",transparent=True,dpi=300)

    # 3) Precision/Recall/F1 bar-plot
    metrics,means,sds = ["Precision","Recall","F1"],[np.mean(precs),np.mean(recs),np.mean(f1s)],[np.std(precs),np.std(recs),np.std(f1s)]
    plt.figure(figsize=(6,4))
    sns.barplot(x=metrics,y=means,ci=None)
    plt.errorbar(metrics,means,yerr=sds,fmt="none",capsize=5,color="black")
    plt.ylim(0,1); plt.ylabel("Score"); plt.title("Precision, Recall & F1 (mean ± SD)",fontweight="bold")
    plt.tight_layout(); plt.savefig("precision_recall_f1_bar.pdf",transparent=True,dpi=300)

    # 4) ROC curves
    mean_fpr = np.linspace(0,1,100)
    interp   = [np.interp(mean_fpr,f,t,left=0) for f,t in rocs]
    mean_tpr,std = np.mean(interp,axis=0), np.std(interp,axis=0); mean_tpr[-1]=1
    plt.figure(figsize=(6,5))
    for i,(f,t) in enumerate(rocs,1): plt.plot(f,t,alpha=0.4,lw=1,label=f"Fold {i}")
    plt.plot(mean_fpr,mean_tpr,color="blue",lw=2,
             label=f"Mean ROC (AUC = {np.nanmean([m[1] for m in folds]):.3f})")
    plt.fill_between(mean_fpr,np.maximum(mean_tpr-std,0),np.minimum(mean_tpr+std,1),
                     color="blue",alpha=0.2,label="±1 SD")
    plt.plot([0,1],[0,1],"--",color="grey",lw=1)
    plt.xlim(0,1); plt.ylim(0,1.05)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curves – 5-fold CV",fontweight="bold"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig("roc_curve_folds.png",dpi=300)

    # ── write summary ────────────────────────────────────────────────────────
    accs,aucs,prs,rcs,f1v = zip(*folds)
    with open("cross_validation_metrics2_pe.txt","w") as f:
        f.write(f"Average Accuracy : {np.mean(accs):.4f}\n")
        f.write(f"Average AUC      : {np.nanmean(aucs):.4f}\n")
        f.write(f"Average Precision: {np.mean(prs):.4f}\n")
        f.write(f"Average Recall   : {np.mean(rcs):.4f}\n")
        f.write(f"Average F1       : {np.mean(f1v):.4f}\n")
        f.write("Learning rate    : 1e-5\nFeatures         : combined_features\n")

    print("\n=== Cross-fold averages ===")
    print(f"Accuracy : {np.mean(accs):.4f}")
    print(f"AUC      : {np.nanmean(aucs):.4f}")
    print(f"Precision: {np.mean(prs):.4f}")
    print(f"Recall   : {np.mean(rcs):.4f}")
    print(f"F1       : {np.mean(f1v):.4f}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="5-fold CV RMS slide classifier")
    p.add_argument("--seed",type=int,required=True,help="Random seed")
    main(p.parse_args().seed)
