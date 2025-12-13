# train.py
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from dataset import CT3DSegDataset, get_file_lists
from model import UNet3D
import numpy as np

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return 1 - ((2.*inter + smooth) / (pred.sum() + target.sum() + smooth))

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0
    for x,y in tqdm(loader):
        x = x.to(device).float()
        y = y.to(device).float()
        opt.zero_grad()
        out = model(x)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
        dloss = dice_loss(out, y)
        loss = bce + dloss
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_dice(model, loader, device):
    model.eval()
    dices = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            out = model(x)
            p = torch.sigmoid(out) > 0.5
            # per-sample dice
            p_flat = p.view(p.size(0), -1).float()
            y_flat = y.view(y.size(0), -1).float()
            inter = (p_flat * y_flat).sum(1)
            dice = (2*inter + 1e-6) / (p_flat.sum(1) + y_flat.sum(1) + 1e-6)
            dices.extend(dice.cpu().numpy().tolist())
    return float(np.mean(dices))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_list, val_list, test_list = get_file_lists("data_synth")
    train_imgs = [p for p,_ in train_list]; train_masks = [m for _,m in train_list]
    val_imgs = [p for p,_ in val_list]; val_masks = [m for _,m in val_list]

    train_ds = CT3DSegDataset(train_imgs, train_masks, augment=True)
    val_ds = CT3DSegDataset(val_imgs, val_masks, augment=False)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    model = UNet3D(in_ch=1, out_ch=1, base_ch=16).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    best_val = 0
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(1, 31):
        print(f"Epoch {epoch}")
        loss = train_one_epoch(model, train_loader, opt, device)
        val_dice = eval_dice(model, val_loader, device)
        print(f"Train loss: {loss:.4f} | Val Dice: {val_dice:.4f}")
        if val_dice > best_val:
            best_val = val_dice
            torch.save(model.state_dict(), f"checkpoints/unet3d_best.pth")
            print("Saved best model")

if __name__ == "__main__":
    main()