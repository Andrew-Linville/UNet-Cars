import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
from pathlib import Path

def save_checkpoint(state, run_dir=None, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    ckpt = run_dir / filename
    torch.save(state, ckpt)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, epoch, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(f"Epoch: {epoch}")
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    
    acc = num_correct/num_pixels
    model.train()
    
    batches = max(len(loader), 1)
    return (dice_score / batches).item(), (num_correct.float() / num_pixels).item()

def save_predictions_as_imgs(
    loader, model, run_dir=None ,folder="saved_images/", device="cuda",
    save=True):
    out_dir = Path(run_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    preds_list = []
    
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        if save:
            torchvision.utils.save_image(
                preds, f"{out_dir}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{out_dir}{idx}.png")
        preds_list.append(preds)
        
    model.train()
    
    return preds_list

def TB_preds_vis(preds_list, writer, epoch):
          
    for i, t in enumerate(preds_list):
        t = t.detach().cpu()
        # pick the first item if it's a batch
        if t.ndim == 4:   # [B,C,H,W]
            t = t[0]
        if t.ndim == 3:   # [C,H,W] ok
            pass
        elif t.ndim == 2: # [H,W] -> add channel
            t = t.unsqueeze(0)
        else:
            raise ValueError("unexpected shape for image tensor")

        writer.add_image(f"val/pred_{i}", t.float(), epoch)

@torch.no_grad()
def log_val_preds_tb(
    loader,
    model,
    writer,
    epoch: int,
    device: str = "cuda",
    threshold: float = 0.5,
    max_batches: int = 2,      # how many val batches to visualize
    samples_per_batch: int = 4,# how many images per batch
    tag_prefix: str = "val/preds"
):
    model.eval()
    for b_idx, (x, y) in enumerate(loader):
        if b_idx >= max_batches:
            break

        x = x.to(device)                              # [B, C, H, W]
        logits = model(x)
        probs  = torch.sigmoid(logits)
        preds  = (probs > threshold).float()          # [B, 1 or C, H, W]

        # ensure single-channel for binary case
        if preds.ndim == 3:
            preds = preds.unsqueeze(1)                # [B,1,H,W]

        k = min(samples_per_batch, preds.size(0))
        # make a grid of k predictions (1ch each). If you want inputs/GT too, see note below.
        grid = torchvision.utils.make_grid(
            preds[:k].detach().cpu(), nrow=k, normalize=True
        )
        writer.add_image(f"{tag_prefix}/batch{b_idx}", grid, epoch)

    model.train()
