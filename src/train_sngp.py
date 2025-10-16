import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    TB_preds_vis,
    log_val_preds_tb,
    uncert_map_TB,
)

#? Get the run directory
import argparse, os, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--logdir",  default=None)   # <-- add this
args = p.parse_args()
RUN_DIR   = Path(args.logdir)

#? Initialize TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(RUN_DIR)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 200
NUM_WORKERS = 4
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch_0=False):
    loop = tqdm(loader)
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            out = model(data)
            predictions = out[0] if isinstance(out, (tuple, list)) else out
            loss = loss_fn(predictions, targets)

        running_loss += loss.item()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    average_loss = running_loss / len(loader)
    return average_loss


@torch.no_grad()
def eval_loss(loader, model, loss_fn, device="cuda", max_batches=None, use_amp=True):
    model.eval()
    total, n = 0.0, 0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches: break
        x = x.to(device)
        y = y.float().unsqueeze(1).to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        total += float(loss.item())
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    epoch = 0
    dice_score, accuracy = check_accuracy(val_loader, model, epoch, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    
    # Initial TB Writes
    # =======================================================
    writer.add_scalars("accuracies",{"dice":dice_score, "accuracy":accuracy}, epoch)
    log_val_preds_tb(val_loader, model, writer, epoch, DEVICE)
    
    initial_train_loss = eval_loss(train_loader, model, loss_fn, device=DEVICE)
    initial_val_loss = eval_loss(val_loader, model, loss_fn, device=DEVICE)
    writer.add_scalars("losses", {"training_loss":initial_train_loss, "val_loss":initial_val_loss}, epoch)
    # =======================================================
    
    for epoch in range(NUM_EPOCHS):
        epoch = epoch+1 # fixes 0 indexing
        training_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss = eval_loss(val_loader, model, loss_fn, device=DEVICE)
        #! Send loss to TB
        writer.add_scalars("losses", {"training_loss":training_loss, "val_loss":val_loss}, epoch)
        
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        
        

        # check accuracy
        dice_score, accuracy = check_accuracy(val_loader, model, epoch, device=DEVICE)
        #! Send these to the TB
        writer.add_scalars("accuracies",{"dice":dice_score, "accuracy":accuracy}, epoch)
        
        if (epoch) % 10 == 0:
            save_checkpoint(checkpoint, run_dir=RUN_DIR)
            
        # print some examples to a folder
        # preds_list = save_predictions_as_imgs(
        #     val_loader, model, run_dir=RUN_DIR, folder="saved_images/", device=DEVICE, save = False
        # )
            
        #! call helper function to display the saved images
        # TB_preds_vis(preds_list, writer, epoch)
        log_val_preds_tb(val_loader, model, writer, epoch, DEVICE)

        # Visualize uncert
        uncert_map_TB(val_loader, model, writer, epoch, device = DEVICE)

if __name__ == "__main__":
    main()