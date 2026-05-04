import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb
from cross_modality_conditional_diffusion import Unet, GaussianDiffusion
from dataset import PairedMRI
from ema_pytorch import EMA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------- Config ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

epochs = 50
batch_size = 8
timesteps = 1000
lr = 1e-4

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

# --- Resume Training Config ---
resume_training = True  # Set True to enable resume training
# Set name of the path, or this will find the latest one, which is cmcd_epoch{epoch}.pth
resume_checkpoint_path = None 

# ---------------- Dataset ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = PairedMRI(
    "datasets/brats19_gen_2_t1",
    phase="train",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# ---------------- Model (ACSG Module Core) ----------------
unet = Unet(
    dim=128,
    channels=1,
    cond_channels=1,
    dim_mults=(1, 2, 4, 8),
    use_cross_attn=False
).to(device)

sample = train_dataset[0]["t1"]
image_size = tuple(sample.shape[-2:])

diffusion = GaussianDiffusion(
    unet,
    image_size=image_size,
    timesteps=timesteps,
    objective="pred_noise",
    auto_normalize=False,
    cond_drop_prob=0.2
).to(device)

optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

# ---------------- EMA ----------------
ema = EMA(unet, beta=0.995, update_every=1)
ema.to(device)

# ---------------- Resume Logic ----------------
start_epoch = 0
global_step = 0

if resume_training:
    if resume_checkpoint_path is None:
        ckpt_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
        if ckpt_files:
            latest_ckpt = max([os.path.join(save_dir, f) for f in ckpt_files], key=os.path.getmtime)
            resume_checkpoint_path = latest_ckpt

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"检测到历史检查点，正在从 {resume_checkpoint_path} 恢复训练...")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        
        unet.load_state_dict(checkpoint["model"])
        if "ema" in checkpoint:
            ema.ema_model.load_state_dict(checkpoint["ema"])
        
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        start_epoch = checkpoint["epoch"] + 1
        global_step = start_epoch * len(train_loader)
        print(f"Successful recovery! Training will continue from the {start_epoch} round.")
    else:
        print("No valid checkpoint was found. A brand-new training will start from Round 0.")

# ---------------- wandb ----------------
wandb.init(
    project="t1-to-t2-cmcd-whole-image",
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "timesteps": timesteps,
        "lr": lr,
        "image_size": image_size,
        "resumed": resume_training
    }
)
wandb.watch(unet, log="all", log_freq=200)

# ---------------- Training Loop ----------------
unet.train()

for epoch in range(start_epoch, epochs):
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)

        t = torch.randint(0, timesteps, (t2.size(0),), device=device).long()
        loss = diffusion.p_losses(t2, t, x_cond=t1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        ema.update()

        epoch_loss += loss.item()
        wandb.log({"loss": loss.item()}, step=global_step)
        global_step += 1
        pbar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} | avg_loss: {avg_loss:.4f}")
    wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch})

    torch.save({
        "model": unet.state_dict(),
        "ema": ema.ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, os.path.join(save_dir, f"cmcd_epoch{epoch}.pth"))

wandb.finish()