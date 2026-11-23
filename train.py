import os
import re
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

data_root = "datasets/brats19_gen_2_t1"

# --- Resume Training Config ---
resume_training = True  # Set True to enable resume training
# Set a checkpoint path, or leave None to use the cmcd_epoch{epoch}.pth in save_dir with the largest epoch
resume_checkpoint_path = None
ckpt_prefix = "cmcd_epoch"

# ---------------- Dataset ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = PairedMRI(
    data_root,
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
def find_checkpoints(directory, prefix):
    """Return paths of {prefix}{epoch}.pth in directory, newest epoch first.

    Sorted by the epoch in the file name rather than by modification time, and other
    .pth files (e.g. old ddpm_epoch*.pth from another experiment) are ignored.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.pth$")
    found = []
    for name in os.listdir(directory):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(directory, name)))
    return [path for _, path in sorted(found, reverse=True)]


def save_checkpoint(state, path):
    # Write to a temp file first, so an interruption during saving can't leave a truncated checkpoint behind
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


start_epoch = 0
global_step = 0
resumed_from = None

if resume_training:
    if resume_checkpoint_path is not None:
        if not os.path.isfile(resume_checkpoint_path):
            raise FileNotFoundError(f"resume_checkpoint_path does not exist: {resume_checkpoint_path}")
        candidates = [resume_checkpoint_path]
    else:
        candidates = find_checkpoints(save_dir, ckpt_prefix)

    checkpoint = None
    for path in candidates:
        try:
            # Load onto CPU: load_state_dict copies the weights to the GPU anyway, and loading with
            # map_location=device would keep a second copy of the weights in GPU memory
            checkpoint = torch.load(path, map_location="cpu")
            resumed_from = path
            break
        except Exception as e:
            if resume_checkpoint_path is not None:
                raise
            print(f"Can't read checkpoint {path} ({e}), trying an older one...")

    if checkpoint is not None:
        print(f"Resuming training from {resumed_from}...")

        ckpt_data_root = checkpoint.get("data_root")
        if ckpt_data_root is not None and ckpt_data_root != data_root:
            raise ValueError(
                f"{resumed_from} was trained on {ckpt_data_root}, but data_root is {data_root}. "
                f"Use another save_dir for this experiment, or set resume_training = False."
            )

        unet.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"] + 1
        # Checkpoints saved before global_step was stored: every epoch has len(train_loader) steps
        global_step = checkpoint.get("global_step", start_epoch * len(train_loader))

        if "ema" in checkpoint:
            ema.ema_model.load_state_dict(checkpoint["ema"])
            # EMA keeps its own counters. Left at initted=False / step=0, the first ema.update()
            # overwrites the loaded EMA weights with the online model and restarts the decay warmup.
            # update_every=1, so the EMA step equals the number of optimizer steps.
            ema.initted.fill_(True)
            ema.step.fill_(checkpoint.get("ema_step", global_step))

        del checkpoint
        print(f"Successful recovery! Training will continue from epoch {start_epoch} (step {global_step}).")
    else:
        print("No valid checkpoint was found. A brand-new training will start from epoch 0.")

if start_epoch >= epochs:
    print(f"The checkpoint already finished epoch {start_epoch - 1}, and epochs = {epochs}. "
          f"Increase epochs to keep training.")
    sys.exit(0)

# ---------------- wandb ----------------
wandb.init(
    project="t1-to-t2-cmcd-whole-image",
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "timesteps": timesteps,
        "lr": lr,
        "image_size": image_size,
        "data_root": data_root,
        "resumed_from": resumed_from,
        "start_epoch": start_epoch
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

    save_checkpoint({
        "model": unet.state_dict(),
        "ema": ema.ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "ema_step": int(ema.step.item()),
        "data_root": data_root
    }, os.path.join(save_dir, f"{ckpt_prefix}{epoch}.pth"))

wandb.finish()