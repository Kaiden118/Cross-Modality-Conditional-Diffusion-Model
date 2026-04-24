import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from cross_modality_conditional_diffusion import Unet, GaussianDiffusion
from dataset import PairedMRI

os.environ["TQDM_DISABLE"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Config ----------------
checkpoint_path = "checkpoints/ddpm_epoch.pth"
save_dir = "results/generated"
os.makedirs(save_dir, exist_ok=True)

timesteps = 1000
cfg_scale = 1.2
max_slices = None

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = PairedMRI(
    "datasets/brats19_gen_2",
    phase="test",
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

sample = test_dataset[0]["t1"]
image_size = tuple(sample.shape[-2:])

# ---------------- Model ----------------
unet = Unet(
    dim=128,
    channels=1,
    cond_channels=1,
    dim_mults=(1, 2, 4, 8),
    use_cross_attn=False
).to(device)

state = torch.load(checkpoint_path, map_location=device)
unet.load_state_dict(state["model"])

sampler_model = unet
if "ema" in state:
    ema_model = Unet(
        dim=128,
        channels=1,
        cond_channels=1,
        dim_mults=(1, 2, 4, 8),
        use_cross_attn=False
    ).to(device)
    ema_model.load_state_dict(state["ema"])
    sampler_model = ema_model
    print("Using EMA model.")

sampler_model.eval()

diffusion = GaussianDiffusion(
    sampler_model,
    image_size=image_size,
    timesteps=timesteps,
    objective="pred_noise",
    auto_normalize=False,
).to(device)


def denorm01(x):
    return (x.clamp(-1, 1) + 1) / 2


torch.set_grad_enabled(False)

count = 0

for batch in test_loader:
    t1 = batch["t1"].to(device)
    t2 = batch["t2"].to(device)

    if "t1_name" in batch:
        in_name = batch["t1_name"][0]
    elif "filename" in batch:
        in_name = batch["filename"][0]
    elif "slice_name" in batch:
        in_name = batch["slice_name"][0]
    elif "slice_id" in batch:
        slice_id = batch["slice_id"].item()
        in_name = f"sample_{slice_id}"
    else:
        in_name = f"sample_{count + 1:04d}"

    in_name = os.path.splitext(in_name)[0]
    if in_name.endswith(".nii"):
        in_name = os.path.splitext(in_name)[0]

    out_name = f"{in_name}_t2.png"

    # ===== The entire graph T1 -> T2 is generated =====
    pred = diffusion.sample(x_cond=t1, batch_size=1, cond_scale=cfg_scale)

    pred_01 = denorm01(pred)

    save_image(pred_01, os.path.join(save_dir, out_name))

    count += 1

    if (max_slices is not None) and (count >= max_slices):
        break

print("Done.")