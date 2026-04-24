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

# ---------------- PSNR ----------------
def calc_psnr(pred, target, eps=1e-8):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(1.0 / (mse + eps))

# ---------------- SSIM ----------------
def gaussian(window_size, sigma, device):
    x = torch.arange(window_size, device=device).float()
    gauss = torch.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
    return gauss / gauss.sum()


def create_window(window_size, channel, device):
    _1d = gaussian(window_size, 1.5, device).unsqueeze(1)
    _2d = _1d @ _1d.t()
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calc_ssim(img1, img2, window_size=11):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel, img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


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

psnr_total = 0.0
ssim_total = 0.0
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

    t2_01 = denorm01(t2)
    pred_01 = denorm01(pred)

    psnr = calc_psnr(pred_01, t2_01)
    ssim = calc_ssim(pred_01, t2_01)

    psnr_total += psnr.item()
    ssim_total += ssim.item()
    count += 1

    save_image(pred_01, os.path.join(save_dir, out_name))

    print(f"[{count}] {out_name} | PSNR {psnr:.2f} | SSIM {ssim:.4f}")
    print(f"Current Average -> PSNR: {psnr_total / count:.2f}, SSIM: {ssim_total / count:.4f}")

    if (max_slices is not None) and (count >= max_slices):
        break

# print("\n====== FINAL RESULT ======")
# if count > 0:
#     print(f"Evaluated slices: {count}")
#     print(f"Average PSNR: {psnr_total / count:.2f}")
#     print(f"Average SSIM: {ssim_total / count:.4f}")
# else:
#     print("No slice was evaluated.")
print("Done.")