import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class PairedMRI(Dataset):
    def __init__(
        self,
        root,
        phase="train",
        transform=None,
        min_intensity=5
    ):
        assert phase in ["train", "test", "val"]
        self.phase = phase
        self.transform = transform

        self.t1_dir = os.path.join(root, phase, "A")
        self.t2_dir = os.path.join(root, phase, "B")

        self.valid_pairs = []
        all_files = sorted(os.listdir(self.t1_dir))

        for t1_name in all_files:
            t2_name = t1_name.replace("_t1_", "_t2_")
            t1_path = os.path.join(self.t1_dir, t1_name)
            t2_path = os.path.join(self.t2_dir, t2_name)

            if not os.path.exists(t2_path):
                continue

            try:
                t1 = np.array(Image.open(t1_path).convert("L"))
                t2 = np.array(Image.open(t2_path).convert("L"))

                if t1.max() > min_intensity and t2.max() > min_intensity:
                    self.valid_pairs.append((t1_name, t2_name))
            except Exception as e:
                print(f"Skip the damaged image {t1_name}: {e}")

        sample_img = Image.open(os.path.join(self.t1_dir, self.valid_pairs[0][0])).convert("L")
        self.image_w, self.image_h = sample_img.size

        print(
            f"[PairedMRI-{phase}] slices={len(self.valid_pairs)}, "
            f"image_size=({self.image_h},{self.image_w})"
        )

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        t1_name, t2_name = self.valid_pairs[idx]

        t1 = Image.open(os.path.join(self.t1_dir, t1_name)).convert("L")
        t2 = Image.open(os.path.join(self.t2_dir, t2_name)).convert("L")

        if self.transform:
            t1 = self.transform(t1)
            t2 = self.transform(t2)

        return {
            "t1": t1,
            "t2": t2,
            "slice_id": idx,
            "t1_name": t1_name,
            "t2_name": t2_name,
        }