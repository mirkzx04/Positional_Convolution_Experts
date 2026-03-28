import os
import json
from PIL import Image
from torch.utils.data import Dataset

class InferenceDataset(Dataset):
    def __init__(self, images_dir, annotations_file, mapping_json_path, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []

        with open(mapping_json_path, "r") as f:
            self.class_to_idx = json.load(f)

        self._load_annotations(annotations_file)

    def _load_annotations(self, annotations_file):
        with open(annotations_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                image_name = parts[0]
                class_name = parts[1]

                if class_name not in self.class_to_idx:
                    continue

                label = self.class_to_idx[class_name]
                image_path = os.path.join(self.images_dir, image_name)

                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label