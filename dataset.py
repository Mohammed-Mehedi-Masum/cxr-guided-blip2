import os
import random
import re
from pathlib import Path
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms

BASE_PATH = Path("/kaggle/input/datasets/mehedithedreamer/x-ray-vlm/Chest-X-Ray Epic Hospital Chittagong, Bangladesh pneumonia")
IMG_SIZE = 224
CLASSES = ['Normal', 'Pneumonia']

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, -2.5, 2.5)

portable_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=1.5))),
    transforms.ColorJitter(contrast=0.7),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    AddGaussianNoise(std=0.05)
])

class CXRDataset(Dataset):
    def __init__(self, samples, transform=test_transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['image']).convert('RGB')
        return self.transform(img), s['label']

def load_dataset_samples(base_dir=BASE_PATH):
    def _get_files(split):
        items = []
        for lbl, name in enumerate(['Normal', 'Pneumonia']):
            p = base_dir / split / name
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for f in p.glob(ext):
                    items.append({'image': str(f), 'label': lbl})
        random.shuffle(items)
        return items

    raw_train = _get_files("Training")
    test_data = _get_files("Testing")

    normals = [s for s in raw_train if s['label'] == 0]
    pneumonias = [s for s in raw_train if s['label'] == 1]

    val_n = int(len(normals) * 0.20)
    val_p = int(len(pneumonias) * 0.20)

    val_data = normals[:val_n] + pneumonias[:val_p]
    train_data = normals[val_n:] + pneumonias[val_p:]

    random.shuffle(train_data)
    random.shuffle(val_data)
    return train_data, val_data, test_data

NORMAL_TEMPLATES = [
    "FINDINGS: Normal chest radiograph. Clear lung fields bilaterally. No focal consolidation, pleural effusion, or pneumothorax. No atelectasis or edema. Heart size and mediastinal contours within normal limits. No acute bony abnormality. No support devices identified.\nIMPRESSION: No acute cardiopulmonary abnormality. Normal study.",
    "FINDINGS: The lungs are clear bilaterally without evidence of focal consolidation, effusion, or pneumothorax. No atelectasis or pulmonary edema. Cardiomediastinal silhouette is within normal limits. Osseous structures are intact. No support devices.\nIMPRESSION: Normal chest radiograph. No acute cardiopulmonary process.",
    "FINDINGS: Both lung fields are well aerated with no areas of consolidation or collapse. No pleural effusion or pneumothorax identified. No edema or vascular congestion. Heart size is normal. The mediastinum is unremarkable. No acute osseous abnormality. No fracture.\nIMPRESSION: Unremarkable chest X-ray. No evidence of acute disease."
]

PNEUMONIA_TEMPLATES = [
    "FINDINGS: Airspace opacity in the right lower lobe consistent with consolidation. Air bronchograms present. Mild perihilar infiltrates bilaterally. No significant pleural effusion. Heart size normal. Findings consistent with pneumonia.\nIMPRESSION: Right lower lobe pneumonia. Recommend appropriate antibiotic therapy and clinical follow-up in 4-6 weeks.",
    "FINDINGS: Focal consolidation in the left lower lobe with associated air bronchograms. Mild left-sided perihilar haziness. No large pleural effusion or pneumothorax. Heart size is within normal limits. No acute osseous abnormality.\nIMPRESSION: Left lower lobe pneumonia. Clinical correlation and follow-up chest radiograph in 4-6 weeks recommended.",
    "FINDINGS: Bilateral patchy airspace opacities involving the lower zones, more prominent on the right. Air bronchograms noted. No significant pleural effusion identified. Cardiac silhouette is normal. No pneumothorax. Osseous structures appear intact.\nIMPRESSION: Bilateral pneumonia, predominantly right-sided. Antibiotic therapy and interval imaging advised."
]

_AUGMENT_SYNONYMS = {
    'consolidation': ['consolidation', 'dense opacity', 'airspace disease'],
    'clear': ['clear', 'well-aerated', 'lucent'],
    'normal': ['normal', 'within normal limits', 'unremarkable'],
    'pneumonia': ['pneumonia', 'infectious process', 'pulmonary infection'],
    'infiltrates': ['infiltrates', 'opacities', 'parenchymal changes'],
}

def augment_template(template):
    text = template
    for word, synonyms in _AUGMENT_SYNONYMS.items():
        if word in text.lower():
            rep = random.choice(synonyms)
            text = re.sub(re.escape(word), rep, text, count=1, flags=re.IGNORECASE)
    return text

def build_guidance_tag(cls_name, confidence):
    return f"[CLS={cls_name}, conf={confidence:.2f}] Describe this chest X-ray."
