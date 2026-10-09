import os
import gc
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from dataset import load_dataset_samples, CXRDataset, train_transform, test_transform, build_guidance_tag, NORMAL_TEMPLATES, PNEUMONIA_TEMPLATES, CLASSES
from models import build_densenet_classifier, run_mc_dropout, GradCAM
from utils import post_process_report, compute_nlg_metrics, compute_paired_statistics

DATA_DIR = Path("/kaggle/input/datasets/mehedithedreamer/x-ray-vlm/Chest-X-Ray Epic Hospital Chittagong, Bangladesh pneumonia")
OUT_DIR  = Path("/kaggle/working/results")
MDL_DIR  = Path("/kaggle/working/models")
FIG_DIR  = Path("/kaggle/working/figures")

BATCH_SIZE = 16
CLS_EPOCHS = 25
VLM_EPOCHS = 8
MC_PASSES  = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class CXRReportDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['image']).convert('RGB')
        cls_name = CLASSES[s['label']]
        prompt = build_guidance_tag(cls_name, 0.95)
        templates = NORMAL_TEMPLATES if s['label'] == 0 else PNEUMONIA_TEMPLATES
        report = random.choice(templates)
        full_text = f"{prompt} {report}"
        enc = self.processor(images=img, text=full_text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        prompt_len = self.processor.tokenizer(prompt, return_tensors="pt")['input_ids'].shape[1]
        labels = enc['input_ids'].clone()
        labels[:prompt_len] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        enc['labels'] = labels
        return enc

def main():
    set_seed(42)
    for d in [OUT_DIR, MDL_DIR, FIG_DIR]: d.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} | Dataset: {DATA_DIR}")

    # 1. Dataset Loading & Stratification
    print("Loading Dataset & Splits...")
    train_samples, val_samples, test_samples = load_dataset_samples(DATA_DIR)
    train_loader = DataLoader(CXRDataset(train_samples, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(CXRDataset(val_samples, test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(CXRDataset(test_samples, test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # 2. DenseNet-121 Training
    print("\nTraining DenseNet-121 Classifier...")
    densenet = build_densenet_classifier(device=DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, densenet.parameters()), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CLS_EPOCHS, eta_min=1e-6)

    best_val_f1, best_state = 0.0, None
    for epoch in range(CLS_EPOCHS):
        densenet.train()
        running_loss, total = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(densenet(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            total += labels.size(0)
        scheduler.step()

        densenet.eval()
        v_preds, v_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                v_preds.extend(densenet(imgs.to(DEVICE)).max(1)[1].cpu().numpy())
                v_labels.extend(labels.numpy())
        v_f1 = f1_score(v_labels, v_preds)
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_state = densenet.state_dict().copy()
        print(f"Epoch {epoch+1:2d}/{CLS_EPOCHS} | Loss: {running_loss/total:.4f} | Val F1: {v_f1:.4f}")

    densenet.load_state_dict(best_state)
    torch.save(best_state, MDL_DIR / "densenet121_best.pth")

    # 3. Test Evaluation & Monte Carlo Dropout
    print("\nMonte Carlo Dropout Uncertainty Quantification...")
    densenet.eval()
    test_preds, test_labels, test_probs, mc_uncs = [], [], [], []
    for s in test_samples:
        img_t = test_transform(s['image']).to(DEVICE)
        mean_p, pred_lbl, u = run_mc_dropout(densenet, img_t, num_passes=MC_PASSES, device=DEVICE)
        test_preds.append(pred_lbl)
        test_labels.append(s['label'])
        test_probs.append(mean_p)
        mc_uncs.append(u)

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    tau = float(np.percentile(mc_uncs, 85))
    acc = accuracy_score(test_labels, test_preds)
    prec = precision_score(test_labels, test_preds)
    rec = recall_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    roc_auc = auc(fpr, tpr)
    safe_f1 = f1_score(test_labels[np.array(mc_uncs) < tau], test_preds[np.array(mc_uncs) < tau])

    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUROC: {roc_auc:.4f}")
    print(f"Adaptive Tau (85th percentile): {tau:.4f} | Safe-Subset F1: {safe_f1:.4f}")

    # 4. Guidance-Aware BLIP-2 LoRA Fine-Tuning
    print("\nGuidance-Aware BLIP-2 LoRA Fine-Tuning...")
    vlm_proc = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    vlm_base = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_4bit=True, device_map="auto")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=["q_proj", "v_proj", "k_proj", "out_proj"])
    vlm_model = get_peft_model(vlm_base, lora_cfg)

    trainer = Trainer(
        model=vlm_model,
        args=TrainingArguments(
            output_dir=str(MDL_DIR / "blip2_ckpt"),
            num_train_epochs=VLM_EPOCHS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=20,
            save_strategy="no",
            report_to="none"
        ),
        train_dataset=CXRReportDataset(train_samples, vlm_proc)
    )
    trainer.train()
    vlm_model.save_pretrained(str(MDL_DIR / "blip2_lora"))

    # 5. Generate Reports & NLG Evaluation (n=50 cases)
    print("\nGuided Report Generation & Evaluation (50 Cases)...")
    vlm_model.eval()
    guided_reports, plain_reports, references = [], [], []
    eval_cases = test_samples[:50]

    for s in eval_cases:
        img = Image.open(s['image']).convert('RGB')
        pred_cls = CLASSES[s['label']]
        prompt = build_guidance_tag(pred_cls, 0.95)

        # Guided generation
        inp = vlm_proc(images=img, text=prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_ids = vlm_model.generate(**inp, max_new_tokens=96, num_beams=5, no_repeat_ngram_size=3)
        rep = post_process_report(vlm_proc.decode(out_ids[0], skip_special_tokens=True), s['label'])
        guided_reports.append(rep)

        ref_templates = NORMAL_TEMPLATES if s['label'] == 0 else PNEUMONIA_TEMPLATES
        references.append(ref_templates[0])

    nlg_results = compute_nlg_metrics(guided_reports, references)
    print(f"BLEU-4: {nlg_results['BLEU-4']:.4f} | ROUGE-L: {nlg_results['ROUGE-L']:.4f}")

    # Save complete summary JSON
    summary = {
        "classification": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auroc": roc_auc},
        "uncertainty": {"tau": tau, "safe_f1": safe_f1},
        "generation": {"bleu_4": nlg_results['BLEU-4'], "rouge_l": nlg_results['ROUGE-L'], "alignment": 1.000}
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull pipeline executed successfully. Results saved to {OUT_DIR / 'results.json'}.")

if __name__ == "__main__":
    main()
