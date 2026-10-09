# Classifier Guided Report Generation with Uncertainty Gating for Chest Radiographs in a Bangladeshi Hospital Cohort

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-BLIP--2-orange)](https://huggingface.co/Salesforce/blip2-opt-2.7b)

Official source code and clinical evaluation artifacts for the paper:  
**"Classifier Guided Report Generation with Uncertainty Gating for Chest Radiographs in a Bangladeshi Hospital Cohort"**  
*Discover Artificial Intelligence* (Springer Nature), 2026.

---

## 📌 Overview

Automated chest radiograph interpretation tools often achieve high classification accuracy, yet vision-language models (VLMs) that generate free-text radiology reports remain susceptible to hallucination, weakly coupled to the underlying image decision, and rarely benchmarked on degraded images typical of low-resource emergency departments.

This repository implements a **guided hybrid pipeline** evaluated on a single-centre Bangladeshi cohort of 2,626 chest radiographs from **Epic Hospital (Chittagong)**:
1. **DenseNet-121 Classifier:** Detects pneumonia with **92.2% accuracy**, **0.992 recall**, and **0.972 AUROC** on clean test images, maintaining **0.912 F1** under simulated degradation (noise, blur, low contrast).
2. **Bayesian Uncertainty Quantification:** Uses Monte Carlo (MC) Dropout ($T=30$ forward passes) to estimate epistemic predictive uncertainty $u(x)$. An adaptive 85th-percentile threshold ($\tau=0.0647$) distinguishes auto-acceptable reports from high-uncertainty cases flagged for radiologist review (safe-subset F1 = **0.965**).
3. **Guidance-Aware BLIP-2 with LoRA:** Exposes the classifier's pathology decision via an in-prompt guidance tag (`[CLS=..., conf=...]`), producing reports **100% faithful to the classifier's decision (alignment = 1.000)** and raising **BLEU-4 to 0.800** (+76.6% relative gain) and **BERTScore F1 to 0.968** ($p < 0.01$, Cohen's $d > 1.3$).
4. **Explainable AI (Grad-CAM):** Extracts visual attention heatmaps from `denseblock4`, embedded directly into dual-format (HTML & PDF) clinical reports.
5. **Human Clinical Validation:** Includes a self-contained 50-case HTML radiologist review tool yielding median clinical accuracy of **5 [IQR: 4–5]** and safety of **4 [IQR: 4–5]** on 5-point Likert scales, with an **88% hallucination-free rate**.

---

## 📂 Repository Structure

```text
├── run_pipeline.py                   # Modular end-to-end execution script
├── dataset.py                        # Dataset loading, stratified 20% validation split, & templates
├── models.py                         # DenseNet-121 classifier, MC Dropout, Grad-CAM, & BLIP-2 LoRA
├── utils.py                          # NLG metrics (BLEU, BERTScore), statistical tests, & post-processing
├── doctor_survey/                    # Self-contained 50-case radiologist evaluation web package
│   ├── index.html                    # Interactive browser survey interface
│   └── cases/                        # Embedded HTML/CSS case reports with Likert rating forms
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # Repository documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/mehedimasum/cxr-guided-blip2.git
cd cxr-guided-blip2
pip install -r requirements.txt
```

### 2. Dataset Preparation
The chest X-ray images were obtained from the publicly accessible Mendeley Data archive:
* **Dataset:** Epic Hospital Chittagong Chest X-Ray Archive (Normal and Pneumonia)
* **DOI:** [10.17632/wndbd5r26y.2](https://doi.org/10.17632/wndbd5r26y.2)
* **Citation:** Epic Hospital Chittagong (2024), Mendeley Data, V2.

Organise the dataset in your workspace or Kaggle input:
```text
Chest-X-Ray Epic Hospital Chittagong, Bangladesh pneumonia/
├── Training/
│   ├── Normal/
│   └── Pneumonia/
└── Testing/
    ├── Normal/
    └── Pneumonia/
```

### 3. Running the Pipeline
You can run the full pipeline in this way:

```bash
python run_pipeline.py
```
Preconfigured to run seamlessly within the Kaggle environment.

---

## 📊 Key Results

### Classification Performance (Clean vs. Degraded)
| Partition | Accuracy | Precision | Recall | F1-Score | AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Test Set (513 images)** | **0.922** | **0.870** | **0.992** | **0.927** | **0.972** |
| **Simulated Degraded (513 images)** | **0.905** | **0.842** | **0.996** | **0.912** | — |
| **Safe-Subset (Auto-Accepted, 85%)** | **0.966** | **0.938** | **0.995** | **0.965** | — |

### Report Generation Ablation (Epic Chittagong Cohort, $n=50$)
| Method | BLEU-4 | BERTScore F1 | CheXbert-14 F1 | Alignment |
| :--- | :---: | :---: | :---: | :---: |
| BLIP-2 (Zero-shot) | 0.000 | — | — | — |
| FlanT5-XL (Zero-shot) | 0.000 | — | — | — |
| BLIP-2 + LoRA (Plain) | 0.453 | 0.918 | — | 0.520 |
| **BLIP-2 + LoRA + Guidance (Ours)** | **0.800** | **0.968** | **0.325$^\dagger$** | **1.000** |

*$^\dagger$ Macro-F1 reflects only 5 clinically active labels; micro-F1 across active labels is 0.911.*  
*Paired statistical comparisons (LoRA vs. Guided LoRA) confirmed significance at $p < 0.01$ (Paired $t$-test $t(49)=6.72$, Wilcoxon $W=120$, Cohen's $d > 1.30$).*

### Radiologist Evaluation (50 Cases)
* **Clinical Accuracy (1–5 Likert):** Median **5 [IQR: 4–5]** (Mean: $4.58 \pm 0.97$)
* **Clinical Safety (1–5 Likert):** Median **4 [IQR: 4–5]** (Mean: $4.28 \pm 0.88$)
* **Hallucination-Free Rate:** **88% (44/50)**
* All 4 major hallucinations stemmed from upstream classifier false positives (normal misclassified as pneumonia), proving that the guidance tag eliminates stochastic generative drift and converts hallucination risk into an auditable function of classifier accuracy.

---

## 🩺 Doctor Evaluation Package

To inspect the clinical survey package evaluated by radiologists:
1. Navigate to the `doctor_survey/` folder.
2. Open `index.html` in any web browser.
3. Review the radiograph, classifier prediction, MC Dropout uncertainty gate status, Grad-CAM attention map, and generated findings/impression with interactive 5-point Likert rating forms.

---

## 📜 Citation

If you use this codebase, models, or methodology in your research, please cite our paper:

```bibtex
@article{Masum2026CXRGuidedBLIP2,
  title={Classifier-Conditioned BLIP-2 with Adaptive Uncertainty Thresholding for Faithful Chest X-Ray Report Generation in Low-Resource Settings},
  author={Masum, Mohammed Mehedi and Chowdhury, Mahfuzulhoq},
  journal={Discover Artificial Intelligence},
  year={2026},
  publisher={Springer Nature},
  doi={10.1007/s44163-026-XXXXX-X}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
