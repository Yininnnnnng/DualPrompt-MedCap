# DualPrompt-MedCap

**DualPrompt-MedCap** is a dual-prompt enhanced medical image captioning framework that integrates:

- 🧠 **Modality-Aware Prompting** via a semi-supervised classifier
- 🧾 **Question-Guided Clinical Prompting** via semantic understanding

These prompts are injected into the **BLIP-3** backbone for improved clinical relevance and modality alignment. A novel **ground-truth-free evaluation metric** is also provided to assess medical captions without requiring reference reports.

---

## 📁 Repository Structure

```
DualPrompt-MedCap/
├── modality_classifier/           # Semi-supervised modality classifier (used to produce prompts)
│   ├── train.py
│   ├── ablation/                  # Auxiliary experiments to compare base vs attention classifier
│   └── Pretrain/                  # Save the .pth file here (download link below)
│
├── dualprompt_generator/          # BLIP-3 captioning model with injected dual prompts
│   └── dualprompt_blip3_slake.py
│
├── gtfree_eval/                   # Ground-truth-free evaluation metrics for captions
│   └── evaluate_gtfree.py
│
├── baselines/                     # Baseline models for comparison
│   ├── blip2/
│   │   └── blip2_caption_rad_slake.py
│   └── tag2text/
│       └── tag2text_caption_slake_rad.py
```

---

## 🔧 Key Components

### 🏷️ Modality Classifier (`modality_classifier/`)

This classifier is trained using FixMatch with a medical attention mechanism to recognize image modality (MRI / CT / X-ray) from the [RAD dataset](https://huggingface.co/datasets/flaviagiammarino/vqa-rad). Its output is used to guide caption generation.

```bash
cd modality_classifier
python train.py
```

> 📥 Or download pretrained model:

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1--Czotnuvo003XQHik8AVI-cCWiC3exf
# Place it at: modality_classifier/Pretrain/slake_modality_model_improved.pth
```

> 🔬 **Ablation**: To compare the base FixMatch vs attention-enhanced classifier, refer to:
> ```
> modality_classifier/ablation/
> ```
> These experiments are diagnostic only and not part of the captioning pipeline.

---

### 🧠 Caption Generation (`dualprompt_generator/`)

Injects dual prompts into the BLIP-3 captioning pipeline:  
- **Modality Prompt** ← from classifier  
- **Question Prompt** ← via PubMedBERT semantic analysis

```bash
cd dualprompt_generator
python dualprompt_blip3_slake.py
```

---

### 📊 Ground-Truth-Free Evaluation (`gtfree_eval/`)

No-reference metric for evaluating medical captions. It assesses:
- ✅ Clinical completeness
- ✅ Anatomical structure and logic
- ✅ Visual and question relevance (via BiomedCLIP)

```bash
cd gtfree_eval
python evaluate_gtfree.py
```

---

### 🧪 Baselines (`baselines/`)

Implemented for fair comparison using same datasets and settings:

| Model | Script |
|-------|--------|
| BLIP-2 | `baselines/blip2/blip2_caption_rad_slake.py` |
| Tag2Text | `baselines/tag2text/tag2text_caption_slake_rad.py` |

---

## 📦 Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

Includes:
- `transformers==4.41.1`
- `torch==2.2.1`
- `open_clip_torch`, `einops`, `spacy`, `scispacy`, etc.

> For GT-free evaluation, install the large SciSpacy model:

```bash
python -m spacy download en_core_sci_lg
```

---

## 📂 Datasets

| Dataset | Used For | Source |
|---------|----------|--------|
| **RAD** | Modality classifier training | [Hugging Face](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) |
| **SLAKE** | Captioning + evaluation | [Hugging Face](https://huggingface.co/datasets/BoKelvin/SLAKE) |

Data loading logic is embedded inside each script.

---

## 📄 Citation

If you find this work useful, please consider citing us:

```bibtex
@inproceedings{zhao2025dualprompt,
  title     = {DualPrompt-MedCap: A Dual-Prompt Enhanced Approach for Medical Image Captioning},
  author    = {Zhao, Yining and Ali, ...},
  booktitle = {To appear},
  year      = {2025}
}
```

---

## 📬 Contact

For questions or feedback, please open an [issue](https://github.com/Yininnnnnng/DualPrompt-MedCap/issues).
