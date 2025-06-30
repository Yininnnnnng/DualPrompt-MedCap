# DualPrompt-MedCap

**DualPrompt-MedCap** is a dual-prompt enhanced medical image captioning framework designed and implemented in **Google Colab notebooks**. It integrates:

- 🧠 **Modality-Aware Prompting** via a semi-supervised modality classifier
- 🧾 **Question-Guided Clinical Prompting** via PubMedBERT-based semantic reasoning

All components are organized into modular notebooks and designed to be executed inside Colab with **Google Drive mounted**. A novel **ground-truth-free evaluation metric** is also provided.

---

## 🗂️ Repository Structure (Notebook-based)

```
DualPrompt-MedCap/
├── modality_classifier/
│   ├── train.ipynb                   # Semi-supervised modality classifier (RAD only)
│   ├── Pretrain/                     # Place pretrained .pth file here (Google Drive)
│   └── ablation/
│       └── ablation_for_semi_supervised_classifier.ipynb

├── dualprompt_generator/
│   └── dualprompt_blip3_slake.ipynb  # Main captioning module (BLIP-3 + DualPrompt)

├── gtfree_eval/
│   └── evaluate_gtfree.ipynb         # Ground-truth-free evaluation metrics

├── baselines/
│   ├── blip2/
│   │   └── blip2_caption_rad_slake.ipynb
│   └── tag2text/
│       └── tag2text_caption_slake_rad.ipynb

├── README.md
├── requirements.txt
```

---

## 🚀 Usage Instructions (Colab Workflow)

All notebooks are designed for Google Colab. Follow these steps for each module:

### 1. Mount Google Drive

Every notebook starts with:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Please ensure your files (data, checkpoints, outputs) are correctly placed under your own Google Drive path.

---

### 2. Install Dependencies (Inside Notebook)

Each notebook begins with appropriate `!pip install` commands. You don't need to install manually via terminal.

You can optionally use:

```bash
pip install -r requirements.txt
```

to prepare your Colab or local environment ahead of time.

---

### 3. Pretrained Classifier Weights (`.pth`)

The DualPrompt captioning module relies on a pretrained modality classifier. Please download from:

🔗 https://drive.google.com/file/d/1--Czotnuvo003XQHik8AVI-cCWiC3exf/view?usp=sharing

and upload it to your Colab Drive under:

```
/content/drive/MyDrive/DualPrompt-MedCap/modality_classifier/Pretrain/slake_modality_model_improved.pth
```

Make sure the path in `dualprompt_blip3_slake.ipynb` matches your upload location.

---

## 📊 Modules

### 🔹 Modality Classifier

Train or evaluate the semi-supervised modality classifier using `train.ipynb`.  
Supports ablation between FixMatch baseline and attention-augmented version.

### 🔹 DualPrompt Generation

Run `dualprompt_blip3_slake.ipynb` to generate captions using BLIP-3 with modality + question-guided prompts.

### 🔹 Ground-Truth-Free Evaluation

Evaluate generated captions using `evaluate_gtfree.ipynb`.  
Assesses medical structure, logic, and image/question relevance—no reference reports required.

### 🔹 Baselines

Compare to:
- BLIP-2: `blip2_caption_rad_slake.ipynb`
- Tag2Text: `tag2text_caption_slake_rad.ipynb`

---

## 📂 Datasets

| Dataset | Usage | Source |
|---------|-------|--------|
| RAD     | Modality classifier only | [Hugging Face](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) |
| SLAKE   | Caption generation + evaluation | [Hugging Face](https://huggingface.co/datasets/BoKelvin/SLAKE) |

Each notebook includes its own data loader and handling logic.

---

## 📄 Citation
If you find DualPrompt-MedCap useful in your research, please consider citing our paper:

📌 Plain Text Citation
Yining Zhao, Ali Braytee, and Mukesh Prasad.
DualPrompt-MedCap: A Dual-Prompt Enhanced Approach for Medical Image Captioning.
arXiv preprint arXiv:2504.09598, 2025. https://doi.org/10.48550/arXiv.2504.09598
```bibtex
@article{zhao2025dualprompt,
  title={DualPrompt-MedCap: A Dual-Prompt Enhanced Approach for Medical Image Captioning},
  author={Zhao, Yining and Braytee, Ali and Prasad, Mukesh},
  journal={arXiv preprint arXiv:2504.09598},
  year={2025}
}
```

---

## 📬 Contact

For questions or issues, please open a [GitHub issue](https://github.com/Yininnnnnng/DualPrompt-MedCap/issues).
