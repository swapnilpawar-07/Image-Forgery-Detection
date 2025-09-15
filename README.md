# Forensic Verification to Differentiate Between Real and Fake Images

A Python toolkit for **digital image forgery detection** focused on three common scenarios:

- **Copy–Move** (regions copied within the same image) — DBSCAN-based clustering + CFA (Color Filter Array) artifacts
- **Splicing** (regions pasted from *another* image) — ELA (Error Level Analysis)
- **Double Compression** — JPEG double-encoding detection

The system outputs visual **heatmaps/masks**, quantitative **scores**, and a short **report** per image.

---

## 🔍 Why this matters

Easy-to-use editing tools make tampering common. Verifying image authenticity is vital for **law enforcement**, **multimedia platforms**, and **digital forensics**. This project offers reproducible baselines you can run locally to flag **copy–move**, **splicing**, and **double-JPEG** signatures.

---

## ✅ Pre-requisites

1. **Project folder ready** (e.g., `Forgery_Detector/`) with application files
2. **Python installed** (3.9–3.12 recommended)
3. **Python on PATH**  
   Windows: *This PC → Right Click → Properties → Advanced system settings → Environment Variables → Path → ensure Python is listed*
4. **Install dependencies** with `pip` from `requirements.txt`

---

## 📦 Installation

# 1) Clone or copy this repo into Forgery_Detector/
cd Forgery_Detector
# 2) Download CASIA2 dataset and save it in same folder
# 3) (Recommended) Create & activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4) Install dependencies
pip install -r requirements.txt
