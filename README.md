
# Speech Biomarkers for Physical Deficit Classification

This repository contains the computational methodology for the **submitted** paper:  
**"Spontaneous speech as a digital biomarker of multidimensional physical decline in aging"**

**Authors:**  
Éloïse Da Cunha¹²³⁴ (corresponding author), Raphaël Zory⁵⁶, Frédéric Chorin⁴, Valeria Manera¹², Auriane Gros¹²⁴

**Affiliations:**  
1. Université Côte d’Azur, Speech and Language Pathology department of Nice, Faculty of Medicine, Nice, France  
2. Université Côte d’Azur, CoBTeK Laboratory, France  
3. Université Côte d’Azur, Interdisciplinary Institute of Artificial Intelligence Côte d’Azur (3IA Côte d’Azur), Sophia Antipolis, France  
4. Centre Hospitalier Universitaire de Nice, 06000 Nice, France  
5. Université Côte d’Azur, LAMHESS, France  
6. Institut Universitaire de France (IUF), Paris, France

**Corresponding author:** Éloïse Da Cunha — [eloise.da-cunha@univ-cotedazur.fr](mailto:eloise.da-cunha@univ-cotedazur.fr)

> ⚠️ **Note**: This paper has been submitted and is currently under review. It has **not yet been published**.

---

## 🧠 Methodology Overview

This project explores how speech biomarkers can predict physical functional deficits in older adults, based on acoustic, temporal, and linguistic characteristics of spontaneous speech.

### Feature Extraction Pipeline

| Feature Type | Key Characteristics                                |
|--------------|---------------------------------------------------|
| Acoustic     | Pitch, jitter, formants, harmonics-to-noise ratio|
| Temporal     | Speech rate, pause duration and frequency          |
| Linguistic   | Lexical diversity, syntactic complexity            |

### Classification Framework

- Separate models trained on:  
  - **Positive-emotion recordings (POS)**  
  - **Negative-emotion recordings (NEG)**  
- A **stacking ensemble** integrates predictions from individual models  
- **SHAP values** used for model interpretability  

---

## 📁 Repository Structure

```
speech-aging-biomarkers/
├── features/          # Feature extraction modules
│   ├── acoustic.py
│   ├── temporal.py
│   └── linguistic.py
├── models/            # Machine learning implementation
│   ├── train_model.py
│   ├── stacking_model.py
│   └── explain_model.py
├── scripts/           # End-to-end processing pipeline
│   ├── 1_transcribe_align.py
│   ├── 2_extract_features.py
│   ├── 3_prepare_classifications.py
│   └── 4_run_modeling.py
├── utils/             # Configs and helpers
│   └── config.py
├── requirements.txt   # Python package dependencies
└── README.md          # Project documentation
```

---

## ⚙️ How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/speech-aging-biomarkers.git
cd speech-aging-biomarkers
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔒 Data and Code Availability

| Resource             | Availability                                                 |
|----------------------|--------------------------------------------------------------|
| **Clinical data**    | Not shared — protected under ethical and legal restrictions  |
| **Speech recordings**| Not publicly available                                       |
| **Code**             | Publicly available (methodology only, non-executable alone)  |

> ⚠️ **Important**: The code demonstrates the methodology used in the study, but **cannot be executed** without access to the original clinical dataset, which is not publicly shared due to privacy and ethical constraints.

---

## ✅ Ethical Compliance

- Data anonymization procedures implemented  
- Secure storage of sensitive health information  
- Use restricted to authorized and approved personnel

---

## 📬 Contact

For any questions about the methodology or code, please contact:

**Éloïse Da Cunha**  
Corresponding Author — [eloise.da-cunha@univ-cotedazur.fr](mailto:eloise.da-cunha@univ-cotedazur.fr)  
CoBTeK Laboratory, Université Côte d’Azur


```
