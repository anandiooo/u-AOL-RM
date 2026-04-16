# Temporal Causal Mental Health Chatbot (Template)

Template ini disiapkan untuk proyek riset:

"Temporal Causal Mental Health Chatbot Using NLP, Personal Knowledge Graph, and Explainable AI"

Fokus template ini adalah proof-of-concept yang:

1. Mengekstrak `emotion`, `symptom`, `trigger`, dan `mechanism` dari percakapan.
2. Membangun `Temporal Personal Causal Graph (TPCG)` yang time-aware.
3. Menyediakan visualisasi graph sebagai komponen Explainable AI (XAI).
4. Memberikan early-risk prediction berbasis rule-based (plus stub GNN).

## 1) Scope Penelitian

- Non-klinis, tidak melakukan diagnosis medis.
- Ditujukan untuk dukungan refleksi mental health mahasiswa.
- Dibuat agar mudah diperluas ke IndoBERT / model lain dan evaluasi eksperimen.

## 2) Struktur Proyek

```text
.
|- configs/
|  |- model_config.yaml
|  |- risk_rules.yaml
|- data/
|  |- raw/
|  |- processed/
|  |- sample/
|     |- sample_conversations.jsonl
|- docs/
|  |- research_protocol_template.md
|  |- questionnaire_template.md
|- outputs/
|- scripts/
|  |- run_pipeline.py
|  |- evaluate_template.py
|- streamlit_app.py
|- src/
|  |- tcmh_chatbot/
|     |- chatbot/
|     |- core/
|     |- evaluation/
|     |- graph/
|     |- nlp/
|     |- prediction/
|- tests/
|- .env.example
|- .gitignore
|- pyproject.toml
|- requirements.txt
|- requirements-ml.txt
```

## 3) Quick Start

### A. Setup environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### B. Jalankan demo pipeline

```powershell
python scripts/run_pipeline.py
```

Output:

- JSON graph di `outputs/tpcg_student_01.json`
- HTML visual XAI di `outputs/tpcg_student_01.html`

### C. Jalankan Streamlit App

```powershell
streamlit run streamlit_app.py
```

Kemudian buka URL lokal yang ditampilkan Streamlit (biasanya `http://localhost:8501`).

## 4) Fitur Dashboard Utama

- Input percakapan per turn (longitudinal)
- Ekstraksi `emotion`, `trigger`, `mechanism`, `symptom`
- Visualisasi TPCG (XAI) langsung di halaman
- Risk score + alasan utama (rule-based)
- Export graph JSON/HTML

## 5) Mapping ke SMART Metrics

Template ini sudah menyiapkan pondasi untuk evaluasi:

- Emotion detection F1 (`src/tcmh_chatbot/evaluation/metrics.py`)
- Symptom extraction precision
- Cause-effect validity
- Early warning accuracy
- User understanding rate (kuesioner)

## 6) Saran Pengembangan Lanjutan

1. Ganti lexical baseline dengan fine-tuned IndoBERT untuk emotion + entity extraction.
2. Tambahkan annotasi temporal-causal pada dataset longitudinal.
3. Implement GNN predictor dari graph sequence (mis. GAT / GraphSAGE).
4. Uji perbandingan dengan chatbot generative tanpa causal memory.
