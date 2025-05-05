
# 📄 NDA Analyzer – Projet IA

## 🎯 Objectif
Construire un outil IA en ligne permettant d’analyser des contrats de confidentialité (NDA), avec :
- Résumé automatique
- Extraction des clauses critiques
- Évaluation du risque (scoring + alertes)
- Interface web simple via Streamlit
- Utilisation exclusive de **modèles open-source** ou **low-cost API** (DeepSeek + BGE)

---

## 🧱 Stack technique

| Composant      | Choix retenu                                |
|----------------|----------------------------------------------|
| UI             | Streamlit                                    |
| LLM            | DeepSeek Reasoning (via Fireworks)           |
| Embeddings     | `BAAI/bge-small-en-v1.5` (Hugging Face)      |
| Vector DB      | FAISS (local)                                |
| PDF / DOCX     | PyPDF2, pdfminer.six, python-docx            |
| OCR (optionnel)| pytesseract                                  |
| Dev Tools      | VS Code + Cursor, Docker (optionnel)         |

---

## 🗂 Structure du projet

```
nda-analyzer/
│
├── app.py                 # Streamlit front-end
├── backend/
│   ├── loader.py          # Parsing des PDF/DOCX
│   ├── embedder.py        # Embeddings avec BGE + FAISS
│   ├── analyzer.py        # Analyse avec DeepSeek
│   └── utils.py           # Fonctions auxiliaires
├── prompts/
│   ├── summarizer.txt     # Prompt de résumé
│   ├── risk_assessor.txt  # Prompt scoring
│   └── clause_extractor.txt
├── .env.example           # FIREWORKS_API_KEY=...
├── requirements.txt
└── Dockerfile             # (optionnel)
```

---

## 🧠 Fonctionnalités

### 🔼 Upload
- Prise en charge de plusieurs formats : PDF, Word
- OCR automatique si besoin (via Tesseract)

### 🔎 Analyse par LLM
- Résumé exécutif du NDA
- Extraction de clauses critiques :
  - Non-concurrence
  - Durée
  - Loi applicable
  - Tribunal compétent
  - Pénalités
  - Confidentialité excessive
- Risk-score global + 3 drapeaux (rouge/orange)
- Réponse structurée en JSON

### 🧬 Embedding + Search
- Utilisation de `bge-small-en-v1.5`
- FAISS pour recherche sémantique dans le texte

### 🖥 Interface
- Résumé, Score, Tableau des clauses
- Recherche plein texte
- Export PDF ou Markdown

---

## 🔐 Sécurité & confidentialité

- Suppression des fichiers upload après analyse
- Pas de logs de contenu utilisateur
- Possibilité d’hébergement local ou Dockerisé

---

## 🚀 Setup rapide

### 1. Cloner & installer
```bash
git clone https://github.com/ton-org/nda-analyzer
cd nda-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Créer le fichier `.env`
```env
FIREWORKS_API_KEY=sk-xxxxxx
```

### 3. Lancer
```bash
streamlit run app.py
```

---

## 🧪 À tester
- Analyse de 3 NDA avec complexité croissante
- Vérification manuelle des alertes
- Test multi-doc + performances embeddings

---

## 📌 Backlog (priorisation agile)

- [ ] 🔒 Authentification utilisateur (session)
- [ ] 🌍 Support multilingue (français / anglais)
- [ ] 🧠 Fine-tuning du prompt sur corpus d’accords
- [ ] 📊 Dashboard comparatif de plusieurs NDA
- [ ] 🤝 Mode “partage d’analyse” (lien sécurisé)

---

## 🔗 Ressources utiles

- [DeepSeek Reasoning on Fireworks](https://app.fireworks.ai/models/deepseek-ai/deepseek-llm-67b-chat)
- [bge-small-en sur HuggingFace](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Streamlit Docs](https://docs.streamlit.io/)
- [FAISS Quickstart](https://github.com/facebookresearch/faiss/wiki/Quick-Start)

---

## ✍️ Auteur / Contact

- Projet initié par Kevin AIE – I&P (Private Equity / IA Ops)
- Développement assisté par ChatGPT / Cursor / Fireworks AI
