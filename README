# 📘 README - Fine-tuning d'un modèle LLaMA 3 avec LoRA  

## 📌 Description  
Ce projet permet d'affiner (`fine-tuner`) un modèle **LLaMA 3.2-1B-Instruct** en utilisant **LoRA** (Low-Rank Adaptation). L'entraînement est réalisé sur un dataset de conversations, représentant des interactions avec **Quera**, une agence spécialisée en Web et Data.  

L'objectif est d'entraîner un modèle capable de répondre de manière pertinente aux utilisateurs en utilisant le contexte de l'agence.  

---

## 📂 Structure du projet  

```bash
📦 projet-finetune-llama3
├── 📂 data
│   ├── dataset.jsonl        # Dataset des conversations pour l'entraînement
├── fintetune.py             # Script principal d'entraînement du modèle
├── train.sh                 # Script pour la conversion et le déploiement du modèle
├── requirements.txt         # Liste des dépendances Python
└── README.md                # Documentation du projet
```

---

## 📜 Données d'entraînement (`data/dataset.jsonl`)  

Le dataset est un fichier **JSONL** contenant des dialogues entre un utilisateur et l'assistant **Quera**. Voici un exemple :  

```json
{
  "messages": [
    {"role": "system", "content": "Vous êtes Quera, une agence spécialisée en Web et Data, située à Paris."},
    {"role": "user", "content": "Quelles sont vos expertises ?"},
    {"role": "assistant", "content": "Nous sommes experts en création de sites WordPress, SEO, Data Science, IA et développement d'APIs."}
  ]
}
```

---

## 🚀 Installation et Environnement  

### 1️⃣ Cloner le projet  
```bash
git clone https://github.com/votre-repo/projet-finetune-llama3.git
cd projet-finetune-llama3
git clone https://github.com/ggerganov/llama.cpp.git
```

### 2️⃣ Installer les dépendances  
Créer un environnement virtuel (recommandé) et installer les bibliothèques nécessaires :  
```bash
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
venv\Scripts\activate     # Sur Windows

pip install -r requirements.txt
```

---

## 🔧 Entraînement du modèle (`fintetune.py`)  

Le script **`fintetune.py`** effectue les étapes suivantes :  
✅ **Chargement du dataset** (fichier JSONL)  
✅ **Prétraitement des données** (conversion des conversations en texte formaté)  
✅ **Tokenization avec un modèle LLaMA 3**  
✅ **Fine-tuning avec LoRA**  
✅ **Sauvegarde du modèle ajusté**  
✅ **Fusion des poids LoRA avec le modèle principal**  

### 🔥 Lancer l'entraînement  
```bash
python fintetune.py
```

⏳ L'entraînement peut prendre du temps selon votre machine.  

---

## 🏗️ Conversion et Déploiement (`train.sh`)  

Une fois le modèle entraîné et fusionné, vous pouvez le convertir au format **GGUF** pour l'utiliser avec **Ollama** :  

```bash
bash train.sh
```

Ce script exécute :  
1. **Conversion du modèle** en GGUF via `convert_hf_to_gguf.py`  
2. **Création du modèle Ollama** avec `ollama create`  

---

## 📌 Dépendances (`requirements.txt`)  

Le projet utilise les bibliothèques suivantes :  

```
torch
transformers
datasets
peft
accelerate
tensorboard
```

Installez-les avec :  
```bash
pip install -r requirements.txt
```

---

## 📬 Contact  

📧 **Kevin Duranty** - [kevin.duranty@quera.fr](mailto:kevin.duranty@quera.fr)  
🌍 **Agence Quera** - [www.quera.fr](https://www.quera.fr)  

🚀 *La technologie au service du développement des activités humaines.*