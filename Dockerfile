# ✅ 1️⃣ Utilisation d'une image optimisée pour PyTorch
FROM continuumio/miniconda3
# ✅ 2️⃣ Définition du répertoire de travail
WORKDIR /app

# ✅ 3️⃣ Copie des fichiers nécessaires
COPY . .

# ✅ 4️⃣ Installation des dépendances
RUN pip install -r ./requirements.txt

# ✅ 6️⃣ Lancement de l'entraînement
CMD ["python", "fintetune.py"]
