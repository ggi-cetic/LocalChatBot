FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements et le wheel local
COPY requirements.txt .
COPY ddgs_fork-1.0.0-py3-none-any.whl .

# Installer les dépendances Python et le wheel local
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir ddgs_fork-1.0.0-py3-none-any.whl

# Copier le reste de l'application
COPY . .

# Créer le dossier history s'il n'existe pas
RUN mkdir -p /app/history

# Exposer le port Flask
EXPOSE 6060

# Variables d'environnement pour Ollama
ENV OLLAMA_HOST=host.docker.internal:11434

# Lancer l'application
CMD ["python", "chatLlama3.py"]
