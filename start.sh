#!/bin/bash

# Quantum Trading System - Installation Universelle
# Ce script configure l'environnement local ou production

set -e

echo "🌌 Bienvenue dans le Quantum Trading System Installer"
echo "==================================================="

# Vérification des pré-requis
command -v python3 >/dev/null 2>&1 || { echo >&2 "Python 3 est requis mais n'est pas installé. Aborting."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo >&2 "Node.js (npm) est requis mais n'est pas installé. Aborting."; exit 1; }

echo "[1/4] Copie du fichier d'environnement..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Fichier .env créé. Pensez à configurer vos clés API."
else
    echo "Fichier .env déjà présent."
fi

# Demander à l'utilisateur le mode de lancement
echo ""
echo "Choisissez le mode de déploiement :"
echo "1) Mode Local (SQLite, Rapide, Sans Docker)"
echo "2) Mode Production (PostgreSQL, Redis, Docker)"
read -p "Sélectionnez une option (1 ou 2) : " deploy_mode

if [ "$deploy_mode" = "2" ]; then
    echo "[2/4] Vérification de Docker..."
    command -v docker-compose >/dev/null 2>&1 || { echo >&2 "docker-compose est requis pour le mode production. Aborting."; exit 1; }
    
    echo "[3/4] Construction et lancement des conteneurs..."
    docker-compose up -d --build
    
    echo "==================================================="
    echo "✅ Déploiement Production Réussi !"
    echo "API: http://localhost:8000"
    echo "Frontend: http://localhost:3000"
    echo "==================================================="
else
    echo "[2/4] Configuration de l'environnement Python local..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo "[3/4] Initialisation de la Base de Données (SQLite)..."
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    python scripts/init_db.py
    python scripts/academy_init.py
    
    echo "[4/4] Configuration du Frontend..."
    cd frontend
    npm install
    
    echo "==================================================="
    echo "✅ Installation Locale Réussie !"
    echo "Pour lancer le système :"
    echo "Term 1 (Backend): source venv/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)/src && uvicorn quantum.infrastructure.api.main:app --reload"
    echo "Term 2 (Frontend): cd frontend && npm run dev"
    echo "==================================================="
fi
