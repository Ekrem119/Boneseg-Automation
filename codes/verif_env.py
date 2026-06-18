# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 10:17:21 2025

@author: Bicel service
"""

import subprocess, sys, os, warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

def bootstrap_env():
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

def ensure_torch():
    try:
        import torch
        print("✅ PyTorch déjà installé :", torch.__version__)
    except ImportError:
        print("Installation de PyTorch (CPU par défaut)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])

REQUIRED = {
    "numpy": ">=1.23.0",
    "scipy": ">=1.13.0",
    "scikit-image": ">=0.24.0",
    "matplotlib": ">=3.9.0",
    "tifffile": ">=2024.5.22",
    "opencv-python": ">=4.10.0",
    "Pillow": ">=10.4.0",
    "segment-anything": None,  # SAM
}

_verified_flag = os.path.join(os.path.dirname(__file__), ".env_verified")

def check_and_install():
    if os.path.exists(_verified_flag):
        print("Environnement déjà vérifié.")
        return

    print("🔍 Vérification de l'environnement SAM...")
    bootstrap_env()
    ensure_torch()

    import pkg_resources
    missing = []
    for pkg, version in REQUIRED.items():
        try:
            if version:
                pkg_resources.require(f"{pkg}{version}")
            else:
                __import__(pkg.replace("-", "_"))
        except Exception:
            missing.append(f"{pkg}{version}" if version else pkg)

    if missing:
        print("📦 Installation des dépendances manquantes...")
        subprocess.run([sys.executable, "-m", "pip", "install", *missing])
        print("✅ Mise à jour terminée.")
    else:
        print("✅ Tous les modules sont déjà à jour.")

    with open(_verified_flag, "w") as f:
        f.write("checked")

if __name__ == "__main__":
    check_and_install()
