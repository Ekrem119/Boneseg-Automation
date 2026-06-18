# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 10:16:56 2025

@author: Bicel service
"""

import os, sys, subprocess, urllib.request, tempfile, platform

TARGET_VERSION = (3, 12)
PYTHON_INSTALLER_URL = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"

def is_python312():
    return sys.version_info[:2] == TARGET_VERSION

def download_python_installer():
    print("⬇ Téléchargement de l’installeur Python 3.12.0 depuis python.org ...")
    tmp_path = os.path.join(tempfile.gettempdir(), "python312_installer.exe")
    urllib.request.urlretrieve(PYTHON_INSTALLER_URL, tmp_path)
    print("Téléchargement terminé :", tmp_path)
    return tmp_path

def install_python312(installer_path):
    print("Installation silencieuse de Python 3.12...")
    cmd = f'"{installer_path}" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0'
    subprocess.run(cmd, shell=True)
    print("Installation terminée.")

def main():
    print(f"Version actuelle : Python {platform.python_version()}")
    if not is_python312():
        print("⚠️ Version incompatible — installez Python 3.12 manuellement si besoin.")
        # Ici on ne relance pas automatiquement, on avertit seulement
    else:
        print("✅ Version compatible : Python 3.12 détecté.")

if __name__ == "__main__":
    main()
