import sys
import os

# ✅ Garante que o diretório do FaceRoots está no sys.path
sys.path.insert(0, "/var/www/faceroots")

# ✅ Ajusta o diretório atual (alguns servidores precisam disso)
os.chdir("/var/www/faceroots")

# ✅ Importa o app Flask do app.py
from app import app