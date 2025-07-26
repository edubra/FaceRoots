#!/bin/bash
echo "🔄 [FaceRoots] Iniciando deploy automático..."

APP_DIR="/var/www/faceroots"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="faceroots"

cd $APP_DIR || { echo "❌ Erro: Pasta $APP_DIR não encontrada!"; exit 1; }

echo "🛑 Parando serviço $SERVICE_NAME..."
sudo systemctl stop $SERVICE_NAME

echo "📥 Atualizando repositório (git pull)..."
git reset --hard
git pull

echo "📦 Ativando ambiente virtual e instalando dependências..."
source $VENV_DIR/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

echo "🚀 Reiniciando serviço $SERVICE_NAME..."
sudo systemctl start $SERVICE_NAME

echo "✅ Deploy concluído! Use 'sudo systemctl status $SERVICE_NAME' para verificar."
