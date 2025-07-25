from flask import Flask, request, render_template, url_for
import os
import sys
import face_recognition
import numpy as np
import pickle
import uuid
import time
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError, ImageOps, ImageEnhance, ImageFilter
import pillow_heif  # ✅ necessário para abrir HEIC/HEIF corretamente
from group_labels import group_labels

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pickle")
DATASET_DIR = os.path.join(BASE_DIR, "data")

# ✅ Detecta automaticamente ambiente (VPS ou Local)
if os.path.exists("/var/www/faceroots"):
    UPLOAD_FOLDER = "/var/www/faceroots/static/uploads"
else:
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_url_path="/static", static_folder="static")
app.config["APPLICATION_ROOT"] = "/faceroots"

# ✅ Registro do opener HEIF (fundamental para abrir HEIC)
pillow_heif.register_heif_opener()

encodings, labels = [], []


def limpar_uploads_antigos(max_age_seconds=3600):
    agora = time.time()
    for arquivo in os.listdir(UPLOAD_FOLDER):
        caminho = os.path.join(UPLOAD_FOLDER, arquivo)
        if os.path.isfile(caminho) and agora - os.path.getmtime(caminho) > max_age_seconds:
            os.remove(caminho)


def load_or_create_encodings():
    global encodings, labels
    if os.path.exists(ENCODINGS_FILE):
        print("🔄 Carregando encodings do arquivo...", flush=True)
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
            encodings = data["encodings"]
            labels = data["labels"]
        print(f"✅ {len(encodings)} rostos carregados.", flush=True)
        return

    print("⏳ Criando encodings a partir do dataset...", flush=True)
    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)
        if os.path.isdir(folder_path):
            for img in os.listdir(folder_path):
                if img.lower().endswith(("jpg", "jpeg", "png", "webp")):
                    img_path = os.path.join(folder_path, img)
                    image = face_recognition.load_image_file(img_path)
                    face_locations = face_recognition.face_locations(image)
                    if face_locations:
                        enc = face_recognition.face_encodings(image, face_locations)[0]
                        encodings.append(enc)
                        labels.append(folder)

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "labels": labels}, f)
    print(f"✅ Encodings salvos ({len(encodings)} rostos).", flush=True)


def compactar_imagem(input_path, max_size=1200):
    try:
        print(f"🔄 Abrindo imagem: {input_path}", flush=True)
        img = Image.open(input_path)
        print(f"✅ Imagem aberta - Formato: {img.format}, Tamanho: {img.size}", flush=True)

        img = ImageOps.exif_transpose(img).convert("RGB")

        # ✅ Se for HEIC/HEIF, converte para JPG antes de qualquer coisa
        if input_path.lower().endswith((".heic", ".heif")):
            new_path = input_path.rsplit(".", 1)[0] + "_converted.jpg"
            img.save(new_path, "JPEG", quality=90)
            print(f"✅ HEIC convertido para JPG: {new_path}", flush=True)
            os.remove(input_path)
            input_path = new_path

        # ✅ Reduz tamanho para facilitar reconhecimento
        if img.width > max_size:
            ratio = max_size / float(img.width)
            new_height = int(float(img.height) * ratio)
            img = img.resize((max_size, new_height), Image.LANCZOS)
            print(f"✅ Imagem redimensionada para: {max_size}x{new_height}", flush=True)

        img.save(input_path, optimize=True, quality=85)
        print(f"✅ Imagem final compactada: {input_path}, tamanho: {os.path.getsize(input_path)/1024/1024:.2f} MB", flush=True)
        return input_path

    except UnidentifiedImageError:
        print("❌ Erro: Arquivo não é uma imagem válida!", flush=True)
        if os.path.exists(input_path):
            os.remove(input_path)
        raise ValueError("Arquivo enviado não é uma imagem válida.")
    except Exception as e:
        print(f"❌ Erro inesperado ao processar imagem: {e}", flush=True)
        if os.path.exists(input_path):
            os.remove(input_path)
        raise ValueError("Erro ao processar imagem.")


def calcular_miscigenacao(macro_percent):
    total = sum(macro_percent.values()) or 1
    proporcoes = [v / total for v in macro_percent.values()]
    diversidade = 1 - sum([p ** 2 for p in proporcoes])
    return round(diversidade * 100, 1)

from PIL import ImageEnhance, ImageFilter
import math

def draw_text_with_outline(draw, text, position, font, fill, outline, outline_width=2):
    """Desenha texto com contorno (outline)"""
    x, y = position
    # Desenha o contorno (stroke)
    for ox in range(-outline_width, outline_width + 1):
        for oy in range(-outline_width, outline_width + 1):
            if ox != 0 or oy != 0:
                draw.text((x + ox, y + oy), text, font=font, fill=outline)
    # Desenha o texto principal
    draw.text((x, y), text, font=font, fill=fill)

def gerar_imagem_resultado(selfie_path, resultados):
    print(f"🖼️ Gerando imagem 1080x1920 centralizada no rosto: {selfie_path}", flush=True)

    # 1) ABRE A IMAGEM E APLICA EFEITOS
    original = Image.open(selfie_path).convert("RGB")
    original = ImageOps.exif_transpose(original)
    original = ImageEnhance.Color(original).enhance(0.01)

    # 2) DETECTA ROSTO PARA CENTRALIZAR
    img_array = np.array(original)
    face_locations = face_recognition.face_locations(img_array)

    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_center_x = (left + right) // 2
        face_center_y = (top + bottom) // 2
    else:
        print("⚠️ Nenhum rosto detectado na centralização. Usando corte central padrão.", flush=True)
        face_center_x = original.width // 2
        face_center_y = original.height // 2

    # 3) CALCULA CORTE 9:16 EM TORNO DO ROSTO
    target_w, target_h = 1080, 1920
    selfie_ratio = original.width / original.height
    target_ratio = target_w / target_h

    if selfie_ratio > target_ratio:
        new_height = original.height
        new_width = int(new_height * target_ratio)
    else:
        new_width = original.width
        new_height = int(new_width / target_ratio)

    left = max(0, face_center_x - new_width // 2)
    top = max(0, face_center_y - new_height // 2)
    right = left + new_width
    bottom = top + new_height

    if right > original.width:
        right = original.width
        left = right - new_width
    if bottom > original.height:
        bottom = original.height
        top = bottom - new_height

    selfie = original.crop((int(left), int(top), int(right), int(bottom)))
    selfie = selfie.resize((target_w, target_h), Image.LANCZOS)

    draw = ImageDraw.Draw(selfie)

    # 4) DESENHA OS PONTOS FACIAIS
    landmarks = face_recognition.face_landmarks(img_array)
    if landmarks:
        scale_x = target_w / (right - left)
        scale_y = target_h / (bottom - top)
        for face in landmarks:
            for part in ["left_eye", "right_eye", "nose_bridge", "nose_tip", "top_lip", "bottom_lip"]:
                if part in face:
                    scaled_points = [
                        ((x - left) * scale_x, (y - top) * scale_y)
                        for (x, y) in face[part]
                    ]
                    draw.line(scaled_points, fill=(0, 255, 0), width=3)

    # 5) ADICIONA LOGO (proporcional)
    logo_path = os.path.join(BASE_DIR, "static", "logo2.png")
    pos_y = selfie.height - 100
    if os.path.exists(logo_path):
        logo = Image.open(logo_path).convert("RGBA")
        logo_w = int(selfie.width * 0.35)  # antes era 0.25
        ratio = logo_w / logo.width
        logo_h = int(logo.height * ratio)
        logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
        pos_x = (selfie.width - logo_w) // 2
        pos_y = selfie.height - logo_h - int(selfie.height * 0.03)
        selfie.paste(logo, (pos_x, pos_y), logo)

    # 6) ESCREVE O TOP 3 RESULTADOS (proporcional)
    try:
        font_size = int(selfie.width * 0.2)  # antes era fixo (60)
        font_texto = ImageFont.truetype("arial.ttf", font_size)
    except:
        font_texto = ImageFont.load_default()

    y_text = pos_y - (len(resultados) * (font_texto.size + int(selfie.height * 0.01))) - int(selfie.height * 0.02)

    for grupo, score in resultados:
        label = group_labels.get(grupo, {}).get("label", grupo)
        texto = f"{label}: {score}%"
        text_w = draw.textlength(texto, font=font_texto)
        draw_text_with_outline(
            draw,
            texto,
            ((selfie.width - text_w) // 2, y_text),
            font=font_texto,
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            outline_width=int(selfie.width * 0.003)
        )
        y_text += font_texto.size + int(selfie.height * 0.01)

    # 7) SALVA
    unique_result_name = f"{uuid.uuid4().hex}_resultado.jpg"
    output_path = os.path.join(UPLOAD_FOLDER, unique_result_name)
    selfie.save(output_path, "JPEG", quality=95, optimize=True)
    print(f"✅ Imagem 1080x1920 com rosto centralizado e pontos faciais salva: {output_path}", flush=True)
    return unique_result_name


@app.route("/", methods=["GET", "POST"])
def index():
    limpar_uploads_antigos()

    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.lower().endswith(("jpg", "jpeg", "png", "webp", "heic", "heif")):
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(path)
            print(f"✅ Foto recebida: {path}, tamanho: {os.path.getsize(path)/1024/1024:.2f} MB", flush=True)

            try:
                path = compactar_imagem(path)
            except ValueError as e:
                print(f"❌ Erro ao compactar imagem: {e}", flush=True)
                return str(e)

            print(f"🔍 Carregando imagem no face_recognition: {path}", flush=True)
            img = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(img)
            print(f"✅ Face locations detectadas: {face_locations}", flush=True)

            if not face_locations:
                os.remove(path)
                print("⚠️ Nenhum rosto detectado!", flush=True)
                return "Nenhum rosto detectado. Tente outra foto."

            img_enc = face_recognition.face_encodings(img, face_locations)[0]
            distances = face_recognition.face_distance(encodings, img_enc)
            similarities = 1 - distances

            group_scores = {}
            for label, sim in zip(labels, similarities):
                group_scores.setdefault(label, []).append(sim)

            percentages = {
                g: round(np.mean(s) * 100, 1)
                for g, s in group_scores.items()
                if np.mean(s) * 100 >= 20
            }

            macro_groups = {
                "EUROPA": ["european_"],
                "ÁSIA": ["asian_"],
                "ÁFRICA": ["african_"],
                "AMÉRICA INDÍGENA": ["indigenous_"],
                "ORIENTE MÉDIO": ["middle_east_"],
                "OCEANIA": ["oceanian_", "aboriginal_australian"]
            }

            grouped_scores, detailed_groups = {}, {}
            for macro, keywords in macro_groups.items():
                etnias_macro = {g: v for g, v in percentages.items() if any(g.startswith(k) for k in keywords)}
                if etnias_macro:
                    grouped_scores[macro] = sum(etnias_macro.values())
                    detailed_groups[macro] = sorted(etnias_macro.items(), key=lambda x: x[1], reverse=True)

            total_macro = sum(grouped_scores.values()) or 1
            normalized_scores = {m: round((v / total_macro) * 100, 1) for m, v in grouped_scores.items()}
            macro_sorted = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

            miscigenacao = calcular_miscigenacao({m: v for m, v in normalized_scores.items() if v >= 5})

            top3 = sorted(percentages.items(), key=lambda x: x[1], reverse=True)[:3]
            if top3:
                maior_etnia, maior_porcentagem = top3[0]
            else:
                maior_etnia, maior_porcentagem = ("", 0)

            img_compartilhavel = gerar_imagem_resultado(path, top3)

            return render_template(
                "result.html",
                image_path=os.path.basename(path),
                macro_sorted=macro_sorted,
                detailed_groups=detailed_groups,
                miscigenacao=miscigenacao,
                group_labels=group_labels,
                img_compartilhavel=img_compartilhavel,
                top3=top3,
                maior_etnia=maior_etnia,
                maior_porcentagem=maior_porcentagem
            )

    return render_template("index.html")


load_or_create_encodings()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
#blablabla