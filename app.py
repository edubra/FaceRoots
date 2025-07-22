from flask import Flask, request, render_template, url_for
import os
import sys
import face_recognition
import numpy as np
import pickle
import uuid
import time
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError, ImageOps
import pillow_heif  # ✅ necessário para abrir HEIC/HEIF corretamente
from group_labels import group_labels

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pickle")
DATASET_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Mantendo subrota /faceroots
app = Flask(__name__, static_url_path="/faceroots/static", static_folder="static")
app.config["APPLICATION_ROOT"] = "/faceroots"

# ✅ Registro do opener HEIF (fundamental para evitar erros)
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
        print("🔄 Carregando encodings do arquivo...")
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
            encodings = data["encodings"]
            labels = data["labels"]
        print(f"✅ {len(encodings)} rostos carregados.")
        return

    print("⏳ Criando encodings a partir do dataset...")
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
    print(f"✅ Encodings salvos ({len(encodings)} rostos).")

def compactar_imagem(input_path, max_size=1200):
    try:
        img = Image.open(input_path).convert("RGB")
       
        # ✅ Corrige automaticamente a rotação EXIF
        img = ImageOps.exif_transpose(img).convert("RGB")

        # ✅ Converte HEIC/HEIF para JPG
        if input_path.lower().endswith((".heic", ".heif")):
            new_path = input_path.rsplit(".", 1)[0] + ".jpg"
            img.save(new_path, "JPEG", quality=90)
            os.remove(input_path)
            input_path = new_path

        # ✅ Reduz tamanho para facilitar reconhecimento
        if img.width > max_size:
            ratio = max_size / float(img.width)
            new_height = int(float(img.height) * ratio)
            img = img.resize((max_size, new_height), Image.LANCZOS)

        img.save(input_path, optimize=True, quality=85)
        return input_path

    except UnidentifiedImageError:
        os.remove(input_path)
        raise ValueError("Arquivo enviado não é uma imagem válida.")

def calcular_miscigenacao(macro_percent):
    total = sum(macro_percent.values()) or 1
    proporcoes = [v / total for v in macro_percent.values()]
    diversidade = 1 - sum([p ** 2 for p in proporcoes])
    return round(diversidade * 100, 1)

def gerar_imagem_resultado(selfie_path, resultados):
    selfie = Image.open(selfie_path).convert("RGB").resize((250, 250))
    largura = 600
    altura = 400 + (len(resultados) * 40)
    img_final = Image.new("RGB", (largura, altura), (245, 243, 235))
    draw = ImageDraw.Draw(img_final)

    try:
        font_titulo = ImageFont.truetype("arial.ttf", 32)
        font_texto = ImageFont.truetype("arial.ttf", 24)
    except:
        font_titulo = ImageFont.load_default()
        font_texto = ImageFont.load_default()

    draw.text((20, 20), "FaceRoots - Suas Origens", fill=(62, 74, 44), font=font_titulo)
    img_final.paste(selfie, (20, 80))

    y_text = 80
    for grupo, score in resultados:
        label = group_labels.get(grupo, {}).get("label", grupo)
        draw.text((300, y_text), f"{label}: {score}%", fill=(74, 100, 61), font=font_texto)
        y_text += 40

    unique_result_name = f"{uuid.uuid4().hex}_resultado.png"
    output_path = os.path.join(UPLOAD_FOLDER, unique_result_name)
    img_final.save(output_path)
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
            print(f"✅ Foto recebida: {path}, tamanho: {os.path.getsize(path)/1024/1024:.2f} MB")

            try:
                path = compactar_imagem(path)  # ✅ Atualiza path se converter HEIC
            except ValueError as e:
                return str(e)

            img = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(img)

            if not face_locations:
                os.remove(path)
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
