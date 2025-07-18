from flask import Flask, request, render_template
import os
import face_recognition
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
from group_labels import group_labels

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ENCODINGS_FILE = "encodings.pickle"
DATASET_DIR = "data"

encodings = []
labels = []


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

    print("⏳ Criando encodings a partir do dataset (primeira vez)...")
    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)
        if os.path.isdir(folder_path):
            print(f"🔹 Processando {folder}...")
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


# ✅ Cálculo da miscigenação com índice ajustado (0% = homogêneo, 100% = extremamente miscigenado)
def calcular_miscigenacao(percentages, limite=20):
    """
    Calcula um índice de miscigenação mais realista.
    - Ignora grupos com menos de 'limite'% (considerados ruído).
    - Se um grupo domina (>70%), miscigenação quase nula.
    """

    # 🔹 1. Filtrar grupos irrelevantes
    filtrados = {g: v for g, v in percentages.items() if v >= limite}

    if not filtrados:
        return 0.0

    max_grupo = max(filtrados.values())
    total = sum(filtrados.values())

    # 🔹 2. Penalizar dominância
    if max_grupo >= 70:
        return round(5 + (100 - max_grupo) * 0.1, 1)  # ~5-10% miscigenação
    if max_grupo >= 60:
        return round(10 + (100 - max_grupo) * 0.3, 1)  # ~10-20% miscigenação

    # 🔹 3. Calcular miscigenação só para casos realmente misturados
    proporcoes = [v / total for v in filtrados.values()]
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

    output_path = os.path.join(UPLOAD_FOLDER, "resultado_compartilhavel.png")
    img_final.save(output_path)
    return output_path


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            img = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(img)

            if not face_locations:
                return "Nenhum rosto detectado. Tente outra foto."

            img_enc = face_recognition.face_encodings(img, face_locations)[0]
            distances = face_recognition.face_distance(encodings, img_enc)
            similarities = 1 - distances

            # ✅ Percentuais por etnia
            group_scores = {}
            for label, sim in zip(labels, similarities):
                group_scores.setdefault(label, []).append(sim)
            percentages = {g: round(np.mean(s) * 100, 1) for g, s in group_scores.items()}

            # ✅ Agrupamento por macro-região
            macro_groups = {
                "EUROPA": ["european_"],
                "ÁSIA": ["asian_"],
                "ÁFRICA": ["african_"],
                "AMÉRICA INDÍGENA": ["indigenous_"],
                "ORIENTE MÉDIO": ["middle_east_"],
                "OCEANIA": ["oceanian_", "aboriginal_australian"]
            }

            grouped_scores = {}
            for label, score in percentages.items():
                added = False
                for macro, keywords in macro_groups.items():
                    if any(label.startswith(k) for k in keywords):
                        grouped_scores.setdefault(macro, 0)
                        grouped_scores[macro] += score
                        added = True
                        break
                if not added:
                    grouped_scores.setdefault("OUTROS", 0)
                    grouped_scores["OUTROS"] += score

            # ✅ Normalização para somar 100%
            total_score = sum(grouped_scores.values()) or 1
            normalized_scores = {m: round((v / total_score) * 100, 1) for m, v in grouped_scores.items()}
            macro_sorted = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

            # ✅ Grupos detalhados
            detailed_groups = {}
            for macro, _ in macro_sorted:
                detailed_groups[macro] = [
                    (lbl, percentages[lbl]) for lbl in percentages.keys()
                    if any(lbl.startswith(k) for k in macro_groups.get(macro, []))
                ]

            # ✅ Mapa (top 3 etnias)
            top3 = sorted(percentages.items(), key=lambda x: x[1], reverse=True)[:3]
            map_data = [
                {
                    "label": group_labels.get(g, {}).get("label", g),
                    "desc": group_labels.get(g, {}).get("desc", ""),
                    "lat": group_labels.get(g, {}).get("coords", [0, 0])[0],
                    "lon": group_labels.get(g, {}).get("coords", [0, 0])[1],
                    "score": s
                }
                for g, s in top3
            ]

            # ✅ Miscigenação
            miscigenacao = calcular_miscigenacao(percentages)

            # ✅ Imagem para compartilhamento
            img_compartilhavel = gerar_imagem_resultado(path, top3)

            return render_template(
                "result.html",
                image_path=path,
                macro_sorted=macro_sorted,
                detailed_groups=detailed_groups,
                map_data=map_data,
                top3=top3,
                miscigenacao=miscigenacao,
                group_labels=group_labels,
                img_compartilhavel=img_compartilhavel
            )

    return render_template("index.html")


if __name__ == "__main__":
    load_or_create_encodings()
    app.run(host="0.0.0.0", port=8000, debug=True)
