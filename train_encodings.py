import face_recognition
import os
import pickle

# Caminho para a pasta de dados
DATASET_DIR = "data"
ENCODINGS_FILE = "encodings.pickle"

encodings = []
labels = []
image_paths = []

# Percorre cada pasta (grupo)
for group in os.listdir(DATASET_DIR):
    group_path = os.path.join(DATASET_DIR, group)
    
    if not os.path.isdir(group_path):
        continue
    
    print(f"Processando grupo: {group}")

    for image_name in os.listdir(group_path):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        image_path = os.path.join(group_path, image_name)
        try:
            image = face_recognition.load_image_file(image_path)
            face_encs = face_recognition.face_encodings(image)

            if len(face_encs) > 0:
                encodings.append(face_encs[0])
                labels.append(group)
                image_paths.append(image_path)  # ✅ Salva o caminho da foto
        except Exception as e:
            print(f"Erro com {image_path}: {e}")

# Salva tudo em um pickle
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(
        {"encodings": encodings, "labels": labels, "paths": image_paths}, f
    )

print(f"\n✅ Treinamento concluído!")
print(f"Total de rostos processados: {len(encodings)}")
print(f"Arquivo salvo: {ENCODINGS_FILE}")