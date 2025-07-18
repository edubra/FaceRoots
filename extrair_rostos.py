import face_recognition
import cv2
import os

# Pastas
input_folder = 'Fotos'
output_folder = 'Rostos'

# Garante que a pasta de saída existe
os.makedirs(output_folder, exist_ok=True)

# Tipos de arquivo suportados
valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')

# Loop em todos os arquivos da pasta de entrada
for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_extensions):
        # Caminhos
        input_path = os.path.join(input_folder, filename)
        print(f"Processando: {input_path}")
        image = face_recognition.load_image_file(input_path)

        # Detecta rostos
        face_locations = face_recognition.face_locations(image)

        # Extrai cada rosto encontrado
        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location

            # Recorta
            face_image = image[top:bottom, left:right]

            # Salva
            output_path = os.path.join(
                output_folder, f"{os.path.splitext(filename)[0]}_face{i + 1}.jpg"
            )
            cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
            print(f"Rosto salvo em: {output_path}")
