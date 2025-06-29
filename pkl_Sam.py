import os
import pickle
import numpy as np
from PIL import Image

# Para procesar las imágenes y las máscaras
def procesar_datos(ruta_imagenes_origen, ruta_mascaras_origen, nombre_salida, i):
    datos_diccionario = []
    # Obtener las listas con nombres de imágenes y máscaras
    nombres_imgs = sorted(os.listdir(ruta_imagenes_origen))
    nombres_masks = sorted(os.listdir(ruta_mascaras_origen))
    # Crear directorio de salida
    output_dir = os.path.dirname(nombre_salida)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Iterar sobre las imágenes y máscaras
    for nombre_img, nombre_mask in zip(nombres_imgs, nombres_masks):
        ruta_origen_img = os.path.join(ruta_imagenes_origen, nombre_img)
        ruta_origen_mask = os.path.join(ruta_mascaras_origen, nombre_mask)
        img = Image.open(ruta_origen_img).convert("RGB")
        mask = Image.open(ruta_origen_mask).convert("L")
        mask_resized = mask.resize((256, 256), Image.NEAREST)
        # Convertir la máscara a un array numpy
        img_array = np.array(img)
        mask_array = np.array(mask_resized)
        
        datos_diccionario.append({
            "image": img_array,
            "label": mask_array,
            "filename": nombre_img 
        })

    # Guardar los datos en pickle
    with open(f"{nombre_salida}_{i}.pkl", "wb") as f:
        pickle.dump(datos_diccionario, f)
    print(f"Datos guardados en {nombre_salida}_{i}.pkl")
    return f"{nombre_salida}_{i}.pkl"
