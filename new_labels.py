import os
import shutil
import pandas as pd
import numpy as np
import cv2

def select_and_copy_images(image_list, source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    copied_images = []

    for image_name in image_list:
        source_path = os.path.join(source_dir, image_name)
        dest_path = os.path.join(dest_dir, image_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
            copied_images.append(image_name)
        else:
            print(f"Advertencia: La imagen {image_name} no se encontró en {source_path}")
    return copied_images

def generate_intersection_masks(df_iou, ruta_sam_out, n, output_dir, i, ruta_data0, threshold=60):
    # Convertir el threshold a un valor entre 0 y 1
    iou_threshold = threshold / 100  
    os.makedirs(output_dir, exist_ok=True)
    # Filtra las imágenes del df que cumplan con el threshold
    df_filtered = df_iou[df_iou['IoU'] >= iou_threshold]
    print(f"Imágenes que cumplen con el threshold ({threshold}%): {len(df_filtered)}")
    # Selecciona las primeras n imágenes (chunk_train) que cumplan
    df_top_n = df_filtered.head(n)
    processed_images = []
    print("Procesando imágenes según el umbral: ", len(df_top_n))

    # Procesa las imágenes 
    for idx, row in df_top_n.iterrows():
        image_name = row['Nombre_Imagen']
        bbox_samRuidoso = row['BBoxSam_ruidoso']
        bbox_sam = row['BBoxSam']
        # Calcula la intersección de los BBs con margen de 10 pixeles
        x_min = max(bbox_samRuidoso[0], bbox_sam[0]) - 10
        y_min = max(bbox_samRuidoso[1], bbox_sam[1]) - 10
        x_max = min(bbox_samRuidoso[2], bbox_sam[2]) + 10
        y_max = min(bbox_samRuidoso[3], bbox_sam[3]) + 10
        # Asegurando que los valores estén dentro de los límites de la imagen
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(255, x_max)
        y_max = min(255, y_max)

        if x_min >= x_max or y_min >= y_max:
            print(f"No hay intersección válida para {image_name}.")
            continue
        sam_mask_path = os.path.join(ruta_sam_out, image_name)
        if not os.path.exists(sam_mask_path):
            print(f"Máscara SAM no encontrada para {image_name}.")
            continue
        sam_mask = cv2.imread(sam_mask_path, cv2.IMREAD_GRAYSCALE)
        if sam_mask is None:
            print(f"Error al cargar la máscara de SAM para {image_name}.")
            continue

        # Crea una nueva máscara negra y copia el área de interés desde SAM
        new_mask = np.zeros_like(sam_mask, dtype=np.uint8)
        new_mask[y_min:y_max, x_min:x_max] = sam_mask[y_min:y_max, x_min:x_max]

        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, new_mask)
        print(f"Máscara de intersección guardada en: {output_path}")
        processed_images.append(image_name)

    processed_data = []
    for img in processed_images:
        iou_value = df_iou[df_iou['Nombre_Imagen'] == img]['IoU'].values
        if iou_value.size > 0:
            processed_data.append({
                'Nombre_Imagen': img,
                'IoU': iou_value[0]
            })
    if not processed_images:
        print("No se puede aprender más, resultados bajo el threshold.")
        return None
    folder_iou = rf"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\df_IoU"
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_excel(os.path.join(folder_iou, f'processed_masks{i}.xlsx'), index=False)
    print(f"Se guardó el archivo de resultados en: {os.path.join(folder_iou, f'processed_masks{i}.xlsx')}")
    print(f"Se procesaron {len(processed_images)} imágenes.")
    return processed_images
