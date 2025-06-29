import pandas as pd
import os

def calculate_iou(bbox1, bbox2):
    """Calcula el Índice de Intersección sobre la Unión (IoU) entre dos bounding boxes.
    - bbox1: Bounding box de SAM Ruidoso [x_min, y_min, x_max, y_max].
    - bbox2: Bounding box de SAM [x_min, y_min, x_max, y_max].
    Out:
    - IoU: Índice de intersección sobre la unión (IoU)."""
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2
    # Calculando las coordenadas de la intersección
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)
    # Si no hay intersección
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0
    # Calcula el área de intersección
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    # Calcula el área de cada bounding box
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    # Calcula el área de la unión
    union_area = area1 + area2 - inter_area
    # Calcula el IoU
    iou = inter_area / union_area
    return iou

def calculate_iou_dataframe(yolo_df, sam_df, iteracion, output_dir):
    if not {'Image', 'BBox'}.issubset(yolo_df.columns):
        print("Error: El DataFrame de YOLO debe contener las columnas 'Image' y 'BBox'.")
        print("Finalizando ejecución.")
        return None
    if not {'Image', 'BBox'}.issubset(sam_df.columns):
        print("Error: El DataFrame de SAM debe contener las columnas 'Image' y 'BBox'.")
        print("Finalizando ejecución.")
        return None
    results = []
    for _, yolo_row in yolo_df.iterrows():
        image_name = yolo_row["Image"]
        yolo_bbox = yolo_row["BBox"]
        # Buscar el bounding box correspondiente de SAM
        sam_row = sam_df[sam_df["Image"] == image_name]
        if sam_row.empty:
            continue  # Si no se encuentra la imagen en SAM, continuar
        sam_bbox = sam_row.iloc[0]["BBox"]
        iou = calculate_iou(yolo_bbox, sam_bbox)
        results.append({
            "Nombre_Imagen": image_name,
            "IoU": iou,
            "BBoxSam_ruidoso": yolo_bbox,
            "BBoxSam": sam_bbox
        })
    df_iou = pd.DataFrame(results)
    
    if df_iou.empty: 
        print("No se encontraron coincidencias de imágenes o IoU válidos después de filtrar.")
        return None
    # Ordenar de mayor a menor IoU
    df_iou = df_iou.sort_values(by="IoU", ascending=False)
    output_filename = f"df_IoU{iteracion}.xlsx"
    output_path = os.path.join(output_dir, output_filename)
    df_iou.to_excel(output_path, index=False)
    print(f"Archivo de IoU guardado en: {output_path}")
    return df_iou
