import os
import cv2
import numpy as np
import pandas as pd

def extract_bboxes_from_mask(mask):
    """Genera los bounding boxes a partir de la máscara binaria."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, x + w, y + h])  # [x_min, y_min, x_max, y_max]
    return bboxes

def extract_red_bboxes(image):
    lower_red = np.array([0, 0, 255])  # Valor mínimo para rojo
    upper_red = np.array([0, 0, 255])  # Valor máximo para rojo
    mask = cv2.inRange(image, lower_red, upper_red)
    # Detecta contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # BBs a partir de los contornos
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, x + w, y + h])  # Formato [x_min, y_min, x_max, y_max]
    return bboxes

def merge_bboxes(bboxes):
    """Fusionando múltiples BBs en uno solo que cubra el área total."""
    if not bboxes:
        return None
    x_min = min([box[0] for box in bboxes])
    y_min = min([box[1] for box in bboxes])
    x_max = max([box[2] for box in bboxes])
    y_max = max([box[3] for box in bboxes])
    return [x_min, y_min, x_max, y_max]

def process_mask_images(mask_dir):
    """Procesando las máscaras de segmentación y generando el DataFrame con los bounding boxes fusionados."""
    data = []
    for filename in os.listdir(mask_dir):
        if filename.endswith(".png"): 
            image_path = os.path.join(mask_dir, filename)
            mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Carga en escala de grises
            if mask is None:
                print(f"No se pudo cargar la máscara: {filename}")
                continue
            # Asegura que sea binaria
            _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            # Generando los BBs a partir de la máscara
            bboxes = extract_bboxes_from_mask(mask_binary)
            merged_bbox = merge_bboxes(bboxes)
            if merged_bbox:
                data.append({"Image": filename, "BBox": merged_bbox, "Mask": mask_binary})
    df = pd.DataFrame(data)
    return df

def bboxes_df(sam_ruidoso_dir, sam_dir):
    sam_ruidoso = process_mask_images(sam_ruidoso_dir)
    sam_df = process_mask_images(sam_dir)
    return sam_ruidoso, sam_df
