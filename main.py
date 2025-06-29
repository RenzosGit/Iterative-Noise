import datetime
import os
import cv2
from pathlib import Path
import numpy as np
from pkl_Sam import procesar_datos
from Infe_Sam_Solo import run_inference
from BBoxes import bboxes_df
from IoU import calculate_iou_dataframe
from new_labels import select_and_copy_images, generate_intersection_masks, yolo_labels
from Trainings import train_sam
from mover_imgs import mover_a_dataset, mover_train_yolo, mover_imgs_y_masks_val, mover_train_a_dataset, mover_archivos
from limpiar import borrar_datos_en_carpeta
from actualizacion import particion_Original


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Data original
imgs_train = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\DataSimulada\images\train"
imgs_val = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\DataSimulada\images\valid"
masks_train = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\DataSimulada\masks\train"
masks_val = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\DataSimulada\masks\valid"

save_imgsdir = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\test_BBOXES"
ruta_sam_out = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\test_Inferencias"
sam_val_out = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\infes_Sam_val"
ruta_pkls = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\pkls_Sam"
# Rutas de pesos
base_sam = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo" 
data_etiquetada = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\data_etiquetada"
folder_IoU = f"C:\\Users\\Desktop\\Project_Tesis\\Flujo_Comparativo\\df_IoU"
data_source = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\Data_Compuesta\images\train"
print("fin de rutas")

def draw_bounding_box(image, bbox, color=(0, 0, 255), thickness=2):
    # Crear una imagen negra
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

def get_ruta_sam(nombre_archivo):
    return os.path.join(base_sam, nombre_archivo)

def train_labeled(iteration, imgs_val, imgs_train, masks_val, masks_train, weights_path, epochs, batch_size):
    ruta_out_train = os.path.join(ruta_pkls, f"train{iteration}")
    ruta_out_val = os.path.join(ruta_pkls, f"valid{iteration}")
    output_name_train = os.path.join(ruta_out_train, "pkl_train")
    output_name_valid = os.path.join(ruta_out_val, "pkl_val")
    ruta_pkl_train = procesar_datos(imgs_train, masks_train, output_name_train, iteration)
    ruta_pkl_val = procesar_datos(imgs_val, masks_val, output_name_valid, iteration) # procesar_datos(ruta_imagenes_origen, ruta_mascaras_origen, nombre_salida)
    weights_file = train_sam(iteration, save_folder, ruta_pkl_train, ruta_pkl_val, weights_path, epochs =epochs, lr=lr, wd=wd, batch_size=batch_size) # (iteration, train_path, val_path, weights_path=None, epochs=30, lr=1e-4, wd=0)
    pesos_sam.append(weights_file)
    print("PESOS: ", pesos_sam)

def eliminar_imagenes(image_names, carpeta):
    for image_name in image_names:
        image_path = os.path.join(carpeta, f"{image_name}")
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Imagen {image_name} eliminada.")
        else:
            print(f"Imagen {image_name} no encontrada.")


def apply_salt_and_pepper_to_images(image_list, input_folder, output_folder, salt_pepper_ratio=0.01):
    os.makedirs(output_folder, exist_ok=True)
    for image_name in image_list:
        input_path = os.path.join(input_folder, image_name)
        if not os.path.exists(input_path):
            print(f"Imagen no encontrada: {input_path}. Saltando...")
            continue
        image = cv2.imread(input_path)
        if image is None:
            print(f"No se pudo leer la imagen: {input_path}. Saltando...")
            continue
        # Agregar ruido sal y pimienta
        noisy_image = add_salt_and_pepper_noise(image, salt_pepper_ratio)
        # Guardar la imagen 
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, noisy_image)
        print(f"Imagen procesada guardada en: {output_path}")

def add_salt_and_pepper_noise(image, ratio):
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = int(ratio * total_pixels / 2)
    num_pepper = int(ratio * total_pixels / 2)
    # Agregar ruido sal 
    coords_salt = [
        np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]
    ]
    noisy_image[coords_salt[0], coords_salt[1]] = 255
    # Agregar ruido pimienta 
    coords_pepper = [
        np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]
    ]
    noisy_image[coords_pepper[0], coords_pepper[1]] = 0
    return noisy_image


# Listas de pesos
pesos_sam = []
n = 5
lr=0.0001  # learning rate
wd=0  # weight drop
epochs = 50
batch_size = 2
n_labeled = 250
print(f"{n} For(s), {epochs} EPOCHS, {n_labeled} IMGS")
# # Obtener las listas de archivos
labeled_imgs, labeled_masks, labeled_imgs_val, labeled_masks_val,total_imgs_train, total_imgs_val,total_masks_train, total_masks_val = particion_Original(imgs_train, masks_train, imgs_val, masks_val, n_labeled)
save_folder = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\metricas_SAM_Train2"
borrar_datos_en_carpeta(folder_IoU)
borrar_datos_en_carpeta(data_etiquetada)
borrar_datos_en_carpeta(save_folder)
borrar_datos_en_carpeta(ruta_pkls)
borrar_datos_en_carpeta(ruta_sam_out)
borrar_datos_en_carpeta(save_imgsdir)

start_time = datetime.datetime.now() 
print(f"Hora de inicio: {start_time}")
check = []
for i in range(n):
    print("\n FOR N°: ", i)
    if i == 0:
        train_imgs_dir0, train_masks_dir0, val_imgs_dir0, val_masks_dir0 = mover_a_dataset(i,labeled_imgs, labeled_masks,
            labeled_imgs_val, labeled_masks_val,data_etiquetada,imgs_train, masks_train,imgs_val, masks_val)
        train_labeled(i, val_imgs_dir0, train_imgs_dir0, val_masks_dir0, train_masks_dir0, weights_path="None", epochs=epochs, batch_size=batch_size)
        print("pesos_sam", pesos_sam)

        print("TOTAL DE IMAGENES PARA INFERENCIAS: ")
        print("total_imgs_val",len(total_imgs_val), ", total_masks_val",len(total_masks_val), "total_imgs_train",len(total_imgs_train), 
              "total_masks_train",len(total_masks_train))
        folder0 = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\data_etiquetada\data_0"
        
    else:
        print("\nCREANDO SET VALIDACION\n")    
        print(f"iteración{i}: {len(total_imgs_train)} imágenes")
        # Crear rutas de la iteración
        iter_dir = os.path.join(data_etiquetada, f"data_{i}")
        train_imgs_dir = os.path.join(iter_dir, "images", "train")
        train_masks_dir = os.path.join(iter_dir, "masks", "train")
        val_imgs_dir = os.path.join(iter_dir, "images", "valid")
        val_masks_dir = os.path.join(iter_dir, "masks", "valid")

        # VALIDACION
        q_imgs_val = len(total_imgs_val) // (n - i)
        print(f" Iteracion {i}, cantidad de elementos de val: {q_imgs_val}")
        # Mover data de validacion
        total_imgs_val, total_masks_val = mover_imgs_y_masks_val(total_imgs_val, total_masks_val, imgs_val, masks_val, val_imgs_dir, val_masks_dir, q_imgs_val)
        print("IMGS Y MASKS VALID QUE QUEDAN: ",len(total_masks_val))
        
        print(f"cantidad de imagenes en train para {i} iteracion: ", len(total_imgs_train))
        borrar_datos_en_carpeta(ruta_sam_out) # Vaciar antes de hacer inferencia
        #------------- SAM
        print("\nINFERENCIAS SAM\n")
        run_inference(total_imgs_train, imgs_train, pesos_sam[i-1], ruta_sam_out) 
        output_folder = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\Salt_Pepper"
        borrar_datos_en_carpeta(output_folder)
        percentage = 0.1
        apply_salt_and_pepper_to_images(total_imgs_train, imgs_train, output_folder, percentage)
        snp_folder = r"C:\Users\Desktop\Project_Tesis\Flujo_Comparativo\Infes_SnP"
        borrar_datos_en_carpeta(snp_folder)
        run_inference(total_imgs_train, output_folder, pesos_sam[i-1], snp_folder)
        #------------- BBOX COMPARATIVO
        sam_ruido, sam_df = bboxes_df(snp_folder, ruta_sam_out) # ruta_sam_out o sam_val_out
        
        print("DataFrame SAM Ruidoso:")
        print(sam_ruido.head(3))
        print("DataFrame SAM:")
        print(sam_df.head(3))
        #----------------- IoU
        print("\nCALCULANDO IoU\n")
        df_iou = calculate_iou_dataframe(sam_ruido, sam_df, i, folder_IoU)
        if df_iou is None:
            print("Ejecución finalizada debido a que no puede seguir aprendiendo con la data conseguida")
            exit()
        print(df_iou.shape)
        
        #----------------- New Labels
        print("\nCREANDO NUEVAS ETIQUETAS \n")
        chunk_train = len(total_imgs_train) // (n - i)
        print("Total de imágenes Train:", len(total_imgs_train))
        print("chunk_train", chunk_train)
        
        image_names = generate_intersection_masks(df_iou, ruta_sam_out, chunk_train, train_masks_dir, i, train_masks_dir0, threshold=90) 
        if image_names is None:
            print("Ejecución finalizada debido a que no puede seguir aprendiendo con la data conseguida")
            exit()
        selected_imgs = select_and_copy_images(image_names, imgs_train, train_imgs_dir) # select_and_copy_images(image_list, source_dir, dest_dir)
        print("Processed Images: ", image_names)
        print("Cantidad de imgs procesadas: ", len(image_names))
        eliminar_imagenes(image_names, data_source)
        print(f"Antes de eliminar: {len(total_imgs_train)} imágenes")
        total_imgs_train = [img for img in total_imgs_train if img not in image_names]
        print(f"Después de eliminar: {len(total_imgs_train)} imágenes")
        check.append(len(total_imgs_train))
        print(f"check: {check}")
        print(f"Imágenes restantes en total_imgs_train: {len(total_imgs_train)}")
        #-----------------Mover dataset
        print("\n GUARDANDO DATA NUEVA \n")
        # Mueve imagenes de train
        mover_archivos(imgs_train, image_names, train_imgs_dir) # mover_archivos(origen_dir, archivos, destino_dir):

        #---------------- Training SAM
        print("\n TRAINING SAM \n")
        train_labeled(i, val_imgs_dir, train_imgs_dir, val_masks_dir, train_masks_dir, pesos_sam[i-1], epochs, batch_size)
        print(f"FIN ITERACION {i}")

end_time = datetime.datetime.now() 
print(f"Hora de término: {end_time}") 
# Calcular la duración de la ejecución 
duration = end_time - start_time 
print(f"Duración de la ejecución: {duration}")