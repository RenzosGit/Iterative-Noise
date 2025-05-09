import os
import shutil

def mover_a_dataset(iteracion, labeled_imgs_train, labeled_masks_train,
    labeled_imgs_val, labeled_masks_val, base_dir, imgs_train_dir, masks_train_dir, imgs_val_dir, masks_val_dir):

    iter_dir = os.path.join(base_dir, f"data_{iteracion}")
    train_imgs_dir = os.path.join(iter_dir, "images", "train")
    train_masks_dir = os.path.join(iter_dir, "masks", "train")
    val_imgs_dir = os.path.join(iter_dir, "images", "valid")
    val_masks_dir = os.path.join(iter_dir, "masks", "valid")
    
    os.makedirs(train_imgs_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(val_imgs_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)
    mover_archivos(imgs_train_dir, labeled_imgs_train, train_imgs_dir)
    mover_archivos(masks_train_dir, labeled_masks_train, train_masks_dir)
    mover_archivos(imgs_val_dir, labeled_imgs_val, val_imgs_dir)
    mover_archivos(masks_val_dir, labeled_masks_val, val_masks_dir)
    print(f"Datos de la iteración {iteracion} guardados en {iter_dir}")
    return train_imgs_dir, train_masks_dir, val_imgs_dir, val_masks_dir

# Función para mover archivos
def mover_archivos(origen_dir, archivos, destino_dir):
    for archivo in archivos:
        origen = os.path.join(origen_dir, archivo)
        destino = os.path.join(destino_dir, archivo)
        if os.path.exists(origen):
            shutil.copy(origen, destino)
        else:
            print(f"Archivo no encontrado: {origen}")

def mover_imgs_y_masks_val(total_imgs_val, total_masks_val, ruta_base_imgs, ruta_base_masks, ruta_destino_imgs, ruta_destino_masks, 
                           cantidad_a_mover):

    os.makedirs(ruta_destino_imgs, exist_ok=True)
    os.makedirs(ruta_destino_masks, exist_ok=True)

    # Vaciando las carpetas de destino antes de copiar nuevos archivos
    for archivo in os.listdir(ruta_destino_imgs):
        os.remove(os.path.join(ruta_destino_imgs, archivo))
    for archivo in os.listdir(ruta_destino_masks):
        os.remove(os.path.join(ruta_destino_masks, archivo))
    # Archivos a mover
    imgs_a_mover = total_imgs_val[:cantidad_a_mover]
    masks_a_mover = total_masks_val[:cantidad_a_mover]
    # Copiando imágenes a destino
    for img, mask in zip(imgs_a_mover, masks_a_mover):
        origen_img = os.path.join(ruta_base_imgs, img)
        origen_mask = os.path.join(ruta_base_masks, mask)
        destino_img = os.path.join(ruta_destino_imgs, img)
        destino_mask = os.path.join(ruta_destino_masks, mask)
        if os.path.exists(origen_img):
            shutil.copy(origen_img, destino_img)
        else:
            print(f"No encontrado: {origen_img}")
        if os.path.exists(origen_mask):
            shutil.copy(origen_mask, destino_mask)
        else:
            print(f"No encontrado: {origen_mask}")

    # Actualizar las listas, eliminando los archivos ya procesados
    total_imgs_val = total_imgs_val[cantidad_a_mover:]
    total_masks_val = total_masks_val[cantidad_a_mover:]

    return total_imgs_val, total_masks_val
