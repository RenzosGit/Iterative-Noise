import random
import os

def particion_Original(imgs_train, masks_train, imgs_val, masks_val, n_labeled):
    imagenes_train = sorted(os.listdir(imgs_train))
    mascaras_train = sorted(os.listdir(masks_train))
    imagenes_val = sorted(os.listdir(imgs_val))
    mascaras_val = sorted(os.listdir(masks_val))
    # Coincide el n° de imágenes y máscaras?
    if len(imagenes_train) != len(mascaras_train):
        raise ValueError("El número de imágenes y máscaras de train no coincide.")
    if len(imagenes_val) != len(mascaras_val):
        raise ValueError("El número de imágenes y máscaras de valid no coincide.")
    # Creando pares de train y validación
    pares_train = list(zip(imagenes_train, mascaras_train))
    pares_val = list(zip(imagenes_val, mascaras_val))
    # Mezclando
    random.shuffle(pares_train)
    random.shuffle(pares_val)

    # Dividir train: n imágenes/máscaras como labeled y el resto se van a total_imgs_train/total_masks_train
    labeled_pares_train = pares_train[:n_labeled]
    total_pares_train = pares_train[n_labeled:]
    labeled_imgs, labeled_masks = zip(*labeled_pares_train) if labeled_pares_train else ([], [])
    total_imgs_train, total_masks_train = zip(*total_pares_train) if total_pares_train else ([], [])
    max_val_labeled = min(len(pares_val), 250)
    labeled_pares_val = pares_val[:max_val_labeled]
    total_pares_val = pares_val[max_val_labeled:]
    labeled_imgs_val, labeled_masks_val = zip(*labeled_pares_val) if labeled_pares_val else ([], [])
    total_imgs_val, total_masks_val = zip(*total_pares_val) if total_pares_val else ([], [])

    # Regresa tuplas a listas
    labeled_imgs = list(labeled_imgs)
    labeled_masks = list(labeled_masks)
    labeled_imgs_val = list(labeled_imgs_val)
    labeled_masks_val = list(labeled_masks_val)
    total_imgs_val = list(total_imgs_val)
    total_masks_val = list(total_masks_val)
    total_imgs_train = list(total_imgs_train)
    total_masks_train = list(total_masks_train)
    return (labeled_imgs, labeled_masks, labeled_imgs_val, labeled_masks_val, total_imgs_train, total_imgs_val, total_masks_train, total_masks_val)
