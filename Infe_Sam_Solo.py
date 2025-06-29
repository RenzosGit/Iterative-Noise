import matplotlib.pyplot as plt
import numpy as np
from transformers import SamProcessor, SamModel
import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
import os
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

# Obtener bounding box
def get_bounding_box(image):
    width, height = image.size
    return [0, 0, width, height]

# Redimensionar la máscara ground truth
def resize_gt_mask(gt_mask, target_shape):
    if len(gt_mask.shape) == 2:
        gt_mask = np.expand_dims(gt_mask, axis=-1)
    gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    resized_mask_tensor = interpolate(gt_mask_tensor, size=target_shape, mode="nearest")
    resized_mask = resized_mask_tensor.squeeze(0).permute(1, 2, 0).numpy()
    if resized_mask.shape[-1] == 1:
        resized_mask = resized_mask[:, :, 0]
    return resized_mask

# Mostrar la máscara
def show_mask(mask, ax, title):
    ax.imshow(mask, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

# Realizar la inferencia
def run_inference(image_names, base_path, ruta_pesos, ruta_img_out):
    print(f"Pesos en sam training: {ruta_pesos}")
    print(f"Pesos usados: {ruta_pesos}")
    print(f"Ruta base de imágenes: {base_path}")
    print(f"Ruta de salida: {ruta_img_out}")
    class SAMDataset(Dataset):
        def __init__(self, image_names, base_path, processor):
            self.image_names = image_names
            self.base_path = base_path
            self.processor = processor
        def __len__(self):
            return len(self.image_names)
        def __getitem__(self, idx):
            image_name = self.image_names[idx]
            image_path = os.path.join(self.base_path, image_name)
            try:
                imagen = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                print(f"Error: Imagen no encontrada en {image_path}")
                return None
            # BB prompt
            prompt = get_bounding_box(image)
            inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
            # Remover batch dimension (agregada por default)
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}
            return inputs
        
        def resize_mask(mask, target_size):
            current_size = mask.shape
            if current_size == target_size: # Si tiene el tamaño correcto, devolver sin modificar
                return mask
            mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float() # Convertir a un tensor y agregar dimensión de lote
            resized_mask = F.interpolate(mask_tensor, size=target_size, mode='nearest')  # Redimensionar a tamaño objetivo
            return resized_mask.squeeze(0).squeeze(0).numpy() # Quitando dimensiones de lote y canal, y devolver como numpy array

    # Cargando el modelo preentrenado y los pesos
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ruta_pesos != "None":
        model.load_state_dict(torch.load(ruta_pesos, map_location=device), strict=False)
    else:
        print("Usando SAM por default.")
        model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base", image_size=(256, 256), format="channels_last")
    model.eval()
    # Procesando cada imagen en la lista
    for idx, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_path = os.path.join(base_path, image_name)
        try:
            # Carga imagen RGB
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Imagen no encontrada en {image_path}")
            continue

        prompt = get_bounding_box(image)
        inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)
        predicted_masks = torch.sigmoid(outputs.pred_masks.squeeze(1))
        predicted_masks = (predicted_masks > 0.5).cpu().numpy().squeeze()
        # Guardando la inferencia
        output_path = os.path.join(ruta_img_out, f"{os.path.splitext(image_name)[0]}.png")
        plt.imsave(output_path, predicted_masks, cmap="gray")
