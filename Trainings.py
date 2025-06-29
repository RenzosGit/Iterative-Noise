import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from transformers import SamProcessor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import SamModel
from tqdm import tqdm
import torch
from torch.optim import Adam
from PIL import Image
import torch.nn as nn
import sys

def train_sam(iteration, save_folder, train_path, val_path, weights_path, epochs=30, lr=1e-4, wd=0, batch_size=2):
    print(f"Pesos en sam training: {weights_path}")
    # Crear archivo de registro y redirigir stdout
    output_dir = f"{save_folder}/iteracion{iteration}"
    os.makedirs(output_dir, exist_ok=True)
    registro_nombre = os.path.join(output_dir,f"registro_{iteration}.txt")
    with open(registro_nombre, "w") as registro_file:
        original_stdout = sys.stdout
        sys.stdout = registro_file
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            with open(train_path, 'rb') as train_file, open(val_path, 'rb') as val_file:
                train_dataset = pickle.load(train_file)
                val_dataset = pickle.load(val_file)
            for example in train_dataset + val_dataset:
                if isinstance(example['image'], np.ndarray):
                    example['image'] = Image.fromarray(example['image'])
                example['image'] = example['image'].convert('RGB')
            def get_bounding_box(image):
                return[0,0,256,256]

            class SAMDataset(Dataset):
                def __init__(self, dataset, processor):
                    self.dataset = dataset
                    self.processor = processor
                def __len__(self):
                    return len(self.dataset)
                def __getitem__(self, idx):
                    item = self.dataset[idx]
                    image = item["image"]
                    ground_truth_mask = np.array(item["label"])
                    image = image.resize((256, 256), Image.Resampling.LANCZOS)
                    ground_truth_mask = Image.fromarray(ground_truth_mask).resize((256, 256), Image.NEAREST)
                    ground_truth_mask = np.array(ground_truth_mask)
                    ground_truth_mask = ground_truth_mask[np.newaxis, :, :]
                    ground_truth_mask = torch.tensor(ground_truth_mask, dtype=torch.float32)
                    prompt = get_bounding_box(image)
                    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
                    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                    inputs["ground_truth_mask"] = ground_truth_mask
                    return inputs
            processor = SamProcessor.from_pretrained("facebook/sam-vit-base", image_size=(256, 256), format="channels_last")
            train_dataloader = DataLoader(SAMDataset(dataset=train_dataset, processor=processor), batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(SAMDataset(dataset=val_dataset, processor=processor), batch_size=batch_size, shuffle=False)

            model = SamModel.from_pretrained("facebook/sam-vit-base")
            if weights_path != "None":
                model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
            else:
                print("No weights provided. Using default SAM weights.")
                # El modelo se carga con los pesos predeterminados al usar from_pretrained
                model = SamModel.from_pretrained("facebook/sam-vit-base")
            for name, param in model.named_parameters():
                if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                    param.requires_grad_(False)
            model.to(device)
            optimizer = Adam(model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

            def normalize_mask(mask):
                return mask / 255.0

            def threshold_mask(mask, threshold=0.5):
                return (mask > threshold).float()

            def dice_coefficient(pred, target):
                smooth = 1.0
                pred_flat = pred.view(-1)
                target_flat = target.view(-1)
                intersection = (pred_flat * target_flat).sum()
                return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

            def iou(y_true, y_pred):
                y_true = threshold_mask(y_true)
                y_pred = threshold_mask(y_pred)
                intersection = torch.sum(y_true * y_pred)
                union = torch.sum((y_true + y_pred) > 0)
                return intersection / (union + 1e-8)

            criterion_bce = nn.BCELoss()
            train_losses, val_losses = [], []
            train_dice, val_dice = [], []
            train_iou, val_iou = [], []

            for epoch in range(epochs):
                model.train()
                epoch_train_losses, epoch_train_dice, epoch_train_iou = [], [], []
                for batch in tqdm(train_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    outputs = model(pixel_values=batch["pixel_values"], input_boxes=batch["input_boxes"], multimask_output=False)
                    predicted_masks = torch.sigmoid(outputs.pred_masks.squeeze(1))
                    ground_truth_masks = normalize_mask(batch["ground_truth_mask"].float()) #.squeeze(2)[..., 0]
                    loss = criterion_bce(predicted_masks, ground_truth_masks)
                    loss.backward()
                    optimizer.step()
                    epoch_train_losses.append(loss.item())
                    epoch_train_dice.append(dice_coefficient(predicted_masks, ground_truth_masks).item())
                    epoch_train_iou.append(iou(predicted_masks, ground_truth_masks).item())
                train_losses.append(np.mean(epoch_train_losses))
                train_dice.append(np.mean(epoch_train_dice))
                train_iou.append(np.mean(epoch_train_iou))
                model.eval()
                epoch_val_losses, epoch_val_dice, epoch_val_iou = [], [], []
                with torch.no_grad():
                    for batch in tqdm(val_dataloader):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(pixel_values=batch["pixel_values"], input_boxes=batch["input_boxes"], multimask_output=False)
                        predicted_masks = torch.sigmoid(outputs.pred_masks.squeeze(1))
                        ground_truth_masks = normalize_mask(batch["ground_truth_mask"].float()) #.squeeze(2)[..., 0]
                        val_loss = criterion_bce(predicted_masks, ground_truth_masks)
                        epoch_val_losses.append(val_loss.item())
                        epoch_val_dice.append(dice_coefficient(predicted_masks, ground_truth_masks).item())
                        epoch_val_iou.append(iou(predicted_masks, ground_truth_masks).item())
                val_losses.append(np.mean(epoch_val_losses))
                val_dice.append(np.mean(epoch_val_dice))
                val_iou.append(np.mean(epoch_val_iou))
                print(f'EPOCH: {epoch}')
                print(f'Train Loss: {train_losses[-1]}, Dice: {train_dice[-1]}, IoU: {train_iou[-1]}')
                print(f'Val Loss: {val_losses[-1]}, Dice: {val_dice[-1]}, IoU: {val_iou[-1]}')

            metrics_data = {
                'Epoch': list(range(1, epochs + 1)),
                'Train Loss': train_losses,
                'Train Dice Coefficient': train_dice,
                'Train IoU': train_iou,
                'Val Loss': val_losses,
                'Val Dice Coefficient': val_dice,
                'Val IoU': val_iou
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_file = os.path.join(output_dir, f"metrics_iter_{iteration}.xlsx")
            metrics_df.to_excel(metrics_file, index=False)
            print(f"Metrics saved to {metrics_file}")

            weights_file = os.path.join(output_dir, f"sam_weights_iter_{iteration}.pth")
            torch.save(model.state_dict(), weights_file)
            print(f"Model weights saved to {weights_file}")
            return weights_file

        finally:
            # Restaurar stdout
            sys.stdout = original_stdout
            print(f"Log saved in {registro_nombre}")
