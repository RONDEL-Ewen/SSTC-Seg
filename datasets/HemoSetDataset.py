import random
import glob
import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class HemoSetDataset(Dataset):
    def __init__(self, data_dir, history_length=4, augment=True):
        """
        Args:
            data_dir (str): Répertoire contenant les fichiers '.pt' du dataset HemoSet.
            history_length (int): Nombre de frames passées (et leurs masks) à inclure par séquence.
            augment (bool): Applique l'augmentation des données.
        """
        self.data_dir = data_dir
        self.history_length = history_length
        self.augment = augment
        self.video_frames = self._load_video_frames()
        self.sequences = self._generate_sequences()

    def _load_video_frames(self):
        files = glob.glob(os.path.join(self.data_dir, "*.pt"))
        video_dict = {}
        for file in files:
            filename = os.path.basename(file).replace('.pt', '')
            # On attend un format "XX__YY" où XX est le numéro de vidéo et YY l'indice de la frame
            parts = filename.split('__')
            if len(parts) != 2:
                print(f"Nom de fichier inattendu: {filename}")
                continue
            try:
                vid = int(parts[0])
                frame = int(parts[1])
            except ValueError:
                print(f"Erreur de conversion pour le fichier: {filename}")
                continue
            if vid not in video_dict:
                video_dict[vid] = []
            video_dict[vid].append((frame, file))
        for vid in video_dict:
            video_dict[vid].sort(key=lambda x: x[0])
        return video_dict

    def _generate_sequences(self):
        sequences = []
        for vid, frames in self.video_frames.items():
            if len(frames) < self.history_length + 1:
                continue
            for i in range(self.history_length, len(frames)):
                original_seq = [frames[j][1] for j in range(i - self.history_length, i + 1)]
                sequences.append(original_seq)
                if self.augment:
                    if random.random() < 0.20:  # 20% de chance pour rotation
                        sequences.append((original_seq, "rotate"))
                    elif random.random() < 0.35:  # 15% de chance pour brightness/contrast
                        sequences.append((original_seq, "brightness_contrast"))
                    elif random.random() < 0.45:  # 10% de chance pour Gaussian noise
                        sequences.append((original_seq, "gaussian_noise"))
        return sequences

    def _apply_augmentation(self, frames, masks, augmentation_type):
        if augmentation_type == "rotate":
            angle = random.choice([90, 180, 270])
            frames = [TF.rotate(frame, angle) for frame in frames]
            masks = [TF.rotate(mask, angle) for mask in masks]
        elif augmentation_type == "brightness_contrast":
            brightness_factor = random.uniform(0.85, 1.15)
            contrast_factor = random.uniform(0.85, 1.15)
            frames = [TF.adjust_brightness(frame, brightness_factor) for frame in frames]
            frames = [TF.adjust_contrast(frame, contrast_factor) for frame in frames]
        elif augmentation_type == "gaussian_noise":
            frames = [frame + torch.randn_like(frame) * 0.025 for frame in frames]
        return frames, masks

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        augmentation_type = None
        if isinstance(sequence_info, tuple):
            sequence_files, augmentation_type = sequence_info
        else:
            sequence_files = sequence_info

        frames = []
        masks = []
        transform = transforms.Resize((80, 80))
        for file in sequence_files:
            data = torch.load(file)
            # Transformation des images
            frames.append(transform(data['image']).unsqueeze(0))
            # Transformation du mask
            mask = transform(data['mask'])
            # Création d'un tenseur à 9 canaux:
            # les 8 premiers canaux sont remplis de zéros et le 9ème contient la classe "bleeding"
            #zeros = torch.zeros(8, mask.shape[1], mask.shape[2])
            #mask_9ch = torch.cat([zeros, mask], dim=0)
            #masks.append(mask_9ch)

            mask_2d = mask.squeeze(0)
            binary_mask = torch.stack([(mask_2d == 0).float(), (mask_2d != 0).float()], dim=0)  # forme (2, H, W)
            masks.append(binary_mask)
            
        if augmentation_type:
            frames, masks = self._apply_augmentation(frames, masks, augmentation_type)
        #print("------------------------")
        #print("HemoSet:")
        #print(frames[0].shape)
        #print(masks[0].shape)
        return frames, masks