import random
import glob
import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class CholecSeg8kDataset(Dataset):
    def __init__(self, data_dir, history_length=4, augment=True):
        """
        Args:
            data_dir (str): Répertoire contenant les fichiers '.pt' (format "XX_YYYYYY.pt").
            history_length (int): Nombre de frames historiques à inclure dans chaque séquence.
            augment (bool): Appliquer ou non les augmentations de données.
        """
        self.data_dir = data_dir
        self.history_length = history_length
        self.augment = augment

        # Charger et organiser les fichiers par vidéo
        self.video_frames = self._load_video_frames()

        # Générer toutes les séquences possibles uniquement si elles sont consécutives
        self.sequences = self._generate_sequences()

    def _load_video_frames(self):
        files = glob.glob(os.path.join(self.data_dir, "*.pt"))
        video_dict = {}
        for file in files:
            filename = os.path.basename(file).replace('.pt', '')
            try:
                vid_str, frame_str = filename.split('_')
                # Convertir en entier (les zéros initiaux sont ignorés)
                vid, frame = int(vid_str), int(frame_str)
            except Exception as e:
                print(f"Nom de fichier inattendu {filename}: {e}")
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
            # On parcourt chaque séquence candidate
            for i in range(self.history_length, len(frames)):
                candidate_seq = frames[i - self.history_length : i + 1]
                frame_numbers = [t[0] for t in candidate_seq]
                # Vérifier que les frames sont consécutives (différence de 1 entre chaque)
                if all(frame_numbers[j+1] - frame_numbers[j] == 1 for j in range(len(frame_numbers)-1)):
                    original_seq = [t[1] for t in candidate_seq]
                    sequences.append(original_seq)
                    if self.augment:
                        # Ajout d'augmentations aléatoires
                        if random.random() < 0.20:
                            sequences.append((original_seq, "rotate"))
                        elif random.random() < 0.35:
                            sequences.append((original_seq, "brightness_contrast"))
                        elif random.random() < 0.45:
                            sequences.append((original_seq, "gaussian_noise"))
                #else:
                #    print(f"Sequence ignorée dans la vidéo {vid} car frames manquantes: {frame_numbers}")
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
        resize_transform = transforms.Resize((60, 100)) #60 100
        for file in sequence_files:
            data = torch.load(file)
            # Redimensionnement de l'image et du masque
            frame = resize_transform(data['image'])
            mask = resize_transform(data['mask'])
            
            # Création du nouveau masque à 9 canaux selon le mapping demandé
            #new_mask = torch.zeros(9, mask.shape[1], mask.shape[2], dtype=mask.dtype)
            #new_mask[0] = mask[1]   # ancien canal 1 -> nouveau canal 0
            #new_mask[2] = mask[2]   # ancien canal 2 -> nouveau canal 2
            #new_mask[7] = mask[0]   # ancien canal 0 -> nouveau canal 7
            #new_mask[8] = mask[7]   # ancien canal 7 -> nouveau canal 8
            # Les canaux 1, 3, 4, 5 et 6 restent à zéro

            #mask = mask[[0, 1, 2, 3, 4, 5, 9, 10]]

            frames.append(frame)
            #masks.append(new_mask)
            masks.append(mask)

        if augmentation_type:
            frames, masks = self._apply_augmentation(frames, masks, augmentation_type)
        #print("------------------------")
        #print("CholecSeg8k:")
        #print(frames[0].shape)
        #print(masks[0].shape)

        # Extraction des métadonnées depuis le nom du fichier de la frame cible (la dernière)
        target_file = sequence_files[-1]
        filename = os.path.basename(target_file).replace('.pt', '')
        try:
            # Pour le format "XX_YYYYYY"
            vid_str, frame_str = filename.split('_')
            meta = {
                "video_id": int(vid_str),
                "frame_id": int(frame_str)
            }
        except Exception as e:
            print("Nom de fichier inattendu", filename, e)
            meta = {"video_id": -1, "frame_id": -1}

        return frames, masks#, meta
    






























class CholecSeg8kDataset2(Dataset):
    def __init__(self, data_dir, history_length=4, augment=True):
        """
        Args:
            data_dir (str): Répertoire contenant les fichiers '.pt' (format "XX_YYYYYY.pt").
            history_length (int): Nombre de frames historiques à inclure dans chaque séquence.
            augment (bool): Appliquer ou non les augmentations de données.
        """
        self.data_dir = data_dir
        self.history_length = history_length
        self.augment = augment

        self.class_names = [
            "Background",
            "Abdominal Wall",
            "Liver",
            "Gastrointestinal Tract",
            "Fat",
            "Grasper",
            "Connective Tissue",
            "Blood",
            "Cystic Duct",
            "L-hook Electrocautery",
            "Gallbladder",
            "Hepatic Vein",
            "Liver Ligament"
        ]

        # Charger et organiser les fichiers par vidéo
        self.video_frames = self._load_video_frames()

        # Générer toutes les séquences possibles uniquement si elles sont consécutives
        self.sequences = self._generate_sequences()

    def _load_video_frames(self):
        files = glob.glob(os.path.join(self.data_dir, "*.pt"))
        video_dict = {}
        for file in files:
            filename = os.path.basename(file).replace('.pt', '')
            try:
                vid_str, frame_str = filename.split('_')
                # Convertir en entier (les zéros initiaux sont ignorés)
                vid, frame = int(vid_str), int(frame_str)
            except Exception as e:
                print(f"Nom de fichier inattendu {filename}: {e}")
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
            # On parcourt chaque séquence candidate
            for i in range(self.history_length, len(frames)):
                candidate_seq = frames[i - self.history_length : i + 1]
                frame_numbers = [t[0] for t in candidate_seq]
                # Vérifier que les frames sont consécutives (différence de 1 entre chaque)
                if all(frame_numbers[j+1] - frame_numbers[j] == 1 for j in range(len(frame_numbers)-1)):
                    original_seq = [t[1] for t in candidate_seq]
                    sequences.append(original_seq)
                    if self.augment:
                        # Ajout d'augmentations aléatoires
                        if random.random() < 0.20:
                            sequences.append((original_seq, "rotate"))
                        elif random.random() < 0.35:
                            sequences.append((original_seq, "brightness_contrast"))
                        elif random.random() < 0.45:
                            sequences.append((original_seq, "gaussian_noise"))
                #else:
                #    print(f"Sequence ignorée dans la vidéo {vid} car frames manquantes: {frame_numbers}")
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
        resize_transform = transforms.Resize((60, 100))
        for file in sequence_files:
            data = torch.load(file)
            # Redimensionnement de l'image et du masque
            frame = resize_transform(data['image'])
            mask = resize_transform(data['mask'])
            
            # Création du nouveau masque à 9 canaux selon le mapping demandé
            #new_mask = torch.zeros(9, mask.shape[1], mask.shape[2], dtype=mask.dtype)
            #new_mask[0] = mask[1]   # ancien canal 1 -> nouveau canal 0
            #new_mask[2] = mask[2]   # ancien canal 2 -> nouveau canal 2
            #new_mask[7] = mask[0]   # ancien canal 0 -> nouveau canal 7
            #new_mask[8] = mask[7]   # ancien canal 7 -> nouveau canal 8
            # Les canaux 1, 3, 4, 5 et 6 restent à zéro

            #mask = mask[[0, 1, 2, 3, 4, 5, 9, 10]]

            frames.append(frame)
            #masks.append(new_mask)
            masks.append(mask)

        if augmentation_type:
            frames, masks = self._apply_augmentation(frames, masks, augmentation_type)
        #print("------------------------")
        #print("CholecSeg8k:")
        #print(frames[0].shape)
        #print(masks[0].shape)

        # Extraction des métadonnées depuis le nom du fichier de la frame cible (la dernière)
        target_file = sequence_files[-1]
        filename = os.path.basename(target_file).replace('.pt', '')
        try:
            # Pour le format "XX_YYYYYY"
            vid_str, frame_str = filename.split('_')
            meta = {
                "video_id": int(vid_str),
                "frame_id": int(frame_str)
            }
        except Exception as e:
            print("Nom de fichier inattendu", filename, e)
            meta = {"video_id": -1, "frame_id": -1}

        #return frames, masks#, meta

        last_mask = masks[-1]
        present = []
        for c in range(last_mask.shape[0]):
            if last_mask[c].sum() > 0:
                present.append(self.class_names[c])
        label_str = " ".join(present)

        return frames, masks, label_str