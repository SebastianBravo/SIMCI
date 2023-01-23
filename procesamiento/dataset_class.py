import os
import torch
import numpy as np
from data_augmentation import get_augmentation

class Dataset(torch.utils.data.Dataset):
  # Characterizes a dataset for PyTorch
  def __init__(self, list_IDs, shape_aug=False, intensity_aug=False):
        #Inicialización lista de IDs
        self.list_IDs = list_IDs
        self.shape_aug = shape_aug
        self.intensity_aug = intensity_aug
        
  def __len__(self):
        # Número total de muestras por dataset
        return len(self.list_IDs)

  def __getitem__(self, index):
        # Obejeto para realizar data augmentation
        augmentation = get_augmentation(shape=self.shape_aug, intensity=self.intensity_aug)
        
        # Seleccionar una muestra del dataset
        img_path = self.list_IDs[index]
        
        # Cargar imagen
        img = np.load(img_path)
        
        # Cargar etiqueta de la muestra
        label_name = os.path.basename(os.path.dirname(self.list_IDs[index]))
        label = 0 if label_name == 'CN' else 1
        
        # aplicar data augmentation
        if self.shape_aug or self.intensity_aug: 
            data = {'image': img}
            aug_img = augmentation(**data)['image']
            
            return aug_img, label
        else:
            return img, label
            
        
    