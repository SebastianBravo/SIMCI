import os
import torch
import resnet
import numpy as np
from torch import nn
from torchsummary import summary
from dataset_class import Dataset
from sklearn.model_selection import KFold

# Path de dataset
data_dir = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/data_no_scores_split/train'

# Extracción de path de cada item del dataset
data_files = []
for class_dir in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_dir):
        data_files += [os.path.join(class_dir, file) for file in os.listdir(class_dir) if file.endswith('.npy')]
data_files = np.array(data_files)

'''--------------------------Definición de modelo---------------------------'''
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Definición de modelo mednet
mednet_model = resnet.resnet50(sample_input_D=192, sample_input_H=256, sample_input_W=256, num_seg_classes=2, no_cuda = False)

# Clase para agregar capa totalmente conectada
class simci_net(nn.Module):
    def __init__(self, mednet_model):
        super(simci_net, self).__init__()
        
        self.pretrained_model = mednet_model
        
        self.pretrained_model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
                                                       nn.Flatten(start_dim=1))
        
        self.fc = nn.Linear(2048, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.pretrained_model(x)   
        x = self.fc(x)
        out = self.activation(x)
        
        return out

# Definición modelo simci 
simci_model = simci_net(mednet_model)

# Diccionario state
net_dict = simci_model.state_dict()

# Cargar pesos
weigth= torch.load('F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/procesamiento/mednet_weights/pretrain/resnet_50_23dataset.pth', map_location=torch.device('cpu'))

# Transferencia de aprendizaje
pretrain_dict = {k.replace("module.", ""): v for k, v in weigth['state_dict'].items() if k.replace("module.", "") in net_dict.keys()}
net_dict.update(pretrain_dict)
simci_model.load_state_dict(net_dict)

'''-------------------------------------------------------------------------'''
# Número de folds para k-fold cross-validation
num_folds = 2

# Cración objeto para validación cruzada
kfold = KFold(n_splits=num_folds, shuffle=True)

# Parametros para generación de datos
gen_param = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 4}

# Número de épocas
epochs = 2

# Lista para almacenar scores de cada fold
scores = dict() # [Presicion, recall, F1-score] --> (weighted macro f1 score), almacenar matríz confusión

# Loss: Binary cross-entropy
loss_func = nn.BCELoss()

# Optimizador Adam
learning_rate = 0.001
optimizer = torch.optim.Adam(simci_model.parameters(), lr=learning_rate)

# Ciclo de entrenamiento con validación cruzada
for train_index, val_index in kfold.split(data_files):
    # Para cada fold se define un dataset de entrenamiento por cada tipo de data augmentation
    train_set = Dataset(data_files[train_index], shape_aug=False, intensity_aug=False) # No augmentation
    train_set_shape = Dataset(data_files[train_index], shape_aug=True, intensity_aug=False) # Shape augmentation
    train_set_inte = Dataset(data_files[train_index], shape_aug=False, intensity_aug=True) # Intensity augmentation
    
    # Concatenar los 3 datasets de entrenamiento
    train_aug_set = torch.utils.data.ConcatDataset([train_set,train_set_shape,train_set_inte])
    
    # Para cada fold se define un dataset de validación
    val_set = Dataset(data_files[val_index], shape_aug=False, intensity_aug=False) # No augmentation
    
    # Dataloaders para entrenamiento y validación
    train_generator = torch.utils.data.DataLoader(train_aug_set, **gen_param)
    val_generator = torch.utils.data.DataLoader(val_set, **gen_param)
    
    # Ciclo de entrenamiento por épocas
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        
        # Ciclo de entreamiento
        for data, labels in train_generator:      
            optimizer.zero_grad() # Poner gradientes en cero          
            target = simci_model(data) # Forward            
            loss = loss_func(target,labels) # Calcular pérdida           
            loss.backward() # Calcular gradientes           
            optimizer.step() # Actualizar pesos           
            epoch_train_loss += loss.item() # registro de loss
        
        train_loss = epoch_train_loss/len(train_generator)
    
        # # Ciclo de validación
        # with torch.no_grad():
        #     simci_model.eval()       
            
    # Print the statistics of the epoch 
    print(f'Completed training batch ---- epoch: {epoch} ------- Training Loss is: {train_loss:.4f}', flush=True)
            
