import os
import torch
import resnet
import torchinfo
import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics
from torch import nn
from torchsummary import summary
from dataset_class import Dataset
from sklearn.model_selection import KFold

# Path de dataset
data_dir = '/HDDmedia/simci/datasets/data_no_scores_split/train'

# Extracción de path de cada item del dataset
data_files = []
for class_dir in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_dir):
        data_files += [os.path.join(class_dir, file) for file in os.listdir(class_dir) if file.endswith('.npy')]
data_files = np.array(data_files)

'''--------------------------Definición de modelo---------------------------'''
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [6,7]

# Definición de modelo mednet
mednet_model = resnet.resnet50(sample_input_D=192, sample_input_H=256, sample_input_W=256, num_seg_classes=2, no_cuda = False)

# Clase para agregar capa totalmente conectada
class simci_net(nn.Module):
    def __init__(self, mednet_model):
        super(simci_net, self).__init__()
        
        self.pretrained_model = mednet_model
        
        self.pretrained_model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
                                                       nn.Flatten(start_dim=1))
        
        self.classifier = nn.Sequential(nn.Linear(2048, 1),
                                        nn.Sigmoid())

    def forward(self, x):
        x = self.pretrained_model(x)   
        out = self.classifier(x)
        
        return out

# Definición modelo simci 
simci_model = simci_net(mednet_model)

# Diccionario state
net_dict = simci_model.state_dict()

# Cargar pesos
weigth = torch.load('/HDDmedia/simci/SIMCI/Procesamiento/mednet_weights/resnet_50_23dataset.pth', map_location=torch.device('cpu'))

# Transferencia de aprendizaje
pretrain_dict = {k.replace("module.", ""): v for k, v in weigth['state_dict'].items() if k.replace("module.", "") in net_dict.keys()}
net_dict.update(pretrain_dict)
simci_model.load_state_dict(net_dict)

# Summary sin bloquear parametros mednet
torchinfo.summary(simci_model, (1,192, 256, 256), batch_dim = 0, col_names = ('input_size', 'output_size', 'num_params', 'trainable'), verbose = 0)

# Bloqueo de parametros mednet
for param in simci_model.pretrained_model.parameters():
    param.requires_grad = False

# Summary sin bloquear parametros mednet
torchinfo.summary(simci_model, (1,192, 256, 256), batch_dim = 0, col_names = ('input_size', 'output_size', 'num_params', 'trainable'), verbose = 0)

# Data parallel
simci_model = torch.nn.DataParallel(simci_model, device_ids = device_ids)
simci_model.to(f'cuda:{simci_model.device_ids[0]}')

'''-------------------------------------------------------------------------'''
# Número de folds para k-fold cross-validation
num_folds = 2

# Cración objeto para validación cruzada
kfold = KFold(n_splits=num_folds, shuffle=True)

# Parametros para generación de datos
gen_param = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 8}

# Número de épocas
epochs = 2

# Lista para almacenar scores de cada fold
scores = dict() # [Presicion, recall, F1-score] --> (weighted macro f1 score), almacenar matríz confusión

# Loss: Binary cross-entropy
loss_func = nn.BCELoss()
loss_func.to(f'cuda:{simci_model.device_ids[0]}')

# Optimizador Adam
learning_rate = 0.001
optimizer = torch.optim.Adam(simci_model.parameters(), lr=learning_rate)

# Ciclo de entrenamiento con validación cruzada
fold_idx = 0
for train_index, val_index in kfold.split(data_files):
    print(f'Fold {fold_idx}/{num_folds}--------------------------------------')
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
        total_train_loss = 0.0
        total_val_loss = 0.0
        
        n_batches = len(train_generator)
        print(f'Epoch {epoch+1}/{epochs}')
        pbar = tf.keras.utils.Progbar(target=n_batches)

        # Ciclo de entreamiento
        simci_model.train()
        for i_batch, (data, labels) in enumerate(train_generator,1):
            data, labels = data.to(f'cuda:{simci_model.device_ids[0]}'), labels.to(f'cuda:{simci_model.device_ids[0]}').float()
            
            optimizer.zero_grad() # Poner gradientes en cero          
            target = simci_model(data) # Forward            
            loss = loss_func(target.squeeze(-1),labels) # Calcular pérdida           
            loss.backward() # Calcular gradientes           
            optimizer.step() # Actualizar pesos
            
            labels_npy = labels.detach().cpu().numpy().reshape(-1,1)
            target_npy = target.detach().cpu().numpy()
            
            total_train_loss += loss.item() # Suma de loss para cada batch
            train_f1_score = metrics.f1_score(labels_npy, target_npy, average='weighted')
            train_recall = metrics.recall_score(labels_npy, target_npy, average='weighted')
            train_precision = metrics.precision_score(labels_npy, target_npy, average='weighted')
            
            pbar.update(i_batch, values=[("loss",loss.item()), 
                                         ("f1_score", train_f1_score),
                                         ("recall", train_recall),
                                         ("precision", train_precision)]) # Actualizar barra de progreso
        
        train_loss = total_train_loss/len(train_generator)
        
        # Ciclo de validación
        with torch.no_grad():
            simci_model.eval() # Modelo en modo evaluación
            
            for i_batch, (data, labels) in enumerate(val_generator,1):
                predicted_labels = simci_model(data) # Predicciones
                
                val_loss = loss_func(predicted_labels.squeeze(-1),labels) # Calcular pérdida
                
                real_labels = labels.detach().cpu().numpy().reshape(-1,1) # Pasar tensor a numpy
                predicted_labels = np.round(predicted_labels.detach().cpu().numpy()) # Pasar tensor a numpy
                
                total_val_loss += val_loss.item()
                
                # Almacenar las predicciones y los labels
                if i_batch == 0:
                    total_labels = real_labels
                    total_predictions = predicted_labels
                else:
                    total_labels = np.concatenate((total_labels, real_labels), axis=0)
                    total_predictions = np.concatenate((total_predictions, predicted_labels), axis=0)
            
            final_val_loss = total_val_loss/len(val_generator)
            val_f1_score = metrics.f1_score(total_labels, total_predictions, average='weighted')
            val_recall = metrics.recall_score(total_labels, total_predictions, average='weighted')
            val_precision = metrics.precision_score(total_labels, total_predictions, average='weighted')
            
        pbar.update(n_batches, values=[("val_loss",final_val_loss), 
                                       ("val_f1_score", val_f1_score),
                                       ("val_recall", val_recall),
                                       ("val_precision", val_precision)]) # Actualizar barra de progreso       

    print('-----------------------------------------------------------------')
    fold_idx += 1

"""-------------------------------Test-------------------------------"""

# # Para cada fold se define un dataset de entrenamiento por cada tipo de data augmentation
# train_set = Dataset(data_files, shape_aug=False, intensity_aug=False) # No augmentation
# train_set_shape = Dataset(data_files, shape_aug=True, intensity_aug=False) # Shape augmentation
# train_set_inte = Dataset(data_files, shape_aug=False, intensity_aug=True) # Intensity augmentation

# # Concatenar los 3 datasets de entrenamiento
# train_aug_set = torch.utils.data.ConcatDataset([train_set,train_set_shape,train_set_inte])

# # # Para cada fold se define un dataset de validación
# # val_set = Dataset(data_files[val_index], shape_aug=False, intensity_aug=False) # No augmentation

# # Dataloaders para entrenamiento y validación
# train_generator = torch.utils.data.DataLoader(train_aug_set, **gen_param)
# # val_generator = torch.utils.data.DataLoader(val_set, **gen_param)

# # Ciclo de entrenamiento por épocas
# for epoch in range(epochs):
#     total_train_loss = 0.0
#     total_val_loss = 0.0
    
#     n_batches = len(train_generator)
#     print(f'Epoch {epoch+1}/{epochs}')
#     pbar = tf.keras.utils.Progbar(target=n_batches)

#     # Ciclo de entreamiento
#     simci_model.train()
#     for i_batch, (data, labels) in enumerate(train_generator,1):
#         data, labels = data.to(f'cuda:{simci_model.device_ids[0]}'), labels.to(f'cuda:{simci_model.device_ids[0]}').float()
        
#         optimizer.zero_grad() # Poner gradientes en cero          
#         target = simci_model(data) # Forward            
#         loss = loss_func(target.squeeze(-1),labels) # Calcular pérdida           
#         loss.backward() # Calcular gradientes           
#         optimizer.step() # Actualizar pesos
        
#         labels_npy = labels.detach().cpu().numpy().reshape(-1,1)
#         target_npy = target.detach().cpu().numpy()
        
#         total_train_loss += loss.item() # Suma de loss para cada batch
#         train_f1_score = metrics.f1_score(labels_npy, target_npy, average='weighted')
#         train_recall = metrics.recall_score(labels_npy, target_npy, average='weighted')
#         train_precision = metrics.precision_score(labels_npy, target_npy, average='weighted')
        
#         pbar.update(i_batch, values=[("loss",loss.item()), 
#                                      ("f1_score", train_f1_score),
#                                      ("recall", train_recall),
#                                      ("precision", train_precision)]) # Actualizar barra de progreso
    
#     train_loss = total_train_loss/len(train_generator)
    
#     # Ciclo de validación
#     with torch.no_grad():
#         simci_model.eval() # Modelo en modo evaluación
        
#         for i_batch, (data, labels) in enumerate(val_generator,1):
#             predicted_labels = simci_model(data) # Predicciones
            
#             val_loss = loss_func(predicted_labels.squeeze(-1),labels) # Calcular pérdida
            
#             real_labels = labels.detach().cpu().numpy().reshape(-1,1) # Pasar tensor a numpy
#             predicted_labels = np.round(predicted_labels.detach().cpu().numpy()) # Pasar tensor a numpy
            
#             total_val_loss += val_loss.item()
            
#             # Almacenar las predicciones y los labels
#             if i_batch == 0:
#                 total_labels = real_labels
#                 total_predictions = predicted_labels
#             else:
#                 total_labels = np.concatenate((total_labels, real_labels), axis=0)
#                 total_predictions = np.concatenate((total_predictions, predicted_labels), axis=0)
        
#         final_val_loss = total_val_loss/len(val_generator)
#         val_f1_score = metrics.f1_score(total_labels, total_predictions, average='weighted')
#         val_recall = metrics.recall_score(total_labels, total_predictions, average='weighted')
#         val_precision = metrics.precision_score(total_labels, total_predictions, average='weighted')
        
#     pbar.update(n_batches, values=[("val_loss",final_val_loss), 
#                                    ("val_f1_score", val_f1_score),
#                                    ("val_recall", val_recall),
#                                    ("val_precision", val_precision)]) # Actualizar barra de progreso