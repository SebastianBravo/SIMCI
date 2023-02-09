import os
import csv
import wandb
import torch
import resnet
import logging
import torchinfo
import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics
from torch import nn
from torchsummary import summary
from dataset_class import Dataset
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

# Login para log de metricas
wandb.login(key='d03749a4afd40541d752e73b1b8ef4d40358d4d1')
os.environ["WANDB_SILENT"] = "true"
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

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

# Weight path
weight_path = '/HDDmedia/simci/SIMCI/Procesamiento/mednet_weights/resnet_50_23dataset.pth'

# Clase para agregar capa totalmente conectada
class simci_net(nn.Module):
    def __init__(self):
        super(simci_net, self).__init__()
        
        self.pretrained_model = resnet.resnet50(sample_input_D=192, sample_input_H=256, sample_input_W=256, num_seg_classes=2, no_cuda = False)
        
        self.pretrained_model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
                                                       nn.Flatten(start_dim=1))
        
        self.classifier = nn.Sequential(nn.Linear(2048, 1),
                                        nn.Sigmoid())

    def forward(self, x):
        x = self.pretrained_model(x)   
        out = self.classifier(x)
        
        return out

def init_model(weigth_path):
    # Definición modelo simci 
    simci_model = simci_net()
    
    # Diccionario state
    net_dict = simci_model.state_dict()
    
    # Cargar pesos
    weigth = torch.load(weigth_path, map_location=torch.device('cpu'))
    
    # Transferencia de aprendizaje
    pretrain_dict = {k.replace("module.", ""): v for k, v in weigth['state_dict'].items() if k.replace("module.", "") in net_dict.keys()}
    net_dict.update(pretrain_dict)
    simci_model.load_state_dict(net_dict)
    
    # Summary sin bloquear parametros mednet
    # torchinfo.summary(simci_model, (1,192, 256, 256), batch_dim = 0, col_names = ('input_size', 'output_size', 'num_params', 'trainable'), verbose = 0)
    
    # Bloqueo de parametros mednet
    for param in simci_model.pretrained_model.parameters():
        param.requires_grad = False
        
    # Summary bloqueando parametros mednet
    # torchinfo.summary(simci_model, (1,192, 256, 256), batch_dim = 0, col_names = ('input_size', 'output_size', 'num_params', 'trainable'), verbose = 0)
        
    return simci_model
'''-------------------------------------------------------------------------'''

# Número de folds para k-fold cross-validation
num_folds = 2

# Cración objeto para validación cruzada
kfold = KFold(n_splits=num_folds, shuffle=True)

# Parametros para generación de datos
gen_param = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 8}

# Número de épocas
epochs = 2

# Lista para almacenar scores de cada fold
scores = dict() # [Presicion, recall, F1-score] --> (weighted macro f1 score), almacenar matríz confusión

# Cliclo de experimentos
exp_idx = 1
# Crear carpeta de experimento
os.makedirs(f"Modelos/Experimento_{exp_idx}")

# Ciclo de entrenamiento con validación cruzada
fold_idx = 1
for train_index, val_index in kfold.split(data_files):
    # Crear carpeta para fold
    fold_path = f"Modelos/Experimento_{exp_idx}/fold_{fold_idx}"
    os.makedirs(fold_path)
    
    print('Inicializando modelo...')
    # Inicializar modelo
    simci_model = init_model(weight_path)

    # Data parallel
    simci_model = torch.nn.DataParallel(simci_model, device_ids = device_ids)
    simci_model.to(f'cuda:{simci_model.device_ids[0]}')
    
    # Loss: Binary cross-entropy
    loss_func = nn.BCELoss()
    loss_func.to(f'cuda:{simci_model.device_ids[0]}')

    # Optimizador Adam
    learning_rate = 0.001
    optimizer = torch.optim.Adam(simci_model.parameters(), lr=learning_rate)

    print('Modelo inicializado')
    
    print('Inicializando Dashboard...')
    # # Inicializar dashboard
    # wandb.init(
    #   # Set the project where this run will be logged
    #   project=f"Experimento_{exp_idx}", 
    #   # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    #   name=f"fold_{fold_idx}", 
    #   # Track hyperparameters and run metadata
    #   config={
    #   "learning_rate": learning_rate,
    #   "architecture": "CNN",
    #   "epochs": epochs,
    #   })
    
    writer = SummaryWriter(f'runs/Experimento_{exp_idx}/fold_{fold_idx}')
    
    print('Dashboard inicializado')
    
    print(f'-------------------------------------------------------- Fold {fold_idx}/{num_folds} --------------------------------------------------------')
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
        print(f'Fold {fold_idx}/{num_folds} - Epoch {epoch+1}/{epochs}')
        pbar = tf.keras.utils.Progbar(target=n_batches, stateful_metrics=["val_loss","val_f1_score","val_recall","val_precision"])

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
            target_npy = np.round(target.detach().cpu().numpy())
            
            total_train_loss += loss.item() # Suma de loss para cada batch
            
            pbar.update(i_batch, values=[("loss",loss.item())]) 
            
            # Almacenar las predicciones y los labels
            if i_batch == 1:
                total_train_labels = labels_npy
                total_train_predictions = target_npy
            else:
                total_train_labels = np.concatenate((total_train_labels, labels_npy), axis=0)
                total_train_predictions = np.concatenate((total_train_predictions, target_npy), axis=0)
               
        # Calculo metricas
        final_train_loss = total_train_loss/n_batches
        train_cf_matrix = metrics.confusion_matrix(total_train_labels, total_train_predictions)
        train_f1_score = metrics.f1_score(total_train_labels, total_train_predictions, average='weighted')
        train_recall = metrics.recall_score(total_train_labels, total_train_predictions, average='weighted')
        train_precision = metrics.precision_score(total_train_labels, total_train_predictions, average='weighted')
        
        # Ciclo de validación
        with torch.no_grad():
            simci_model.eval() # Modelo en modo evaluación
            
            for i_batch, (data, labels) in enumerate(val_generator,0):
                data, labels = data.to(f'cuda:{simci_model.device_ids[0]}'), labels.to(f'cuda:{simci_model.device_ids[0]}').float()
                
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
            
            # Calculo metricas
            final_val_loss = total_val_loss/len(val_generator)
            val_cf_matrix = metrics.confusion_matrix(total_labels, total_predictions)
            val_f1_score = metrics.f1_score(total_labels, total_predictions, average='weighted')
            val_recall = metrics.recall_score(total_labels, total_predictions, average='weighted')
            val_precision = metrics.precision_score(total_labels, total_predictions, average='weighted', zero_division=1)
        
        pbar.update(n_batches, values=[("train_f1_score", train_f1_score),
                                       ("train_recall", train_recall),
                                       ("train_precision", train_precision),
                                       ("val_loss",final_val_loss),
                                       ("val_f1_score", val_f1_score),
                                       ("val_recall", val_recall),
                                       ("val_precision", val_precision)]) # Actualizar barra de progreso 
        
        # Guardado de metricas
        metrics_dict = {"train_loss": final_train_loss,
                        "train_f1_score": train_f1_score,
                        "train_recall": train_recall,
                        "train_precision": train_precision,
                        "val_loss": final_val_loss, 
                        "val_f1_score": val_f1_score,
                        "val_recall": val_recall,
                        "val_precision": val_precision}
        
        csv_file_name = f"metrics_fold_{fold_idx}.csv"
        
        with open(os.path.join(fold_path, csv_file_name), 'a') as f:
            w = csv.DictWriter(f, metrics_dict.keys())
            
            if epoch == 0:
                w.writeheader()
            
            w.writerow(metrics_dict)
        
        # wandb.log(metrics_dict)
        writer.add_scalar("train_loss", final_train_loss, epoch)
        writer.add_scalar("train_f1_score", train_f1_score, epoch)
        writer.add_scalar("train_recall", train_recall, epoch)
        writer.add_scalar("train_precision", train_precision, epoch)
        writer.add_scalar("val_loss", final_val_loss, epoch)
        writer.add_scalar("val_f1_score", val_f1_score, epoch)
        writer.add_scalar("val_recall", val_recall, epoch)
        writer.add_scalar("val_precision", val_precision, epoch)

    print('---------------------------------------------------------------------------------------------------------------------------')
    # wandb.finish(quiet=True)
    writer.flush()
    
    # Guardar Modelo
    print("Guardando modelo...")
    model_filename = f"model_fold_{fold_idx}_epochs_{epochs}.pth"
    torch.save({'model_state_dict': simci_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),}, os.path.join(fold_path, model_filename))
    print("Modelo guardado")
    
    print("Guardando matrices...")
    cf_train_filename = f"cf_train_fold_{fold_idx}_epochs_{epochs}.npy"
    cf_val_filename = f"cf_val_fold_{fold_idx}_epochs_{epochs}.npy"
    
    np.save(os.path.join(fold_path, cf_train_filename), train_cf_matrix)
    np.save(os.path.join(fold_path, cf_val_filename), val_cf_matrix)
    print("Matrices guardadas")
    
    fold_idx += 1
    
    # Para empezar dashboard -> tensorboard --logdir runs