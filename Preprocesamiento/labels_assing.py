import os
import shutil
import numpy as np
import pandas as pd
import splitfolders

"""-------------------Asiganación de labels para cada MRI'------------------"""
# Path imágenes MRI
# mri_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/dataset_brain_all_scores'
mri_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/dataset_brain_no_scores'

# Path labels
# MCI_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/data_all_scores/MCI'
# CN_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/data_all_scores/CN'

MCI_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/data_no_scores/MCI'
CN_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/data_no_scores/CN'

# Importación csv con labels
# labels = pd.read_csv(r'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/labels_all_scores.csv', delimiter=',')
labels = pd.read_csv(r'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/labels_no_scores.csv', delimiter=',')
# labels['sospechosa'] = list(np.zeros(len(labels)))

error_mri = []
idx = 0
for patient in os.listdir(mri_path):
    try:
        # Obtener índice de paciente
        patient_idx = list(labels['serie']).index(patient[:-4])
        
        # Obtener etiqueta del paciente
        label = labels['Group'][patient_idx]
        
        # Path de archivo de paciente
        source_path = os.path.join(mri_path,patient)
        
        if label == 'CN':
            # Path de guardado con etiqueta NC
            out_path = os.path.join(CN_path,patient)
             
        if label == 'MCI':
            # Path de guardado con etiqueta MCI
            out_path = os.path.join(MCI_path,patient)
            
        # Guardar copia
        shutil.copy(source_path, out_path)
        
        # Conteo de imágenes guardadas
        print(f'Paciente {idx} {patient} guardado en carpeta {label}', flush=True)
        idx = idx+1
        
        # Cargar imágen
        # mri_image = np.load(os.path.join(mri_path,patient))
        
        # if np.sum(mri_image) < 120000:
        #     labels['sospechosa'][patient_idx] = 1
        #     print(f'Paciente {idx} {patient}: Revisar', flush=True)
        #     error_mri.append(patient)
        # else:
        #     labels['sospechosa'][patient_idx] = 0
        #     print(f'Paciente {idx} {patient}: Correcto', flush=True)
        
        
    except:
        # imagenes con error
        print(f'Falló el guardado de paciente: {patient}', flush=True)
        error_mri.append(patient)
        
# Segmentación de conjuntos
# splitfolders.ratio('F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/data_all_scores', output='data_all_scores_split', seed=1337, ratio=(0.60, 0.20, 0.20))
splitfolders.ratio('F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/data_no_scores', output='data_no_scores_split', seed=1337, ratio=(0.80, 0.20))
