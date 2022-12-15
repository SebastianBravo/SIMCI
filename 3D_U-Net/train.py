import keras
import unet
import data_generator
import numpy as np
import tensorflow as tf
import segmentation_models_3D as sm
from keras import backend as K
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from matplotlib import pyplot as plt

# GPU para entrenamiento
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Learning rate
LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)

# Función de perdida
loss = sm.losses.BinaryCELoss()

# Métricas de desempeño
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

metrics = [dice_coefficient, 'accuracy', sm.metrics.IOUScore(threshold=0.5)]

#Define the model. Here we use Unet but we can also use other model architectures from the library.
model = unet.build_unet((64,64,64,1), n_classes=1)

model.compile(optimizer = optim, loss=loss, metrics=metrics)
print(model.summary())

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/3D_U-Net/outputs/checkpoints2/model.{epoch:02d}-{val_dice_coefficient:.2f}.h5'),
    # tf.keras.callbacks.ModelCheckpoint(filepath='outputs/checkpoints2/model.{epoch:02d + 30}-{val_dice_coefficient:.2f}.h5'),
    tf.keras.callbacks.CSVLogger('F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/3D_U-Net/outputs/metrics2/metrics2.csv', separator=",", append=False),
]

# Data generator params
params = {'dim': (64,64,64),
          'batch_size': 4,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': False}

train_gen = data_generator.DataGenerator(path='F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/3D_U-Net/Data_split/train', **params)
val_gen = data_generator.DataGenerator(path='F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/3D_U-Net/Data_split/val', **params)

# Fit the model
history = model.fit(train_gen,
                    epochs=13,
                    workers=4, 
                    verbose=1,
                    validation_data = val_gen,
                    callbacks=callbacks)

# Tiempo 8125 segundos por epoca

# Modelo parcialmente entrenado
model.save(f'outputs/modelo/3d_Unet_final2.h5')

# Gráficas de los scores y losses
train_iou = history.history['iou_score']
val_iou = history.history['val_iou_score']

loss = history.history['loss']
val_loss = history.history['val_loss']

train_dice = history.history['dice_coefficient']
val_dice = history.history['val_dice_coefficient']

epochs = range(1, len(train_iou) + 1)

plt.figure(1)
plt.plot(epochs, train_iou, 'y', label='Training IOU')
plt.plot(epochs, val_iou, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.ylim([0,1])
plt.legend()
plt.grid()
plt.savefig(f'outputs/graphs/iou/3d_Unet_final_iou.svg')

plt.figure(2)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0,1])
plt.legend()
plt.grid()
plt.savefig(f'outputs/graphs/loss/3d_Unet_final_loss.svg')

plt.figure(3)
plt.plot(epochs, train_dice, 'y', label='Training DICE')
plt.plot(epochs, val_dice, 'r', label='Validation DICE')
plt.title('Training and validation Dice score')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0,1])
plt.legend()
plt.grid()
plt.savefig(f'outputs/graphs/dice/3d_Unet_final_loss.svg')