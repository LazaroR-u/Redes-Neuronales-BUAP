# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 17:52:07 2022

@author: Lazaro Diaz
"""

import mnist_loader
import network
import pickle

#Dividimos los datos en datos de entrenamiento, de validacion y de prueba
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

#nos aseguramos que tengan el formato de listas 
training_data = list(training_data)
test_data = list(test_data)

#definimos la red neuronal con 784 neuronas de entrada, 30 intermedias y 10 de salida
#784 es la cantidad de pixeles de la imagen y 10 es la cantidad de digitos a reconocer
net = network.Network([784,30,10])



#ENTRENAMIENTO 
#usamos el algoritmo Stochastic Gradient Descent con momentum para entrenar el modelo, se 
#usaron los datos de entrenamiento con 30 epocas, 10 mini batch y  un learning rate de 3.0 y un momentum de 0.91
# y se compararon los datos esperados con los datos de prueba
net.SGD_momentum( training_data, 30, 10, 3.0,0.91, test_data=test_data)


#archivo = open("red_prueba_RMS.pkl",'wb')


#pickle.dump(net,archivo)
#archivo.close()
#exit()
#leer el archivo
#archivo_lectura = open("red_prueba_RMS_1.pkl",'rb')
#net = pickle.load(archivo_lectura)
#archivo_lectura.close()
#net.SGD( training_data, 10, 50, 0.5, test_data=test_data)
#archivo = open("red_prueba_RMS_1.pkl",'wb')
#pickle.dump(net,archivo)

#archivo.close()
#exit()