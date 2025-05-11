__authors__ = ['1632753', '1633672', '1634802']
__group__ = ['DJ.17', 'DL.10']

import numpy as np
import KNN
from KNN import *
from Kmeans import *
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud, read_extended_dataset, crop_images
from operator import itemgetter
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from operator import itemgetter
import Kmeans


def retrieval_by_color(lista_imagenes, etiquetas, pregunta):
    resultados = []

    for i in range(len(lista_imagenes)):
        imagen = lista_imagenes[i]
        etiqueta = etiquetas[i]
        if set(pregunta).issubset(etiqueta):
            resultados.append(imagen)

    return resultados


def retrieval_by_shape(images, labels, question):
    image_list = []

    for index, label in enumerate(labels):
        if all(q in label for q in question):
            image_list.append(images[index])

    return image_list


def kmean_statistics(kmeans, kmax, test_imgs):

    config=['Fischer','Intra','Inter']
    for i in config:
        kmeans.find_bestKBetter(kmax, i)
        visualize_k_means(kmeans, [80, 60, 3])


def get_shape_accuracy(result, real):
    count = 0
    for predicted, true_label in zip(result, real):
        if predicted == true_label:
            count += 1
    precision = (count / len(result)) * 100
    return precision


def get_color_accuracy(predicted, real):
    correct_labels = 0.0
    total = len(predicted)
    for i in range(len(predicted)):

        pred_set = set(predicted[i])
        true_set = set(real[i])
        intersection = pred_set.intersection(true_set)
        similarity = len(intersection) / max(len(pred_set), len(true_set))

        if similarity >= 0.5:
            correct_labels += 1

    precision = (correct_labels/ total) * 100
    return precision


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    # List with all the existent classes
    lista = []
    preconfig=['first','random','custom']
    config=['Fischer','Intra','Inter']
    listaColores=[]
    precision=[]
    subset = class_labels[0:180]
    listaTipo=[]
    '''
    # Test temps i precisió configuració inicial i heurística (versió millorada)
    listaTiempo = []
    for i in preconfig:
        for j in config:
            start_time = time.time()
            for k in range(0,180):
                km = KMeans(cropped_images[k], k)
                km.find_bestKBetter(10, j)
                listaColores.append(get_colors(km.centroids))
            end_time = time.time()
            listaTiempo.append(end_time-start_time)
            precision.append(get_color_accuracy(listaColores, color_labels))
            listaColores = []
            listaTipo.append(i + ' i ' + j)

    plt.figure(figsize=(15,10))
    plt.plot(listaTipo, precision)
    plt.title("Tasa d'encert Kmeans segons heurística i configuració inicial")
    plt.xlabel("Configuració inicial i Heurística")
    plt.ylabel("% d'encert")
    plt.show()
    plt.figure(figsize=(15, 10))
    plt.plot(listaTipo, listaTiempo)
    plt.title("Temps empleat")
    plt.xlabel("Configuració inicial i Heurística")
    plt.ylabel("Segons")
    plt.show()
    '''
    # Test temps i precisió configuració inicial (versió inicial)
    '''
    
    for i in preconfig:
        for k in range(0,180):
            km = KMeans(cropped_images[k], i)
            km.find_bestK(10)
            listaColores.append(get_colors(km.centroids))
        precision.append(get_color_accuracy(listaColores, color_labels))
        listaColores = []
    plt.figure(figsize=(15, 10))
    plt.plot(preconfig, precision)
    plt.title("Tasa d'encert Kmeans segons configuració inicial")
    plt.xlabel("Configuració inicial")
    plt.ylabel("% d'encert")
    plt.show()
    '''
    '''
    #Test kmean_statistics

    for i in range(20,30):
        km=KMeans(imgs[i])
        kmean_statistics(km, 10, imgs[i])
    '''

    #Test retrieval_by_color
    '''
    lista = []
    for i in range(0, 180):
        km = KMeans(cropped_images[i])
        km.find_bestKBetter(10,'Inter')
        lista.append(get_colors(km.centroids))
    matching = retrieval_by_color(imgs[:80], lista, ['Pink'])
    visualize_retrieval(np.array(matching), len(matching), title='Mostra per color d´imatge')
    '''
    #Test retrieval_by_shape
    '''
    knn = KNN(imgs[:180], class_labels[:180])
    preds = knn.predict(imgs[:180], 2)
    matching_shape = retrieval_by_shape(imgs[:180], preds, ['Sandals'])
    visualize_retrieval(np.array(matching_shape), len(matching_shape), title='Mostra per classe d´imatge')
    '''
    # Test retrieval_by_color
    '''
    lista = []
    for i in range(0, 180):
        km = KMeans(cropped_images[i])
        km.find_bestK(10)
        lista.append(get_colors(km.centroids))
    matching = retrieval_by_color(imgs[:180], lista, ['Pink'])
    visualize_retrieval(np.array(matching), len(matching), title='Mostra per color d´imatge(Rosa)')
    '''
    # Test retrieval_by_shape
    '''
    knn = KNN(imgs[:180], class_labels[:180])
    preds = knn.predict(imgs[:180], 2)
    matching_shape = retrieval_by_shape(imgs[:180], preds, ['Sandals'])
    visualize_retrieval(np.array(matching_shape), len(matching_shape), title='Mostra per classe d´imatge(Sandalies)')
    '''
    #Test get_shape_accuracy variant la K
    '''
    listaPrecicion=[]
    listaK=[]
    knn = KNN(imgs[:180], class_labels[:180])
    for i in range(2,10):
        preds = knn.predict(imgs[:180], i)
        precision=get_shape_accuracy(preds, class_labels[:180])
        listaPrecicion.append(precision)
        listaK.append(i)
    

    plt.figure(figsize=(15, 10))
    plt.plot(listaK, listaPrecicion)
    plt.title("Tasa d'encert KNN segons K")
    plt.xlabel("K")
    plt.ylabel("% d'encert")
    plt.show()
    '''
    #Presició Kmeans i KNN sense millores
    '''
    for k in range(0, 180):
        km = KMeans(cropped_images[k])
        km.find_bestK(10)
        listaColores.append(get_colors(km.centroids))
    print("La presició de Kmeans és: ", (get_color_accuracy(listaColores, color_labels)))
    knn = KNN(imgs[:180], class_labels[:180])
    preds = knn.predict(imgs[:180], 2)
    precision = get_shape_accuracy(preds, class_labels[:180])
    print ("La presició de KNN és: ", precision)
    '''