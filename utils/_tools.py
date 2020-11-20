#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import confusion_matrix

from scipy.stats import kurtosis
from scipy.stats import skew

# ----------------------------------------------------------------------------
def get_time_windows(data, nperwd, nleap):
    """
    -> np.array
    
    generates a numpy array of time windows, of length nperwd, extracted
    from data.
    
    :param pd.Series data:
      time series of measurement values.
    :param int nperwd:
      length of samples of each time window.
    :param int nleap:
      length of leap between time windows.
    
    :returns:
      a numpy array of size (n_windows, nperwd).
    """
    
    # obtener np.array de la serie de datos
    x = data.values
    n_data = x.shape[0]
    
    # determinar cantidad de ventanas a generar
    n_windows = np.floor( (n_data - nperwd)/nleap ) + 1
    n_windows = int(n_windows)
    
    # inicializar dataset
    X = np.zeros( (n_windows, nperwd) )
    
    # generar time windows
    for i in range(n_windows):
        # obtener index de la ventana
        idx_start, idx_end = i*nleap, i*nleap + nperwd
      
        # asignar datos a X
        X[i, :] = x[idx_start:idx_end]
    
    return X

# ----------------------------------------------------------------------------
def plot_confusion_matrix(Y_true, Y_pred, target_names,
                          title='Confusion matrix',
                          cmap=None, normalize=False,
                          figsize=(5,5)):
    
    """
    given the true (Y_true) and the predicted (Y_pred) labels,
    makes the confusion matrix.
    
    :param np.array Y_true:
        the true labels of the data. (no one hot encoding).
    :param np.array Y_pred:
        the predicted labels of the data by the model. (no one hot encoding).
    :param list target_names:
        given classification classes such as [0, 1, 2] the class names,
        for example: ['high', 'medium', 'low'].
    :param str title:
        the text to display at the top of the matrix.
    :param str cmap:
        the gradient of the values displayed from matplotlib.pyplot.cm
        see http://matplotlib.org/examples/color/colormaps_reference.html
        plt.get_cmap('jet') or plt.cm.Blues.
    :param bool normalize:
        if False, plot the raw numbers, if True, plot the proportions.
        
    :reference:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        
    """
    import itertools
    
    cm = confusion_matrix(Y_true, Y_pred)
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                      verticalalignment="center",
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                      verticalalignment="center",
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")
    
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
# ----------------------------------------------------------------------------
def plot_loss_function(train_info, figsize=(5,5)):
    """
    -> None
    
    this function plots de evolution of the loss function of the model 
    during the training epochs.
    
    :param train_info:
        training history of the classification model.
        
    """
    # crear figura
    plt.figure(figsize=figsize)
    
    plt.plot(train_info.history['loss'])
    plt.plot(train_info.history['val_loss'])
    
    # caracteristicas del plot
    plt.title('Model loss')
    plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
 
# ----------------------------------------------------------------------------
def extract_features(x):
    """
    -> np.array
    
    compute 9 signal features for each sample along the data x:
    - mean, variance.
    - rms, peak, valley, peak2peak.
    - crest factor, kurtosis, skewness.
    
    :param np.array x:
      data of shape (n_samples, nperwd) containing de samples.
    
    :returns:
      np.ndarray of shape (n_samples, n_features) containing
      the extracted features.
    """
    
    # mean
    mean = np.mean( x, axis=1 )
    mean = mean.reshape( (-1, 1) )
    
    # varianza
    var = np.var( x, axis=1 )
    var = var.reshape( (-1, 1) )
    
    # -----------------------------------------------------------
    # valor eficaz rms
    rms = np.sqrt( np.mean( np.square(x), axis=1 ) )
    rms = rms.reshape( (-1, 1) )
    
    # peak
    peak = np.reshape( np.max( x, axis=1 ), (-1, 1) )
    
    # valley
    valley = np.reshape( np.min( x, axis=1 ), (-1, 1) )
    
    # peak2peak
    p2p = np.reshape( np.abs( peak - valley ), (-1, 1) )
    
    # -----------------------------------------------------------
    # crest factor
    cf = np.divide( peak, rms )
    
    # kurtosis
    ktsis = np.reshape( kurtosis( x, axis=1 ), (-1, 1) )
    
    # skewness
    skwn = np.reshape( skew( x, axis=1 ), (-1, 1) )
    
    # -----------------------------------------------------------
    # concatenar features
    out = np.hstack( [mean, var, rms, peak, valley, p2p, cf, ktsis, skwn] )
    return out

# ----------------------------------------------------------------------------
def plot_img_samples(dataset, index, grid=None,
                     figsize=(5,5), title=''):
    """
    -> None
    
    this function concatenates and plot the index samples of the dataset, 
    following the rows and columns of the grid parameter.
    
    :param np.array dataset:
        dataset containing the image samples.
        it is assumed a (samples, height, width) shape.
    :param array-like index:
        list of the samples indexes to plot.
    :param tuple grid:
        (rows, cols) of images to follow in the concatenation.
        if None, it is assumed only a row of images.
        
    :returns:
        None.
    """
    
    index = np.array( index )
    
    rows, cols = grid
    _, h, w = dataset.shape
    
    # verificar que la cantidad de imagenes coincide con el grid
    assert index.size >= rows*cols
    
    # concatenar imágenes
    img = np.zeros( (rows*h, cols*w) )
    
    for i, idx in enumerate(index):
        
        # extraer imagen del dataset
        image = dataset[idx, :, :]
        image = np.reshape( image, (h, w) )
        
        vmin, vmax = np.min(image, axis=None), np.max(image, axis=None)
        
        if (vmax - vmin)!=0.0:
            image = (image - vmin)/(vmax - vmin)
        else:
            image = np.zeros_like(image)
        
        # agregar imagen a img
        k, j = i%cols, i//cols
        img[j*h:(j+1)*h, k*w:(k+1)*w] = image
        
    # plotear
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='magma')
    plt.title(title)

# ----------------------------------------------------------------------------    
def plot_predict_scatter(Y_true, Y_pred, eps=0.05, figsize=(12, 5), title='', **kargs):
    """
    -> None
    
    Plotea la disperción entre Y_true e Y_pred (este último correspondiente
    a la predicción realizada por el modelo) con el fin de visualizar el 
    desempeño del modelo.
    
    :param DataFrame Y_true:
        set de datos reales con los cuales comparar las estimaciones.
    :param DataFrame Y_pred:
        set de datos obtenidos con el modelo.
    :param float eps:
        radio que determina la vecindad de los puntos.
    :param str title:
        titulo a poner en el plot.
        
    :returns:
        None
    """
    
    # reordenar datos
    Y_true = np.array(Y_true).flatten()
    Y_pred = np.array(Y_pred).flatten()
    
    # calcular densidad
    z = np.zeros_like(Y_true)
    
    for i in range( Y_true.shape[0] ):
        # distancia
        dist = np.sqrt( (Y_true[i] - Y_true)**2 + (Y_pred[i] - Y_pred)**2 )
        # cantidad de vecinos
        z[i] = np.sum( dist < eps )
        
    # ordenar datos
    idx = np.argsort(z)
    Y_true, Y_pred, z = Y_true[idx], Y_pred[idx], z[idx]
    
    # obtener colores
    cmap = cm.get_cmap('jet')
    z_min, z_max = np.min(z), np.max(z)
    
    colors = cmap( (z - z_min)/(z_max - z_min) )
    
    # inicializar plot
    fig = plt.figure(figsize=figsize)
    
    # plotear Y_true vs Y_pred
    plt.scatter(Y_true, Y_pred, c=colors,**kargs)
    
    # plotear linea 1:1
    plt.plot([0, 1], [0, 1], c='k')
    
    # añadir limites
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    #plt.title(title)
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.show()
    return None

# ----------------------------------------------------------------------------    
def plot_RUL_scatter(Y_true, Y_pred, eps=0.05, figsize=(12, 5), title='', **kargs):
    """
    -> None
    
    Plotea la disperción entre Y_true e Y_pred (este último correspondiente
    a la predicción realizada por el modelo) con el fin de visualizar el 
    desempeño del modelo para estimar el RUL de la turbina.
    
    :param DataFrame Y_true:
        set de datos reales con los cuales comparar las estimaciones.
    :param DataFrame Y_pred:
        set de datos obtenidos con el modelo.
    :param float eps:
        radio que determina la vecindad de los puntos.
    :param str title:
        titulo a poner en el plot.
        
    :returns:
        None
    """
    
    # reordenar datos
    Y_true = np.array(Y_true).flatten()
    Y_pred = np.array(Y_pred).flatten()
    
    # calcular densidad
    z = np.zeros_like(Y_true)
    
    for i in range( Y_true.shape[0] ):
        # distancia
        dist = np.sqrt( (Y_true[i] - Y_true)**2 + (Y_pred[i] - Y_pred)**2 )
        # cantidad de vecinos
        z[i] = np.sum( dist < eps )
        
    # ordenar datos
    idx = np.argsort(z)
    Y_true, Y_pred, z = Y_true[idx], Y_pred[idx], z[idx]
    
    # obtener colores
    cmap = cm.get_cmap('jet')
    z_min, z_max = np.min(z), np.max(z)
    
    colors = cmap( (z - z_min)/(z_max - z_min) )
    
    # inicializar plot
    fig = plt.figure(figsize=figsize)
    
    # plotear Y_true vs Y_pred
    plt.scatter(Y_true, Y_pred, c=colors,**kargs)
    
    # obtener máximos y mínimos
    pred_min, pred_max = np.min(Y_pred), np.max(Y_pred)
    true_min, true_max = np.min(Y_true), np.max(Y_true)
    
    # plotear linea 1:1
    plt.plot([true_min, true_max], [pred_min, pred_max], c='k')
    
    # añadir limites
    plt.xlim([true_min, true_max])
    plt.ylim([pred_min, pred_max])
    
    #plt.title(title)
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.show()
    return None

# ----------------------------------------------------------------------------    
def plot_RUL_sorted(Y_true, Y_pred, figsize=(12, 5), title='', **kargs):
    """
    -> None
    
    Plotea una comparación ordenada entre Y_true e Y_pred (este último
    corresponditente a la predicción realizada por el modelo) con el fin de
    visualizar el desempeño del modelo para estimar el RUL de la turbina.
    
    Plotea la disperción entre Y_true e Y_pred (este último correspondiente
    a la predicción realizada por el modelo) con el fin de visualizar el 
    desempeño del modelo para estimar el RUL de la turbina.
    
    :param DataFrame Y_true:
        set de datos reales con los cuales comparar las estimaciones.
    :param DataFrame Y_pred:
        set de datos obtenidos con el modelo.
    :param str title:
        titulo a poner en el plot.
        
    :returns:
        None
    """
    
    # reordenar datos
    Y_true = np.array(Y_true).flatten()
    Y_pred = np.array(Y_pred).flatten()
    
    # sorting
    sort_idx = np.argsort(Y_true)
    Y_true = Y_true[sort_idx]
    Y_pred = Y_pred[sort_idx]
    
    # plot
    fig = plt.figure(figsize=figsize)
    
    plt.plot(Y_true, linestyle='-', linewidth=0.5, marker='o', markersize=3, label = 'True')
    plt.plot(Y_pred, linestyle='-', linewidth=0.5, marker='^', markersize=3, label = 'Prediction')
    
    #plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('RUL')
    plt.grid(True)
    plt.show()
    
    return None
