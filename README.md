![banner](bin/banner.png)
Este repositorio contiene los talleres prácticos y el material complementario del curso ME7260 Aprendizaje Profundo en Diagnóstico y Pronóstico de Fallas.

## Google Colab
Al desarrollar proyectos de Deep Learning, la mayor limitante suele ser el poder computacional disponible (CPU/GPU) para ejecutarlos. [**Google Colaboratory**](https://colab.research.google.com/notebooks/welcome.ipynb) es un entorno de Jupyter notebook gratuito que se ejecuta completamente en la nube. La plataforma permite tanto escribir como ejecutar código, y solo se requiere de una cuenta google.

https://colab.research.google.com/notebooks/welcome.ipynb

Dentro de este repositorio podrá encontrar los notebooks (\*.ipynb) de los distintos workshops a realizar a lo largo del curso. Descárguelos y ábralos dentro de su propia sesión de Colab.

## Clonar GitHub
En caso de requerir importar archivos y/o elementos de este repositorio a un entorno de Google Colab, puede clonarlo directamente mediante:

`! git clone https://github.com/cherrerab/deeplearningfallas.git`

Si el repositorio se ha clonado correctamente, notará que en la carpeta de archivos `/content` se habrá creado el directorio `deeplearningfallas`. Para utilizar este directorio dentro de la sesión, utilice el siguiente comando:

`%cd /content/deeplearningfallas`

De este modo será posible importar las funciones y utilidades contenidas dentro del repositorio:

`>>> from utils._tools import *`

## Tutorial Data Manipulation
Numpy y Pandas son librerías utilizadas extensamente en la ciencia de datos y el Machine Learning. Fundamentalmente, ambas están orientadas a la manipulación de arreglos multidimensionales y bases de datos, por supuesto, cada una con sus estructuras de datos y operaciones particulares. En este tutorial aprenderá las funcionalidades básicas de estas librerías que le permitirán abordar los próximos workshops del curso.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cherrerab/deeplearningfallas/blob/master/workshop_01/tutorial_01.ipynb)

## Workshop MFPT Data Manipulation
La Machinery Failure Prevention Technology es una organización orientada a la integración y al intercambio de información técnica entre distintas comunidades científicas y de ingeniería, con el fin de avanzar en el estudio de los mecánismos de falla. En este workshop trabajaremos con su Bearing Fault Dataset, el cual contiene mediciones experimentales de vibración de rodamientos de bola bajo distintos estados de daño.

Mediante este workshop aprenderá el framework clásico de exploración, análisis y preprocesamiento de bases de datos. Estos pasos son fundamentales para desarrollar con éxito cualquier proyecto de Data Science y Deep Learning.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cherrerab/deeplearningfallas/blob/master/workshop_01/workshop_01.ipynb)

[![Open In Youtube](https://raw.githubusercontent.com/cherrerab/deeplearningfallas/master/workshop_01/bin/auxvid.png)](https://youtu.be/p483_exDlVY)

## Workshop MFPT Classification Model
Continuando con el procesamiento de datos del workshop anterior, en esta oportunidad aprenderá como construir y configurar un modelo de clasificación de Deep Learning mediante TensorFlow y Keras. Entender el framework de estas herramientas le permitirá seguir avanzando con modelos y arquitecturas de mayor complejidad a medida que estas sean introducidas en el curso.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cherrerab/deeplearningfallas/blob/master/workshop_02/workshop_02_pauta.ipynb)

[![Open In Youtube](https://raw.githubusercontent.com/cherrerab/deeplearningfallas/master/workshop_02/bin/auxvid_2.png)](https://youtu.be/kDrmyoKVmJE)

## Workshop Defective PV Module Cells
Entre los métodos que permiten llevar a cabo una inspección visual de la condición de las celdas solares, las imágenes de electroluminiscencia (EL) proveen de imágenes de alta resolución que posibilitan la detección de defectos de menor tamaño en la superficie de las celdas. En este workshop dispondremos de un dataset público de imágenes de electroluminiscencia de celdas solares fotovoltaicas en distintos estado de daño con el cual desarrollaremos y entrenaremos un modelo de diagnóstico en base a redes convolucionales (CNN) mediante TensorFlow y Keras.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cherrerab/deeplearningfallas/blob/master/workshop_03/workshop_03.ipynb)

[![Open In Youtube](https://raw.githubusercontent.com/cherrerab/deeplearningfallas/master/workshop_03/bin/auxvid.png)](https://youtu.be/xk-wfL7W8D0)


## Workshop ELPV Transfer Learning
En ocasiones, no resulta del todo eficiente entrenar un modelo convolucional desde cero pues es posible que para alcanzar una mejor abstracción en la extracción de features dentro del modelo se requieran una mayor cantidad de datos e incluso un mayor tiempo de entrenamiento, lo que no siempre será factible. El Transfer Learning consiste en adaptar modelos previamente desarrollados para facilitar el aprendizaje sobre un nuevo problema de interés, del mismo modo en que alguien que sabe guitarra podría aprender a tocar bajo más rapidamente que alguien sin ninguna experiencia musical. Así, en este workshop volveremos a usar el dataset de electroluminiscencias de celdas solares fotovoltaicas para el desarrollo de un modelo convolucional clasificación mediante transfer learning.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cherrerab/deeplearningfallas/blob/master/workshop_04/workshop_04.ipynb)

[![Open In Youtube](https://raw.githubusercontent.com/cherrerab/deeplearningfallas/master/workshop_04/bin/auxvid.png)](https://youtu.be/PVufdn9dwx8)

## Workshop C-MAPSS Dataset
El Comercial Modular Aero-Propulsion System Simulation (C-MAPSS) es un software desarrollado por NASA como ambiente de simulación de motores de reacción tipo turbofán. Así, esta herramienta permite la implementación y evaluación de algoritmos de control y diagnóstico sobre la operación de un motor turbofán de 90.000 lbf de propulsión. En este workshop implementaremos un modelo recurrente mediante `tensorflow` y `keras` para la predicción del RUL de turbofanes de aviación, a partir de la serie temporal de mediciones de sus sensores.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cherrerab/deeplearningfallas/blob/master/workshop_05/workshop_05.ipynb)

[![Open In Youtube](https://raw.githubusercontent.com/cherrerab/deeplearningfallas/master/workshop_05/bin/auxvid.png)](https://youtu.be/OVQxnClnmO8)

## Workshop C-MAPSS Anomaly Detection Autoencoder
La detección de anomalías (Anomaly Detection) sobre sistemas con una alta dimensionalidad de datos, es un problema de particular interés tanto en el campo del Machine Learning como también en diversas áreas de la ingeniería e industria. En particular, los algoritmos de detección de anomalías resultan de suma relevancia a la hora de desarrollar e implementar sistemas de monitoreo de equipos o activos industriales de mayor complejidad. En este workshop volveremos a utilizar el C-MAPSS Dataset de la NASA para desarrollar un detector de anomalías mediante un Autoencoder. Así, en este caso reconfiguraremos los datos de las simulaciones ya no para predecir el tiempo de vida remanente o Remaining Useful Life (RUL) de cada turbina, sino para clasificar si la turbina se encuentra en un estado de operación nominal o en uno de degradación crítica.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cherrerab/deeplearningfallas/blob/master/workshop_06/workshop_06.ipynb)

[![Open In Youtube](https://raw.githubusercontent.com/cherrerab/deeplearningfallas/master/workshop_06/bin/auxvid.png)](https://youtu.be/iwr5JfoJQVs)


## Workshop C-MAPSS Anomaly Detection Variational Autoencoder
Si bien los Autoencoders son capaces de reducir la dimensionalidad de una determinada estructura de datos a un vector latente mediante su etapa de `encoding`, estos tienen el problema de que finalmente procesan o codifican cada sample de manera independiente. En este sentido, considerando que todos los datos o samples utilizados para el `training` del modelo son de la misma clase o describen el mismo sistema, nos gustaría que el `encoding` del vector latente sea capaz de capturar la distribución de los datos en su nueva representación. Aquí es donde entran los Variational Autoencoders (VAE) en los cuales, en términos generales, se impone (forzadamente) una distribución normal sobre cada uno de los componentes del vector latente.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cherrerab/deeplearningfallas/blob/master/workshop_07/workshop_07.ipynb)

[![Open In Youtube](https://raw.githubusercontent.com/cherrerab/deeplearningfallas/master/workshop_07/bin/auxvid_1.png)](https://youtu.be/3JFGHw6dBAE)
[![Open In Youtube](https://raw.githubusercontent.com/cherrerab/deeplearningfallas/master/workshop_07/bin/auxvid_2.png)](https://youtu.be/IdaVZZsbpw4)

## Workshop Generative Adversarial Networks
En términos simples, un modelo GAN se compone de dos modelos independientes: el `Discriminador` y, por supuesto, el `Generador`. Por un lado, el `Discriminador`, como su nombre lo indica, está diseñado para discriminar las imágenes creadas artificalmente por el `Generador` de las imágenes reales contenidas en el dataset. Así, el `Discriminador` procesa secuencialmente la información contenida en la imagen de entrada mediante una arquitectura Conv2D, para finalmente retornar la etiqueta de clasificación de la imagen (`real` o `fake`) mediante una capa softmax.

Por el otro lado, el `Generador` está diseñado para construir imágenes a partir de un vector de ruido uniforme `NOISE_VECTOR`. Por supuesto, el tamaño de este vector o `NOISE_DIM` será otro de los hiperparámetros que constituirá el modelo. De este modo, de manera inversa a un modelo Convolucional convencional, el `Generador` procesa este vector de ruido mediante un arquitectura Conv2DTranspose con la finalidad de ir construyendo secuencialmente la imagen RGB de salida.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cherrerab/deeplearningfallas/blob/master/workshop_08/workshop_08.ipynb)


[![Open In Youtube](https://raw.githubusercontent.com/cherrerab/deeplearningfallas/master/workshop_08/bin/auxvid.png)](https://youtu.be/tA-SDLDzUu8)
