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

[![HOLA](https://raw.githubusercontent.com/cherrerab/deeplearningfallas/master/workshop_01/bin/video.png)](https://youtu.be/p483_exDlVY)

