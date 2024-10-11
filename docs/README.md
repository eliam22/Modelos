# Modelos de Machine Learning

## Descripción General
Este repositorio contiene varios modelos de machine learning aplicados a problemas de predicción, clasificación y análisis de clústeres. Los modelos incluyen regresión lineal, K-Nearest Neighbors (KNN), un árbol de decisión y un análisis de clústeres utilizando K-Means. Cada modelo está documentado en su propio notebook, que ofrece una explicación detallada de la implementación, los pasos seguidos y los resultados obtenidos.


## Contenido del Repositorio
- **ArbolDecision.ipynb**: Implementación de un modelo de árbol de decisión para la clasificación de estrellas.
- **KNN.ipynb**: Modelo KNN para la clasificación de estrellas.
- **RegresionLineal.ipynb**: Modelo de regresión lineal para predecir el precio del oro.
- **KMeans_Clustering.ipynb**: Análisis de clústeres utilizando K-Means para segmentar datos de ventas electrónicas.

## Requisitos
Para ejecutar los notebooks, asegúrate de tener instalados los siguientes componentes:

- **Python 3.x**
- **Bibliotecas**:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

## Instrucciones para Ejecutar los Notebooks
1. Clona el repositorio:

    ```bash
    git clone https://github.com/eliam22/Modelos.git
    ```

2. Navega al directorio de los notebooks:

    ```bash
    cd Modelos/notebooks
    ```

3. Inicia el servidor Jupyter:

    ```bash
    jupyter notebook
    ```

## Proyectos

### 1. Proyecto: Predicción del Precio del Oro (Regresión Lineal)
**Introducción**  
Este proyecto busca predecir el precio del oro utilizando un conjunto de datos históricos. Se aplicaron técnicas de regresión para optimizar la precisión del modelo.

**Pasos**  
- **Análisis Exploratorio**: Evaluación de la correlación entre las características y el precio del oro.
- **Manejo de Valores Faltantes**: Imputación de valores faltantes mediante la media.
- **Modelado**: Entrenamiento y evaluación de múltiples modelos de regresión, utilizando métricas como el MAE y el MSE.

**Conclusiones**  
Se identificaron correlaciones significativas entre las características y el precio del oro, y se seleccionó el modelo más preciso según diversas métricas.

### 2. Proyecto: Clasificación de Estrellas (KNN)
**Introducción**  
Este proyecto clasifica diferentes tipos de estrellas mediante el algoritmo K-Nearest Neighbors (KNN).

**Pasos**  
- **Preprocesamiento**: Limpieza de datos y conversión a formato numérico.
- **Modelado**: Entrenamiento del modelo KNN y ajuste de hiperparámetros.
- **Evaluación**: Cálculo de precisión y generación de una matriz de confusión para evaluar el rendimiento del modelo.

**Conclusiones**  
El modelo KNN logró una precisión del 72.73% con un valor de k=5.

### 3. Proyecto: Clasificación de Estrellas (Árbol de Decisión)
**Introducción**  
Este proyecto implementa un modelo de árbol de decisión para clasificar estrellas en diversas categorías, basándose en sus características físicas.

**Pasos**  
- **Preprocesamiento**: Normalización de características y balanceo de clases.
- **Modelado**: Entrenamiento del modelo de árbol de decisión.
- **Resultados**: Evaluación de la precisión y análisis de la importancia de las características.

**Conclusiones**  
La precisión del modelo fue insatisfactoria, sugiriendo la necesidad de ajustes adicionales para mejorar el rendimiento.

### 4. Proyecto: Análisis de Clústeres (K-Means)
**Introducción**  
Este proyecto realiza un análisis de clústeres utilizando el algoritmo K-Means para segmentar datos de ventas electrónicas.

**Pasos**  
- **Preprocesamiento**: Eliminación de columnas no numéricas y codificación de variables categóricas.
- **Aplicación de K-Means**: Determinación del número óptimo de clústeres mediante el método del codo y ajuste del modelo K-Means.
- **Visualización**: Visualización de los clústeres formados en gráficos de dispersión.

**Conclusiones**  
Los resultados del análisis de clústeres permiten identificar patrones en los datos de ventas, facilitando la segmentación de clientes.

**Pasos**  
- **Preprocesamiento**: Normalización de características y balanceo de clases.
- **Modelado**: Entrenamiento del modelo de Random Forest.
- **Resultados**: Evaluación de la precisión y análisis de la importancia de las características.

**Conclusiones**  
La precisión del modelo fue insatisfactoria, sugiriendo la necesidad de ajustes adicionales para mejorar el rendimiento.
