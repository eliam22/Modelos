# Modelos de Machine Learning

## Descripción General
Este repositorio contiene varios modelos de machine learning aplicados a diferentes problemas de predicción y clasificación, incluyendo regresión lineal, KNN, y un árbol de decisión. Cada modelo tiene su propio notebook que incluye la explicación detallada de su implementación y los resultados obtenidos.

## Contenido del Repositorio
1. **ArbolDecision.ipynb**: Modelo de Árbol de Decisión para clasificación de estrellas.
2. **KNN.ipynb**: Modelo KNN para clasificación de estrellas.
3. **RegresionLineal.ipynb**: Modelo de regresión lineal para predecir el precio del oro.

---

## 1. Proyecto: Predicción del Precio del Oro (Regresión Lineal)

### Introducción
Este proyecto tiene como objetivo predecir el precio del oro usando un conjunto de datos históricos. Se exploraron técnicas de regresión para mejorar la precisión del modelo.

### Pasos
- **Análisis Exploratorio**: Correlación entre características y precio del oro.
- **Manejo de Valores Faltantes**: Imputación de valores utilizando la media.
- **Modelado**: Se entrenaron y evaluaron varios modelos de regresión con métricas como MAE y MSE.

### Conclusiones
- Se identificaron correlaciones significativas entre características.
- El modelo más preciso fue seleccionado utilizando múltiples métricas.

---

## 2. Proyecto: Clasificación de Estrellas (KNN)

### Introducción
Este proyecto clasifica diferentes tipos de estrellas usando el algoritmo K-Nearest Neighbors (KNN).

### Pasos
- **Preprocesamiento**: Limpieza de datos y transformación a formato numérico.
- **Modelado**: Entrenamiento de un modelo KNN y ajuste de hiperparámetros.
- **Evaluación**: Precisión y matriz de confusión para evaluar el rendimiento del modelo.

### Conclusiones
- El modelo KNN logró una precisión de 72.73% con k=5.

---

## 3. Proyecto: Clasificación de Estrellas (Árbol de Decisión)

### Introducción
En este proyecto se utilizó un modelo de Random Forest para clasificar estrellas en diferentes categorías, basado en sus características físicas.

### Pasos
- **Preprocesamiento**: Normalización de características y balanceo de clases.
- **Modelado**: Entrenamiento de un modelo de Random Forest.
- **Resultados**: Precisión y análisis de la importancia de características.

### Conclusiones
- La precisión fue baja, lo que sugiere la necesidad de ajustes adicionales.

---

## Instrucciones para Ejecutar

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/usuario/repo.git

