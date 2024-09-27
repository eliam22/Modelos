# Modelos
Modelo de regresión líneal
Modelo KNN
Modelo Árbol de decisión 

Técnicas de regresión 
Reporte de Proyecto: Predicción del Precio del Oro
Introducción
El objetivo de este proyecto es desarrollar un modelo que prediga el precio del oro utilizando un conjunto de datos históricos. La importancia del oro como activo financiero y su volatilidad en el mercado motivan la necesidad de contar con herramientas que permitan anticipar su comportamiento.
Análisis Exploratorio de Datos
Correlaciones
Se realizó un análisis exploratorio inicial mediante un mapa de calor para identificar las correlaciones entre las diferentes características del conjunto de datos y el precio del oro. Este análisis ayudó a comprender cuáles variables influyen más en el precio, permitiendo orientar el enfoque del modelado.
Manejo de Valores Faltantes
Durante el análisis se detectaron valores faltantes en el conjunto de datos. Se optó por imputar estos valores utilizando la media, aunque también se consideraron otras estrategias como la mediana o la moda, dependiendo de la distribución de cada característica.
Preparación de Datos
Los datos fueron divididos en conjuntos de entrenamiento y validación. Esta división es crucial para poder evaluar el rendimiento del modelo y asegurarse de que generalice bien a datos no vistos.
Entrenamiento de Modelos
Se entrenaron varios modelos de regresión para predecir el precio del oro. Las métricas de rendimiento, como el Error Absoluto Medio (MAE), el Error Cuadrático Medio (MSE) y el coeficiente de determinación (R²), fueron utilizadas para evaluar la efectividad de cada modelo. Esta etapa es fundamental para identificar cuál modelo se ajusta mejor a los datos.
Visualización de Resultados
Se recomendó crear visualizaciones que ayuden a interpretar los resultados. Las gráficas permiten observar cómo diferentes factores influyen en el precio del oro, facilitando la comprensión de los patrones identificados en el modelo.
Conclusiones
•	Correlaciones: Se identificaron relaciones significativas entre varias características y el precio del oro, lo que puede utilizarse para mejorar la precisión de las predicciones.
•	Manejo de Valores Faltantes: La imputación de valores faltantes es un paso crítico que puede influir en la calidad del modelo.
•	Rendimiento del Modelo: La evaluación del rendimiento utilizando diferentes métricas permite seleccionar el modelo más adecuado para realizar predicciones precisas.
•	Visualización: Las visualizaciones son herramientas poderosas para comunicar los hallazgos y comprender la influencia de diferentes factores en el precio del oro.
Este proyecto resalta la importancia del análisis de datos en la predicción de precios y proporciona una base sólida para futuros estudios en este ámbito.

KNN
Reporte de Análisis de Datos de Estrellas
1. Carga de Datos
•	Se montó Google Drive y se cargó el conjunto de datos desde el archivo cleaned_star_data.csv, el cual contiene información sobre 240 estrellas.
2. Información General del Dataset
•	El conjunto de datos contiene las siguientes columnas:
o	Temperature (K): Temperatura en Kelvin.
o	Luminosity (L/Lo): Luminosidad en términos de la luminosidad solar.
o	Radius (R/Ro): Radio en términos del radio solar.
o	Absolute magnitude (Mv): Magnitud absoluta.
o	Star type: Tipo de estrella (como variable objetivo).
o	Star color: Color de la estrella.
o	Spectral Class: Clase espectral.
Todas las columnas tienen 239 valores no nulos, y los tipos de datos son mayormente objetos y un solo float.
3. Preprocesamiento de Datos
•	Se identificaron y eliminaron filas con valores faltantes en columnas críticas: Luminosity (L/Lo), Radius (R/Ro), Absolute magnitude (Mv), y Temperature (K).
•	Se reemplazaron espacios en blanco en los datos con NaN y se transformaron las columnas a tipo float para el análisis numérico.
4. Análisis Exploratorio
•	Se generaron gráficos de correlación y un pairplot para observar las relaciones entre las variables.
•	Se realizó un análisis de componentes principales (PCA) para reducir la dimensionalidad del conjunto de datos y facilitar la visualización.
5. Modelado de Datos
•	Se dividió el conjunto de datos en conjuntos de entrenamiento y prueba utilizando train_test_split.
•	Se entrenó un modelo k-NN (k-vecinos más cercanos) con el conjunto de entrenamiento, utilizando la distancia de Manhattan como métrica.
•	Se realizó una búsqueda de hiperparámetros para determinar el valor óptimo de k, encontrando que k=1 logró la mayor precisión del 86.36%.
6. Evaluación del Modelo
•	La matriz de confusión mostró el desempeño del modelo en la predicción de los tipos de estrellas en el conjunto de prueba.
•	Se calculó la precisión del modelo, obteniendo un 72.73% de precisión para k=5.
7. Predicción de Nuevos Datos
•	Se proporcionó un mecanismo para ingresar nuevos datos de luminosidad, radio y magnitud absoluta, lo que resultó en una predicción de tipo de estrella.
Resultados de Predicción
•	Para los valores ingresados:
o	Luminosidad: 13143 L/Lo
o	Radio: 13 R/Ro
o	Magnitud Absoluta: 3131
•	El modelo predijo que el tipo de estrella es 3.
Conclusiones
El análisis y modelado realizado proporciona una comprensión sólida sobre la clasificación de estrellas, utilizando técnicas de aprendizaje automático. El modelo k-NN se mostró efectivo, aunque su precisión podría mejorarse con más datos o utilizando otros modelos de aprendizaje automático.

Árbol de decisión 
Informe sobre Clasificación de Estrellas
1. Objetivo del Proyecto
El objetivo de este proyecto es clasificar diferentes tipos de estrellas utilizando datos sobre sus características físicas, como temperatura, luminosidad, radio y magnitud absoluta. La clasificación se realiza mediante el uso de un modelo de aprendizaje automático, específicamente un clasificador de Random Forest.
2. Datos Utilizados
•	Fuente: El conjunto de datos fue importado desde un archivo CSV llamado cleaned_star_data.csv.
•	Características (X):
o	Temperature (K): Temperatura de la estrella en Kelvin.
o	Luminosity (L/Lo): Luminosidad de la estrella en relación con la luminosidad del Sol.
o	Radius (R/Ro): Radio de la estrella en relación con el radio del Sol.
o	Absolute magnitude (Mv): Magnitud absoluta de la estrella.
•	Variable Objetivo (y):
o	Star type: Tipo de estrella (categoría).
3. Preprocesamiento de Datos
•	Eliminación de Valores NaN: Se eliminaron filas con valores NaN para asegurar la integridad de los datos.
•	Conversión a Numérico: Las características fueron convertidas a formato numérico, gestionando errores mediante la conversión de entradas inválidas a NaN.
•	Normalización: Las características fueron normalizadas utilizando StandardScaler para asegurar que todos los atributos tuvieran la misma escala.
4. Balanceo de Clases
Se utilizó SMOTE (Synthetic Minority Over-sampling Technique) para balancear el conjunto de datos, aumentando las instancias de las clases minoritarias y ayudando al modelo a aprender mejor las características de cada tipo de estrella.
5. División del Conjunto de Datos
El conjunto de datos fue dividido en un conjunto de entrenamiento (70%) y un conjunto de prueba (30%), asegurando que la estratificación se mantuviera para reflejar la distribución original de las clases.
6. Entrenamiento del Modelo
Se entrenó un modelo de Random Forest utilizando las características normalizadas y balanceadas. Este clasificador es robusto y adecuado para problemas de clasificación, especialmente en conjuntos de datos con múltiples clases.
7. Resultados
•	Precisión del Modelo: Accuracy: 0.2222
•	Matriz de Confusión:
css
Copy code
[[ 0  0 12  0  0  0]
 [ 0  4  8  0  0  0]
 [ 0  0 12  0  0  0]
 [ 0 12  0  0  0  0]
 [ 0 11  1  0  0  0]
 [ 0 12  0  0  0  0]]
•	Informe de Clasificación:
markdown
Copy code
              precision    recall  f1-score   support

         0.0       0.00      0.00      0.00        12
         1.0       0.10      0.33      0.16        12
         2.0       0.36      1.00      0.53        12
         3.0       0.00      0.00      0.00        12
         4.0       0.00      0.00      0.00        12
         5.0       0.00      0.00      0.00        12

    accuracy                           0.22        72
   macro avg       0.08      0.22      0.12        72
weighted avg       0.08      0.22      0.12        72
8. Importancia de las Características
Feature	Importance
Radius(R/Ro)	0.400623
Absolute magnitude(Mv)	0.347418
Luminosity(L/Lo)	0.161313
Temperature (K)	0.090646
9. Conclusiones
•	El modelo actual muestra una precisión relativamente baja, sugiriendo que puede haber necesidad de mejorar la calidad del conjunto de datos o explorar otros modelos más complejos o ajustar hiperparámetros.
•	La importancia de las características indica que Radius(R/Ro) y Absolute magnitude(Mv) son las más influyentes en la clasificación del tipo de estrella.
•	Recomendaciones para mejorar el modelo incluyen:
o	Recolectar más datos para balancear mejor las clases.
o	Probar otros algoritmos de clasificación, como SVM o redes neuronales.
o	Realizar un ajuste de hiperparámetros para el modelo de Random Forest.
10. Siguientes Pasos
•	Implementar mejoras basadas en las recomendaciones.
•	Probar diferentes enfoques para manejar las clases desbalanceadas.
•	Realizar validación cruzada para asegurar la robustez del modelo.
![image](https://github.com/user-attachments/assets/fba2a35f-6802-4d03-b4e7-8a495896b7e0)

