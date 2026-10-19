# Clasificación tabular: árboles, ensembles y AutoML

Laboratorio práctico para comparar modelos de clasificación y distintas estrategias de combinación usando el dataset **Heart Disease**: 918 observaciones, 11 variables predictoras y una etiqueta binaria (`HeartDisease`). El preprocesamiento mediante one-hot encoding genera 20 columnas predictoras.

El objetivo es aprender **cuándo usar cada método, cómo ajustarlo y cómo evaluar si su complejidad aporta una mejora real**.

## Guía rápida de métodos

| Método | Para qué volver a esta sección |
|---|---|
| **Decision Tree** | Construir un clasificador interpretable y estudiar cómo `max_depth` y `min_samples_split` controlan su complejidad. |
| **Random Forest** | Combinar árboles mediante muestreo aleatorio de filas y características; analizar profundidad, número de árboles e importancia de variables. |
| **XGBoost** | Entrenar árboles secuencialmente con boosting; ajustar `learning_rate`, `n_estimators`, `max_depth` y `min_child_weight`; utilizar early stopping. |
| **Stacking** | Entrenar un meta-modelo con predicciones out-of-fold de KNN, regresión logística y SVM. |
| **Ensemble ponderado** | Combinar probabilidades de MLP y KNN con distintos pesos y explorar su impacto en el rendimiento. |
| **Hard y soft voting** | Comparar la mayoría de etiquetas con el promedio de probabilidades de varios clasificadores. |
| **Bagging de SVM** | Entrenar 20 SVM sobre muestras bootstrap y comprobar si mejoran frente a una SVM individual optimizada. |
| **AutoGluon** | Automatizar preprocesamiento, entrenamiento, selección y combinación de modelos bajo un presupuesto de cómputo. |
| **TabPFN v2 — pendiente** | Explorar un modelo preentrenado para datos tabulares mediante in-context learning. La ejecución se interrumpió y quedó pendiente de evaluación. |

## Objetivos de aprendizaje

- **Reconocer sobreajuste y subajuste** comparando el rendimiento en entrenamiento y validación, sin asumir que una brecha pequeña implica un mejor modelo.
- **Optimizar hiperparámetros con GridSearchCV y validación cruzada estratificada de 5 folds**, manteniendo el test fuera de la selección.
- **Construir pipelines con escalado** para KNN, SVM, regresión logística y MLP, ajustando el preprocesamiento dentro de cada fold.
- **Distinguir bagging, boosting, voting y stacking**, comprendiendo cómo generan diversidad y combinan predicciones.
- **Generar predicciones out-of-fold** para entrenar un meta-modelo o explorar pesos sin utilizar directamente las predicciones sobre los datos de ajuste.
- **Interpretar accuracy, ROC AUC y logloss**: aciertos de clasificación, capacidad de ordenamiento y calidad de probabilidades.
- **Interpretar importancia de características**, diferenciando reducción de impureza y permutación, sin confundir importancia predictiva con causalidad.
- **Evaluar el costo de la complejidad**: más árboles, más modelos o mayor automatización no garantizan mejores resultados.
- **Gestionar un entorno local con uv**, registrar dependencias y reconocer cuándo un cambio de bibliotecas requiere reiniciar el kernel.

## Aprendizajes principales

En la comparación final, la **SVM individual optimizada alcanzó un accuracy de 89,67 %**, frente al **89,13 % del bagging de 20 SVM**. La diferencia equivale a una sola observación del test: un ensemble más costoso no necesariamente mejora un modelo individual bien ajustado.

La selección de características puede ayudar a simplificar un modelo y reducir costos de adquisición de datos, pero requiere validación. Eliminar columnas de one-hot encoding no implica necesariamente eliminar mediciones originales.

## Alcance de la evaluación

Este es un laboratorio exploratorio, no una validación clínica. Se reutilizó el mismo test para comparar numerosas alternativas, por lo que los resultados no representan una evaluación final totalmente independiente.

En stacking y en la selección de pesos se optimizaron primero los modelos base sobre todo el entrenamiento y después se generaron predicciones out-of-fold. Se utilizó esta simplificación para reducir cómputo; una evaluación más rigurosa del proceso completo requeriría validación anidada.

## Entorno

Python 3.11, uv y notebooks de Jupyter en VS Code, ejecutados localmente en macOS. Bibliotecas principales: pandas, NumPy, scikit-learn, XGBoost, Matplotlib, AutoGluon y TabPFN.
