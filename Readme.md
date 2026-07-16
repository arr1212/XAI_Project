# Comparación de SHAP y LIME en la Explicabilidad de Modelos QSAR
## Consistencia y Estabilidad de las Explicaciones en la Predicción de Toxicidad Molecular

Este documento detalla el propósito, la estructura y la metodología implementada en el notebook [0-QSAR_XAI_Notebook.ipynb]. 

---

## 📌 Descripción General del Proyecto

El notebook tiene como objetivo principal evaluar y comparar cuantitativamente dos de los métodos de explicabilidad local más utilizados en aprendizaje automático: **SHAP (SHapley Additive exPlanations)** y **LIME (Local Interpretable Model-agnostic Explanations)**. 

La evaluación se realiza sobre modelos **QSAR (Quantitative Structure-Activity Relationship)** entrenados para predecir la **toxicidad molecular** de diferentes compuestos. Para medir la calidad y robustez de las explicaciones generadas, el notebook utiliza métricas avanzadas de consistencia y estabilidad, basadas en el framework de investigación **HolisticAI**.

---

## 🛠️ Tecnologías y Librerías Utilizadas

El desarrollo del notebook se apoya en el siguiente stack de librerías científicas y de machine learning en Python:

*   **Explicabilidad (XAI):** `shap` y `lime` (con `lime_tabular`).
*   **Modelado y Evaluación:** `scikit-learn` (Pipelines, Standard Scaler, Stratified K-Fold y múltiples clasificadores).
*   **Métricas Robustas:** Framework **HolisticAI** (implementaciones adaptadas de consistencia y estabilidad).
*   **Procesamiento y Estadística:** `numpy`, `pandas`, `scipy` (cálculo de distancias, Spearman, Kendall's Tau, Jensen-Shannon).
*   **Visualización:** `matplotlib` (incluyendo `gridspec` para layouts complejos) y `seaborn`.

---

## 📊 Estructura del Flujo de Trabajo

El notebook está organizado de forma secuencial en las siguientes secciones clave:

### 1. Instalación de Dependencias
Configuración del entorno e instalación de librerías esenciales (especialmente aquellas orientadas a explicabilidad y métricas de robustez).

### 2. Carga y Exploración de Datos
Lectura de los conjuntos de datos de entrenamiento (`train.csv`) y prueba (`test.csv`) que contienen descriptores moleculares de entrada y etiquetas binarias de toxicidad (`tox` vs. `nontox`).

### 3. Entrenamiento de Modelos con Validación Cruzada
Entrenamiento de 8 clasificadores distintos usando validación cruzada estratificada de 5 pliegues (5-Fold Stratified CV):
1.  **LR:** Regresión Logística (con escalado previo).
2.  **DT:** Árbol de Decisión.
3.  **RF:** Random Forest.
4.  **KNN:** K-Vecinos Más Cercanos (con escalado previo).
5.  **NB:** Naive Bayes Gaussiano.
6.  **GB:** Gradient Boosting.
7.  **SVM:** Máquina de Vectores de Soporte (con probabilidad habilitada y escalado previo).
8.  **MLP:** Perceptrón Multicapa (con escalado previo).

Se registran métricas de rendimiento: *Accuracy*, *Precision*, *Recall*, *F1-Score* y *ROC AUC*.

### 4. Evaluación sobre el Conjunto de Prueba
Validación final del desempeño predictivo de cada modelo sobre el conjunto de test independiente.

### 5. Selección de los Mejores Modelos
Filtrado automático de los mejores clasificadores basándose en un umbral mínimo de rendimiento (por defecto `F1-Score >= umbral`). Esto asegura que solo se expliquen modelos con alta capacidad predictiva.

### 6. Generación de Explicaciones (SHAP vs. LIME)
Generación de matrices de atribución local de características para todas las instancias del conjunto de prueba:
*   **SHAP:** Utiliza `TreeExplainer` optimizado para modelos basados en árboles (RF, GB, DT). Para otros modelos (SVM, MLP, KNN, etc.), implementa `KernelSHAP` optimizado mediante resúmenes de fondo con *k-means* (por ejemplo, 100 clusters) y un número adaptativo de muestras para garantizar estabilidad numérica.
*   **LIME:** Utiliza `LimeTabularExplainer` configurado de forma agnóstica para estimar explicaciones locales lineales perturbando el entorno de cada instancia.

### 7. Métricas de Explicabilidad (Robustez)
Evaluación de las explicaciones mediante las métricas propuestas por Muñoz et al. (2023) — Ecuaciones (4) a (8) del paper —, implementadas directamente a partir de esas fórmulas:
*   **Rank Consistency (RC):** compara el ranking de importancia de cada feature en cada instancia contra su ranking modal (el más frecuente entre todas las instancias), normalizando la desviación por el rango máximo observado ($D_{max,j}$):
    $$RC = \frac{1}{d}\sum_{j=1}^{d}\left(1 - \frac{\bar D_j}{D_{max,j}}\right)$$
*   **Importance Stability (IS):** normaliza la varianza poblacional de la importancia de cada feature ($V_j$) respecto a la varianza máxima teórica de una variable acotada en [0,1] con esa misma media (cota de Bernoulli, $V_{max,j}=\mu_j(1-\mu_j)$):
    $$IS = \frac{1}{d}\sum_{j=1}^{d}\left(1 - \frac{V_j}{V_{max,j}}\right)$$

Ambas métricas se normalizan en el rango **[0, 1]**, donde **1** representa consistencia o estabilidad perfecta.

> **Nota de implementación:** las funciones equivalentes del repositorio `holisticai` (`_rank_consistency.py::rank_consistency()` e `_importance_stability.py::importance_stability()`) no reproducen exactamente estas ecuaciones tal como están escritas en el paper — a ambas les falta el signo `1 -`, y la segunda además usa varianza muestral (`ddof=1`) y promedio ponderado por importancia en vez de la varianza poblacional y el promedio simple que define el texto. El notebook usa versiones corregidas de esas funciones (`rank_consistency_holistic()` e `importance_stability_holistic()`) para que coincidan exactamente con las ecuaciones del paper.

### 8. Análisis Comparativo y Criterio de Selección
Cálculo de las diferencias absolutas ($\Delta$) entre SHAP y LIME para cada métrica:
*   $\Delta\text{RankConsistency} = |RC_{\text{SHAP}} - RC_{\text{LIME}}|$
*   $\Delta\text{Stability} = |Stab_{\text{SHAP}} - Stab_{\text{LIME}}|$

Esto permite identificar de manera objetiva cuál método XAI ofrece explicaciones más robustas y confiables para cada arquitectura de modelo entrenada.

### Bloques de Diagnóstico Profundo (Análisis de Fallas)
A continuación de la Sección 9 (Conclusión Final), el notebook incluye tres bloques de diagnóstico avanzado, sin numeración propia, para entender las discrepancias entre explicadores:
*   **Bloque 1 – Casos de Mayor Desacuerdo:** Identifica las moléculas donde SHAP y LIME discrepan más fuertemente en su orden de importancia (usando correlaciones como Kendall's Tau). Evalúa la hipótesis de si las discrepancias aumentan cerca de la frontera de decisión (probabilidad del modelo $\approx 0.5$) o en moléculas atípicas (outliers).
*   **Bloque 2 – Inestabilidad Local:** Identifica qué instancias específicas inducen mayor varianza local en cada método, analizándolo en función de la distancia al centroide del conjunto de entrenamiento.
*   **Bloque 3 – Análisis de Características Conflictivas:** Detecta las características (descriptores químicos) donde SHAP y LIME asignan signos de impacto opuestos (por ejemplo, un método indica que la característica aporta a la toxicidad y el otro que la reduce). Analiza la relación de estos conflictos con variables de distribución bimodal o interacciones no lineales fuertes.

---

## 📈 Visualizaciones Clave

El notebook genera dashboards visuales interactivos y estáticos de alta calidad:
1.  **Comparativa de Performance Predictiva:** Gráficos de barra que comparan las métricas tradicionales de clasificación.
2.  **Bump Charts (Gráfico de Rango de Features):** Muestra visualmente cómo cambia el orden de importancia de las 27 características moleculares al pasar de SHAP a LIME, resaltando en rojo las líneas de las características con mayores discrepancias de rango.
3.  **Heatmaps de Conflicto de Signo:** Matrices de instancias vs. características que exponen gráficamente dónde ocurren desacuerdos de dirección entre explicadores.

---

## 📚 Referencias

El estudio se fundamenta en la siguiente bibliografía destacada:
1.  **Framework de Explicabilidad:** Munoz, C., da Costa, K., Modenesi, B., & Koshiyama, A. (2023). *Evaluating explainability in machine learning predictions through explainer-agnostic metrics*. arXiv:2302.12094.
2.  **Aplicación QSAR / Toxicidad:** Togo, M. V. et al. (2022). TIRESIA: an explainable artificial intelligence platform for predicting developmental toxicity. *Journal of Chemical Information and Modeling*, 63(1), 56–66.
3.  **SHAP:** Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
4.  **LIME:** Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD*.
