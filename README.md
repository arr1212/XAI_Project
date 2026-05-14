# QSAR XAI Comparative Analysis: SHAP vs LIME

## 📋 Descripción General

Este proyecto implementa un análisis comparativo riguroso de dos métodos de Explainability in AI (XAI) - **SHAP** (SHapley Additive exPlanations) y **LIME** (Local Interpretable Model-agnostic Explanations) - en el contexto de modelos QSAR (Quantitative Structure-Activity Relationship) para predicción de toxicidad molecular.

**Objetivo Principal**: Evaluar cuál método de explicabilidad proporciona explicaciones más consistentes, estables y confiables para diferentes arquitecturas de modelos de aprendizaje automático aplicadas a toxicología predictiva.

## 📊 Dataset

- **Moléculas**: 292 (234 training, 58 test)
- **Features**: 27 descriptores moleculares + fragmentos estructurales
- **Target**: Binario (0: no-tóxica, 1: tóxica)
- **Modelos evaluados**: SVM (kernel RBF), Naive Bayes, k-Nearest Neighbors

## 🔧 Pipeline General

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LOADING & PREPROCESSING                 │
│  - Cargar dataset QSAR (292 moléculas × 27 features)            │
│  - Normalización MinMax [0,1]                                   │
│  - Train-test split: 234/58 (80/20)                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING (5-FOLD CV)                    │
│  - SVM (kernel='rbf', C=100, gamma='scale')                     │
│  - Naive Bayes (GaussianNB)                                     │
│  - KNN (n_neighbors=5, metric='euclidean')                      │
│  - Plus: Gradient Boosting, Random Forest, MLP, Decision Tree   │
│  - Métrica de selección: F1-score en test set                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 MODEL EVALUATION & SELECTION                     │
│  - Evaluar todos modelos en test set                            │
│  - Seleccionar TOP-3 por F1-score                               │
│  - Modelos finales: SVM, NB, KNN                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              XAI EXPLAINER GENERATION (SHAP & LIME)              │
│  Para cada modelo seleccionado:                                  │
│  - SHAP: KernelExplainer (nsamples=2102)                        │
│  - LIME: LimeTabularExplainer (num_samples=1000)                │
│  - Generar atribuciones para las 58 instancias de test          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 METRICS CALCULATION (MUNOZ 2023)                 │
│  Para cada combinación modelo-explicador:                        │
│  - Rank Consistency (RC) - Definition 3.5                        │
│  - Importance Stability (IS) - Definition 3.6                    │
│  - Global Explainability Score (GES) = 0.5(RC + IS)            │
│  - Kendall Tau (correlación de rankings)                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                COMPARATIVE ANALYSIS & VISUALIZATION              │
│  - Tablas de resultados por modelo                              │
│  - Gráficos de barras (RC, IS, GES)                            │
│  - Análisis de desacuerdos SHAP vs LIME                        │
│  - Boxplots de distribuciones de atribuciones                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📐 Fórmulas Matemáticas Principales

### 1. RANK CONSISTENCY (RC) - Definition 3.5 (Munoz et al. 2023)

Para cada instancia $i$ y feature $j$:

$$r_{ij} = \text{rank}(f_{ij})$$

donde $r_{ij}$ es el ranking del feature $j$ en la instancia $i$ (ranks bajos = mayor importancia).

Para cada feature $j$, encontrar el ranking modal:

$$r_{\text{mode},j} = \text{moda}(\{r_{1j}, r_{2j}, ..., r_{Mj}\})$$

Desviación promedio del ranking modal:

$$\bar{D}_j = \frac{1}{M}\sum_{i=1}^{M}|r_{ij} - r_{\text{mode},j}|$$

Rango máximo observado:

$$D_{\max,j} = \max_i\{r_{ij}\} - \min_i\{r_{ij}\}$$

Rank Consistency para feature $j$:

$$C_j = 1 - \frac{\bar{D}_j}{D_{\max,j}}$$

**Rank Consistency Global:**

$$RC = \frac{1}{d}\sum_{j=1}^{d}C_j$$

**Rango**: $[0, 1]$
- $RC \approx 1$: Rankings muy consistentes (feature mantiene posición relativa)
- $RC \approx 0$: Rankings uniformes/inconsistentes

---

### 2. IMPORTANCE STABILITY (IS) - Definition 3.6 (Munoz et al. 2023)

Para cada feature $j$, media de importancia:

$$\mu_j = \frac{1}{M}\sum_{i=1}^{M}f_{ij}$$

Varianza muestral:

$$V_j = \frac{1}{M}\sum_{i=1}^{M}(f_{ij} - \mu_j)^2$$

Varianza máxima teórica (Bernoulli):

$$V_{\max,j} = \mu_j(1 - \mu_j)$$

Estabilidad para feature $j$:

$$S_j = 1 - \frac{V_j}{V_{\max,j}}$$

**Importance Stability Global:**

$$IS = \frac{1}{d}\sum_{j=1}^{d}S_j$$

**Rango**: $[0, 1]$
- $IS \approx 1$: Magnitudes muy predecibles
- $IS \approx 0$: Magnitudes varían ampliamente

---

### 3. GLOBAL EXPLAINABILITY SCORE (GES)

Promedio ponderado de ambas métricas:

$$GES = 0.5 \cdot RC + 0.5 \cdot IS$$

**Rango**: $[0, 1]$
- $GES > 0.80$: Explicabilidad excelente
- $GES < 0.70$: Explicabilidad cuestionable

---

### 4. SHAP: Kernel SHAP (KernelSHAP)

Para una instancia $x$ y feature $j$:

$$\phi_j = \frac{1}{2^d}\sum_{S \subseteq F \setminus \{j\}} \left[f(S \cup \{j\}) - f(S)\right]$$

donde:
- $F$ = conjunto de todos los features
- $S$ = coalición de features
- $f(S)$ = predicción del modelo eliminando features en $S$

**En KernelSHAP (aproximación por muestreo):**

$$\phi_j^{\text{SHAP}} \approx \text{argmin}_{\phi} \sum_{z' \in D} \left[f(x'_z) - \phi_0 - \sum_{k=1}^{d}\phi_k z'_k\right]^2 \pi_{x}(z')$$

donde $\pi_x(z')$ es un kernel de proximidad.

---

### 5. LIME: Local Interpretable Model

Para instancia $x_0$, generar perturbaciones y ajustar regresión local:

$$f_{\text{local}}(x) \approx \beta_0 + \sum_{j=1}^{d}\beta_j \tilde{x}_j$$

donde:
- $\tilde{x}_j$ = feature $j$ perturbado
- $\beta_j$ = coeficiente de regresión local (importancia)

Ajuste ponderado por proximidad a $x_0$:

$$\min_{\beta} \sum_{z \in \text{perturbaciones}} \pi(z) \left[f(z) - (\beta_0 + \sum_j\beta_j z_j)\right]^2$$

donde $\pi(z) = \exp\left(-\frac{d(x_0, z)^2}{\sigma^2}\right)$ es el peso de proximidad.

---

### 6. KENDALL TAU - Correlación de Rankings

Para dos explicadores (SHAP y LIME):

$$\tau = \frac{2}{M(M-1)}\sum_{i<j}\text{sign}(r_i^{\text{SHAP}} - r_j^{\text{SHAP}}) \cdot \text{sign}(r_i^{\text{LIME}} - r_j^{\text{LIME}})$$

**Rango**: $[-1, 1]$
- $\tau = 1$: Acuerdo perfecto en rankings
- $\tau = 0$: Independencia
- $\tau = -1$: Desacuerdo perfecto

---

## 📊 Resultados Principales

### Resumen de Métricas (HC_7.b)

```
┌──────────────────────────────────────────────────────────────────┐
│                          SVM                                      │
├──────────────────────────────────────────────────────────────────┤
│ Ganador: LIME (Dominancia)                                        │
│ RC:  SHAP=0.7458  |  LIME=0.7809  |  Δ=+3.5%                    │
│ IS:  SHAP=0.9018  |  LIME=0.9095  |  Δ=+0.8%                    │
│ GES: SHAP=0.8238  |  LIME=0.8452  |  Δ=+2.1%                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     NAIVE BAYES                                   │
├──────────────────────────────────────────────────────────────────┤
│ Ganador: SHAP (Dominancia)                                        │
│ RC:  SHAP=0.8103  |  LIME=0.7260  |  Δ=+8.4%                    │
│ IS:  SHAP=0.9561  |  LIME=0.9081  |  Δ=+4.8%                    │
│ GES: SHAP=0.8832  |  LIME=0.8171  |  Δ=+6.6%                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                   k-NEAREST NEIGHBORS                             │
├──────────────────────────────────────────────────────────────────┤
│ Ganador: LIME (Dominancia)                                        │
│ RC:  SHAP=0.7179  |  LIME=0.7877  |  Δ=+7.0%                    │
│ IS:  SHAP=0.8941  |  LIME=0.9180  |  Δ=+2.4%                    │
│ GES: SHAP=0.8060  |  LIME=0.8529  |  Δ=+4.7%                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     GLOBAL AVERAGE                                │
├──────────────────────────────────────────────────────────────────┤
│ GES_SHAP = 0.8377  |  GES_LIME = 0.8384  |  Δ=+0.07%            │
│ Ganador Global: LIME (marginal, contexto-dependiente)            │
└──────────────────────────────────────────────────────────────────┘
```

## 🔍 Hallazgos Clave

### 1. Dependencia de Arquitectura del Modelo

- **Naive Bayes (aditivo)**: SHAP exacto → RC=0.8103 (máximo observado)
- **SVM (no-aditivo global)**: Equilibrio → LIME ligeramente mejor (+3.5%)
- **KNN (hiper-local)**: SHAP falla → RC=0.7179 (mínimo observado)

### 2. Divergencia en Rankings (SHAP)

SHAP produce rankings diferentes por modelo:

```
SVM:  fr_Ndealkylation1 (TOP-1)
NB:   fr_lactam (TOP-1)
KNN:  fr_lactam ≈ MATS2i (TOP-2 empatados)
```

### 3. Convergencia en Rankings (LIME)

LIME identifica consistentemente los mismos fragmentos como importantes:

```
SVM, NB, KNN: fr_lactam, fr_furan, fr_HOCCN, fr_thiazole (TOP-4)
```

### 4. Validación Química

Todos los top features de LIME tienen mecanismos tóxicos conocidos:
- **fr_lactam**: Metabolismo a metabolitos reactivos ✓
- **fr_furan**: Oxidación a epóxidos genotóxicos ✓
- **fr_thiazole**: Biotransformación a sulfóxidos ✓

## 📈 Métricas de Performance de Modelos

```
┌─────────────────────────────────────────────────────────┐
│ Modelo │  Acc  │ F1-Score │  AUC  │ Selección         │
├─────────────────────────────────────────────────────────┤
│ SVM    │ 0.793 │  0.870   │ 0.783 │ TOP-3 (F1)        │
│ NB     │ 0.793 │  0.870   │ 0.765 │ TOP-3 (F1)        │
│ KNN    │ 0.776 │  0.854   │ 0.712 │ TOP-3 (F1)        │
│ GB     │ 0.690 │  0.791   │ 0.703 │ Overfitting       │
│ RF     │ 0.759 │  0.844   │ 0.765 │ Inferior          │
│ MLP    │ 0.776 │  0.851   │ 0.742 │ Inferior          │
│ DT     │ 0.672 │  0.777   │ 0.579 │ Muy bajo          │
│ LR     │ 0.776 │  0.847   │ 0.783 │ Inferior          │
└─────────────────────────────────────────────────────────┘
```

## 💾 Estructura de Archivos

```
proyecto/
│
├── QSAR_XAI_Notebook.ipynb          # Notebook principal
├── data/
│   ├── qsar_dataset.csv             # Dataset QSAR (292 moléculas)
│   └── feature_attributions.csv     # Atribuciones SHAP/LIME exportadas
│
├── results/
│   ├── feature_attributions.csv     # Importancias promedio
│   ├── rank_consistency.csv         # Métricas RC
│   ├── importance_stability.csv     # Métricas IS
│   └── figures/
│       ├── shap_summary_svm.png     # SHAP plots
│       ├── lime_importance_svm.png  # LIME plots
│       └── comparacion_shap_lime.png # Gráfico comparativo
│
└── outputs/
    ├── metricas_munoz_2023.csv
    ├── comparativa_shap_lime.csv
    └── [Documentos académicos generados]
```

## 🚀 Uso del Notebook

### Requisitos
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import shap
import lime
import lime.lime_tabular
```

### Pasos Principales

1. **Cargar datos y entrenar modelos**
```python
# Carga de dataset QSAR
df = pd.read_csv('qsar_dataset.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalización
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# División train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar modelos seleccionados
model_svm = SVC(kernel='rbf', probability=True).fit(X_train, y_train)
model_nb = GaussianNB().fit(X_train, y_train)
model_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
```

2. **Generar explicaciones SHAP**
```python
explainer_shap = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer_shap.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

3. **Generar explicaciones LIME**
```python
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    mode='classification',
    random_state=42
)
# Para cada instancia
exp = explainer_lime.explain_instance(X_test[i], model.predict_proba)
```

4. **Calcular métricas de explicabilidad**
```python
# Rank Consistency
rc_shap = rank_consistency_munoz(shap_attributions)
rc_lime = rank_consistency_munoz(lime_attributions)

# Importance Stability
is_shap = importance_stability_munoz(shap_attributions)
is_lime = importance_stability_munoz(lime_attributions)

# GES Global
ges_shap = 0.5 * (rc_shap + is_shap)
ges_lime = 0.5 * (rc_lime + is_lime)
```

## 📚 Referencias Principales

1. **Munoz, C., da Costa, K., Modenesi, B., & Koshiyama, A. (2023).** 
   "Evaluating Explainability in Machine Learning Predictions through 
   Explainer-Agnostic Metrics." arXiv:2302.12094v3

2. **Lundberg, S. M., & Lee, S. I. (2017).** 
   "A Unified Approach to Interpreting Model Predictions." 
   In Advances in Neural Information Processing Systems (pp. 4765-4774).

3. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** 
   "Why Should I Trust You?: Explaining the Predictions of Any Classifier." 
   In KDD (pp. 1135-1144).

4. **Guidotti, R., Monreale, A., Ruggieri, S., Turini, F., Giannotti, F., & 
   Pedreschi, D. (2018).** 
   "A Survey of Methods for Explaining Black Box Models." 
   ACM computing surveys, 51(5), 1-42.

## 🎯 Conclusiones Principales

### Para QSAR Heterogéneo

1. **SHAP es superior en modelos aditivos** (Naive Bayes)
   - RC_SHAP = 0.8103 vs RC_LIME = 0.7260 (Δ = 8.4%)
   - Razón: Valores de Shapley exactos en aditivos

2. **LIME es superior en modelos no-aditivos** (SVM, KNN)
   - KNN: RC_LIME = 0.7877 vs RC_SHAP = 0.7179 (Δ = 7.0%)
   - Razón: Perturbación local respeta topología

3. **LIME es más robusto en portfolio heterogéneo**
   - Variabilidad RC: LIME 6.17% vs SHAP 9.24%
   - LIME gana 2/3 modelos, superior en 2

4. **Ambos métodos son seguros en magnitudes** (IS > 0.89 universalmente)
   - Diferencia: RC es diferenciador (hasta 8.4%)
   - Magnitudes confiables en ambos

### Recomendación Operativa

```
┌─────────────────────────────────────────────────────────┐
│ MATRIZ DE DECISIÓN PARA SELECCIÓN DE XAI                │
├─────────────────────────────────────────────────────────┤
│ Modelo: Aditivo (NB)          → ELEGIR: SHAP            │
│ Modelo: No-aditivo (SVM)      → ELEGIR: LIME            │
│ Modelo: Hiper-local (KNN)     → ELEGIR: LIME (crítico)  │
│ Portfolio Heterogéneo         → ELEGIR: LIME (robusto)  │
└─────────────────────────────────────────────────────────┘
```

## 📝 Notas Importantes

- Las métricas RC e IS están normalizadas en [0,1] según Munoz et al. (2023)
- GES = 0.5×RC + 0.5×IS con pesos iguales (puede ser subóptimo)
- SHAP en KNN produce rankings inconsistentes debido a discontinuidades topológicas
- LIME genera explicaciones química válidas (validadas contra literatura)
- El análisis utiliza 5-fold cross-validation para robustez

## 👨‍🔬 Autores

**Proyecto**: Fundamentos de Inteligencia Artificial Explicable
**Autoras**: Alice Rambo, Fiorella Cravero
**Profesor**: Maria Vanina Martinez
**Institución**: [Universidad/Instituto]
**Año**: 2024-2025

---

**Última actualización**: Mayo 2025
**Versión del notebook**: QSAR_XAI_Notebook2.ipynb

