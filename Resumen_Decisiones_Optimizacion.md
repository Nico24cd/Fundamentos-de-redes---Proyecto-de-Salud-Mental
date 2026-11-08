# Resumen de decisiones de optimización y sus justificaciones

**Proyecto:** Entrenamiento de Red Neuronal Enfocado a Salud Mental en Estudiantes Universitarios
**Notebook:** `Entrenamiento_de_Red_Neuronal_Proyecto-3.ipynb`
**Objetivo:** clasificación binaria del riesgo de depresión (`riesgo_dep`: 0 = sin riesgo, 1 = riesgo).

---

## 1. Preprocesamiento de los datos

| Decisión | Justificación dada en el notebook |
|---|---|
| Abreviar los nombres de columna y crear un diccionario de variables en español | Los nombres originales (las preguntas completas de la encuesta) eran demasiado largos para trabajar con ellos. |
| Eliminar `educ_padre` y `educ_madre` | Son las únicas columnas con faltantes (954 y 1059 de 2992 filas, más del 30 %). Se consideran datos **MNAR** que no se pueden inferir, por lo que se opta por borrar las columnas en lugar de imputar. |
| *One-hot encoding* de `etnia` (6 columnas binarias) | La columna tenía una codificación ordinal simple (1–6) de categorías que no tienen orden; el one-hot evita que la red interprete un orden falso. Se fuerzan las 6 columnas aunque alguna categoría no aparezca. |
| Recodificar `genero` de {1, 2} a {0, 1} | Dejar las variables binarias en formato 0/1. |
| División 80/20 entrenamiento/prueba **estratificada** (`random_state=42`) | Mantener la misma proporción de clases (≈64 % / 36 %) en ambos conjuntos. |
| `StandardScaler` solo en `anio_nac` y `estrato`, ajustado **después** del split y solo con entrenamiento | Son las variables de mayor escala numérica; las escalas PSS (0–4), GAD (0–3) y las demás preguntas ya tienen rangos pequeños. Ajustar después del split evita fuga de información (*data leakage*). |
| Split **fijo** de validación (80/20 de `X_train`, semilla 42) definido una sola vez | Que todos los experimentos usen exactamente los mismos datos de entrenamiento y validación para que sean comparables. El conjunto de prueba se reserva para el resultado final. |

## 2. Modelo baseline y red básica

- **Perceptrón** (1 neurona sigmoide, Adam, 10 épocas) como *baseline*: accuracy en prueba ≈ **0.82** (el texto lo redondea a 80 %); recall de la clase 1 = 0.68. Justificación: tener un punto de referencia sobre el cual mejorar sin perder desempeño.
- **Red básica** (1 capa oculta de 32 ReLU, SGD lr = 0.01, 50 épocas): accuracy en prueba ≈ **0.81**. Limitaciones observadas que motivan la optimización:
  - aprendizaje inestable (picos y caídas inconsistentes),
  - brecha creciente entre la pérdida de entrenamiento y la de validación.

## 3. Experimentos individuales (un factor a la vez)

Todos en PyTorch, con el mismo split fijo de validación y, desde el experimento de optimizadores, con los **mismos pesos iniciales** (`initial_weights`) para aislar el efecto de cada factor.

### 3.1 Arquitectura
Se probaron 9 MLP: 1, 2 y 3 capas × 32, 64 y 128 neuronas (Adam, lr = 0.01, 50 épocas, CrossEntropy).
- Todas quedan en un rango estrecho (0.843 – 0.866 de accuracy de validación).
- Mejor: `[128]` (0.866). Aumentar profundidad/ancho **no** mejora; las de 3 capas grandes empeoran levemente.
- Se graficó complejidad (neuronas totales) vs. accuracy para visualizar que más complejidad no compensa.

### 3.2 Optimizador
Arquitectura fija `[64, 64]`, lr = 0.01, 50 épocas, BCEWithLogitsLoss.

| Optimizador | Val. acc. |
|---|---|
| Mini-Batch GD (batch 32) | **0.856** |
| RMSProp | 0.845 |
| AdaGrad | 0.841 |
| SGD + Momentum (0.9) | 0.833 |
| Adam | 0.826 |
| GD (batch completo) | 0.755 |

El GD de batch completo es claramente el peor con 50 épocas (pocas actualizaciones por época); Mini-Batch GD sin momentum es el mejor.

### 3.3 Learning rate
SGD mini-batch, `[64, 64]`: mejor **0.005** (0.860), luego 0.05 (0.856) y 0.01 (0.849). Con 0.1 se reduce la validación (0.833) mientras el entrenamiento sube (0.867) → indicio de sobreajuste/inestabilidad. Con 0.001 aprende demasiado lento (0.841).

### 3.4 Función de pérdida
`BCEWithLogitsLoss` (0.862) vs. `BCELoss` + sigmoide (0.856). Se elige **BCEWithLogitsLoss** por mejor resultado y por ser numéricamente más estable (aplica la sigmoide dentro de la pérdida).

### 3.5 Regularización
L1 (λ = 1e-4) vs. L2 (weight decay = 1e-4): 0.860 vs. 0.862. Diferencias muy pequeñas; ambas se llevan a la búsqueda final junto con "sin regularización".

## 4. Búsqueda final en malla (grid search)

Con base en los experimentos anteriores se seleccionaron los mejores valores de cada factor y se probaron **todas las combinaciones**:

| Factor | Valores |
|---|---|
| Arquitectura | `[32]`, `[64, 64]`, `[128, 128]`, `[128, 128, 128]` |
| Optimizador | AdaGrad, Mini-Batch GD, GD (batch completo) |
| Learning rate | 0.005, 0.01, 0.05 |
| Pérdida | BCEWithLogitsLoss |
| Regularización | L1, L2, Ninguna (λ = 1e-4, solo sobre pesos, no sesgos) |
| Épocas | 75 (batch 32) |

4 × 3 × 3 × 3 = **108 entrenamientos**. Criterios de justicia de la comparación:
- mismo split de entrenamiento/validación para todas las corridas;
- mismos pesos iniciales por arquitectura y mismo orden de mini-batches (generador con semilla);
- la pérdida que se reporta es la BCE pura (sin la penalización), para comparar L1/L2/Ninguna y train/val en la misma escala;
- el mejor modelo se elige por **validación** (accuracy y, en empate, menor pérdida); el conjunto de prueba solo se usa para reportar.

### Hallazgos por factor (promedios sobre las 108 corridas)
- **Optimizador:** Mini-Batch GD es el mejor y el más estable (val. acc. 0.862 ± 0.010, menor pérdida 0.340). AdaGrad tiene mayor brecha train–val (sobreajuste). GD de batch completo es el peor (0.792), aunque mejora mucho con lr = 0.05.
- **Arquitectura:** `[32]` tiene el mejor promedio (0.847) y la menor varianza; `[128, 128, 128]` el peor (0.818) y la mayor brecha → más capacidad tiende a sobreajustar en este conjunto pequeño (~1900 muestras de entrenamiento).
- **Learning rate:** 0.05 tiene mejor promedio, pero porque rescata a GD; para Mini-Batch GD los mejores valores son 0.005–0.01.
- **Regularización:** L1 da el mejor promedio y la menor pérdida de validación, sobre todo porque estabiliza a AdaGrad en redes grandes. Con Mini-Batch GD las tres opciones dan resultados prácticamente idénticos (λ = 1e-4 es una penalización muy suave).

### Mejor configuración (según validación)

| Parámetro | Valor |
|---|---|
| Arquitectura | `[128, 128]` |
| Optimizador | Mini-Batch GD (SGD sin momentum, batch 32) |
| Learning rate | 0.01 |
| Pérdida | BCEWithLogitsLoss |
| Regularización | Ninguna (empata con L2) |
| Épocas | 75 |

**Resultados:** val. acc. = 0.8745, val. loss = 0.326
**Prueba:** accuracy = 0.805, F1 (clase 1) = 0.717, recall (clase 1) = 0.692, AUC = 0.895, loss = 0.390.

## 5. Observaciones para la discusión

1. **El modelo optimizado no supera claramente al baseline en prueba.** En la corrida de Colab el perceptrón obtuvo 0.816 de accuracy y 0.73 de F1 en la clase 1; la mejor red obtuvo 0.805 y 0.717. Además, el baseline de Keras **no fija semilla**: al re-ejecutar el notebook localmente dio 0.748 de accuracy (F1 clase 1 = 0.63), mientras que los resultados de PyTorch se reprodujeron exactamente. Conviene fijar `tf.keras.utils.set_random_seed(42)` en el baseline, o reportarlo como promedio de varias corridas, para que la comparación sea justa. La caída de 0.87 (validación) a 0.80 (prueba) sugiere que elegir entre 108 configuraciones con un único conjunto de validación de 478 muestras sobreajusta la selección. Vale la pena mencionarlo y, si hay tiempo, usar validación cruzada o promediar varias semillas.
2. **Las diferencias entre configuraciones son pequeñas** (1–2 puntos), del orden del ruido del muestreo; conviene interpretarlas como tendencias y no como ganadores definitivos.
3. **El recall de la clase de riesgo (≈0.69) es la métrica más relevante** para una detección temprana: se están perdiendo ~31 % de los casos en riesgo. Se podría justificar ajustar el umbral de decisión (< 0.5) o usar `pos_weight` en la BCE para priorizar recall.
4. **Inconsistencias menores en el texto del notebook:**
   - La tabla del pipeline final lista 3 arquitecturas y 2 regularizaciones (y dice "4 × 3 × 3 × 2 = 108"), pero el código usa 4 arquitecturas (incluye `[64, 64]`) y 3 opciones de regularización (incluye "Ninguna"); el cálculo correcto es 4 × 3 × 3 × 3 = 108.
   - El texto del baseline dice "accuracy del 80 %", el resultado impreso es 81.6 %.
   - El experimento de arquitecturas usa Adam y CrossEntropy (2 salidas), mientras que los siguientes usan SGD y BCE (1 salida); las arquitecturas finales no son exactamente las ganadoras de ese primer experimento, y la elección de `[32]`, `[128, 128]` y `[128, 128, 128]` no se justifica explícitamente en el texto.
   - Varias celdas de experimentos no tienen una conclusión escrita en markdown (optimizadores, learning rate, pérdida, regularización y pipeline final); los hallazgos de este resumen se pueden usar para completarlas.
