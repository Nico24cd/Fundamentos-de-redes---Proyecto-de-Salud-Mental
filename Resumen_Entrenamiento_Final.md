# Resumen completo del entrenamiento — notebook final

**Proyecto:** Entrenamiento de Red Neuronal Enfocado a Salud Mental en Estudiantes Universitarios
**Notebook:** `Entrenamiento_de_Red_Neuronal_Proyecto (final).ipynb`
**Estudiantes:** Nicoll Quesada, Daniel Hurtado y Santiago Lozano
**Tarea:** clasificación binaria del riesgo de depresión (`riesgo_dep`: 0 = sin riesgo, 1 = riesgo).

> **Pregunta de investigación:** ¿En qué medida variables como el comportamiento del estudiante, la estabilidad financiera, el ambiente social y el desempeño académico pueden usarse para detectar de forma temprana la necesidad de ayuda psicológica?

---

## Índice

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Datos y variable objetivo](#2-datos-y-variable-objetivo)
3. [Preprocesamiento](#3-preprocesamiento)
4. [Partición de los datos y escalado](#4-partición-de-los-datos-y-escalado)
5. [Infraestructura común de modelado](#5-infraestructura-común-de-modelado)
6. [Criterio de selección de los experimentos](#6-criterio-de-selección-de-los-experimentos)
7. [Experimentos de un solo factor](#7-experimentos-de-un-solo-factor)
8. [Pipeline final: búsqueda en grilla de 216 configuraciones](#8-pipeline-final-búsqueda-en-grilla-de-216-configuraciones)
9. [Evaluación final en prueba](#9-evaluación-final-en-prueba)
10. [Discusión y conclusiones](#10-discusión-y-conclusiones)
11. [Tabla resumen: problema → técnica → efecto](#11-tabla-resumen-problema--técnica--efecto)
12. [Oportunidades de mejora](#12-oportunidades-de-mejora)

---

## 1. Resumen ejecutivo

- **Modelo final:** MLP `[32]` (1 capa oculta de 32 neuronas ReLU) | **AdaGrad** | **lr = 0.05** | sin L1/L2 | **dropout = 0.1** | `BCEWithLogitsLoss` | early stopping (mejor época 37, parada en la 52) | **umbral de decisión 0.40** elegido en validación. 1 217 parámetros.
- **Desempeño en prueba (599 estudiantes, usado una sola vez):** accuracy 0.8080, F1 (clase 1) 0.7527, **recall 0.8178**, precision 0.6972, AUC 0.8992.
- **Hallazgo central:** el **perceptrón (regresión logística, 37 parámetros) iguala o supera al modelo final** en accuracy (0.8280), pérdida, F1 (0.7553) y AUC (0.9022). El modelo final solo gana en recall, y eso se debe sobre todo al umbral de 0.40.
- **Por qué:** (1) la relación entre las entradas y la etiqueta es casi lineal, en parte por una **circularidad parcial** (la etiqueta viene del PHQ-9 y las entradas incluyen estrés, ansiedad y "qué tan deprimido se sintió ayer"); (2) hay **sesgo de selección** al elegir el mejor de 216 modelos con un único conjunto de validación de 479 muestras.
- **Respuesta a la pregunta de investigación:** las variables permiten detectar el riesgo con AUC ≈ 0.90 en datos no vistos, pero la señal es mayormente lineal y cercana a los instrumentos que definen la etiqueta. El modelo debe usarse como **apoyo**, no como reemplazo del criterio profesional.

---

## 2. Datos y variable objetivo

| Aspecto | Detalle |
|---|---|
| Fuente | *Self-reported subjective well-being and mental health of college students – 2022* (Mendeley Data, Universidad Icesi; DOI 10.17632/bytb22nf7m.1). |
| Población | Estudiantes universitarios colombianos. |
| Tamaño | 2 992 filas × 34 columnas. |
| Contenido | Demografía y nivel socioeconómico, bienestar subjetivo del día anterior (preguntas 6–15), PSS-10 (estrés, 16.x), GAD-7 (ansiedad, 17.x). |
| Variable objetivo | `riesgo_dep`, construida con **k-means sobre las preguntas PHQ-9** (0 = bajo riesgo, 1 = alto riesgo). |
| Distribución | 1 923 sin riesgo (64.3 %) / 1 069 con riesgo (35.7 %). |

**Decisiones y razonamiento sobre la etiqueta:**

- Las preguntas **PHQ-9 no se usan como entradas**, para que el modelo no vea directamente la información con la que se generó la etiqueta.
- Sí se conservan PSS-10, GAD-7 y la pregunta 12 (`depresion_ayer`). El notebook advierte desde el inicio que esto genera una **posible circularidad parcial**: son constructos muy cercanos a la depresión, así que es esperable que estén fuertemente asociados con el objetivo y que parte del desempeño se explique por eso. Esta advertencia vuelve a aparecer en las conclusiones.

---

## 3. Preprocesamiento

| Decisión | Razonamiento |
|---|---|
| **Abreviar los nombres de columna** (`anio_nac`, `estrato`, `pss_1`…`pss_10`, `gad_1`…`gad_7`, etc.) y crear un diccionario de variables en español. | Los nombres originales eran las preguntas completas de la encuesta, imposibles de manejar en código. El diccionario conserva el significado de cada variable. Se verifica que no haya nombres duplicados. |
| **Eliminar `educ_padre` y `educ_madre`.** | Son las únicas columnas con faltantes (954 y 1 059 de 2 992, más del 30 %). Se consideran **MNAR** (no faltan al azar) y no se pueden inferir de forma confiable, así que se eliminan en lugar de imputarlas. Quedan 32 columnas. |
| **One-hot encoding de `etnia`** (6 columnas: indígena, gitano, raizal, palenquero, afrocolombiano, ninguno). | La codificación original (1–6) es nominal, sin orden. Usarla como número haría que la red asumiera un orden falso. Se fuerzan las 6 columnas aunque alguna categoría no aparezca. |
| **Recodificar `genero` de {1, 2} a {0, 1}.** | Dejar todas las variables binarias en formato 0/1 (con un `assert` que verifica que solo existan esos dos valores). |

Resultado: **36 variables de entrada**, de las cuales 7 son binarias (`genero` + 6 dummies de etnia) y 29 no binarias.

---

## 4. Partición de los datos y escalado

### 4.1 Partición única

Los conjuntos se definen **una sola vez** y todos los experimentos los reutilizan:

1. Split **estratificado** 80/20 entrenamiento/prueba (`random_state = 42`).
2. Split **estratificado** 80/20 del entrenamiento en entrenamiento de experimentos y validación.

| Conjunto | Filas | Proporción clase 1 |
|---|---|---|
| Entrenamiento (experimentos) `X_train_exp` | 1 914 | 0.3574 |
| Validación `X_val` | 479 | 0.3570 |
| Prueba `X_test` | 599 | 0.3573 |

**Razonamiento:** la estratificación mantiene la proporción 64/36 en los tres conjuntos. El split fijo garantiza que las diferencias entre experimentos vengan del factor que se cambia y no de los datos. El conjunto de prueba **solo se usa en la evaluación final**.

### 4.2 Corrección de fuga de información

En la versión anterior del notebook el `StandardScaler` se ajustaba con todo `X_train`, que incluía las filas de validación. Así, la media y la desviación estándar ya "habían visto" la validación. **Ahora la validación se separa antes de escalar** y el escalador se ajusta **solo con `X_train_exp`**; después se aplica a validación y prueba.

### 4.3 Estrategias de escalado

La función `preparar_conjuntos(estrategia)` genera los tensores para cuatro estrategias. Las columnas binarias **nunca se escalan**.

| Estrategia | Transformación |
|---|---|
| `sin_escalar` | Datos crudos. |
| `parcial` | `StandardScaler` solo en `anio_nac` y `estrato` (lo que se hacía antes). |
| `estandar_completo` | `StandardScaler` en todas las columnas no binarias. |
| `minmax_completo` | `MinMaxScaler` (0–1) en todas las columnas no binarias. |

Hasta el experimento de escalado se usa `parcial`; después, la estrategia ganadora (ver §7.3).

---

## 5. Infraestructura común de modelado

Todo se hace en **PyTorch**, con el mismo framework, el mismo split y el mismo protocolo para todos los modelos. Estas decisiones existen para que **las comparaciones sean justas y reproducibles**:

| Componente | Qué hace | Por qué |
|---|---|---|
| `fijar_semilla(seed=42)` | Fija `random`, `numpy` y `torch`, y activa algoritmos deterministas. Se llama antes de crear cada modelo y antes de cada entrenamiento. | Reproducibilidad. La versión anterior tenía un baseline de Keras sin semilla. |
| `MLP(input_size, hidden_layers, dropout)` | Red con salida de 1 neurona (logit). `hidden_layers = []` es el perceptrón (regresión logística). Si `dropout > 0`, se agrega `nn.Dropout(p)` después de cada ReLU. | Una sola clase para todas las arquitecturas. |
| `crear_modelo(...)` + `PESOS_INICIALES` | Guarda una sola vez el `state_dict` inicial de cada arquitectura y lo copia capa lineal por capa lineal a cada modelo nuevo. | **Todas las corridas de una misma arquitectura parten de los mismos pesos**, así que una diferencia de resultados se debe al factor que cambia y no a la inicialización. Se copia capa por capa porque con dropout cambian los índices del `nn.Sequential`. |
| `DataLoader` con generador con semilla | Mismo orden de mini-batches en todas las corridas. | Elimina otra fuente de variación aleatoria. |
| `entrenar_modelo(...)` | Única función de entrenamiento: L1/L2 sobre los **pesos** (no sesgos), `pos_weight` opcional, early stopping con restauración de la mejor época. | Un único protocolo para todo. |
| Pérdidas registradas | Siempre la **BCE pura**, sin ponderar y sin penalización. | Corridas con y sin L1/L2 o `pos_weight` quedan en la misma escala y son comparables. |
| Métricas de entrenamiento por época | Se calculan en modo `eval()` sobre todo `X_train_exp`. | Con dropout, el promedio de mini-batches en modo `train()` subestima el desempeño real. |
| `torch.set_num_threads(1)` | Un solo hilo de CPU. | Las redes son pequeñas; con un hilo el entrenamiento es más rápido. |
| Batch size | 32 (GD de batch completo usa las 1 914 filas). | 60 actualizaciones por época con mini-batch. |

---

## 6. Criterio de selección de los experimentos

- **Experimentos de un solo factor:** se elige la opción con **menor pérdida BCE de validación en la mejor época**.
  - **Razón:** en un conjunto de 479 muestras, una sola muestra mal clasificada mueve la accuracy 0.21 puntos porcentuales; la pérdida refleja de forma continua la calidad de las probabilidades y es más estable. Siempre se reportan también accuracy, F1 y recall de la clase 1.
- **Pipeline final:** se ordena por **F1 de la clase 1 en validación** y, en caso de empate, por menor pérdida de validación (ver §8.2).

Cada experimento termina con un bloque **Observación → Por qué importa → Decisión**.

---

## 7. Experimentos de un solo factor

### 7.1 Baseline: perceptrón

Perceptrón (MLP sin capas ocultas = regresión logística), Mini-Batch GD, lr = 0.01, 50 épocas, sin early stopping.

| Modelo | Pérdida val | Accuracy val | F1 val | Recall val | AUC val | Brecha pérdida | Parámetros |
|---|---|---|---|---|---|---|---|
| Perceptrón | 0.3908 | 0.8351 | 0.7642 | 0.7485 | 0.8944 | 0.0303 | 37 |

- **Observación:** un modelo lineal ya supera el 83 % de accuracy. La brecha es pequeña y la mejor época es la 47 de 50 (todavía mejoraba lentamente).
- **Por qué importa:** es una referencia **exigente**; cualquier red más compleja tiene que justificarse frente a ella.
- **Decisión:** agregar una capa oculta (`[32]`) con el mismo optimizador, lr y épocas para ver si la no linealidad aporta algo.

### 7.2 Red básica `[32]`

| Modelo | Pérdida val | Accuracy val | F1 val | Brecha pérdida | Parámetros |
|---|---|---|---|---|---|
| Perceptrón | 0.3908 | 0.8351 | 0.7642 | 0.0303 | 37 |
| Red básica `[32]` | 0.3869 | 0.8351 | 0.7656 | 0.0346 | 1 217 |

- **Observación:** la red apenas mejora al perceptrón (misma accuracy). Las curvas muestran **aprendizaje inestable**: picos de pérdida de validación de hasta ≈ 0.415 entre las épocas 20 y 50, mientras la de entrenamiento sigue bajando. La brecha crece con el entrenamiento.
- **Por qué importa:** si la pérdida de entrenamiento baja y la de validación se estanca u oscila, la red empieza a memorizar (indicio de sobreajuste). Además, el resultado depende de la época en que se corte.
- **Decisión:** introducir **early stopping** con restauración de la mejor época.

### 7.3 Early stopping

`[32]` y `[128, 128, 128]`, Mini-Batch GD, lr = 0.01, hasta 150 épocas, con y sin early stopping (mismos pesos iniciales y orden de batches).

| Arquitectura | Pérdida val época 150 (sin ES) | Pérdida val mejor época (con ES) | Mejor época | Parada | Brecha época 150 | Brecha mejor época |
|---|---|---|---|---|---|---|
| `[32]` | 0.3998 | 0.3866 | 47 | 62 | 0.0725 | 0.0325 |
| `[128, 128, 128]` | 0.4275 | 0.3922 | 12 | 27 | **0.1622** | 0.0177 |

- **Observación:** sin early stopping la brecha crece mucho, sobre todo en la red grande (accuracy de entrenamiento 0.8871 frente a 0.8267 en validación). La red grande alcanza su mínimo en la época 12 y desde ahí la validación empeora con picos cada vez mayores.
- **Por qué importa:** el sobreajuste es claro y **crece con la capacidad**. Sin un criterio de parada se compararían modelos sobreentrenados en épocas arbitrarias.
- **Decisión:** **desde aquí todos los experimentos usan early stopping** (paciencia 15, `min_delta` = 1e-4, máximo 150 épocas, con restauración de los mejores pesos).

### 7.4 Escalado de las variables de entrada

Red `[32]`, Mini-Batch GD, lr = 0.01, early stopping. Solo cambia la estrategia de escalado.

| Estrategia | Pérdida val | Accuracy val | F1 val | AUC val | Mejor época |
|---|---|---|---|---|---|
| `sin_escalar` | 0.6518 | 0.6430 | **0.0000** | 0.5000 | 23 |
| `parcial` | 0.3866 | 0.8288 | 0.7545 | 0.8967 | 47 |
| `estandar_completo` | 0.3873 | **0.8330** | **0.7619** | 0.8974 | 45 |
| `minmax_completo` | **0.3857** | 0.8225 | 0.7416 | 0.8969 | 149 |

- **Observación:** sin escalar, la red **no aprende**: predice siempre la clase 0. `anio_nac` (≈ 2000) domina la entrada y satura las neuronas. Entre las estrategias que escalan, la diferencia de pérdida es de solo 0.0016. `minmax_completo` converge más lento (mejor época 149).
- **Por qué importa:** escalar es **indispensable**; entre estrategias que escalan, las diferencias son menores que el ruido de un conjunto de 479 muestras.
- **Decisión:** según el criterio fijado (menor pérdida de validación) se adopta **`minmax_completo`**, que además lleva las variables no binarias al mismo rango [0, 1] que las binarias. Se tiene en cuenta su convergencia lenta al leer mejores épocas cercanas a 150.

**Modelos de referencia.** El perceptrón y la red `[32]` se reentrenan con `minmax_completo` y early stopping, para compararlos con el modelo final en igualdad de condiciones:

| Modelo | Pérdida val | Accuracy val | F1 val | Recall val | Mejor época |
|---|---|---|---|---|---|
| Perceptrón (ref.) | 0.3915 | 0.8288 | 0.7515 | 0.7251 | 149 |
| Red básica `[32]` (ref.) | 0.3857 | 0.8225 | 0.7416 | 0.7135 | 149 |

### 7.5 Arquitecturas

9 arquitecturas: 1–3 capas × 32/64/128 neuronas. Mini-Batch GD, lr = 0.01, early stopping.

| Modelo | Pérdida val | Accuracy val | F1 val | Mejor época | Brecha pérdida | Parámetros |
|---|---|---|---|---|---|---|
| MLP_1x32 | 0.3857 | 0.8225 | 0.7416 | 149 | 0.0308 | 1 217 |
| MLP_1x64 | 0.3871 | 0.8205 | 0.7394 | 115 | 0.0294 | 2 433 |
| MLP_1x128 | 0.3862 | 0.8267 | 0.7508 | 115 | 0.0296 | 4 865 |
| MLP_2x32 | 0.3879 | 0.8288 | 0.7530 | 129 | 0.0353 | 2 273 |
| MLP_2x64 | 0.3867 | 0.8309 | 0.7538 | 129 | 0.0363 | 6 593 |
| MLP_2x128 | 0.3853 | 0.8225 | 0.7432 | 91 | 0.0316 | 21 377 |
| MLP_3x32 | 0.3871 | **0.8372** | **0.7692** | 137 | 0.0377 | 3 329 |
| **MLP_3x64** | **0.3836** | 0.8330 | 0.7561 | 90 | 0.0307 | 10 753 |
| MLP_3x128 | 0.3847 | 0.8267 | 0.7462 | 90 | 0.0336 | 37 889 |

- **Observación:** las 9 quedan en un rango muy estrecho (pérdida 0.3836–0.3879, accuracy 0.8205–0.8372). Pasar de 1 217 a 37 889 parámetros **no mejora** la validación. Con early stopping la brecha se mantiene entre 0.029 y 0.038: las redes grandes no sobreajustan más **porque el entrenamiento se corta antes**. Varias llegan cerca de 150 épocas por la convergencia lenta de SGD con lr = 0.01 sobre datos en [0, 1].
- **Por qué importa:** la capacidad adicional no se traduce en mejor generalización; ninguna arquitectura domina.
- **Decisión:** se registra `MLP_3x64` como la de menor pérdida, pero el pipeline final conserva arquitecturas de distinto tamaño (`[32]`, `[64, 64]`, `[128, 128]`, `[128, 128, 128]`). **A igualdad de desempeño se prefiere la red más pequeña (parsimonia).**

### 7.6 Optimizadores

Arquitectura `[64, 64]`, lr = 0.01, mismos pesos iniciales.

| Optimizador | Actualizaciones/época | Pérdida val | Accuracy val | F1 val | Mejor época | Pérdida val última época | Brecha |
|---|---|---|---|---|---|---|---|
| GD (batch completo) | 1 | 0.6514 | 0.6430 | 0.0000 | 150 | 0.6514 | 0.0005 |
| Mini-Batch GD | 60 | 0.3867 | 0.8309 | 0.7538 | 129 | 0.3868 | 0.0363 |
| **SGD con momentum** | 60 | **0.3863** | 0.8330 | 0.7619 | 20 | 0.3923 | 0.0379 |
| AdaGrad | 60 | 0.3901 | 0.8267 | 0.7462 | 26 | 0.3946 | 0.0424 |
| RMSProp | 60 | 0.3917 | 0.8267 | 0.7537 | 3 | 0.3962 | 0.0260 |
| Adam | 60 | 0.3876 | **0.8392** | **0.7688** | 19 | **0.4537** | 0.0478 |

- **Observación:**
  - El GD de batch completo hace 1 actualización por época frente a 60 del mini-batch. En 150 épocas solo da 150 pasos (≈ 2.5 épocas de mini-batch) y no sale de predecir la clase 0. **Su desventaja viene del número de actualizaciones, no de que el algoritmo sea peor.**
  - Los optimizadores adaptativos llegan al mínimo muy rápido (RMSProp en la época 3, Adam en la 19) y después **la validación empeora**: Adam termina en 0.4537 y tiene la mayor brecha.
- **Por qué importa:** los optimizadores rápidos también sobreajustan rápido; early stopping es imprescindible con ellos. Entre los mejores, las diferencias son del orden de 0.001.
- **Decisión:** se adopta **SGD con momentum** como referencia para el bloque de regularización. El pipeline final mantiene **AdaGrad, Mini-Batch GD y GD**, como estaba definido en el diseño del proyecto.

### 7.7 Learning rate

`[64, 64]`, Mini-Batch GD.

| lr | Pérdida val | Accuracy val | F1 val | Recall val | Mejor época | Parada |
|---|---|---|---|---|---|---|
| 0.001 | 0.4496 | 0.7996 | 0.6863 | 0.6140 | 150 | 150 |
| 0.005 | 0.3889 | 0.8246 | 0.7455 | 0.7193 | 149 | 150 |
| 0.01 | 0.3867 | 0.8309 | 0.7538 | 0.7251 | 129 | 144 |
| **0.05** | **0.3861** | **0.8372** | **0.7651** | 0.7427 | 37 | 52 |
| 0.1 | 0.3866 | 0.8225 | 0.7432 | 0.7193 | 12 | 27 |

- **Observación:** con 0.001 el modelo no converge en 150 épocas. Con 0.05 se obtiene la menor pérdida, el mejor F1 y la mejor accuracy. Con 0.1 el mínimo llega en la época 12 y la validación empeora mientras el entrenamiento mejora: **inestabilidad y sobreajuste temprano**.
- **Por qué importa:** un lr pequeño desperdicia épocas y uno grande vuelve el entrenamiento inestable. El rango útil es **0.005–0.05**.
- **Decisión:** el mejor lr para Mini-Batch GD es **0.05**. En el bloque de regularización (SGD con momentum 0.9, que amplifica el paso efectivo ≈ 10 veces) se mantiene **lr = 0.01**, el valor con el que se evaluó ese optimizador. El pipeline final explora 0.005, 0.01 y 0.05.

### 7.8 Función de pérdida

`[64, 64]`, Mini-Batch GD, lr = 0.01. `BCEWithLogitsLoss` (salida logit) frente a `BCELoss` + sigmoide dentro del modelo (`MLPSigmoide`). Se verifica que ambos modelos parten de los mismos pesos (`True`).

| Pérdida | Pérdida val | Accuracy val | F1 val | Mejor época |
|---|---|---|---|---|
| `BCEWithLogitsLoss` | 0.3867 | 0.8309 | 0.7538 | 129 |
| `BCELoss` + sigmoide | 0.3867 | 0.8309 | 0.7538 | 129 |

- **Observación:** resultados **idénticos** hasta el cuarto decimal.
- **Por qué importa:** son matemáticamente la misma pérdida. La diferencia que se veía en la versión anterior del notebook venía de que **los modelos no partían de los mismos pesos**, no de la función de pérdida.
- **Decisión:** se mantiene **`BCEWithLogitsLoss`**, que aplica la sigmoide dentro de la pérdida con el truco *log-sum-exp* y es numéricamente más estable con salidas muy grandes o muy pequeñas.

### 7.9 Regularización: L1, L2 y dropout

**Motivación.** Los experimentos previos mostraron sobreajuste cuando el entrenamiento se prolonga o el paso es agresivo (`[128, 128, 128]` sin ES: brecha 0.1622; Adam: 0.4537 en la última época; lr = 0.1: mejor época 12). Early stopping **solo corta a tiempo, no cambia lo que la red aprende**; la regularización ataca la causa (limita el tamaño de los pesos o fuerza representaciones redundantes).

**Configuración de prueba:** `[128, 128, 128]` (capacidad suficiente para sobreajustar) | SGD con momentum | lr = 0.01 | early stopping.

**Magnitud de la penalización antes de entrenar.** Se calculó cuánto pesa cada penalización con los pesos iniciales frente a la BCE inicial (≈ ln 2 = 0.693). Σ|w| = 1 839 y Σw² = 127.99:

| λ | Penalización L1 inicial | Penalización L2 inicial |
|---|---|---|
| 1e-5 | 0.018 | 0.001 |
| 1e-4 | 0.184 | 0.013 |
| 1e-3 | 1.839 | 0.128 |
| 1e-2 | 18.39 | 1.280 |

Esto explica de antemano qué λ tendrán efecto y por qué L2 es mucho más suave que L1 con el mismo λ.

#### L1

| Modelo | Pérdida val | Accuracy val | F1 val | Recall val | Brecha | Mejor época |
|---|---|---|---|---|---|---|
| Sin regularización | 0.3840 | 0.8288 | 0.7500 | 0.7193 | 0.0345 | 14 |
| L1 λ = 1e-5 | 0.3838 | 0.8288 | 0.7500 | 0.7193 | 0.0342 | 14 |
| L1 λ = 1e-4 | 0.3838 | 0.8267 | 0.7477 | 0.7193 | 0.0329 | 14 |
| **L1 λ = 1e-3** | **0.3831** | 0.8246 | 0.7407 | 0.7018 | **0.0265** | 26 |
| L1 λ = 1e-2 | 0.6517 | 0.6430 | 0.0000 | 0.0000 | −0.0002 | 3 |

- **Por qué λ = 1e-4 casi no tiene efecto:** la penalización vale 0.18 y su gradiente por peso es solo 1e-4, muy pequeño frente al gradiente de la BCE; en las ~14 épocas útiles no alcanza a reducir los pesos.
- Con **λ = 1e-3** la penalización (1.84) ya compite con la BCE: la brecha baja de 0.0345 a 0.0265 y la mejor época se retrasa a la 26, aunque el F1 baja a 0.7407.
- Con λ = 1e-2 (penalización 18.4) todos los pesos se van a cero y el modelo colapsa a la clase 0.
- **Decisión:** **λ = 1e-3 para L1**. Su efecto principal es reducir la brecha, no mejorar el desempeño.

#### L2

| Modelo | Pérdida val | Accuracy val | F1 val | Recall val | Brecha | Mejor época |
|---|---|---|---|---|---|---|
| Sin regularización | 0.3840 | 0.8288 | 0.7500 | 0.7193 | 0.0345 | 14 |
| L2 λ = 1e-5 | 0.3832 | 0.8288 | 0.7500 | 0.7193 | 0.0335 | 14 |
| L2 λ = 1e-4 | 0.3833 | 0.8288 | 0.7500 | 0.7193 | 0.0334 | 14 |
| **L2 λ = 1e-3** | **0.3826** | 0.8267 | 0.7477 | 0.7193 | 0.0308 | 14 |
| L2 λ = 1e-2 | 0.3836 | 0.8246 | 0.7423 | 0.7076 | **0.0185** | 20 |

- L2 es mucho más suave (Σw² = 128 frente a Σ|w| = 1 839). λ = 1e-2 es la que más reduce la brecha, pero baja F1 y recall.
- **Decisión:** **λ = 1e-3 para L2**. La mejora sobre no regularizar es pequeña (−0.0014): con early stopping, el sobreajuste que queda es limitado.

#### Dropout

| p | Pérdida val | Accuracy val | F1 val | Recall val | Brecha | Pérdida val última época |
|---|---|---|---|---|---|---|
| **0.0** | **0.3840** | 0.8288 | 0.7500 | 0.7193 | 0.0345 | 0.4088 |
| 0.1 | 0.3861 | 0.8309 | 0.7568 | 0.7368 | 0.0274 | 0.3970 |
| 0.2 | 0.3861 | 0.8288 | 0.7545 | 0.7368 | 0.0257 | 0.3982 |
| 0.3 | 0.3909 | 0.8288 | 0.7574 | 0.7485 | 0.0303 | 0.4045 |
| 0.5 | 0.3956 | 0.8309 | 0.7568 | 0.7368 | 0.0469 | 0.4017 |

- **Observación:** por pérdida gana p = 0, pero p = 0.1 y 0.2 **reducen la brecha, mejoran F1 y recall**, y hacen que las curvas se degraden menos después del mínimo (0.3970 frente a 0.4088 en la última época). p = 0.5 empeora.
- **Decisión:** por criterio, el mejor p es 0. Como dropout reduce la brecha y mejora el F1, el pipeline final compara **p = 0 contra p = 0.1** (el mejor valor mayor que cero).

#### Comparación del bloque

| Opción | Pérdida val | F1 val | Brecha |
|---|---|---|---|
| Sin regularización | 0.3840 | 0.7500 | 0.0345 |
| L1 (λ = 1e-3) | 0.3831 | 0.7407 | **0.0265** |
| **L2 (λ = 1e-3)** | **0.3826** | 0.7477 | 0.0308 |
| Dropout (p = 0) | 0.3840 | 0.7500 | 0.0345 |

- **Conclusión:** las cuatro opciones quedan dentro de 0.0014 de pérdida. **Con early stopping activo, la regularización aporta poco desempeño adicional**; su valor está en estabilizar y reducir la brecha en redes grandes.
- **Mejor configuración hasta aquí:** `[128, 128, 128]` + SGD con momentum + lr = 0.01 + **L2 (λ = 1e-3)**. En el pipeline final se prueban Ninguna, L1 (1e-3) y L2 (1e-3).

### 7.10 Desbalance de clases (64 % / 36 %)

- **Observación:** en todos los experimentos el **recall de la clase 1 es la métrica más baja** (0.7193 en la mejor configuración, con precision 0.7785).
- **Por qué importa:** en detección temprana, un **falso negativo** (estudiante en riesgo no detectado) es el error más costoso, porque ese estudiante no recibe apoyo. Un falso positivo solo genera una revisión adicional.
- **Opciones comparadas en validación:**
  - (a) sin ajuste;
  - (b) `pos_weight = n_neg / n_pos = 1230 / 684 = 1.7982` en la pérdida (calculado **solo con `y_train_exp`**);
  - (c) mismo modelo que (a), pero **optimizando el umbral de decisión** entre 0.20 y 0.60 para maximizar el F1 de la clase 1.

| Opción (validación) | Accuracy | Balanced acc. | Precision | Recall | F1 | AUC | Pérdida |
|---|---|---|---|---|---|---|---|
| (a) Sin ajuste (0.5) | 0.8267 | 0.8028 | 0.7785 | 0.7193 | 0.7477 | 0.8974 | 0.3826 |
| (b) `pos_weight` (0.5) | 0.8205 | 0.8214 | 0.7157 | 0.8246 | 0.7663 | 0.8961 | 0.4019 |
| **(c) Umbral optimizado (0.37)** | **0.8288** | **0.8279** | 0.7306 | **0.8246** | **0.7747** | 0.8974 | 0.3826 |

- **Criterio automático de aceptación:** un ajuste se adopta si mejora recall **y** F1 sin bajar la accuracy más de 2 puntos. Ambos cumplen; se elige el de mayor F1.
- **Razonamiento:** `pos_weight` sube el recall pero desplaza las probabilidades hacia la clase 1 (la pérdida sube a 0.4019 y la accuracy baja). El umbral logra **el mismo recall con mejor F1 y mejor accuracy, sin alterar el entrenamiento ni la calibración** (misma pérdida y AUC).
- **Decisión:** se adopta el **ajuste de umbral**. El pipeline final se entrena **sin `pos_weight`** y se ordena con umbral 0.5; después, para el modelo elegido, el umbral se vuelve a ajustar **en validación**. **El conjunto de prueba nunca se usa para elegir el umbral.**

---

## 8. Pipeline final: búsqueda en grilla de 216 configuraciones

### 8.1 Espacio de búsqueda

| Factor | Valores | Origen |
|---|---|---|
| Arquitectura | `[32]`, `[64, 64]`, `[128, 128]`, `[128, 128, 128]` | Distintas capacidades (§7.5) |
| Optimizador | AdaGrad, Mini-Batch GD, GD (batch completo) | §7.6 |
| Learning rate | 0.005, 0.01, 0.05 | Rango útil (§7.7) |
| Regularización | Ninguna, L1 (λ = 1e-3), L2 (λ = 1e-3) | §7.9 |
| Dropout | 0, 0.1 | §7.9 |
| Función de pérdida | `BCEWithLogitsLoss` sin `pos_weight` | §7.8 y §7.10 |
| Early stopping | Paciencia 15, máx. 150 épocas, batch 32 (salvo GD) | §7.3 |

Total: 4 × 3 × 3 × 3 × 2 = **216 entrenamientos** (≈ 30 minutos).

**Garantías de comparación justa:** mismo split y escalado `minmax_completo`; mismos pesos iniciales y orden de batches por arquitectura; L1/L2 solo para gradientes (se reporta BCE pura); selección **solo con validación**.

Además de las métricas usuales, se registró la **oscilación de la pérdida de validación** (cambio absoluto medio entre épocas consecutivas hasta la mejor época) como medida de inestabilidad.

### 8.2 Criterio de ranking

Las configuraciones se ordenan por **F1 de la clase 1 en validación** y, en empate, por **menor pérdida de validación**.

**Razón del cambio de criterio respecto a los experimentos individuales:** el objetivo es detectar a la mayor cantidad de estudiantes en riesgo (recall) sin llenar la lista de falsas alarmas (precision). El F1 de la clase 1 resume ambas cosas, mientras que la accuracy favorece a la clase mayoritaria.

### 8.3 Top 10

| # | Arquitectura | Optimizador | lr | Reg. | Dropout | Pérdida val | Accuracy val | F1 val | Recall val | Mejor época | Brecha |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `[32]` | AdaGrad | 0.05 | Ninguna | 0.1 | 0.3838 | 0.8372 | **0.7706** | 0.7661 | 37 | 0.0430 |
| 2 | `[64, 64]` | AdaGrad | 0.05 | L1 | 0.1 | 0.3892 | 0.8372 | 0.7706 | 0.7661 | 37 | 0.0342 |
| 3 | `[128, 128, 128]` | AdaGrad | 0.01 | L2 | 0.0 | 0.3875 | 0.8309 | 0.7679 | 0.7836 | 18 | 0.0375 |
| 4 | `[128, 128]` | AdaGrad | 0.05 | L1 | 0.1 | 0.3892 | 0.8372 | 0.7679 | 0.7544 | 13 | 0.0279 |
| 5 | `[32]` | Mini-Batch GD | 0.05 | Ninguna | 0.1 | 0.3862 | 0.8330 | 0.7661 | 0.7661 | 37 | 0.0336 |
| 6 | `[64, 64]` | Mini-Batch GD | 0.05 | Ninguna | 0.0 | 0.3861 | 0.8372 | 0.7651 | 0.7427 | 37 | 0.0383 |
| 7 | `[128, 128]` | AdaGrad | 0.01 | Ninguna | 0.1 | 0.3876 | 0.8330 | 0.7647 | 0.7602 | 13 | 0.0412 |
| 8 | `[128, 128]` | Mini-Batch GD | 0.05 | L1 | 0.1 | 0.3853 | 0.8351 | 0.7642 | 0.7485 | 42 | 0.0277 |
| 9 | `[32]` | AdaGrad | 0.005 | Ninguna | 0.1 | 0.3876 | 0.8351 | 0.7642 | 0.7485 | 147 | 0.0260 |
| 10 | `[128, 128, 128]` | AdaGrad | 0.01 | Ninguna | 0.1 | 0.3886 | 0.8267 | 0.7635 | 0.7836 | 14 | 0.0513 |

### 8.4 Efecto promedio de cada factor (las 216 corridas)

| Factor | Nivel | F1 val medio (± d.e.) | Pérdida val media |
|---|---|---|---|
| **Optimizador** | AdaGrad | **0.7536** (± 0.0077) | 0.3868 |
| | Mini-Batch GD | 0.7492 (± 0.0071) | 0.3861 |
| | GD | 0.2228 (± 0.2851) | 0.5975 |
| Arquitectura | `[32]` | 0.6493 (± 0.1936) | 0.4445 |
| | `[64, 64]` | 0.5634 (± 0.3099) | 0.4612 |
| | `[128, 128]` | 0.5793 (± 0.3130) | 0.4523 |
| | `[128, 128, 128]` | 0.5088 (± 0.3470) | 0.4692 |
| Learning rate | 0.005 | 0.5371 | 0.4780 |
| | 0.01 | 0.5146 | 0.4701 |
| | 0.05 | 0.6739 | 0.4223 |
| Regularización | Ninguna / L1 / L2 | 0.5800 / 0.5687 / 0.5769 | ≈ 0.456 |
| Dropout | 0 / 0.1 | 0.5750 / 0.5753 | ≈ 0.457 |

### 8.5 Hallazgos del pipeline

- **El GD de batch completo es el factor que más pesa.** 38 de sus 72 corridas no salen de predecir solo la clase 0 (F1 = 0). Solo con lr = 0.05 logra resultados razonables (F1 hasta 0.7094). Por eso los promedios por arquitectura y lr tienen desviaciones tan grandes: las redes grandes con GD fallan más (más parámetros que mover con pocas actualizaciones).
- **AdaGrad y Mini-Batch GD son casi equivalentes y muy estables** (d.e. ≈ 0.007–0.008).
- **Sin contar el GD (144 corridas), los demás factores mueven muy poco el resultado:**
  - F1 medio por arquitectura: 0.7506 a 0.7522;
  - lr = 0.05 algo mejor (0.7548 frente a ≈ 0.750);
  - sin regularización 0.7531, L2 0.7523, L1 0.7488;
  - dropout 0.1: 0.7537 frente a 0.7491 sin dropout;
  - el rango completo de F1 es 0.7368–0.7706, y **25 configuraciones superan 0.76**.
- **La capacidad no gana:** las vecinas de mayor tamaño de la configuración elegida tienen más brecha (0.047–0.057) y más oscilación (0.012–0.017) que `[32]` (0.043 y 0.004).

### 8.6 Configuración elegida y su justificación

**MLP `[32]` | AdaGrad | lr = 0.05 | sin L1/L2 | dropout = 0.1** (mejor época 37, parada en la 52).

Se justifica con más criterios que la accuracy:

1. **F1 y recall de la clase 1:** mayor F1 de validación (0.7706), recall 0.7661, precision 0.7751. Empata con `[64, 64]` + L1 y gana el desempate por menor pérdida (0.3838 frente a 0.3892).
2. **Pérdida de validación:** 0.3838, por debajo del promedio sin GD (≈ 0.386). El alto F1 no viene de probabilidades mal calibradas.
3. **Estabilidad de las curvas:** oscilación de 0.0043, menor que sus vecinas de mayor capacidad (0.012–0.017), sin el deterioro posterior típico de redes grandes.
4. **Brecha moderada:** 0.0430 en pérdida y 0.0103 en accuracy, menor que la de sus vecinas grandes.
5. **Parsimonia:** 1 217 parámetros, la red más pequeña del grid y 31 veces menor que `[128, 128, 128]`.
6. **Robustez:** cambiar un solo factor da F1 entre 0.7508 y 0.7661 en casi todos los casos; está en una **zona estable** del espacio de búsqueda, no en un pico aislado. Excepciones: pasar a GD (0.7051) y quitar el dropout (0.7492, aunque la pérdida queda igual en 0.3835).

**Configuraciones vecinas** (difieren en un solo factor):

| Ranking | Cambio | F1 val | Pérdida val | Brecha | Oscilación |
|---|---|---|---|---|---|
| 1 | — (elegida) | 0.7706 | 0.3838 | 0.0430 | 0.0043 |
| 5 | Mini-Batch GD | 0.7661 | 0.3862 | 0.0336 | 0.0076 |
| 9 | lr = 0.005 | 0.7642 | 0.3876 | 0.0260 | 0.0010 |
| 12 | L1 | 0.7633 | 0.3865 | 0.0306 | 0.0040 |
| 17 | L2 | 0.7625 | 0.3856 | 0.0334 | 0.0048 |
| 25 | `[64, 64]` | 0.7602 | 0.3902 | 0.0469 | 0.0123 |
| 47 | `[128, 128, 128]` | 0.7544 | 0.3863 | 0.0571 | 0.0169 |
| 54 | `[128, 128]` | 0.7537 | 0.3915 | 0.0504 | 0.0132 |
| 67 | lr = 0.01 | 0.7508 | 0.3864 | 0.0304 | 0.0010 |
| 78 | dropout = 0 | 0.7492 | 0.3835 | 0.0383 | 0.0056 |
| 151 | GD | 0.7051 | 0.4412 | 0.0099 | 0.0018 |

> **Advertencia registrada en el notebook:** la ventaja del modelo elegido sobre las siguientes configuraciones equivale a **1 o 2 estudiantes de validación** bien clasificados. Elegir el máximo entre 216 corridas introduce un **sesgo de selección optimista**, que se confirma en prueba.

---

## 9. Evaluación final en prueba

Única sección que usa el conjunto de prueba. Se evalúan **una sola vez** tres modelos ya entrenados con `X_train_exp`.

**Umbral del modelo final:** se repite la búsqueda 0.20–0.60 **en validación** y da **0.40**.

| Modelo | Umbral | Accuracy | Pérdida | Precision (1) | Recall (1) | F1 (1) | AUC | Parámetros |
|---|---|---|---|---|---|---|---|---|
| Perceptrón | 0.5 | **0.8280** | **0.3724** | **0.7681** | 0.7430 | **0.7553** | **0.9022** | 37 |
| Red básica `[32]` | 0.5 | 0.8214 | 0.3764 | 0.7560 | 0.7383 | 0.7470 | 0.9017 | 1 217 |
| **Modelo final** | **0.40** | 0.8080 | 0.3842 | 0.6972 | **0.8178** | 0.7527 | 0.8992 | 1 217 |
| Modelo final | 0.5 | 0.8164 | 0.3842 | 0.7261 | 0.7804 | 0.7523 | 0.8992 | 1 217 |

**Reporte de clasificación del modelo final en prueba (umbral 0.40):**

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| Sin riesgo (0) | 0.8879 | 0.8026 | 0.8431 | 385 |
| Riesgo (1) | 0.6972 | 0.8178 | 0.7527 | 214 |
| Macro promedio | 0.7926 | 0.8102 | 0.7979 | 599 |
| Accuracy | | | 0.8080 | 599 |

---

## 10. Discusión y conclusiones

### 10.1 Comparación honesta con el perceptrón

**El perceptrón iguala o supera al modelo elegido** en accuracy, pérdida, F1 y AUC. El modelo final solo gana en **recall** (0.8178 frente a 0.7430), principalmente gracias al umbral de 0.40, a cambio de bajar la precision (0.6972 frente a 0.7681). Dos explicaciones:

1. **Relación casi lineal y circularidad parcial.** La etiqueta se construyó con k-means sobre el PHQ-9, y las entradas incluyen escalas muy cercanas (PSS-10, GAD-7 y `depresion_ayer`). Una combinación lineal de esas escalas reproduce casi toda la frontera de decisión: un modelo lineal ya captura casi toda la señal (AUC ≈ 0.90) y las capas ocultas tienen poco que aportar. En todo el notebook, la red básica mejora la pérdida del perceptrón en menos de 0.006.
2. **Sesgo de selección.** El modelo final se eligió como el mejor de 216 configuraciones con un único conjunto de validación de 479 muestras, y además se ajustó su umbral sobre ese mismo conjunto. Su F1 de validación (0.7706 con 0.5 y 0.7889 con 0.40) cae en prueba a 0.7523 y 0.7527. El perceptrón, que no pasó por ninguna selección, mantiene su desempeño (0.7515 → 0.7553). **Gran parte de la ventaja que la red mostraba en validación era ruido seleccionado.**

### 10.2 Brecha entre validación y prueba

| Modelo | Accuracy val → prueba | F1 val → prueba |
|---|---|---|
| Perceptrón | 0.8288 → 0.8280 (−0.1 pp) | 0.7515 → 0.7553 |
| Red básica `[32]` | 0.8225 → 0.8214 (−0.1 pp) | 0.7416 → 0.7470 |
| Modelo final (umbral 0.40) | 0.8413 → 0.8080 (**−3.3 pp**) | 0.7889 → 0.7527 |

- Los modelos de referencia, que **no** se eligieron entre muchas alternativas, generalizan igual en validación y prueba. Solo el modelo seleccionado y con umbral ajustado pierde desempeño, que es exactamente lo que se espera con sesgo de selección.
- Su **AUC casi no cambia** (0.8987 → 0.8992): la capacidad de **ordenar** a los estudiantes por riesgo se mantiene. Lo que estaba sobreajustado a validación es el **punto de corte**.

### 10.3 Conclusiones generales del proceso

1. **Escalar es indispensable**; sin escalar la red no aprende. La estrategia concreta importa poco.
2. **Early stopping es la técnica más eficaz contra el sobreajuste** y es imprescindible con optimizadores rápidos o redes grandes.
3. **Más capacidad no mejora la generalización** en este problema; a igualdad de desempeño se prefiere la red más pequeña.
4. **El número de actualizaciones importa más que el algoritmo:** el GD de batch completo falla por dar solo 1 paso por época.
5. **`BCEWithLogitsLoss` y `BCELoss` + sigmoide son equivalentes**; las diferencias previas venían de pesos iniciales distintos. Controlar la inicialización fue clave para obtener conclusiones válidas.
6. **Con early stopping, L1/L2 y dropout aportan poco desempeño** (≤ 0.0014 en pérdida); su beneficio es reducir la brecha y estabilizar.
7. **Ajustar el umbral es mejor que `pos_weight`** para subir el recall: mismo recall, mejor F1 y sin alterar la calibración.
8. **El perceptrón es tan bueno como la mejor red** en datos no vistos. La señal es esencialmente lineal.
9. **Seleccionar entre muchas configuraciones con una sola validación produce resultados optimistas** que no se sostienen en prueba.

### 10.4 Respuesta a la pregunta de investigación

Las variables de comportamiento, bienestar, estrés y ansiedad permiten detectar el riesgo de depresión con una **AUC cercana a 0.90** en datos no vistos. Con el umbral ajustado, el modelo final identifica al **81.8 %** de los estudiantes en riesgo del conjunto de prueba. Sin embargo, gran parte de esta señal es lineal y está cerca de los instrumentos con los que se construyó la etiqueta, así que el desempeño debe interpretarse con cautela y el modelo debe usarse como **apoyo**, no como reemplazo del criterio profesional.

---

## 11. Tabla resumen: problema → técnica → efecto

| Problema detectado | Técnica aplicada | Efecto observado |
|---|---|---|
| Fuga de información: el escalador veía las filas de validación | Split de validación antes de escalar; escalador ajustado solo con `X_train_exp` | Validación no contaminada (corrección metodológica) |
| Resultados no reproducibles (baseline de Keras sin semilla) | PyTorch en todo el notebook, `fijar_semilla`, mismos pesos iniciales y orden de batches | Condiciones idénticas en todas las corridas; por eso `BCEWithLogitsLoss` y `BCELoss` coinciden |
| Variables en escalas muy distintas (`anio_nac` ≈ 2000) | Escalado `minmax_completo` | Sin escalar: F1 = 0, pérdida 0.6518. Escalado: pérdida 0.3857 |
| Sobreajuste al entrenar demasiadas épocas | Early stopping con restauración de pesos | `[128, 128, 128]`: brecha 0.1622 → 0.0177; pérdida val 0.4275 → 0.3922 |
| GD de batch completo sin avance | Mini-batch (60 actualizaciones/época) u optimizadores adaptativos | GD: F1 = 0 con lr ≤ 0.01. Mini-batch o AdaGrad: F1 ≈ 0.75 |
| Inestabilidad con learning rate alto | Barrido de learning rate | lr = 0.1: mejor época 12; lr = 0.05: mejor pérdida (0.3861) |
| Brecha en redes grandes | L1/L2 (λ = 1e-3) y dropout | Brecha 0.0345 → 0.0265 (L1) y 0.0274 (dropout 0.1); mejora de pérdida ≤ 0.0014 |
| Recall bajo de la clase en riesgo | Umbral de decisión ajustado en validación | Validación: recall 0.7193 → 0.8246. Prueba: 0.7804 → 0.8178 (precision 0.7261 → 0.6972) |

---

## 12. Oportunidades de mejora

1. **Validación cruzada estratificada (k = 5 o 10)** para elegir la configuración y el umbral, en lugar de un único conjunto de 479 muestras, y así reducir el sesgo de selección.
2. **Reportar incertidumbre:** repetir los mejores modelos con varias semillas y calcular intervalos de confianza por *bootstrap* en prueba. Con este tamaño de muestra, diferencias de 0.01 en F1 no son concluyentes.
3. **Búsqueda más pequeña y dirigida:** cada configuración adicional aumenta el riesgo de quedarse con un resultado favorecido por el ruido. Una búsqueda aleatoria o bayesiana sobre pocos factores (learning rate y optimizadores adaptativos) reduciría ese riesgo y el costo. El GD de batch completo puede descartarse.
4. **Calibración de probabilidades y umbral según el costo de los errores:** calibrar (Platt o isotónica) y definir el umbral junto con profesionales de bienestar, según el costo de un falso negativo frente a un falso positivo, en lugar de maximizar el F1.
5. **Comparar con modelos lineales regularizados y de árboles** (regresión logística L1/L2, *gradient boosting*). Como el perceptrón ya alcanza el desempeño de la red, conviene saber si un modelo más interpretable basta. También conviene validar con datos de otra cohorte o universidad.
