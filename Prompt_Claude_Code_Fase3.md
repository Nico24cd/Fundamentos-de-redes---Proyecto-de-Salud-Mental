# Prompt para Claude Code — Fase 3: reestructuración del notebook de entrenamiento

Trabaja en esta carpeta. Archivos:

- `Entrenamiento_de_Red_Neuronal_Proyecto-3.ipynb` → notebook actual (NO lo sobrescribas).
- `data_with_depression_risk (1).csv` → datos (2992 filas; objetivo `Riesgo de depresion`).
- `Proyecto redes.pdf` → contexto del problema y definición de variables (léelo primero).
- `Resumen_Decisiones_Optimizacion.md` → resumen de las decisiones actuales y de los problemas detectados.

## Objetivo

Crear **`Entrenamiento_de_Red_Neuronal_Proyecto-4.ipynb`**, una versión mejorada del notebook que responda a la retroalimentación del profesor y corrija los problemas metodológicos. Todo el texto del notebook (celdas markdown, comentarios de código, títulos de gráficas, nombres de ejes y encabezados de tablas) debe estar **en español**. Mantén la estructura general y el estilo de las secciones existentes; reutiliza el código actual donde sirva.

**Regla narrativa obligatoria:** cada vez que un experimento detecte un problema (sobreajuste, inestabilidad, subajuste, desbalance, etc.) y se introduzca una técnica nueva, agrega justo después una celda markdown breve (3–6 líneas) con la estructura **"Observación → Decisión"**: qué se observó (con los números reales de la salida), por qué es un problema y qué técnica se aplica a continuación y por qué. Estas conclusiones se escriben **después de ejecutar el notebook**, con los valores reales. No inventes números.

## Restricciones (no hacer)

- **No** eliminar variables predictoras ni hacer experimentos de selección o ablación de variables. Se usan todas las variables actuales (solo se siguen eliminando `educ_padre` y `educ_madre`, como ya está justificado).
- **No** agregar una sección de trabajo colaborativo (se hará en otro documento).
- **No** usar el conjunto de prueba en ningún experimento ni en la selección de modelos. El conjunto de prueba se usa **una sola vez**, en la sección final.
- No modificar el CSV ni el PDF.

---

## 1. Cambios generales de infraestructura (al inicio de la sección de modelado)

1. **Eliminar TensorFlow/Keras.** El perceptrón baseline y la red básica se reimplementan en PyTorch con la misma clase `MLP`, para que todo use el mismo framework, el mismo split y el mismo protocolo.
2. **Función de semilla global** `fijar_semilla(seed)`, que fije `random`, `numpy` y `torch` (y `torch.use_deterministic_algorithms` si no rompe nada). Todo modelo se crea después de llamarla, y todo `DataLoader` usa `generator=torch.Generator().manual_seed(SEED)` para que el orden de los mini-batches sea reproducible. `SEED = 42` en todo el notebook.
3. **Clase `MLP` única** con parámetros `hidden_layers` y `dropout` (por defecto 0.0). La capa `nn.Dropout(p)` va después de cada ReLU solo si `p > 0`. Salida de 1 neurona (logit). Elimina las variantes duplicadas (`MLPBinarySigmoid`, salida de 2 neuronas con CrossEntropy), salvo la que necesite el experimento de función de pérdida (ver 5.4).
4. **Una única función de entrenamiento** `entrenar_modelo(...)` que reemplace `train_mlp_ce`, `train_mlp`, `train_mlp_loss_experiment`, `train_mlp_regularization_experiment` y `entrenar_configuracion`. Parámetros: modelo, optimizador, `batch_size`, `max_epocas`, `early_stopping` (bool), `paciencia`, `min_delta`, `tipo_reg` (`"Ninguna"`, `"L1"` o `"L2"`), `lambda_reg`, `pos_weight` (opcional) y función de pérdida. Requisitos:
   - La penalización L1/L2 se aplica **solo a los pesos, no a los sesgos**, y se suma a la pérdida únicamente para el gradiente. Las pérdidas que se registran y grafican son siempre la **BCE pura, sin ponderar**, para que todas las corridas sean comparables.
   - Al final de cada época, las métricas de entrenamiento se calculan en modo `model.eval()` sobre todo `X_train_exp`, no como promedio de los mini-batches en modo `train()`. Con dropout, el promedio en modo `train()` subestima el desempeño y produce comparaciones engañosas train vs. validación.
   - Historial por época: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `train_f1`, `val_f1`.
   - **Early stopping** sobre `val_loss` (paciencia por defecto 15, `min_delta=1e-4`, `max_epocas` por defecto 150) que **restaura los pesos de la mejor época**. Devuelve la mejor época y la época de parada.
5. **Función de evaluación** `evaluar(modelo, X, y, umbral=0.5)` que devuelva loss BCE, accuracy, precision, recall, F1 y balanced accuracy de la clase 1, y AUC.
6. **Funciones de gráficas reutilizables** (con títulos y ejes en español):
   - `graficar_curvas(historiales, titulo)`: pérdida y accuracy de train (línea continua) y validación (discontinua), con una línea vertical en la mejor época si hubo early stopping.
   - `tabla_resumen(resultados)`: DataFrame con métricas de train y validación en la mejor época, mejor época, época de parada, brecha train–val (loss y accuracy) y número de parámetros. **Ninguna celda vacía.**
7. **Criterio de selección** (explicarlo en una celda markdown al inicio de "Pruebas de entrenamiento"): en los experimentos de un solo factor se elige por la **menor pérdida BCE de validación en la mejor época**, porque es una métrica más estable que la accuracy en 478 muestras (una muestra equivale a 0.21 puntos de accuracy). Siempre se reportan también accuracy, F1 y recall de la clase 1. El criterio del pipeline final se define en la sección 7.

## 2. Contexto del problema y variable objetivo (nueva celda markdown al inicio)

Con base en `Proyecto redes.pdf`, agrega después de la pregunta de investigación una celda breve que explique:

- de dónde viene el dataset (Mendeley, Universidad Icesi, 2022);
- **cómo se construyó `riesgo_dep`**: aplicando k-means al grupo de preguntas de depresión (PHQ-9) del cuestionario original y asignando 0 = bajo riesgo y 1 = alto riesgo;
- que las preguntas PHQ-9 usadas para construir la etiqueta **no** se incluyen como variables de entrada, lo que evita que el modelo vea directamente la información con la que se generó la etiqueta;
- que sí se incluyen escalas relacionadas (PSS-10 de estrés, GAD-7 de ansiedad y la pregunta 12 sobre estado depresivo del día anterior). Por eso es esperable que estén fuertemente asociadas al objetivo, y esto se debe tener en cuenta al interpretar el desempeño: es una **posible fuente de circularidad parcial**.

Corrige también los textos actuales que dicen "una versión adaptada que ya incluye la etiqueta" para que remitan a esta explicación.

## 3. Preprocesamiento: corrección de fuga y experimento de escalado

1. Corrige los errores de ortografía de los títulos y textos existentes ("Análsis", "Contrucción", "cateogrias", "tambien", "validadción", etc.).
2. Haz el split del conjunto de validación **antes** de escalar. Hoy el `StandardScaler` se ajusta sobre todo `X_train`, que incluye las filas de validación. Se debe ajustar **solo con `X_train_exp`** y luego transformar validación y prueba. Mantén el split estratificado 80/20 train/test (`random_state=42`) y haz el split de validación 80/20 sobre `X_train` también **estratificado** (`train_test_split(..., stratify=y_train, random_state=42)`).
3. Crea una función `preparar_conjuntos(estrategia)` que devuelva los tensores de entrenamiento, validación y prueba para cada estrategia de escalado:
   - `"sin_escalar"`: datos crudos;
   - `"parcial"`: lo que se hace hoy (`StandardScaler` solo en `anio_nac` y `estrato`);
   - `"estandar_completo"`: `StandardScaler` en todas las columnas no binarias (año de nacimiento, estrato, escalas 0–10, PSS 0–4 y GAD 0–3);
   - `"minmax_completo"`: `MinMaxScaler` (rango 0–1) en las mismas columnas no binarias.
   Las columnas binarias (`genero` y las dummies de `etnia`) no se escalan.
4. **Experimento de escalado** (nueva subsección "Experimento: escalado de las variables de entrada"). Ubícalo después de la red básica y del experimento de early stopping (ver 4). Usa la red básica `[32]`, SGD mini-batch, lr = 0.01, batch 32, early stopping, mismos pesos iniciales y mismo orden de batches, y cambia solo la estrategia de escalado. Entrega una tabla comparativa y las curvas de pérdida y accuracy de validación de las 4 estrategias en una misma figura. Agrega una celda "Observación → Decisión" que explique qué estrategia se adopta. A partir de aquí, todos los experimentos usan la estrategia ganadora.

## 4. Baselines y diagnóstico de sobreajuste con early stopping

1. **Perceptrón baseline** en PyTorch: `MLP(hidden_layers=[])` (equivalente a regresión logística), SGD mini-batch, lr = 0.01, batch 32, entrenado en `X_train_exp` y evaluado en validación (no en prueba). Muestra sus curvas y su tabla resumen.
2. **Red básica** `[32]` ReLU, SGD lr = 0.01, 50 épocas, **sin** early stopping, para mostrar el problema. Muestra curvas y tabla. Actualiza la celda que hoy dice "accuracy del 80 %" para que use los valores reales de validación.
3. Celda "Observación → Decisión": se observa aprendizaje inestable y una brecha creciente entre la pérdida de entrenamiento y la de validación (indicio de sobreajuste). Por eso se introduce early stopping.
4. **Experimento: early stopping.** Entrena la red básica y una red grande `[128, 128, 128]` hasta 150 épocas, con y sin early stopping (4 corridas, mismos pesos iniciales). Grafica las curvas, marca la época de parada y compara en tabla la pérdida de validación en la última época frente a la de la mejor época. Agrega "Observación → Decisión": **desde este punto todos los experimentos usan early stopping** con restauración de los mejores pesos.

## 5. Experimentos de un solo factor (limpieza y control)

Todos usan las mismas condiciones: misma estrategia de escalado, mismo split, `fijar_semilla(SEED)` antes de crear cada modelo, mismos pesos iniciales por arquitectura (guardados con `copy.deepcopy(state_dict())`), `DataLoader` con generador con semilla, `BCEWithLogitsLoss`, early stopping y máximo 150 épocas. Ninguno usa el conjunto de prueba. Cada uno termina con tabla resumen, curvas y una celda "Observación → Decisión".

1. **Arquitecturas (corregir el experimento actual):** las 9 combinaciones de 1–3 capas × 32/64/128 neuronas, ahora con semilla fija, salida de 1 neurona con BCE y el mismo optimizador de referencia que la red básica (SGD mini-batch, lr = 0.01). Así queda comparable con los demás experimentos. Mantén la gráfica de complejidad vs. desempeño, pero usa en el eje Y la pérdida y la accuracy de validación en la mejor época, y agrega el número de parámetros. Si las redes grandes muestran una brecha train–val mayor, menciónalo en la celda de observación como indicio de sobreajuste.
2. **Optimizadores (limpiar):** GD de batch completo, Mini-Batch GD, SGD con momentum (0.9), AdaGrad, RMSProp y Adam, con arquitectura `[64, 64]`, lr = 0.01 y los mismos pesos iniciales. **Elimina el cálculo y la gráfica de pérdida en prueba** de las celdas actuales 36–37 y reemplázalos por curvas de train/val. Presenta las 6 curvas de validación en una sola figura y las 6 de entrenamiento en otra, además de la tabla. En la observación, aclara que el GD de batch completo hace 1 actualización por época frente a unas 60 del mini-batch. Su desventaja se debe al número de actualizaciones, no a que el algoritmo sea peor.
3. **Learning rate:** {0.001, 0.005, 0.01, 0.05, 0.1} con Mini-Batch GD, `[64, 64]` y los mismos pesos iniciales. Gráfica de curvas de validación y tabla. Si con lr alto la accuracy de entrenamiento sube mientras la de validación baja, explícalo como inestabilidad o sobreajuste.
4. **Función de pérdida:** `BCEWithLogitsLoss` vs. `BCELoss` + sigmoide. Asegúrate de que ambos modelos partan de **los mismos pesos iniciales**: las capas lineales tienen las mismas claves en el `state_dict`; si no, cópialas explícitamente.

## 6. Bloque de sobreajuste: regularización L1/L2 y dropout

Introdúcelo con una celda "Observación → Decisión" que cite la evidencia de sobreajuste de los experimentos anteriores (brecha train–val en redes grandes, lr altos o AdaGrad). Usa como configuración de prueba una red con capacidad suficiente para sobreajustar, `[128, 128, 128]`, con el mejor optimizador y lr encontrados, early stopping y máximo 150 épocas.

1. **Barrido de λ para L1:** λ ∈ {1e-5, 1e-4, 1e-3, 1e-2}, más la referencia sin regularización.
2. **Barrido de λ para L2:** λ ∈ {1e-5, 1e-4, 1e-3, 1e-2} (penalización explícita sobre los pesos, igual que L1).
3. Para cada barrido: tabla con val_loss, val_acc, val_F1, brecha train–val y mejor época; gráfica de val_loss y brecha vs. λ (eje X logarítmico); y curvas train/val de cada λ. Selecciona el **mejor λ de L1** y el **mejor λ de L2** (menor val_loss). Explica en la observación por qué λ = 1e-4, el valor actual, prácticamente no tenía efecto.
4. **Dropout:** p ∈ {0.0, 0.1, 0.2, 0.3, 0.5} sobre la misma configuración, con tabla, curvas y "Observación → Decisión". Selecciona el mejor p.
5. Tabla final del bloque: sin regularización vs. mejor L1 vs. mejor L2 vs. mejor dropout, con curvas comparativas en una sola figura.

## 7. Desbalance de clases (64 % / 36 %)

1. "Observación": el recall de la clase 1 (riesgo) es el más bajo de las métricas, y para una detección temprana los falsos negativos son el error más costoso.
2. **Experimento** con la mejor configuración hasta el momento, comparando tres opciones en **validación**:
   (a) sin ajuste;
   (b) `pos_weight = n_neg / n_pos` en `BCEWithLogitsLoss` (calculado solo con `y_train_exp`);
   (c) sin ajuste en el entrenamiento pero con el **umbral de decisión optimizado en validación**, buscando entre 0.2 y 0.6 el que maximiza F1 de la clase 1.
   Reporta accuracy, balanced accuracy, precision, recall y F1 de la clase 1, AUC y la matriz de confusión de validación de cada opción. Incluye la curva precisión–recall de validación marcando el umbral elegido.
3. "Decisión": si `pos_weight` o el umbral mejoran el recall y el F1 de la clase 1 sin una caída importante de la accuracy, adopta esa opción en el pipeline final; si no, justifica por qué no. Aclara que el umbral se ajusta en validación y que el conjunto de prueba nunca se usa para elegirlo.

## 8. Pipeline final (grid search)

1. Factores:
   - Arquitectura: `[32]`, `[64, 64]`, `[128, 128]`, `[128, 128, 128]`.
   - Optimizador: AdaGrad, Mini-Batch GD y GD de batch completo.
   - Learning rate: 0.005, 0.01 y 0.05.
   - Regularización: Ninguna, L1 con el mejor λ y L2 con el mejor λ (de la sección 6).
   - Dropout: 0 y el mejor p (de la sección 6).
   - Pérdida: `BCEWithLogitsLoss` (con `pos_weight` si se adoptó en la sección 7).
   - Early stopping en todas las corridas, máximo 150 épocas, batch 32 (salvo el GD).
   
   Total: 4 × 3 × 3 × 3 × 2 = **216 corridas**. Antes de lanzar el grid, mide el tiempo de una corrida y, si el estimado total supera unos 40 minutos, avísame antes de reducir algo.
2. Corrige la celda markdown del pipeline para que la tabla de factores y la cuenta de combinaciones coincidan **exactamente** con el código. Hoy la tabla lista 3 arquitecturas y 2 regularizaciones con "4 × 3 × 3 × 2 = 108", lo cual es inconsistente.
3. **Criterio de ranking:** F1 de la clase 1 en validación (mejor época); en caso de empate, menor val_loss. Justifícalo en markdown: el objetivo es detectar estudiantes en riesgo. Reporta también val_acc.
4. **Tablas** (con encabezados en español y sin celdas vacías):
   - todas las corridas ordenadas;
   - top 10 con métricas de train y validación, mejor época, época de parada y brecha;
   - efecto promedio ± desviación estándar de cada factor (arquitectura, optimizador, lr, regularización y dropout);
   - tabla pivote de val_F1 por arquitectura × (optimizador, lr) para cada regularización;
   - comparación directa L1 vs. L2 vs. Ninguna y dropout 0 vs. p.
5. **Gráficas:** curvas promedio de validación por nivel de cada factor (una fila de subplots, como en el notebook actual, ahora incluyendo dropout), curvas del top 5 y un mapa de calor de val_F1 por arquitectura × optimizador. Si las grillas de 9 subplots por arquitectura resultan excesivas con 216 corridas, consérvalas solo para la arquitectura ganadora.
6. Celda "Observación → Decisión" con los hallazgos por factor y la configuración elegida. Debe incluir una justificación del modelo elegido **más allá de la accuracy de validación**: estabilidad de sus curvas, brecha train–val, val_loss, F1 y recall de la clase 1, número de parámetros (parsimonia) y su desempeño frente a configuraciones vecinas. Esto responde a la retroalimentación: "justifiquen el modelo base con algo más que la exactitud de validación".

## 9. Evaluación final en prueba (única sección que usa el conjunto de prueba)

1. Toma el modelo final (mejor del grid, con el umbral elegido en validación si aplica), el **perceptrón baseline** y la **red básica**, todos ya entrenados en `X_train_exp`, y evalúalos **una sola vez** en prueba.
2. **Tabla comparativa final, completa y sin celdas vacías:** filas = perceptrón, red básica y modelo final; columnas = accuracy, loss, precision, recall, F1 y AUC de la clase 1 en **train, validación y prueba**, más el número de parámetros y la mejor época. Esto atiende la retroalimentación: "la tabla tiene celdas vacías y no incluye métricas de entrenamiento".
3. **Gráficas del modelo reportado** (atienden "faltan las curvas del modelo reportado"):
   - curvas de pérdida y accuracy train/val **solo del modelo final**, con la época de early stopping marcada;
   - matriz de confusión en prueba;
   - curva ROC en prueba comparando los tres modelos;
   - curva precisión–recall en prueba.
   Titula cada figura con el nombre exacto del modelo y la configuración, para que no se confundan con otras arquitecturas.
4. `classification_report` del modelo final en prueba.

## 10. Discusión final (celda markdown en español, con números reales)

- Comparación honesta del modelo final con el perceptrón. Si el perceptrón iguala o supera a la red, analízalo: posible relación casi lineal entre las escalas PSS/GAD y una etiqueta derivada de PHQ-9 (circularidad parcial, ver sección 2), y el sesgo de selección al elegir el mejor de 216 configuraciones con un solo conjunto de validación de 478 muestras. Esto atiende la retroalimentación: "el perceptrón iguala o supera al modelo elegido, y eso merece ser analizado".
- Brecha entre validación y prueba y su interpretación.
- Qué técnica resolvió cada problema detectado (tabla breve "Problema detectado → Técnica aplicada → Efecto observado").
- Cinco oportunidades de mejora coherentes con el diagnóstico.

## 11. Verificación (obligatoria antes de terminar)

1. Verifica que estén instalados `torch`, `scikit-learn`, `pandas`, `matplotlib`, `jupyter` y `nbconvert`; instala lo que falte.
2. Ejecuta el notebook completo de principio a fin: `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Entrenamiento_de_Red_Neuronal_Proyecto-4.ipynb`. Debe correr sin errores con el CSV en la misma carpeta (`FUENTE_DATOS = "local"`).
3. Después de ejecutarlo, **escribe o actualiza todas las celdas "Observación → Decisión" y la discusión con los valores reales** de las salidas, y revisa que ningún texto contradiga una tabla o figura.
4. Comprueba con búsqueda en el código que `X_test` / `X_test_tensor` / `y_test` **solo** aparecen en la preparación de datos y en la sección 9.
5. Entrégame un resumen corto: secciones nuevas, configuración final elegida, métricas finales en prueba frente al perceptrón, tiempo total de ejecución y cualquier decisión que hayas tenido que tomar por tu cuenta.
