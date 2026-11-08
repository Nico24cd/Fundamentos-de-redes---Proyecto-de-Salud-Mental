"""
Pipeline del modelo final de detección temprana de riesgo de depresión.

Generado a partir de notebooks/pipeline.ipynb: contiene las mismas funciones
de las etapas, la configuración y la inferencia, para usar el modelo sin el notebook.

Uso:
    from src.pipeline_riesgo import cargar_artefactos, predecir
    artefactos = cargar_artefactos()
    predecir(registro_crudo, artefactos)   # registro con las columnas originales del CSV
"""
import copy
import json
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Raíz del repositorio (este archivo vive en src/)
RAIZ = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    semilla: int = 42
    # Partición (estratificada)
    test_size: float = 0.20
    val_size: float = 0.20            # fracción de entrenamiento que se reserva para validación
    # Modelo final (mejor configuración del pipeline de 216 corridas)
    arquitectura: tuple = (32,)
    dropout: float = 0.1
    optimizador: str = "AdaGrad"
    learning_rate: float = 0.05
    batch_size: int = 32
    # Early stopping
    max_epocas: int = 150
    paciencia: int = 15
    min_delta: float = 1e-4
    # Decisión
    umbral: float = 0.40              # elegido en validación (maximiza el F1 de la clase 1)
    # Perceptrón de referencia
    perceptron_optimizador: str = "SGD"
    perceptron_learning_rate: float = 0.01
    # Rutas
    ruta_datos: Path = RAIZ / "datos" / "data_with_depression_risk.csv"
    ruta_modelos: Path = RAIZ / "modelos"
    ruta_figuras: Path = RAIZ / "reportes" / "figuras"


CFG = Config()


def fijar_semilla(semilla=CFG.semilla):
    """Fija todas las fuentes de aleatoriedad para que el entrenamiento sea reproducible."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.use_deterministic_algorithms(True, warn_only=True)


torch.set_num_threads(1)
fijar_semilla()


OBJETIVO = "riesgo_dep"

PREGUNTAS = {
    "1": "anio_nac", "2": "genero", "3": "estrato", "4": "etnia",
    "5.1": "educ_padre", "5.2": "educ_madre",
    "6": "satisf_vida", "7": "felicidad_ayer", "8": "risa_ayer", "9": "aprendio_ayer",
    "10": "disfrute_ayer", "11": "preocupacion_ayer", "12": "depresion_ayer",
    "13": "ira_ayer", "14": "estres_ayer", "15": "soledad",
    **{f"16.{i}": f"pss_{i}" for i in range(1, 11)},   # PSS-10: estrés percibido (0-4)
    **{f"17.{i}": f"gad_{i}" for i in range(1, 8)},    # GAD-7: ansiedad (0-3)
}
COLUMNAS_ELIMINADAS = ["educ_padre", "educ_madre"]


def cargar_datos(ruta=CFG.ruta_datos):
    """Etapa 1: lee el CSV crudo."""
    return pd.read_csv(ruta)


def renombrar_columnas(df):
    """Etapa 2: cambia cada pregunta por un nombre corto según su número.
    Las columnas que ya tienen nombre corto no cambian."""
    def nombre_corto(columna):
        columna = columna.strip()
        if columna.lower().startswith("riesgo"):
            return OBJETIVO
        numero = re.match(r"^(\d+(?:\.\d+)?)", columna)
        return PREGUNTAS.get(numero.group(1), columna) if numero else columna
    return df.rename(columns=nombre_corto)


def limpiar(df):
    """Etapa 3: elimina las columnas de educación de los padres (>30 % de faltantes, MNAR)."""
    return df.drop(columns=COLUMNAS_ELIMINADAS, errors="ignore")


ETNIAS = {1: "indigena", 2: "gitano", 3: "raizal", 4: "palenquero", 5: "afrocolombiano", 6: "ninguno"}
COLUMNAS_ETNIA = [f"etnia_{v}" for v in ETNIAS.values()]


def codificar(df):
    """Etapa 4: one-hot de etnia (6 columnas fijas) y género a 0/1."""
    df = df.copy()
    df["genero"] = df["genero"].map({1: 0, 2: 1})
    dummies = pd.get_dummies(df["etnia"].map(ETNIAS), prefix="etnia", dtype=int)
    dummies = dummies.reindex(columns=COLUMNAS_ETNIA, fill_value=0)
    return pd.concat([df.drop(columns="etnia"), dummies], axis=1)


def preparar(df_crudo):
    """Etapas 2 a 4 juntas: del CSV crudo a variables numéricas."""
    return codificar(limpiar(renombrar_columnas(df_crudo)))


def particionar(df, cfg=CFG):
    """Etapa 5: entrenamiento / validación / prueba, estratificado (64 % / 16 % / 20 %)."""
    X = df.drop(columns=OBJETIVO)
    y = df[OBJETIVO].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.semilla, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=cfg.val_size, random_state=cfg.semilla, stratify=y_train)
    return {"train": (X_tr, y_tr), "val": (X_val, y_val), "test": (X_test, y_test)}


COLUMNAS_BINARIAS = ["genero"] + COLUMNAS_ETNIA


def ajustar_escalador(X_train):
    """Etapa 6a: ajusta MinMaxScaler SOLO con entrenamiento (variables no binarias)."""
    columnas = [c for c in X_train.columns if c not in COLUMNAS_BINARIAS]
    return MinMaxScaler().fit(X_train[columnas].astype(float))


def escalar(X, escalador):
    """Etapa 6b: aplica el escalador ya ajustado (validación, prueba o inferencia)."""
    X = X.astype(float).copy()
    columnas = list(escalador.feature_names_in_)
    X[columnas] = escalador.transform(X[columnas])
    return X


def a_tensores(X, y=None):
    X_t = torch.tensor(X.values, dtype=torch.float32)
    return X_t if y is None else (X_t, torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1))


class MLP(nn.Module):
    """Red con capas ocultas ReLU (+ dropout opcional) y 1 neurona de salida (logit)."""

    def __init__(self, n_entradas, arquitectura, dropout=0.0):
        super().__init__()
        capas, previo = [], n_entradas
        for n in arquitectura:
            capas += [nn.Linear(previo, n), nn.ReLU()]
            if dropout > 0:
                capas.append(nn.Dropout(dropout))
            previo = n
        capas.append(nn.Linear(previo, 1))
        self.network = nn.Sequential(*capas)

    def forward(self, x):
        return self.network(x)


def crear_modelo(n_entradas, arquitectura=CFG.arquitectura, dropout=CFG.dropout):
    fijar_semilla()   # mismos pesos iniciales en cada ejecución
    return MLP(n_entradas, arquitectura, dropout)


def entrenar(modelo, X_tr, y_tr, X_val, y_val, optimizador="AdaGrad", lr=CFG.learning_rate, cfg=CFG):
    """Etapa 7: entrena con mini-batches y early stopping sobre la pérdida de validación."""
    opt = (torch.optim.Adagrad if optimizador == "AdaGrad" else torch.optim.SGD)(modelo.parameters(), lr=lr)
    criterio = nn.BCEWithLogitsLoss()
    fijar_semilla()
    lotes = DataLoader(TensorDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(cfg.semilla))
    historial = {"train_loss": [], "val_loss": []}
    mejor, mejor_epoca, mejor_estado, sin_mejora = float("inf"), 0, None, 0

    for epoca in range(1, cfg.max_epocas + 1):
        modelo.train()
        for X_b, y_b in lotes:
            opt.zero_grad()
            criterio(modelo(X_b), y_b).backward()
            opt.step()
        modelo.eval()
        with torch.no_grad():
            historial["train_loss"].append(criterio(modelo(X_tr), y_tr).item())
            perdida_val = criterio(modelo(X_val), y_val).item()
        historial["val_loss"].append(perdida_val)
        if perdida_val < mejor - cfg.min_delta:
            mejor, mejor_epoca, mejor_estado, sin_mejora = perdida_val, epoca, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= cfg.paciencia:
                break

    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    return {"historial": historial, "mejor_epoca": mejor_epoca, "epoca_parada": epoca}


def predecir_proba(modelo, X_t):
    """Probabilidad de riesgo (clase 1) para cada fila."""
    modelo.eval()
    with torch.no_grad():
        return torch.sigmoid(modelo(X_t)).view(-1).numpy()


def aplicar_umbral(proba, umbral=CFG.umbral):
    """Etapa 8: probabilidad -> clase (1 = riesgo)."""
    return (proba >= umbral).astype(int)


def elegir_umbral(proba_val, y_val, candidatos=np.round(np.arange(0.20, 0.6001, 0.01), 2)):
    """Umbral que maximiza el F1 de la clase 1 en VALIDACIÓN (nunca en prueba)."""
    f1s = [f1_score(y_val, aplicar_umbral(proba_val, u)) for u in candidatos]
    return float(candidatos[int(np.argmax(f1s))])


def evaluar(y, proba, umbral=CFG.umbral):
    """Etapa 9: métricas de la clase 1 y AUC."""
    pred = aplicar_umbral(proba, umbral)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "auc": roc_auc_score(y, proba)}


def guardar_artefactos(modelo, escalador, columnas, metricas, cfg=CFG):
    """Guarda pesos, escalador, columnas, umbral y configuración en modelos/."""
    ruta = Path(cfg.ruta_modelos)
    ruta.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), ruta / "mlp_final_state_dict.pt")
    joblib.dump(escalador, ruta / "escalador_minmax.joblib")
    metadatos = {
        "columnas": list(columnas),
        "umbral": cfg.umbral,
        "arquitectura": list(cfg.arquitectura),
        "dropout": cfg.dropout,
        "configuracion": {k: str(v) for k, v in asdict(cfg).items()},
        "metricas_prueba": metricas,
        "versiones": {"torch": torch.__version__, "scikit-learn": sklearn.__version__,
                      "pandas": pd.__version__, "numpy": np.__version__},
    }
    (ruta / "metadatos.json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False))


def cargar_artefactos(ruta=CFG.ruta_modelos):
    """Reconstruye el modelo entrenado y el escalador desde disco."""
    ruta = Path(ruta)
    meta = json.loads((ruta / "metadatos.json").read_text())
    modelo = MLP(len(meta["columnas"]), meta["arquitectura"], meta["dropout"])
    modelo.load_state_dict(torch.load(ruta / "mlp_final_state_dict.pt"))
    modelo.eval()
    return {"modelo": modelo, "escalador": joblib.load(ruta / "escalador_minmax.joblib"),
            "columnas": meta["columnas"], "umbral": meta["umbral"]}


def predecir(registro_crudo, artefactos):
    """Inferencia: registro(s) crudo(s) (dict, Series o DataFrame) -> probabilidad y clase."""
    if isinstance(registro_crudo, dict):
        registro_crudo = pd.DataFrame([registro_crudo])
    elif isinstance(registro_crudo, pd.Series):
        registro_crudo = registro_crudo.to_frame().T
    X = preparar(registro_crudo).drop(columns=OBJETIVO, errors="ignore")
    faltantes = set(artefactos["columnas"]) - set(X.columns)
    if faltantes:
        raise ValueError(f"Faltan variables en el registro: {sorted(faltantes)}")
    X = escalar(X[artefactos["columnas"]], artefactos["escalador"])
    proba = predecir_proba(artefactos["modelo"], a_tensores(X))
    return pd.DataFrame({"probabilidad_riesgo": proba,
                         "clase": aplicar_umbral(proba, artefactos["umbral"])}, index=X.index)
