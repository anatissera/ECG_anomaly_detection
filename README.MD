# Detección de anomalías en señales ECG

Este repositorio contiene el código, los datos y los resultados del proyecto de análisis y modelado de señales ECG.

---

## Estructura del proyecto

```
ECG_anomaly_detection/
├── data/                         # Datos crudos y procesados
│   ├── ChapmanShaoxing/
│   ├── ChapmanShaoxing_fmm/
│   ├── data_processed/
│   ├── ptb_xl_fmm/
│   └── ptb-xl/
├── src/                          # Módulos y utilidades de Python
├── notebooks/                    # Notebooks de Jupyter
│   ├── data_analysis/
│   │   └── visualizacion_de_datos.ipynb   # Efecto del filtrado FMM y exploración de los datos
│   ├── model_testing.ipynb       # Evaluación en el conjunto de datos no vistos
│   └── pipeline_VAE.ipynb        # Flujo completo: entrenamiento de VAE, cálculo de scores y generación de reportes
├── saved_models_and_params/      # Modelos entrenados y configuraciones de hiperparámetros
│   ├── best_hps.json
│   ├── threshold.json
│   ├── vae_model.pt
│   ├── vae_state.pt
│   └── xgb_clf.joblib
├── Tissera_Couriel_Informe.pdf   # Informe final en formato PDF
├── requirements.txt              # Dependencias de Python
└── README.md                     # Este archivo (descripción y uso)
```

---

## Contenido principal

- **`notebooks/data_analysis/visualizacion_de_datos.ipynb`**: Análisis exploratorio y visualización de la influencia del filtrado FMM en las señales ECG.

- **`notebooks/pipeline_VAE.ipynb`**: Punto de entrada para reproducir el flujo completo del proyecto. Incluye:
  1. Carga y preprocesamiento de datos (FMM).
  2. Entrenamiento y evaluación del VAE.
  3. Cálculo de scores y detección de anomalías.
  4. Exportación de resultados y métricas.

- **`notebooks/model_testing.ipynb`**: Pruebas finales de clasificación y métricas en datos no vistos.

- **`saved_models_and_params/`**: Contiene el modelo VAE entrenado (`vae_model.pt`, `vae_state.pt`), el clasificador XGB (`xgb_clf.joblib`), y los archivos JSON con los mejores hiperparámetros y umbral de detección (`best_hps.json`, `threshold.json`).

- **Informes**:
  - `Tissera_Couriel_Informe.pdf`: Documento con la descripción metodológica, resultados y conclusiones.
  - `Poster.pdf`: Resumen gráfico del proyecto para presentación.

---

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu_usuario/ECG_anomaly_detection.git
   cd ECG_anomaly_detection
   ```
2. Configura un entorno virtual (recomendado) e instala dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate       # Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Datos

Los datos utilizados provienen de:

1. **PTB‑XL ECG Dataset, versión 1.0.3**
   - Patrick Wagner et al. (2020). *PTB-XL: A Large Publicly Available ECG Dataset*. *Scientific Data*. Disponible en PhysioNet: https://doi.org/10.13026/kfzx-aw45

2. **Chapman‑Shaoxing Database de ECG 12 derivados**
   - Jie Zheng et al. (2020). *A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients*. *Scientific Data*. Disponible en Kaggle: https://www.kaggle.com/datasets/erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database

---

