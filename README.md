# TPI Inteligencia Artificial 2026 — Sistema de Búsqueda Multimodal Agéntica

**Materia:** Inteligencia Artificial — Cátedra IA 2026  
**Cuatrimestre:** Otoño 2026  
**Deadline entrega:** 14/06/2026 a las 23:59 hs  
**Kaggle competition:** [enlace oficial](https://www.kaggle.com/t/c08a7b4815a64b6f957dbd2b25bf0d8d)

---

## Descripción del sistema

Sistema de recuperación de imágenes por texto (text-to-image retrieval) sobre Pascal VOC 2012, con tres niveles de sofisticación acumulativos:

| Nivel | Componentes | Target |
|-------|-------------|--------|
| Base | EDA + CLIP embeddings + FAISS baseline | Aprobado |
| Intermedio | + LLM reformulation + integrity validation | Bueno |
| Avanzado | + reranking de negaciones + ablation study | Sobresaliente |

El pipeline transforma una consulta en español en un ranking de imágenes relevantes, con trazabilidad completa de cada decisión tomada por el sistema.

---

## Arquitectura en una línea

```
Consulta (ES) → [LLM Agéntico] → Consulta reformulada (EN) → FAISS → Pool candidatos → [Reranker] → Top-10 resultados
                     ↓ trazas JSON
```

Ver [`docs/data_flow.md`](docs/data_flow.md) para el diagrama completo.

---

## Estructura del proyecto

```
TrabajoPracticoIA-2026/
├── .gitignore
├── README.md                    ← este archivo
├── roadmap.md                   ← plan de iteraciones y cronograma
├── architecture_decision.md     ← decisiones tecnológicas justificadas
├── memory.md                    ← log de sesiones y decisiones inter-sesión
│
├── docs/
│   ├── data_flow.md             ← diagrama de flujo de datos detallado
│   ├── evaluation_schema.md     ← construcción de ground truth y MAP@10
│   └── theoretical_notes.md    ← fundamentos teóricos (CLIP, FAISS, reranking)
│
├── src/                         ← paquete Python modular
│   ├── __init__.py
│   ├── config.py                ← configuración global (paths, seeds, hiperparámetros)
│   ├── eda/                     ← análisis exploratorio del dataset
│   ├── embeddings/              ← extracción de embeddings con CLIP ViT-B/32
│   ├── indexing/                ← gestión del índice FAISS
│   ├── agentic/                 ← pipeline agéntico (traducción, expansión, validación, trazas)
│   ├── reranking/               ← penalización de atributos excluyentes
│   ├── evaluation/              ← MAP@10 y ablation study
│   └── submission/              ← generación de submission.csv para Kaggle
│
├── scripts/                     ← scripts de ejecución standalone
│   ├── 01_build_embeddings.py
│   ├── 02_build_index.py
│   ├── 03_run_search.py
│   └── 04_generate_submission.py
│
├── tests/                       ← unit tests de módulos críticos
│
└── notebook_compiler.py         ← compila src/ en la notebook final de Kaggle
```

---

## Setup local

### Requisitos

```bash
python >= 3.10
```

### Instalación

```bash
# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### Dataset

El dataset Pascal VOC 2012 se monta directamente en Kaggle. Para desarrollo local:

```
# Descargarlo manualmente desde Kaggle y ubicarlo en:
data/pascal-voc-2012-dataset/VOC2012_train_val/VOC2012_train_val/JPEGImages/
```

Ajustar `src/config.py` → `DATASET_ROOT` si el path difiere.

---

## Flujo de ejecución local

```bash
# 1. Generar embeddings de todas las imágenes (una sola vez, ~15 min en CPU)
python scripts/01_build_embeddings.py

# 2. Construir el índice FAISS
python scripts/02_build_index.py

# 3. Probar búsquedas
python scripts/03_run_search.py --query "perro en el jardín"

# 4. Generar submission.csv
python scripts/04_generate_submission.py
```

---

## Despliegue en Kaggle

Ver [`notebook_compiler.py`](notebook_compiler.py) para generar el archivo de notebook.  
La notebook final debe nombrarse: `TPI_IA_2026P_[nro_grupo].ipynb`

Pasos:
1. Subir `src/` como dataset auxiliar en Kaggle (o usar el compiler para inline)
2. Montar `pascal-voc-2012-dataset` como dataset de input
3. Ejecutar la notebook completa → genera `submission.csv`
4. Compartir la notebook con el usuario `martingra`

---

## Métricas objetivo

| Métrica | Descripción |
|---------|-------------|
| MAP@10 | Mean Average Precision en top-10 resultados |
| Δ MAP (baseline → reformulación) | Ganancia del LLM agéntico |
| Δ MAP (reformulación → +reranking) | Ganancia del reranker de negaciones |

---

## Dependencias principales

| Librería | Versión mínima | Uso |
|----------|---------------|-----|
| torch | 2.1.0 | Backend para CLIP y LLM |
| transformers | 4.40.0 | CLIP, MarianMT, Phi-3-mini |
| faiss-cpu / faiss-gpu | 1.7.4 | Índice vectorial |
| Pillow | 10.0.0 | Carga de imágenes |
| numpy | 1.26.0 | Operaciones vectoriales |
| pandas | 2.1.0 | Manipulación de metadata |
| matplotlib / seaborn | - | Visualizaciones EDA |
| scikit-learn | 1.3.0 | K-Means, t-SNE |
| umap-learn | 0.5.5 | Reducción de dimensionalidad |
| lxml | 4.9.0 | Parsing de anotaciones VOC XML |
| bitsandbytes | 0.43.0 | Cuantización 4-bit del LLM (GPU) |
