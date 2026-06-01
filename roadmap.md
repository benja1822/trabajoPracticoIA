# Roadmap — TPI IA 2026 Otoño

## Visión general

El trabajo se estructura en **4 sprints** que construyen el sistema de forma incremental. Cada sprint entrega un subset funcional y testeable del pipeline; el siguiente sprint lo extiende sin romperlo.

```
Sprint 1 ──► Sprint 2 ──► Sprint 3 ──► Sprint 4
Baseline     Agente       Reranking    Entrega
(nivel base) (nivel int.) (nivel avz.) (integración)
```

**¿Por qué 4 sprints y no más?**  
Con deadline el 14/06 y complejidad creciente en cada nivel, 4 sprints de ~3 días de trabajo cada uno equilibran avance visible con margen para imprevistos. Cada sprint cierra con un componente funcional que puede mostrarse/defenderse por separado.

---

## Sprint 1 — Infraestructura + EDA + Baseline FAISS

**Objetivo:** sistema funcional de punta a punta al nivel base. Si algo falla aquí, nada más funciona.

**Duración estimada:** 2–3 sesiones de trabajo  
**Estado:** ✅ COMPLETADO (sesión 2026-05-29)

### Tareas

| # | Tarea | Archivo destino | Estado |
|---|-------|----------------|--------|
| 1.1 | `config.py`: paths, seeds, hiperparámetros | `src/config.py` | ✅ |
| 1.2 | EDA: stats de imágenes (dims, canales, clases) | `src/eda/dataset_stats.py` | ✅ |
| 1.3 | EDA: visualizaciones + silhouette score para K | `src/eda/visualizer.py` | ✅ |
| 1.4 | CLIP extractor con dataloader eficiente | `src/embeddings/clip_extractor.py` | ✅ |
| 1.5 | FAISS manager con IndexType strategy pattern | `src/indexing/faiss_manager.py` | ✅ |
| 1.6–1.8 | Scripts 01–03 | `scripts/` | ✅ |
| 1.9 | Test suite: 82 tests, 0 fallos | `tests/` | ✅ |
| 1.10 | `pytest.ini` con markers y configuración | raíz | ✅ |

### Bugs encontrados y corregidos en este sprint
- **negation_parser.py**: regex capturaba 2 palabras ("red on") → fix: captura 1 token (ver `docs/bug_fixes.md` BUG-004)
- **tracer.py**: `datetime.utcnow()` deprecated → fix: `datetime.now(timezone.utc)`

### Salida del sprint
- 82 tests corriendo en 2.58s, todos verdes
- `IndexType` enum (Strategy pattern) en FAISSManager
- `find_optimal_k()` con silhouette score en visualizer
- Suite de tests cubre: metrics, negation parser, tracer, CSV builder, FAISS manager

### Riesgos
- **Lentitud en CPU**: CLIP sobre ~17k imágenes tarda ~15 min en CPU. Solución: batch size alto + pin_memory.
- **Memoria RAM**: el índice FAISS flat de 17k × 512 pesa ~34 MB, no es problema.
- **Versión de transformers vs torch**: anclar versiones en requirements.txt desde el inicio.

---

## Sprint 2 — Pipeline Agéntico (Traducción + Expansión + Validación + Trazas)

**Objetivo:** integrar el LLM liviano para reformular consultas, detectar negaciones y validar integridad semántica. Todo observable via logs JSON.

**Dependencias:** Sprint 1 completo (necesita el índice FAISS para probar búsquedas reformuladas)

**Estado:** ✅ COMPLETADO (sesión 2026-05-29)

**Duración estimada:** 3 sesiones de trabajo

### Tareas

| # | Tarea | Archivo destino | Complejidad |
|---|-------|----------------|-------------|
| 2.1 | Módulo de traducción ES→EN (MarianMT `opus-mt-es-en`) | `src/agentic/translator.py` | ✅ |
| 2.2 | Parser de negaciones (regex + heurísticas lingüísticas) | `src/agentic/negation_parser.py` | ✅ |
| 2.3 | Expansor semántico via Phi-3-mini (sinónimos + normalización) | `src/agentic/expander.py` | ✅ |
| 2.4 | Validador de integridad (detecta contradicciones, queries vacías) | `src/agentic/validator.py` | ✅ |
| 2.5 | Tracer JSON estructurado | `src/agentic/tracer.py` | ✅ |
| 2.6 | Orquestador del pipeline agéntico | `src/agentic/pipeline.py` | ✅ |
| 2.7 | Tests de integración con 5 queries de ejemplo | `tests/test_pipeline.py` | ✅ |
| 2.8 | Tests unitarios: translator, llm_client, expander, validator | `tests/` | ✅ |

### Bugs encontrados y corregidos en este sprint
- **BUG-006**: `AgentResult` no exponía `has_negation` — fix: `@property has_negation`
- **BUG-007**: `MagicMock` en TraceStep causaba `TypeError` al serializar a JSON — fix: siempre usar `TraceStep` real en mocks de tests

### Salida del sprint
- 123 tests, 0 fallos, 10.62s
- Pipeline completo: detección de idioma → traducción → negación → expansión → validación → FAISS → reranking
- Fallback graceful: si el LLM falla, el pipeline sigue funcionando con la traducción
- Trazas JSON observables por cada paso con latencia

### Decisión de diseño clave: LLM en dos partes
Separamos **traducción** (MarianMT, modelo dedicado, determinista) de **expansión semántica** (Phi-3-mini, modelo generativo). Esto permite:
- Fallback: si Phi-3-mini falla o es muy lento, la traducción sigue funcionando
- Trazabilidad: sabemos exactamente qué cambio vino de cada componente
- Testabilidad: podemos testear traducción sin cargar el LLM pesado

### Riesgos
- **Tiempo de carga de Phi-3-mini en Kaggle**: primera carga ~2 min. Solución: cargar en la inicialización de la notebook, no en cada búsqueda.
- **Calidad del LLM para tareas específicas**: Phi-3-mini puede alucinar sinónimos irrelevantes. Solución: prompt engineering cuidadoso + validación de output.
- **Kaggle sin internet en runtime**: los modelos deben descargarse previamente y subirse como dataset auxiliar, o usar el dataset `huggingface-models` de Kaggle.

---

## Sprint 3 — Reranking + Evaluación + Ablation Study

**Objetivo:** penalizar resultados que contienen atributos excluyentes, medir MAP@10 de forma sistemática y comparar las tres configuraciones del sistema.

**Dependencias:** Sprint 2 completo (necesita el parser de negaciones para saber qué penalizar)

**Duración estimada:** 2–3 sesiones de trabajo

### Tareas

| # | Tarea | Archivo destino | Complejidad |
|---|-------|----------------|-------------|
| 3.1 | Algoritmo de reranking por penalización negativa | `src/reranking/negation_reranker.py` | Alta |
| 3.2 | Implementación de MAP@10 | `src/evaluation/metrics.py` | Media |
| 3.3 | Ground truth q1–q20: parser de anotaciones VOC XML | `src/evaluation/ground_truth.py` | Media |
| 3.4 | Ground truth q21–q40: estrategia de intersección o anotación | `src/evaluation/ground_truth.py` | Alta |
| 3.5 | Runner del ablation study (3 configuraciones × N queries) | `src/evaluation/ablation.py` | Media |
| 3.6 | Script `04_generate_submission.py` | `scripts/` | Baja |
| 3.7 | Visualizaciones comparativas (baseline vs reform. vs reranking) | integrado en notebook | Baja |

### Salida esperada del sprint
- MAP@10 calculado para las 3 configuraciones del sistema
- Tabla de ablation study con diferencias estadísticas
- submission.csv válido para Kaggle (formato exacto: q1–q40, 10 IDs por fila, separador `;`)
- Ejemplos visuales de búsquedas con y sin reranking

### Complejidad del reranking
El algoritmo funciona así:
1. Recuperar top-N (N=50) candidatos de FAISS con la query reformulada
2. Para cada atributo negado (ej. "rojo"), computar similitud CLIP de cada imagen con ese término
3. Score final = `sim_positiva - λ × sim_negativa`
4. Re-ordenar por score final y tomar top-10

El hiperparámetro `λ` controla qué tan agresiva es la penalización. Se calibra con las queries de evaluación.

---

## Sprint 4 — Integración, Notebook de Kaggle y Reporte

**Objetivo:** empaquetar todo en un entregable reproducible y escribir el reporte técnico.

**Dependencias:** Sprints 1–3 completos

**Duración estimada:** 2 sesiones de trabajo

### Tareas

| # | Tarea | Complejidad |
|---|-------|-------------|
| 4.1 | `notebook_compiler.py`: genera notebook Kaggle desde `src/` | Alta |
| 4.2 | Test de reproducibilidad end-to-end en Kaggle (fork + re-run) | Media |
| 4.3 | Reporte técnico en Markdown (secciones según spec PDF) | Media |
| 4.4 | Validación del submission.csv (formato, 10 IDs, sin extensiones) | Baja |
| 4.5 | Compartir notebook con usuario `martingra` en Kaggle | Baja |

### Checklist final antes de entrega

- [ ] Notebook se ejecuta sin errores en Kaggle (botón "Run All")
- [ ] submission.csv tiene exactamente 40 filas (q1–q40) y 10 IDs cada una
- [ ] IDs sin extensión `.jpg` y sin duplicados por fila
- [ ] Semilla aleatoria fijada → resultados reproducibles al 100%
- [ ] Reporte .md con todas las secciones requeridas
- [ ] Notebook compartida con `martingra`
- [ ] Trazas JSON visibles en la notebook para al menos 3 queries de ejemplo

---

## Cronograma tentativo

```
Semana 1 (hasta ~01/06)   → Sprint 1 completo
Semana 2 (hasta ~07/06)   → Sprint 2 completo
Semana 3 (hasta ~11/06)   → Sprint 3 completo
Días 12–14/06             → Sprint 4 + validación final
```

---

## Consideraciones teórico-prácticas que deben quedar claras

1. **CLIP no entiende negaciones**: el espacio de embeddings de CLIP no codifica ausencia de atributos. "auto no rojo" y "auto rojo" tienen embeddings muy similares. Por eso el reranking es necesario y no es trivial.

2. **Reproducibilidad vs aleatoriedad**: K-Means, t-SNE, UMAP y el sampling del LLM son estocásticos. Toda semilla debe estar en `config.py` y aplicarse antes de cada operación.

3. **MAP@10 requiere ground truth binario**: cada imagen es relevante (1) o no (0). Para las 20 clases VOC esto está dado; para queries complejas hay que diseñar el esquema con cuidado.

4. **Kaggle tiene límite de 9h de ejecución por session**: la extracción de embeddings con CLIP en ~17k imágenes tarda ~8 min en GPU T4. Guardar el resultado es crítico para no repetirlo.

5. **Índice FAISS y normalización**: usamos `IndexFlatIP` con vectores L2-normalizados. El producto interno de vectores unitarios = cosine similarity. Esto hay que documentarlo en el reporte.

6. **El agente no es un único LLM monolítico**: cada responsabilidad (traducción, expansión, validación) tiene su propio componente. Esto es lo que la cátedra evalúa como "separación clara de responsabilidades".
