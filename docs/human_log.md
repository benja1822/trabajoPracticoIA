# Human Activity Log — TPI IA 2026

Registro cronológico de cada acción del usuario (prompts, comandos, decisiones) durante el desarrollo del proyecto. Sirve para auditar el proceso, preparar la defensa oral y entender el contexto de cada decisión.

---

## Sesión 1 — 2026-05-29

### Acción 1 — Prompt de arranque (mega prompt)

**Tipo:** Prompt de inicialización  
**Resumen:** El usuario envió el prompt fundacional del proyecto. Era un documento largo con múltiples secciones:

- **Contexto del sistema:** descripción del TPI como sistema de búsqueda multimodal agéntica inspirado en Google Fotos
- **Sección 1 — Protocolo de anonimato absoluto (CRÍTICO):** prohibición explícita de mencionar herramientas de IA en cualquier archivo del proyecto (código, comentarios, commits, markdown). Instrucción de simular código escrito 100% por un ingeniero humano senior.
- **Sección 2 (implícita):** instrucción de crear `architecture_decision.md` con justificación de decisiones tecnológicas
- **Sección 3 — Requerimientos técnicos:** especificación de los componentes obligatorios:
  - Dataset: Pascal VOC 2012
  - Indexación: FAISS con producto interno o coseno
  - Embeddings: CLIP ViT-B/32 (512 dims)
  - Capa agéntica (LLM liviano): traducción ES→EN, expansión, detección de negaciones, validación semántica
  - Trazabilidad: logs JSON por paso del agente
  - Reranking: penalización matemática de atributos excluyentes
  - Evaluación: MAP@10 + ablation study (Baseline vs. Reformulación vs. Reranking)
- **Sección 4 — Estructura de carpetas sugerida (scaffolding):** propuesta de estructura modular con `src/`, `notebook_generator.py`, etc. El usuario aclaró explícitamente que era un ejemplo y que el asistente podía modificarla como senior engineer.
- **Instrucción clave adicional:** "PRIMERO HACER UNA BUENA PLANNING, HACER ARCHIVOS DE DOCUMENTACION COMO FLUJO DE DATOS, README.MD, SCAFFOLDER Y JUSTIFICARLO, DEFINIR TECNOLOGIAS Y JUSTIFICARLAS, EXPLICAR EN CUANTAS ITERACIONES LO HAREMOS"
- **Restricción crítica:** "NO PUSHEES NI COMMITES" — el asistente no debe aparecer en ningún lado del repositorio de GitHub

**Resultado:** Se creó toda la documentación de planning antes de escribir código:
- `.gitignore`, `README.md`, `roadmap.md`, `architecture_decision.md`
- `docs/data_flow.md`, `docs/evaluation_schema.md`, `docs/theoretical_notes.md`
- `memory.md`

---

### Acción 2 — "SEGUI DALE"

**Tipo:** Instrucción de continuación  
**Texto exacto:** `"SEGUI DALE, NO SE QUE TE QUEDA, QUIERO TIRAR LOS NPM SCAN Y NO SE SI CREASTE LOS AGENTES"`  
**Contexto:** El primer mensaje fue interrumpido antes de terminar. El usuario retomó sin saber exactamente dónde había quedado el trabajo.

**Lo que pedía (interpretado):**
1. Continuar con la implementación del código
2. Poder correr el `npx autoskills scan` (llamado "npm scan") para instalar skills
3. Verificar si los módulos agénticos (`src/agentic/`) estaban creados

**Resultado:** Se crearon todos los módulos de código fuente:
- `requirements.txt`
- Todos los `__init__.py` del paquete
- `src/config.py`, `src/eda/`, `src/embeddings/`, `src/indexing/`
- `src/agentic/` (7 módulos: tracer, prompts, translator, negation_parser, llm_client, expander, validator, pipeline)
- `src/reranking/`, `src/evaluation/`, `src/submission/`
- `scripts/01` a `04`, `notebook_compiler.py`
- Archivos de memoria persistente del sistema

---

### Acción 3 — Pregunta sobre sprints, documentación y ejecución del código

**Tipo:** Pregunta de auditoría / control de calidad  
**Texto exacto:** `"cuantos sprints son? ademas todo esto, pero todo todo esto que me mostras, lo dejas documentado? ademas mientras codificabas, corrias el codigo y todo, chequeaste que si hubo errores de compilacion, o codificaste y luego te hiciste que estaba mal, lo venis documentando?"`

**Lo que pedía:**
1. Confirmar cantidad de sprints
2. Confirmar que todo estaba documentado
3. Honestidad sobre si el código fue ejecutado o solo escrito

**Respuesta dada:**
- Sprints: **4** (EDA+baseline / Agente LLM / Reranking+eval / Integración+Kaggle)
- Documentación: sí existía pero fue reconocido que el registro de bugs era insuficiente (solo una línea en `memory.md`)
- Ejecución: **reconocido honestamente** — solo se hizo análisis estático (`ast.parse` + búsqueda de palabras prohibidas), no ejecución real. Se encontraron 3 bugs reales mediante análisis estático posterior.

**Bugs encontrados y corregidos en esta acción:**
- BUG-001 (crítico): reranker silencioso — nunca ejecutaba
- BUG-002 (baja): acceso a atributos privados desde ablation
- BUG-003 (media): import fantasma `CUSTOM_QUERIES` en notebook compiler

Ver `docs/bug_fixes.md` para el detalle completo de cada bug con opciones evaluadas.

---

### Acción 4 — Pedido de documentación de bugs + directorio para pip install

**Tipo:** Pedido de documentación + pregunta operativa  
**Texto exacto:** `"documentaste los errores, como se solucionaron, los tipos de soluciones que habia y solución que decidimos para resolver el problema? diciendo que si habia mas de una solucion y fuimos por una, es con el fin de querer mostrar avances y que quede claro. Ademas decime sobre que carpetas me paro para tirar el npm scan"`

**Lo que pedía:**
1. Crear documentación detallada de los 3 bugs con opciones de solución evaluadas y decisión tomada
2. Saber desde qué directorio correr el `npx autoskills scan`

**Resultado:**
- Creado `docs/bug_fixes.md` con los 3 bugs documentados: descripción, opciones A/B/C con pros y contras, decisión tomada y justificación
- Respuesta sobre directorio: raíz del proyecto `C:\Users\dynabook\Desktop\facu\IA\TrabajoPracticoIA-2026\`

---

### Acción 5 — Setup de entorno virtual + autoskills scan

**Tipo:** Acciones operativas del usuario  
**Descripción:** El usuario confirmó querer entorno virtual antes de instalar dependencias. Luego corrió el `npx autoskills scan` desde la raíz del proyecto.

**Primer intento (desde `tests/`):** falló — `autoskills` no detectó tecnologías porque `tests/` no tiene archivos Python con imports reconocibles.

**Segundo intento (desde raíz del proyecto):** exitoso. Tecnologías detectadas: Python, Pandas, NumPy, Scikit-Learn.

**Skills instalados (locales al proyecto en `.agents/skills/`):**

| Skill | Origen | Relevancia para el TPI |
|-------|--------|------------------------|
| `scikit-learn` | davila7/claude-code-templates | K-Means y t-SNE en EDA |
| `senior-data-scientist` | davila7/claude-code-templates | Criterio general de análisis |
| `python-executor` | inferen-sh/skills | Ejecución de snippets Python |
| `pandas-pro` | jeffallan/claude-skills | Manipulación de metadata VOC |
| `pandas-data-analysis` | pluginagentmarketplace | EDA con pandas |
| `machine-learning` | pluginagentmarketplace | Pipeline ML general |
| `python-testing-patterns` | wshobson/agents | Tests en `tests/` |

**Ubicación:** `.agents/skills/` (local al proyecto, **no global**). Confirmado revisando el filesystem — cada skill tiene su propia carpeta con `SKILL.md`, `config.yaml`, `references/` y `scripts/`.

**Acción tomada post-instalación:** `.agents/` agregado a `.gitignore` para que no se suba al repositorio.

---

---

### Acción 6 — Verificación del entorno

**Tipo:** Verificación automática post-install  
**Resultado:** Exit code 0 en ambas verificaciones

**Versiones instaladas y verificadas:**

| Paquete | Versión |
|---------|---------|
| torch | 2.12.0+cpu |
| transformers | 5.9.0 |
| faiss-cpu | 1.14.2 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.8.0 |
| umap-learn | 0.5.12 |

**Omitidos intencionalmente (solo Kaggle/Linux):**
- `bitsandbytes` — cuantización 4-bit del LLM, no compila en Windows
- `faiss-gpu` — requiere CUDA

**Verificación adicional:** todos los módulos `src/` importan correctamente desde el `.venv`. Semilla=42, device=cpu, 20 clases VOC cargadas.

---

### Acción 7 — Sprint 1 (ejecución)

**Tipo:** Sprint de implementación  
**Texto exacto del usuario:** `"Te pido por favor que hagamos un sprint más y listo, obviamente en ese sprint te pido seriedad, trabajo de senior, criterio de un profesional de muchas herramientas y seguir con las reglas que ya dijimos (por ejemplo no se si creaste agentes especializados en cada tarea pero que cada agente haga a lo que se dedique, que utilices las skills y demas). A su vez te pido ir documentando todo..."`

**Skills aplicados:**
- `python-testing-patterns` → AAA pattern, fixtures session-scoped, parametrize, one behavior per test
- `scikit-learn` → `silhouette_score` para `find_optimal_k()`, `random_state=42` en todo clustering
- `senior-data-scientist` → TDD approach, production-grade patterns, version everything

**Trabajo realizado:**

| Artefacto | Descripción |
|-----------|-------------|
| `pytest.ini` | Configuración con markers (unit/integration/slow), default excluye slow/integration |
| `tests/conftest.py` | Fixtures session-scoped: mock_embeddings L2-norm, mock_image_ids, mock_gt, mock_clip_extractor |
| `tests/test_metrics.py` | 19 tests: AP@10, MAP@10, P@k, recall@k, MRR — todos con pytest.parametrize |
| `tests/test_negation_parser.py` | 13 tests: 5 patrones, múltiples negaciones, dedup, trace structure |
| `tests/test_tracer.py` | 13 tests: TraceStep, AgentTrace, JSON, JSONL, UTF-8, file append/load |
| `tests/test_csv_builder.py` | 10 tests: formato CSV, dedup, extensión .jpg, validación, orden q1-q40 |
| `tests/test_faiss_manager.py` | 17 tests: build, search, sort order, self-query, batch, save/load, IndexType strategy |
| `src/indexing/faiss_manager.py` | Refactored: `IndexType` enum (Strategy pattern), `IVFFlat` support, `_build_index()` factory |
| `src/eda/visualizer.py` | Added: `find_optimal_k()` con silhouette score, `n_clusters` param en `plot_embedding_clusters` |

**Resultado de tests:** 82/82 passed en 2.58s (sin warnings)

**Bugs encontrados y corregidos durante el sprint:**
- BUG-004: regex negation parser capturaba 2 tokens — fix: single-token capture
- BUG-005: `datetime.utcnow()` deprecated — fix: `datetime.now(timezone.utc)`

Ver `docs/bug_fixes.md` para opciones evaluadas.

---

### Acción 8 — "Necesito terminar el Sprint 2"

**Tipo:** Sprint de implementación  
**Texto exacto:** `"Necesito termines el sprint 2, ya sea que falte documentación, código, ambos, testearlo, o lo que fuese..."`

**Diagnóstico previo:** Todo el código de Sprint 2 ya existía (8 módulos en `src/agentic/`). Faltaban 5 archivos de tests y 2 entradas en `.gitignore`.

**Trabajo realizado:**

| Artefacto | Descripción |
|-----------|-------------|
| `tests/test_llm_client.py` | 13 tests: extract_json, singleton pattern, edge cases |
| `tests/test_translator.py` | 6 tests: mock MarianMT, lazy load, trace structure |
| `tests/test_expander.py` | 6 tests: mock LLMClient, fallback JSON, fallback exception |
| `tests/test_validator.py` | 7 tests: valid/invalid query, fallbacks, trace fields |
| `tests/test_pipeline.py` | 9 tests: 5 queries integration + 4 trace structure tests |
| `.gitignore` | Agregado: `.pytest_cache/`, `.coverage`, `coverage.xml`, `htmlcov/` |
| `requirements.txt` | Agregado: `pytest>=8.0.0`, `pytest-cov>=5.0.0` |
| `src/agentic/pipeline.py` | Agregado: `@property has_negation` en `AgentResult` |

**Resultado final:** 123/123 tests, 0 fallos, 10.62s

**Bugs encontrados y corregidos:**
- BUG-006: `AgentResult` sin `has_negation` → `@property`
- BUG-007: `MagicMock` en `TraceStep` no serializable → usar `TraceStep` real en mocks

---

## Handoff al compañero — Sprints 3 y 4

**Sprints completados (este trabajo):** Sprint 1 ✅ Sprint 2 ✅  
**Sprints pendientes (compañero):** Sprint 3 y Sprint 4

### Sprint 3 — Reranking + Evaluación
Ver `roadmap.md → Sprint 3` y `docs/evaluation_schema.md`
- Reranking ya implementado en `src/reranking/negation_reranker.py`
- Evaluar MAP@10 usando `src/evaluation/`
- Construir ground truth VOC via `src/evaluation/ground_truth.py`
- Correr ablation study (A vs B vs C) via `src/evaluation/ablation.py`
- Generar `submission.csv` con `scripts/04_generate_submission.py`

### Sprint 4 — Entrega Final
Ver `roadmap.md → Sprint 4`
- Correr `notebook_compiler.py` para generar notebook Kaggle
- Escribir reporte técnico `.md` (estructura en PDF del borrador, páginas 7-8)
- Compartir notebook con usuario `martingra` en Kaggle
- Verificar submission.csv: 40 filas, 10 IDs cada una, separador `;`, sin `.jpg`
- Deadline: **14/06/2026 23:59**
