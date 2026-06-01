# Project Memory — TPI IA 2026 Otoño

Archivo de registro inter-sesión. Actualizar al inicio y fin de cada sesión de trabajo.

---

## Estado actual del proyecto

**Sprint activo:** Sprint 1 — Infraestructura + EDA + Baseline

**Archivos completados:**
- [x] `.gitignore`
- [x] `README.md`
- [x] `roadmap.md`
- [x] `architecture_decision.md`
- [x] `docs/data_flow.md`
- [x] `docs/evaluation_schema.md`
- [x] `docs/theoretical_notes.md`
- [x] `memory.md` (este archivo)
- [ ] `src/config.py`
- [ ] `src/eda/`
- [ ] `src/embeddings/`
- [ ] `src/indexing/`
- [ ] `src/agentic/`
- [ ] `src/reranking/`
- [ ] `src/evaluation/`
- [ ] `src/submission/`
- [ ] `scripts/`
- [ ] `notebook_compiler.py`

---

## Decisiones fijas (no reabrir)

| Decisión | Valor | Justificado en |
|----------|-------|---------------|
| Modelo de embeddings | CLIP ViT-B/32 | ADR-001 |
| Tipo de índice FAISS | IndexFlatIP + L2-norm | ADR-002 |
| Módulo de traducción | Helsinki-NLP/opus-mt-es-en | ADR-003 |
| LLM agéntico | Phi-3-mini-4k-instruct (4-bit) | ADR-004 |
| Fallback LLM | TinyLlama-1.1B-Chat | ADR-004 |
| Estrategia reranking | Pool N=50 + penalización CLIP | ADR-005 |
| Semilla aleatoria global | 42 | config.py |
| Métrica principal | MAP@10 | evaluation_schema.md |

---

## Parámetros a calibrar (decisiones pendientes)

| Parámetro | Rango a explorar | Sprint |
|-----------|-----------------|--------|
| λ (penalización reranker) | {0.2, 0.3, 0.5, 0.7, 1.0} | Sprint 3 |
| Pool size para reranking | {30, 50, 100} | Sprint 3 |
| Temperatura LLM variantes | {0.2, 0.3, 0.5} | Sprint 2 |
| Batch size CLIP | {32, 64, 128} (depende de GPU) | Sprint 1 |

---

## Contexto del entorno Kaggle

- Dataset input: `pascal-voc-2012-dataset`
- Path imágenes: `/kaggle/input/pascal-voc-2012-dataset/VOC2012_train_val/VOC2012_train_val/JPEGImages`
- Path anotaciones: `/kaggle/input/pascal-voc-2012-dataset/VOC2012_train_val/VOC2012_train_val/Annotations`
- Competencia oficial: https://www.kaggle.com/t/c08a7b4815a64b6f957dbd2b25bf0d8d
- Nombre notebook entregable: `TPI_IA_2026P_[nro_grupo].ipynb`
- Compartir con: usuario `martingra`
- Queries: q1–q20 (clases VOC), q21–q40 (custom, ground truth en solution.csv de cátedra)

---

## Log de sesiones

### Sesión 1 — 2026-05-29
- Completado: scaffolding completo + toda la documentación de planning + todos los módulos de código
- Bugs encontrados y corregidos post-escritura (análisis estático) — ver `docs/bug_fixes.md` para opciones evaluadas y decisión tomada:
  1. BUG-001 (crítico): reranker silencioso — fix: embeddings al constructor del `NegationReranker`
  2. BUG-002 (baja): acceso a privados desde ablation — fix: `@property clip/faiss` en `AgenticPipeline`
  3. BUG-003 (media): import fantasma `CUSTOM_QUERIES` en notebook compiler — fix: eliminado
- Skills instalados localmente vía `npx autoskills scan` → `.agents/skills/` (7 skills: scikit-learn, senior-data-scientist, python-executor, pandas-pro, pandas-data-analysis, machine-learning, python-testing-patterns)
- `.agents/` agregado a `.gitignore` — no se sube al repo
- Entorno virtual `.venv` creado con Python 3.12.10
- `pip install -r requirements.txt` ejecutado y verificado — todos los imports OK (torch 2.12.0, transformers 5.9.0, faiss-cpu 1.14.2, numpy 2.4.6, pandas 3.0.3, sklearn 1.8.0)
- Nota: el código NO fue ejecutado end-to-end (dependencias no instaladas localmente). Ejecutar en Kaggle es el primer test real.
- Sprint 1 COMPLETADO: 82 tests, 0 fallos, 2.58s
- Sprint 2 COMPLETADO: 123 tests totales, 0 fallos, 10.62s
  - Tests: test_llm_client, test_translator, test_expander, test_validator, test_pipeline (5 queries)
  - BUG-006: AgentResult faltaba has_negation (@property)
  - BUG-007: MagicMock en TraceStep no serializable → usar TraceStep real en mocks
  - `IndexType` strategy pattern en FAISSManager (FLAT_IP / IVF_FLAT)
  - `find_optimal_k()` con silhouette score en visualizer
  - 2 bugs más documentados (BUG-004 regex, BUG-005 datetime)
  - Skills aplicados: python-testing-patterns (AAA, fixtures, parametrize), scikit-learn (silhouette), senior-data-scientist (TDD)
- Próximo: Sprint 3 (reranking + MAP@10 + ablation) y Sprint 4 (entrega Kaggle) — a cargo del compañero

---

## Problemas conocidos / Cosas a tener en cuenta

- `bitsandbytes` no funciona bien en Windows; solo Kaggle (Linux) o WSL2
- Al guardar el índice FAISS, guardarlo junto con `image_ids.json` para mantener el mapeo
- El submission.csv usa `;` como separador, no coma — fácil de equivocarse
- Las queries q1–q20 son las 20 clases en inglés exactamente como aparecen en las anotaciones VOC (ej. "diningtable" no "dining table")
