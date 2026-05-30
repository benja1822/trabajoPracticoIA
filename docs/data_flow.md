# Flujo de Datos del Pipeline

## Diagrama de alto nivel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FASE OFFLINE (una sola vez)                        │
│                                                                             │
│  Pascal VOC 2012                                                            │
│  JPEGImages/ ──► [CLIP ViT-B/32] ──► embeddings float32[N×512]            │
│                       batch=64              ──► [L2 normalize]              │
│                                                      ──► [FAISS IndexFlatIP]│
│                                                              ──► faiss.index│
│                                                                             │
│  VOC Annotations/                                                           │
│  (XML) ──► [ground_truth.py] ──► gt_dict {class: [img_ids]}                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ faiss.index + gt_dict cargados en memoria
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FASE ONLINE (por cada query)                       │
│                                                                             │
│  Input: "auto no rojo en la calle"  (español)                              │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    PIPELINE AGÉNTICO                                  │  │
│  │                                                                       │  │
│  │  1. LANGUAGE DETECTOR                                                 │  │
│  │     input:  raw query string                                          │  │
│  │     output: lang_code ("es"), confidence float                        │  │
│  │     tool:   langdetect                                                │  │
│  │                    │                                                  │  │
│  │                    ▼                                                  │  │
│  │  2. TRANSLATOR (solo si lang != "en")                                │  │
│  │     input:  query en español                                          │  │
│  │     output: query en inglés: "car not red on the street"             │  │
│  │     model:  Helsinki-NLP/opus-mt-es-en                               │  │
│  │                    │                                                  │  │
│  │                    ▼                                                  │  │
│  │  3. NEGATION PARSER                                                   │  │
│  │     input:  translated query                                          │  │
│  │     output: {                                                         │  │
│  │               has_negation: true,                                     │  │
│  │               positive_query: "car on the street",                   │  │
│  │               negative_attributes: ["red"]                           │  │
│  │             }                                                         │  │
│  │     tool:   regex patterns + spaCy dependency parsing                │  │
│  │                    │                                                  │  │
│  │                    ▼                                                  │  │
│  │  4. SEMANTIC EXPANDER                                                 │  │
│  │     input:  positive_query                                            │  │
│  │     output: {                                                         │  │
│  │               expanded_query: "car automobile vehicle street road",  │  │
│  │               variants: ["vehicle on the road", "car street scene"]  │  │
│  │             }                                                         │  │
│  │     model:  Phi-3-mini-4k-instruct (T=0.3)                           │  │
│  │                    │                                                  │  │
│  │                    ▼                                                  │  │
│  │  5. INTEGRITY VALIDATOR                                               │  │
│  │     input:  expanded_query + negative_attributes                      │  │
│  │     output: {valid: bool, reason: str, corrected_query: str|null}    │  │
│  │     model:  Phi-3-mini-4k-instruct (T=0)                             │  │
│  │     checks: ¿contradice la query positiva y negativa?                │  │
│  │             ¿la query tiene sentido semántico?                        │  │
│  │             ¿la expansión introdujo términos incoherentes?            │  │
│  │                    │                                                  │  │
│  │                    ▼                                                  │  │
│  │  6. TRACER                                                            │  │
│  │     Registra todos los pasos anteriores en un objeto JSON             │  │
│  │     output: trace_dict (serializable)                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                    │                                                        │
│                    ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    RETRIEVAL (FAISS)                                  │  │
│  │                                                                       │  │
│  │  query_text = expanded_query (o positive_query si expansión falla)  │  │
│  │                    │                                                  │  │
│  │                    ▼                                                  │  │
│  │  [CLIP text encoder]                                                  │  │
│  │     input:  query_text (string)                                       │  │
│  │     output: query_embedding float32[512] (L2-normalized)             │  │
│  │                    │                                                  │  │
│  │                    ▼                                                  │  │
│  │  [FAISS IndexFlatIP.search(query_embedding, k=50)]                   │  │
│  │     output: candidates = [(img_id, score), ...] top-50               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                    │                                                        │
│                    ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    RERANKING (solo si has_negation=true)             │  │
│  │                                                                       │  │
│  │  Para cada negative_attribute (ej. "red"):                           │  │
│  │    neg_embedding = CLIP.encode_text("red")                           │  │
│  │    neg_score[img] = cosine_sim(img_embedding, neg_embedding)         │  │
│  │                                                                       │  │
│  │  final_score[img] = pos_score[img] - λ × max(neg_scores[img])       │  │
│  │  λ = 0.5 (hiperparámetro, ver config.py)                             │  │
│  │                                                                       │  │
│  │  output: candidates re-ordenados por final_score                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                    │                                                        │
│                    ▼                                                        │
│  Output: top-10 image IDs ["2007_000032", "2008_004552", ...]              │
│          + trace_dict (JSON log completo del pipeline)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Especificación de tipos por componente

### Entradas y salidas del sistema

| Componente | Input type | Output type | Side effects |
|-----------|------------|-------------|-------------|
| `CLIPExtractor.encode_images()` | `List[Path]` | `np.ndarray[N, 512]` | Guarda .npy en disco |
| `FAISSManager.build()` | `np.ndarray[N, 512]`, `List[str]` | `None` | Guarda .faiss en disco |
| `FAISSManager.search()` | `np.ndarray[512]`, `k: int` | `List[Tuple[str, float]]` | Ninguno |
| `AgenticPipeline.process()` | `str` (query en español) | `AgentResult` dataclass | Guarda trace JSON |
| `NegationReranker.rerank()` | `candidates: List`, `neg_attrs: List[str]` | `List[Tuple[str, float]]` | Ninguno |
| `EvaluationRunner.map_at_k()` | `results: Dict`, `gt: Dict` | `float` | Ninguno |

### Dataclass `AgentResult`

```python
@dataclass
class AgentResult:
    original_query: str          # query en español, sin modificar
    translated_query: str        # traducción al inglés
    positive_query: str          # query sin negaciones
    negative_attributes: list    # atributos a penalizar, puede ser []
    expanded_query: str          # query expandida con sinónimos
    is_valid: bool               # pasó la validación de integridad
    validation_reason: str       # por qué es válida/inválida
    trace: dict                  # log JSON completo del pipeline
```

---

## Flujo de datos en disco

```
data/
├── embeddings/
│   ├── image_embeddings.npy      ← float32[N, 512], L2-normalizado
│   └── image_ids.json            ← list[str], mismo orden que embeddings
├── index/
│   └── faiss_flat_ip.index       ← índice FAISS serializado
├── ground_truth/
│   ├── voc_gt.json               ← {class_name: [img_ids]} para q1-q20
│   └── complex_gt.json           ← {query_id: [img_ids]} para q21-q40
└── traces/
    └── pipeline_traces.jsonl     ← una línea JSON por query procesada
```

---

## Garantías del sistema

1. **Idempotencia**: dado el mismo estado del índice y la misma query, el pipeline produce exactamente los mismos resultados (temperatura=0 en el LLM para todas las tareas excepto expansión, donde temperatura=0.3 pero con semilla fija).

2. **Graceful degradation**: si Phi-3-mini no está disponible, el pipeline opera sin expansión semántica (solo traducción + FAISS). Si MarianMT falla, el pipeline opera con la query original en español (CLIP maneja algo de español).

3. **Separación de fases offline/online**: los embeddings y el índice FAISS se calculan una sola vez. La fase online solo realiza encode del texto de la query (muy rápido, <20ms).
