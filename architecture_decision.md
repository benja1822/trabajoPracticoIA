# Architecture Decision Records — TPI IA 2026

Este documento registra las decisiones tecnológicas y arquitectónicas tomadas para el sistema, con su justificación, alternativas consideradas y consecuencias.

---

## ADR-001: Modelo de embeddings — CLIP ViT-B/32

**Estado:** Aceptado (requerido por especificación)

**Contexto**  
Necesitamos un modelo que mapee imágenes y texto al mismo espacio semántico para poder comparar queries textuales con imágenes directamente.

**Decisión**  
Usar `openai/clip-vit-base-patch32` de la librería `transformers` (o `open_clip_torch`).

**Alternativas consideradas**

| Modelo | Dims | Calidad retrieval | Velocidad | Motivo de descarte |
|--------|------|------------------|-----------|-------------------|
| CLIP ViT-B/32 | 512 | Buena | Rápida | **Elegido** |
| CLIP ViT-L/14 | 768 | Mejor | 3× más lento | Demasiado lento para ~17k imgs en Kaggle |
| ALIGN | 640 | Comparable | Más lento | Menos soporte en transformers |
| SigLIP | 512–1024 | Mejor en español | Requiere fine-tune | Scope fuera del TP |

**Consecuencias**
- (+) Balance ideal velocidad/calidad para ~17k imágenes de VOC
- (+) Disponible directamente en HuggingFace Hub sin configuración extra
- (+) Soporta Kaggle T4/P100 sin cuantización
- (-) ViT-B/32 usa patches de 32×32px, lo que limita la captura de detalles finos

---

## ADR-002: Índice vectorial — FAISS IndexFlatIP

**Estado:** Aceptado

**Contexto**  
Con ~17.112 imágenes y vectores de 512 dimensiones, necesitamos una estructura de búsqueda eficiente. La decisión es qué tipo de índice FAISS usar.

**Decisión**  
Usar `faiss.IndexFlatIP` (producto interno exacto) con vectores L2-normalizados.

**Por qué producto interno y no L2**  
CLIP produce embeddings que están conceptualmente en una hiperesfera unitaria. La similitud coseno (= producto interno de vectores unitarios) captura la orientación del vector, que es lo que importa semánticamente. La distancia L2 entre vectores unitarios mezcla magnitud y ángulo.

```
cosine_similarity(a, b) = a·b / (||a|| × ||b||)
# Si ||a|| = ||b|| = 1  →  cosine_similarity(a, b) = a·b
```

Normalizamos todos los vectores a norma 1 antes de indexar, y luego usamos `IndexFlatIP`.

**Alternativas consideradas**

| Tipo de índice | Búsqueda | Exactitud | RAM extra | Decisión |
|---------------|---------|----------|-----------|---------|
| `IndexFlatIP` | ~3ms/query | 100% exacta | 0 | **Elegido** |
| `IndexIVFFlat` (nlist=100) | ~0.5ms/query | ~95% | +overhead entrenamiento | Innecesario para 17k imgs |
| `IndexHNSWFlat` | ~0.3ms/query | ~98% | +índice grafo | Más complejidad sin beneficio real |
| `IndexPQ` | <0.1ms/query | ~90% | -compresión | Pérdida de calidad inaceptable para TP |

**Consecuencias**
- (+) Resultados 100% reproducibles (búsqueda exacta)
- (+) Índice de ~34 MB (17k × 512 × 4 bytes), cómodo en RAM y Kaggle
- (+) Implementación simple y fácil de defender en la oral
- (-) No escala bien a millones de vectores, pero no es el caso

**Patrón aplicado: Strategy (IndexType enum)**

En Sprint 1 se introdujo `IndexType` como enum que actúa de selector de estrategia:

```python
class IndexType(Enum):
    FLAT_IP  = "flat_ip"   # exacto, default
    IVF_FLAT = "ivf_flat"  # aproximado, para escala

mgr = FAISSManager(index_type=IndexType.FLAT_IP)   # default
mgr = FAISSManager(index_type=IndexType.IVF_FLAT)  # alternativa
```

La interfaz pública (`add`, `search`, `save`, `load`) es idéntica en ambos casos. El método interno `_build_index()` actúa de factory. Esto permite cambiar el tipo de índice sin modificar el pipeline ni los scripts.

---

## ADR-003: Traducción ES→EN — MarianMT opus-mt-es-en

**Estado:** Aceptado

**Contexto**  
CLIP fue entrenado principalmente en inglés. Las queries del usuario vienen en español. Necesitamos un traductor confiable, rápido y ejecutable offline en Kaggle.

**Decisión**  
Usar `Helsinki-NLP/opus-mt-es-en` (MarianMT, ~300 MB, seq2seq dedicado a traducción).

**Alternativas consideradas**

| Opción | Calidad | Tamaño | Online req. | Decisión |
|--------|---------|--------|-------------|---------|
| MarianMT opus-mt-es-en | Alta | 300 MB | No | **Elegido** |
| Phi-3-mini para traducción | Alta | 2.5 GB cargado | No | Malgastar LLM grande en tarea simple |
| Google Translate API | Muy alta | 0 (API) | Sí | Kaggle no garantiza internet en runtime |
| NLLB-200-distilled-600M | Muy alta | 600 MB | No | Overhead innecesario para ES→EN |

**Por qué no usar directamente Phi-3-mini para traducir**  
MarianMT es un modelo seq2seq entrenado específicamente para traducción ES→EN. Produce salidas más deterministas y rápidas que un LLM generativo. Reservamos Phi-3-mini para tareas donde realmente se necesita razonamiento (expansión semántica, validación de integridad). Separar estas responsabilidades hace el sistema más robusto y testeable.

**Consecuencias**
- (+) Traducción rápida (~50ms por query), determinista, sin aleatoriedad
- (+) Disponible en HuggingFace Hub, fácil de descargar como modelo auxiliar en Kaggle
- (-) Puede cometer errores en vocabulario de dominio muy específico (nombres propios, jerga)

---

## ADR-004: LLM agéntico — Phi-3-mini-4k-instruct (4-bit quantized)

**Estado:** Aceptado (con fallback)

**Contexto**  
Necesitamos un LLM liviano capaz de: expandir sinónimos, normalizar la semántica de la query, detectar incoherencias y generar variantes. Debe correr en Kaggle (T4/P100, ~16GB VRAM).

**Decisión**  
`microsoft/Phi-3-mini-4k-instruct` con cuantización 4-bit via `bitsandbytes` (~2.5 GB VRAM).

**Fallback si Phi-3-mini no está disponible en Kaggle**  
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` (sin cuantización, ~2.2 GB VRAM).

**Alternativas consideradas**

| Modelo | Params | VRAM (4-bit) | Instruction following | Decisión |
|--------|--------|-------------|----------------------|---------|
| Phi-3-mini-4k-instruct | 3.8B | ~2.5 GB | Excelente | **Elegido** |
| TinyLlama-1.1B-Chat | 1.1B | ~1.1 GB | Aceptable | Fallback |
| Mistral-7B-Instruct-v0.2 | 7B | ~4.5 GB | Muy bueno | Más lento, mayor huella |
| Qwen2.5-1.5B-Instruct | 1.5B | ~1.5 GB | Bueno | Alternativa válida |
| Llama-3.2-3B-Instruct | 3.2B | ~2 GB | Muy bueno | Alternativa válida |

**Por qué Phi-3-mini sobre TinyLlama**  
Phi-3-mini supera a TinyLlama en benchmarks de razonamiento y seguimiento de instrucciones estructuradas (JSON output), lo cual es crítico para las trazas del sistema. TinyLlama tiende a "alucinar" en tareas de razonamiento multi-paso.

**Notas de implementación**
- Temperatura = 0 para la mayoría de tareas (reproducibilidad)
- Temperatura = 0.3 para generación de variantes (diversidad controlada)
- Prompt templates en `src/agentic/prompts.py` (versionados)

**Consecuencias**
- (+) Calidad razonamiento notablemente superior a TinyLlama para este tamaño
- (+) Soporta output JSON estructurado de forma confiable
- (-) Primera carga ~2 min en Kaggle; hay que cargar al inicio de la notebook, no en cada búsqueda
- (-) Requiere `bitsandbytes` instalado, que a veces da problemas en Windows (en Kaggle Linux funciona bien)

---

## ADR-005: Estrategia de reranking — Pool + penalización CLIP

**Estado:** Aceptado

**Contexto**  
CLIP no codifica negaciones de forma directa. "auto no rojo" y "auto rojo" producen embeddings casi idénticos porque CLIP aprendió de pares texto-imagen donde el texto siempre describe lo que hay (no lo que no hay).

**Decisión**  
Pool-based reranking en dos pasos:
1. Recuperar top-N (N=50) candidatos via FAISS con la query reformulada (positiva, sin negaciones)
2. Para cada imagen candidata, computar su similitud con los términos negados
3. Score final = `sim_positiva(img, query_positiva) - λ × max(sim_negativas(img, terminos_negados))`
4. Re-rankear por score final, tomar top-10

**Parámetro λ**  
λ controla la fuerza de la penalización. Se ajusta empíricamente. Valor inicial propuesto: λ = 0.5.

**Alternativas consideradas**

| Estrategia | Efectividad | Implementación | Decisión |
|------------|-------------|----------------|---------|
| Pool + penalización CLIP | Media-alta | Media | **Elegida** |
| Buscar directamente "query sin negación" | Igual que baseline | Trivial | No mejora nada |
| Buscar negativos y excluirlos por ID | Baja (CLIP retrieval de negaciones es ruidoso) | Media | Poco confiable |
| Fine-tuning del modelo de retrieval | Alta | Muy alta | Fuera de scope |

**Consecuencias**
- (+) Funciona sin datos de entrenamiento adicionales
- (+) El mecanismo es explicable y justificable en la defensa oral
- (-) Requiere 2 forward passes de CLIP por query con negaciones (más lento)
- (-) La efectividad depende de qué tan bien CLIP separe el atributo negado

---

## ADR-006: Estructura modular vs notebook monolítica

**Estado:** Aceptado

**Contexto**  
La especificación exige una notebook de Kaggle reproducible. Pero desarrollar directamente en una notebook gigante es inmantenible, dificulta los tests y genera conflictos de estado entre celdas.

**Decisión**  
Desarrollar como paquete Python modular en `src/` y compilar/importar en la notebook final via `notebook_compiler.py`.

**Dos opciones de despliegue en Kaggle**

**Opción A — Dataset auxiliar** (recomendada):
- Subir `src/` como dataset de Kaggle `tpi-ia-2026-src`
- En la notebook: `sys.path.insert(0, '/kaggle/input/tpi-ia-2026-src')`
- Importar normalmente

**Opción B — Inline via compiler**:
- `notebook_compiler.py` lee cada módulo y lo incrusta como celda de código
- La notebook queda totalmente auto-contenida
- Más frágil de mantener pero no requiere el dataset auxiliar

**Consecuencias de la arquitectura modular**
- (+) Tests unitarios posibles por módulo
- (+) Control de versiones limpio (cambios en un módulo no afectan otros)
- (+) Colaboración: cada integrante puede trabajar en un módulo distinto
- (+) La defensa oral es mucho más fácil si cada componente tiene responsabilidad clara
- (-) Hay que mantener `notebook_compiler.py` sincronizado con los módulos

---

## ADR-007: Trazabilidad — logs JSON estructurados

**Estado:** Aceptado

**Contexto**  
La cátedra evalúa explícitamente que "las decisiones del sistema sean observables y justificables". Necesitamos un mecanismo de logging que capture cada paso del pipeline agéntico.

**Decisión**  
Cada ejecución del pipeline produce un objeto JSON con este esquema:

```json
{
  "query_id": "q_001",
  "timestamp": "2026-06-01T15:23:01",
  "original_query": "auto no rojo en la calle",
  "steps": [
    {
      "step": "language_detection",
      "result": "es",
      "confidence": 0.99
    },
    {
      "step": "translation",
      "model": "opus-mt-es-en",
      "input": "auto no rojo en la calle",
      "output": "car not red on the street",
      "latency_ms": 48
    },
    {
      "step": "negation_detection",
      "detected": true,
      "positive_query": "car on the street",
      "negative_attributes": ["red"]
    },
    {
      "step": "semantic_expansion",
      "model": "phi-3-mini",
      "variants": ["car on the road", "vehicle street scene"],
      "latency_ms": 312
    },
    {
      "step": "integrity_validation",
      "valid": true,
      "reason": "query is coherent and not contradictory"
    },
    {
      "step": "faiss_retrieval",
      "query_used": "car on the street",
      "top_k": 50,
      "retrieved_ids": ["2007_000032", "..."]
    },
    {
      "step": "reranking",
      "lambda": 0.5,
      "final_top10": ["2007_000032", "..."]
    }
  ],
  "final_results": ["2007_000032", "..."]
}
```

**Consecuencias**
- (+) Permite auditar exactamente por qué el sistema devuelve cada resultado
- (+) Facilita el debug de casos donde el sistema falla
- (+) Los logs pueden mostrarse en la notebook como evidencia del enfoque agéntico
- (-) Leve overhead de serialización JSON (~1ms por query, irrelevante)
