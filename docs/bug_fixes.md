# Registro de Bugs y Decisiones de Corrección

Documento de trazabilidad de bugs encontrados durante el desarrollo, opciones de solución evaluadas y decisión tomada. Se mantiene para poder justificar las decisiones en la defensa oral y como historial de diseño.

---

## BUG-001 — El reranker nunca ejecutaba (bug silencioso crítico)

**Detectado en:** sesión 1, análisis estático post-escritura  
**Severidad:** Crítica — el pipeline compilaba y corría sin error, pero el reranking de negaciones no hacía absolutamente nada

### Descripción del problema

`NegationReranker.rerank()` tenía esta firma:

```python
def rerank(self, candidates, negative_attributes,
           image_embeddings=None, image_ids=None):
    if not negative_attributes or image_embeddings is None:
        return candidates  # ← siempre entraba acá
```

Y `AgenticPipeline` lo llamaba así:

```python
reranked = self._reranker.rerank(
    candidates=candidates,
    negative_attributes=neg_result.negative_attributes,
    # image_embeddings y image_ids nunca se pasaban
)
```

Resultado: `image_embeddings` siempre era `None` → el `if` del principio siempre era verdadero → la función devolvía los candidatos sin modificar → el reranking de negaciones no existía en la práctica, aunque el código "parecía" que sí.

El sistema hubiera entregado exactamente los mismos resultados con o sin negaciones, sin ningún mensaje de error.

### Opciones de solución evaluadas

**Opción A — Pasar embeddings en cada llamada a `rerank()`**
```python
# En pipeline.py
reranked = self._reranker.rerank(
    candidates=candidates,
    negative_attributes=neg_result.negative_attributes,
    image_embeddings=self._image_embeddings,   # nuevo attr en pipeline
    image_ids=self._image_ids,
)
```
- Pro: interface explícita, fácil de ver qué datos se usan
- Con: el pipeline tendría que cargar y guardar la matriz de embeddings (512×17k floats ≈ 34MB en RAM), que ya estaba en el reranker de todas formas
- Con: acoplamiento innecesario — el pipeline no debería saber nada de la representación interna del reranker

**Opción B — Mover los embeddings al constructor del reranker** ← *elegida*
```python
# Al instanciar en scripts/
reranker = NegationReranker(
    clip_extractor=clip,
    image_embeddings=embeddings,   # cargados una sola vez
    image_ids=image_ids,
)

# rerank() ya no los necesita como params
def rerank(self, candidates, negative_attributes):
    if self._image_embeddings is None:
        return candidates
    ...
```
- Pro: el reranker es un objeto auto-contenido con todo lo que necesita para funcionar
- Pro: la llamada en el pipeline queda limpia, sin pasar datos de grandes dimensiones en cada query
- Pro: se puede hacer `reranker.load_embeddings(...)` post-construcción si se necesita (flexibilidad)
- Con: el objeto pesa más en RAM desde su creación (aceptable, son 34MB)

**Opción C — Cargar embeddings desde disco dentro de `rerank()`**
```python
def rerank(self, candidates, negative_attributes):
    if self._embeddings_path is None:
        return candidates
    embeddings = np.load(self._embeddings_path)  # cargar en cada llamada
    ...
```
- Con: leer 34MB de disco en *cada query* es prohibitivo (~100ms por llamada)
- Descartada inmediatamente

### Decisión tomada: Opción B

El reranker es una dependencia que sabe cómo penalizar imágenes. Que tenga la matriz de embeddings encapsulada es coherente con su responsabilidad. Además, agrega `load_embeddings()` como método público por si se necesita actualizar los embeddings sin recrear el objeto.

---

## BUG-002 — Acceso a atributos privados del pipeline desde el ablation study

**Detectado en:** sesión 1, análisis estático  
**Severidad:** Baja — funcionaba, pero era frágil y violaba encapsulamiento

### Descripción del problema

`ablation.run_reformulation()` accedía a `pipeline._clip` y `pipeline._faiss` directamente:

```python
emb  = pipeline._clip.encode_text(expanded)
hits = pipeline._faiss.search(emb, k=k)
```

Nombres con `_` prefijo son convención de Python para "privado/interno". Acceder a ellos desde otro módulo significa que si alguna vez se renombran, el código de ablation falla en runtime sin ninguna advertencia en tiempo de análisis estático.

### Opciones de solución evaluadas

**Opción A — Hacer `_clip` y `_faiss` públicos (quitar el `_`)**
```python
self.clip  = clip_extractor
self.faiss = faiss_manager
```
- Con: expone los internos del pipeline directamente, cualquiera puede modificarlos desde afuera
- Semánticamente incorrecto — son dependencias internas, no parte de la API pública

**Opción B — Agregar `@property` de solo lectura** ← *elegida*
```python
@property
def clip(self) -> CLIPExtractor:
    return self._clip

@property
def faiss(self) -> FAISSManager:
    return self._faiss
```
Y en ablation usar `pipeline.clip` y `pipeline.faiss`.
- Pro: la API pública está explícitamente declarada (intención clara)
- Pro: son de solo lectura (no se puede hacer `pipeline.clip = otra_cosa`)
- Pro: el ablation study puede acceder a clip y faiss para re-run sin necesitar rerun el pipeline completo

**Opción C — Cambiar la firma de `run_reformulation()` para recibir `clip` y `faiss` directamente**
```python
def run_reformulation(queries, clip, faiss, k=10):
```
- Pro: no depende del pipeline en absoluto, más desacoplado
- Con: si se llama desde múltiples lugares hay que pasar más argumentos
- Con: cambiaría la simetría con `run_baseline(queries, clip, faiss, k)` — en ese caso ambas son iguales y pierde sentido tener el pipeline

### Decisión tomada: Opción B

Las properties `@property` son el patrón Python idiomático para exponer acceso controlado a internos. El ablation study tiene una razón legítima para acceder al extractor y al índice (para configuraciones parciales del pipeline), por lo que exponerlos con `@property` de solo lectura es la interface correcta.

---

## BUG-003 — Import de nombre inexistente en notebook compiler

**Detectado en:** sesión 1, análisis estático  
**Severidad:** Media — el notebook generado hubiera fallado al ejecutar en Kaggle con `ImportError`

### Descripción del problema

En `notebook_compiler.py`, el `MAIN_CELL` (la celda de ejecución principal que se inyecta en la notebook) tenía:

```python
from src.config import VOC_CLASSES, CUSTOM_QUERIES  # ← CUSTOM_QUERIES no existe en config.py
```

`CUSTOM_QUERIES` es un diccionario local definido dentro de `scripts/04_generate_submission.py`, no en `src/config.py`. El compiler lo "copió" por error durante la escritura del MAIN_CELL.

### Opciones de solución evaluadas

**Opción A — Agregar `CUSTOM_QUERIES` a `src/config.py`**
```python
# En config.py
CUSTOM_QUERIES = {
    "q21": "perro en el jardín",
    ...
}
```
- Con: las queries son datos que probablemente cambien (la cátedra las publica); ponerlas en config las mezcla con parámetros técnicos del sistema
- Con: `config.py` es para hiperparámetros y paths, no para contenido semántico de búsqueda

**Opción B — Eliminar el import del MAIN_CELL y definir las queries inline en la notebook** ← *elegida*
```python
# En MAIN_CELL, sin import de CUSTOM_QUERIES
from src.config import VOC_CLASSES
# Las queries q21-q40 se definen inline o se cargan de un archivo externo en la notebook
```
- Pro: separación correcta — config técnico vs. contenido de queries
- Pro: en la notebook de Kaggle es más natural ver las queries definidas explícitamente como celda separada, así el usuario puede editarlas fácilmente
- Con: hay que acordarse de actualizar las queries en la notebook cuando la cátedra las publique

### Decisión tomada: Opción B

Las queries de q21-q40 son contenido que va a mutar (la cátedra las publicará). No tienen lugar en `config.py` junto con los hiperparámetros del sistema. La notebook de Kaggle las define inline con un comentario claro de que hay que actualizarlas.

---

---

## BUG-004 — Regex del negation parser capturaba 2 tokens ("red on" en vez de "red")

**Detectado en:** Sprint 1 — primera ejecución de tests (77/82 pasaron)  
**Severidad:** Media — el reranker hubiera recibido atributos incorrectos ("red on" ≠ "red") reduciendo su efectividad silenciosamente

### Descripción

El patrón `r"\bnot\s+(\w+(?:\s+\w+)?)"` captura opcionalmente 2 palabras:
- "car not red on the street" → captura **"red on"** en vez de **"red"**
- "no dogs in the photo" → captura **"dogs in"** en vez de **"dogs"**

### Opciones de solución

**Opción A — Añadir stop words al segundo token** (filtrar preposiciones del capture group)
- Pro: preserva casos donde sí queremos 2 palabras ("not very bright")
- Con: lista de stop words incompleta y frágil; las preposiciones varían entre idiomas

**Opción B — Capturar solo 1 token** ← *elegida*
```python
r"\bnot\s+(\w+)"   # "not red" → "red"
```
- Pro: determinista y simple; el atributo negado es casi siempre 1 palabra
- Pro: reduce ruido en el reranker (atributos más precisos → penalización más precisa)
- Con: no captura "not very red" → solo captura "not very" (aunque "very" sería filtrado por stop words)

### Decisión: Opción B

La semántica del reranker opera sobre atributos discretos ("red", "people", "background"). Dos palabras como "red on" o "dogs in" son siempre incorrectos. La mayoría de los casos relevantes son de 1 palabra, y para los que no lo son, el LLM (expander) puede desambiguar antes.

### Archivos modificados
- `src/agentic/negation_parser.py` — regex simplificado

---

## BUG-005 — `datetime.utcnow()` deprecated en Python 3.12

**Detectado en:** Sprint 1 — warnings en test suite  
**Severidad:** Baja — funcionaba, pero Python 3.12 emite DeprecationWarning; se eliminaría en versión futura

### Fix

```python
# Antes
datetime.utcnow().isoformat()
# Después
datetime.now(timezone.utc).isoformat()
```

La diferencia es que `now(timezone.utc)` retorna un datetime timezone-aware (con `+00:00` en el ISO string). Esto es más correcto y compatible con futuras versiones.

### Archivos modificados
- `src/agentic/tracer.py`

---

---

## BUG-006 — `AgentResult` no tenía campo `has_negation`

**Detectado en:** Sprint 2 — primera corrida de tests del pipeline  
**Severidad:** Baja — faltaba un campo conveniente en la API pública del resultado

### Descripción

Los tests del pipeline accedían a `result.has_negation` para verificar si la query tenía negaciones. `AgentResult` solo tenía `negative_attributes: List[str]`, sin un shortcut booleano.

### Opciones

**Opción A — Cambiar tests para usar `result.negative_attributes == []`**
- Con: hace los tests más verbosos; `has_negation` es semánticamente más claro

**Opción B — Agregar `@property has_negation`** ← *elegida*
```python
@property
def has_negation(self) -> bool:
    return bool(self.negative_attributes)
```
- Pro: API más expresiva, coherente con `NegationResult.has_negation` en el parser
- Pro: los módulos que consumen `AgentResult` no necesitan saber sobre la lista interna

---

## BUG-007 — `MagicMock` en `TraceStep` causaba `TypeError` al serializar JSON

**Detectado en:** Sprint 2 — tests del pipeline  
**Severidad:** Media — hacía fallar la serialización del trace a disco

### Descripción

En `test_pipeline.py`, el mock de `_detect_language` usaba:
```python
return_value=("en", MagicMock())   # MagicMock como TraceStep
```
Cuando el pipeline intentaba serializar el trace a `.jsonl`, `json.dumps` no puede serializar un `MagicMock`.

### Opciones

**Opción A — Suprimir `append_to_file` en tests (mock no-op)**
- Con: oculta el comportamiento real; el test no verifica que la traza es serializable

**Opción B — Usar un `TraceStep` real en el mock** ← *elegida*
```python
from src.agentic.tracer import TraceStep
lang_step = TraceStep(step="language_detection", result="en")
return_value=("en", lang_step)
```
- Pro: el test verifica el comportamiento real de serialización
- Pro: mucho más seguro — si `TraceStep` cambia su estructura, el test lo detecta

**Fix adicional:** `val_client.model_name = "phi-3-mini-mock"` — atributos de MagicMocks que van dentro de TraceStep siempre deben ser strings, no MagicMocks.

---

## Estado post-correcciones

| Bug | Archivo modificado | Tipo de fix |
|-----|-------------------|-------------|
| BUG-001 | `src/reranking/negation_reranker.py` | Constructor recibe embeddings |
| BUG-001 | `src/agentic/pipeline.py` | Import de numpy movido fuera de TYPE_CHECKING |
| BUG-001 | `scripts/03_run_search.py` | Pasa embeddings al constructor del reranker |
| BUG-001 | `scripts/04_generate_submission.py` | Pasa embeddings al constructor, elimina variable redundante |
| BUG-001 | `notebook_compiler.py` MAIN_CELL | Instanciación correcta del reranker |
| BUG-002 | `src/agentic/pipeline.py` | Agrega `@property clip` y `@property faiss` |
| BUG-002 | `src/evaluation/ablation.py` | Usa `pipeline.clip` y `pipeline.faiss` |
| BUG-003 | `notebook_compiler.py` MAIN_CELL | Elimina import de `CUSTOM_QUERIES` |

**Verificación post-fix:** análisis estático completo pasó sin errores (sintaxis, imports, palabras prohibidas, wiring del reranker).
