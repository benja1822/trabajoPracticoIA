# Notas Teóricas — Fundamentos del Sistema

Este documento registra los conceptos teóricos clave que sustentan las decisiones de diseño y que el equipo debe dominar para la defensa oral.

---

## 1. CLIP: Contrastive Language-Image Pre-training

### Qué es

CLIP (Radford et al., OpenAI 2021) es un modelo entrenado con 400 millones de pares (imagen, texto) usando aprendizaje contrastivo. El objetivo de entrenamiento maximiza la similitud entre pares genuinos y la minimiza entre pares falsos.

### Espacio semántico compartido

Tanto el texto encoder (ViT o Transformer) como el image encoder (ViT o ResNet) proyectan sus inputs a un espacio vectorial de dimensión fija (512 para ViT-B/32). Este espacio es semántico: conceptos similares quedan geométricamente cercanos.

```
CLIP("a photo of a dog") ≈ CLIP(imagen_de_perro)
CLIP("a photo of a cat") ≠ CLIP(imagen_de_perro)
```

### Por qué normalizar a norma 1

CLIP produce embeddings cuyos módulos varían entre muestras. Para comparar semántica pura (no magnitud), normalizamos a norma unitaria y usamos producto interno:

```
similarity(a, b) = a·b / (||a|| ||b||)  →  a·b  cuando ||a|| = ||b|| = 1
```

### Limitación crítica: las negaciones

CLIP fue entrenado con texto descriptivo que **describe qué está presente** en la imagen, nunca qué no está. Por lo tanto:

```
CLIP("red car") ≈ CLIP("car not red")  ← ¡problem!
```

El embedding de "not red" no tiene mucho sentido en el espacio de CLIP porque "not red" nunca apareció en los captions de entrenamiento asociados a imágenes. Por eso necesitamos el reranker post-FAISS.

### Limitación: embeddings globales sin información espacial

ViT-B/32 genera **un único vector por imagen** que captura la semántica global. No hay información sobre dónde está cada objeto. Una imagen con un perro pequeño en el fondo y un gato grande en el frente tiene un embedding dominado por el gato.

---

## 2. FAISS: Facebook AI Similarity Search

### Qué es

FAISS es una librería de búsqueda de vecinos más próximos en espacios de alta dimensión, optimizada para operaciones en GPU y CPU con SIMD.

### Tipos de índice relevantes para este TP

**IndexFlatIP** (Búsqueda exacta por producto interno)
- No hay entrenamiento del índice
- Busca en todos los N vectores: O(N·D)
- Resultado: los k vectores con mayor producto interno
- Para N=17k, D=512: ~3ms por query en CPU → perfectamente aceptable

**IndexIVFFlat** (Aproximado, inverted file)
- Agrupa vectores en `nlist` celdas de Voronoi
- En búsqueda, visita solo `nprobe` celdas más cercanas
- Requiere entrenamiento previo (fit con una muestra representativa)
- Para N=17k, el overhead no vale la pena

**Por qué no usamos HNSW o IVF para este dataset**  
A 17k vectores, `IndexFlatIP` es exacto y rápido. Los índices aproximados son para escenarios con millones de vectores donde la exactitud se sacrifica por velocidad. En un TP académico donde MAP@10 importa, exactitud > velocidad.

### Guardado y carga del índice

```python
faiss.write_index(index, "faiss_flat_ip.index")
index = faiss.read_index("faiss_flat_ip.index")
```

El índice serializado incluye todos los vectores pero **no** el mapeo ID→nombre de imagen. Ese mapeo lo guardamos separado en `image_ids.json`.

---

## 3. LLMs Livianos: Phi-3-mini

### Por qué un LLM y no solo reglas

Las reglas regex pueden detectar negaciones simples ("no X", "sin X"), pero fallan en:
- Negaciones implícitas: "evitando el rojo"
- Contradicciones complejas: "auto rápido y lento a la vez"
- Sinonimia de dominio: "felino" ≠ "gato" en un tokenizer sin conocimiento semántico

Un LLM entiende estas sutilezas porque fue entrenado en texto natural.

### Cuantización 4-bit

La cuantización reduce la precisión de los pesos de float32 (4 bytes/param) a int4 (0.5 bytes/param), reduciendo VRAM en ~8×. Con `bitsandbytes`:

```
Phi-3-mini (3.8B params):
- float32: ~14.4 GB  →  inviable en T4
- int8:     ~3.8 GB  →  viable pero lento
- int4:     ~2.0 GB  →  ideal para Kaggle T4/P100
```

La pérdida de calidad en 4-bit para tareas de NLP es típicamente <2% comparado con float32.

### Temperatura y reproducibilidad

- `temperature=0` → output determinista, greedy decoding. Usar para clasificación, detección de negaciones, validación.
- `temperature=0.3` → cierta aleatoriedad controlada. Usar para generación de variantes de query (queremos diversidad).
- Con semilla fija (`torch.manual_seed`) + temperatura 0.3, los resultados son reproducibles.

---

## 4. Reranking por penalización de negaciones

### El problema de la negación en retrieval

El retrieval vectorial opera con similitud positiva: más similar = más relevante. No hay operador "NOT" en el espacio vectorial. Intentar buscar directamente "car NOT red" con CLIP produce embeddings mal definidos.

### Solución: pool + penalización post-retrieval

```
Paso 1: Recuperar top-50 candidatos buscando "car on the street" (query positiva)
Paso 2: Para cada candidato, medir similitud con "red" (atributo negado)
Paso 3: Score final = sim_positiva - λ × sim_negativa
Paso 4: Re-rankear por score final → top-10
```

### Calibración de λ

λ controla cuánto penalizamos. Un λ alto puede excluir imágenes que tienen algo de rojo pero que son claramente de un auto. Un λ bajo no penaliza suficiente.

Estrategia de calibración:
1. Probar λ ∈ {0.2, 0.3, 0.5, 0.7, 1.0} en queries de validación
2. Medir MAP@10 para queries con negaciones
3. Elegir el λ que maximiza MAP@10 sin degradar MAP para queries sin negaciones

**Resultado esperado en ablation:** MAP mejora para queries con negaciones, se mantiene igual para queries sin negaciones.

### Limitación fundamental del reranking

Si la imagen que queremos está en el puesto 51 del FAISS (fuera del pool de 50), el reranker no puede traerla. Aumentar el pool a N=100 mejora el recall pero aumenta el tiempo de procesamiento.

---

## 5. MAP@10 en contexto de retrieval multimodal

### Por qué MAP@10 y no precision@10

Precision@10 es el porcentaje de relevantes en los top-10, pero no distingue si el primer resultado es relevante o el décimo. MAP@10 premia poner los relevantes **antes** en el ranking.

### Diferencia con otras métricas

| Métrica | Captura | Cuando usar |
|---------|---------|-------------|
| MAP@10 | Ranking quality, precisión | Evaluación general del retrieval |
| Recall@10 | Cobertura de relevantes | Cuando el dataset tiene pocos relevantes |
| NDCG@10 | Relevancia gradual (0-2) | Cuando hay niveles de relevancia, no binario |
| MRR | Posición del primer relevante | Sistemas de QA donde el primer resultado importa |

Para Pascal VOC (relevancia binaria: está/no está la clase), MAP@10 es la métrica correcta.

### Interpretación de resultados típicos

Para búsqueda zero-shot con CLIP sobre VOC:
- Baseline esperado: MAP@10 ≈ 0.35–0.55 para clases "fáciles" (dog, cat, car)
- Clases "difíciles" (boat, pottedplant): MAP@10 ≈ 0.15–0.30
- Con reformulación: mejora esperada de 5–15% en MAP promedio

Estos valores son orientativos; los resultados reales dependerán de la implementación.

---

## 6. Pascal VOC 2012: aspectos relevantes del dataset

### Estadísticas

- Total imágenes train+val: ~17.112
- Clases anotadas: 20
- Anotaciones: bounding boxes por objeto (XML), segmentación (PNG masks), clasificación por imagen

### Formato de IDs de imagen

Los IDs siguen el formato `YYYY_NNNNNN`, donde:
- `YYYY`: año de la imagen (2007 o 2008 principalmente, también 2009–2012)
- `NNNNNN`: número de 6 dígitos

El nombre de archivo es `{ID}.jpg`. El submission debe contener el ID **sin** la extensión.

### Distribución desbalanceada de clases

"person" aparece en ~69% de las imágenes. "sheep" y "boat" en ~10%.  
Esto afecta el MAP@10: para clases raras, es más difícil recuperar 10 relevantes y el AP@10 tiende a ser más variable.

### Dificultades del dataset para retrieval

- **Objetos parciales**: parte de un auto, cabeza de perro → embedding global puede ser engañoso
- **Co-ocurrencia frecuente**: person+car, person+bicycle → búsqueda de "bicycle" puede traer muchas imágenes con persona+bicicleta donde la bicicleta es pequeña
- **Variación de escala**: mismo objeto puede ser muy pequeño o ocupar toda la imagen
