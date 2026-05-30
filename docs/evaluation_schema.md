# Esquema de Evaluación — MAP@10 y Ground Truth

## Métrica principal: MAP@10

**Mean Average Precision at 10** mide qué tan bien el sistema ubica las imágenes relevantes en las primeras 10 posiciones del ranking.

### Definiciones formales

Para una query `q` con conjunto de relevantes `R` y ranking de resultados `r_1, ..., r_10`:

```
Precision@k = |{r_1,...,r_k} ∩ R| / k

Average Precision@10 (AP@10) = Σ_{k=1}^{10} Precision@k × is_relevant(r_k)
                                ─────────────────────────────────────────────
                                         min(|R|, 10)
```

```
MAP@10 = (1/|Q|) × Σ_{q ∈ Q} AP@10(q)
```

Donde `|Q|` es el número total de queries evaluadas.

### Ejemplo numérico

Query: "dog" → ground truth: 500 imágenes de perros en VOC  
Sistema devuelve: [dog, cat, dog, dog, horse, dog, dog, cat, dog, dog]
                   R     -    R    R     -    R    R    -    R    R

```
P@1  = 1/1  = 1.0
P@2  = 1/2  = 0.5
P@3  = 2/3  = 0.67
P@4  = 3/4  = 0.75
P@5  = 3/5  = 0.6
P@6  = 4/6  = 0.67
P@7  = 5/7  = 0.71
P@8  = 5/8  = 0.625
P@9  = 6/9  = 0.67
P@10 = 7/10 = 0.7

AP@10 = (1.0 + 0.67 + 0.75 + 0.67 + 0.71 + 0.67 + 0.7) / min(500, 10)
      = 5.17 / 10 = 0.517
```

---

## Construcción del Ground Truth

### q1–q20: Clases VOC (evaluación en Kaggle)

Pascal VOC 2012 provee anotaciones XML por imagen con las clases de objetos presentes.

**20 clases:** aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor

**Proceso:**
1. Leer todos los archivos `Annotations/*.xml`
2. Para cada imagen, extraer las clases de sus objetos
3. Construir `gt_voc = {class_name: Set[img_id]}`
4. Las queries q1–q20 son exactamente las 20 clases en inglés

```python
# Ejemplo de estructura resultante
gt_voc = {
    "dog":    {"2007_000032", "2007_000033", ...},  # ~1200 imágenes
    "car":    {"2007_000027", "2007_000042", ...},  # ~1900 imágenes
    "person": {"2007_000042", "2007_000063", ...},  # ~9000 imágenes
    ...
}
```

**Nota sobre imagen con múltiples clases:** una imagen puede aparecer en el GT de varias clases si tiene múltiples objetos anotados. Esto es correcto y esperado.

### q21–q40: Queries complejas (ground truth provisto por cátedra)

Para las queries q21–q40, la cátedra provee el `solution.csv` con el ground truth oficial. **No necesitamos construirlo nosotros** — se usa directamente para calcular MAP@10 en la evaluación de Kaggle.

Sin embargo, para el **ablation study local** (desarrollo y debugging), diseñamos un ground truth aproximado usando el método de intersección:

**Método de intersección para queries compuestas**

```
Query: "auto rojo" → "car" ∩ estimated_red_images
Query: "perro en jardín" → "dog" ∩ estimated_outdoor_images
Query: "persona con bicicleta" → "person" ∩ "bicycle" (co-occurring in same image)
```

**Implementación:**
- `gt_voc["car"]` da las imágenes con autos
- "rojo" → usamos CLIP para recuperar las top-K imágenes más similares a "red object"
- GT aproximado = intersección de ambos conjuntos

Esta estrategia es una **aproximación**. La cátedra evalúa MAP@10 con su ground truth oficial; nuestro GT aproximado sirve solo para desarrollo.

---

## Diseño del Ablation Study

El ablation study compara 3 configuraciones del sistema para aislar el aporte de cada componente:

### Configuraciones

| Config | ID | Descripción |
|--------|-----|-------------|
| Baseline | `A` | CLIP + FAISS con query original en español |
| + Reformulación | `B` | CLIP + FAISS con query traducida y expandida por LLM |
| + Reranking | `C` | Config B + penalización de atributos negativos |

### Queries de prueba para el ablation

Debemos seleccionar queries que permitan medir el aporte de cada componente:

**Grupo 1: queries simples (para medir ganancia de traducción)**
- "perro" → "dog"
- "automóvil" → "car"
- "pájaro volando" → "bird flying"

**Grupo 2: queries con sinónimos (para medir ganancia de expansión semántica)**
- "vehículo de motor" → ¿mejora vs. solo "vehicle"?
- "felino doméstico" → ¿mejora vs. "cat"?

**Grupo 3: queries con negaciones (para medir ganancia del reranker)**
- "auto no rojo" → busca autos sin penalización vs. con penalización
- "persona no adulta" → niños vs. adultos
- "animal no perro" → cualquier animal excepto perros

### Tabla de resultados esperada

```
| Query                  | MAP@10_A | MAP@10_B | MAP@10_C | ΔMAP(A→B) | ΔMAP(B→C) |
|------------------------|----------|----------|----------|-----------|-----------|
| "perro"                | 0.XX     | 0.XX     | 0.XX     | +X.XX     | 0.00      |
| "automóvil"            | 0.XX     | 0.XX     | 0.XX     | +X.XX     | 0.00      |
| "auto no rojo"         | 0.XX     | 0.XX     | 0.XX     | +X.XX     | +X.XX     |
| ...                    | ...      | ...      | ...      | ...       | ...       |
| PROMEDIO               | 0.XX     | 0.XX     | 0.XX     | +X.XX     | +X.XX     |
```

**Hipótesis esperadas:**
- `MAP_B > MAP_A` para queries donde la traducción/expansión ayuda (la mayoría)
- `MAP_C > MAP_B` solo para queries con negaciones
- `MAP_C ≈ MAP_B` para queries sin negaciones (el reranker no debería degradar)

---

## Consideraciones para el reporte técnico

1. **Limitación del GT aproximado**: para q21–q40 nuestro GT local es una aproximación. Los resultados definitivos vienen de Kaggle.

2. **Sesgo del tamaño del GT**: "person" tiene ~9000 imágenes relevantes en VOC, mientras "sheep" tiene ~500. Esto afecta AP@10 de forma asimétrica (más fácil llegar a AP@10=1.0 para clases raras).

3. **MAP@10 es sensible al orden**: las posiciones 1–3 tienen mucho más peso que las posiciones 8–10. El sistema debería optimizar para poner las más relevantes primero.

4. **Métricas complementarias a incluir en el reporte**:
   - Recall@10: fracción de relevantes recuperados en top-10
   - MRR (Mean Reciprocal Rank): posición del primer resultado correcto
   - Visualizaciones cualitativas (más convincentes para la defensa oral)
