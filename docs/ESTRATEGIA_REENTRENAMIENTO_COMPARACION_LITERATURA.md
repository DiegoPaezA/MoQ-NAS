# Estrategia de reentrenamiento y comparación con la literatura (MoQ-NAS 2026)

Fecha: 2026-10-03. Documento para llevar al clúster. **Todavía no se reentrena nada**: aquí se define
*qué* reentrenar, *con qué protocolo*, *qué corregir antes* y *contra qué métodos* comparar.

Archivos asociados:
- `retrain_matrices/literature_retrain_candidates.csv`: IDs exactos por corrida (2 386 filas) con su rol
  (`best_acc`, `knee`, `compact`, `strat10`, `front`) y los objetivos proxy.
- `../scripts/build_literature_retrain_candidates.py` (repo de análisis): regenera el CSV a partir de `data/`.

---

## 0. Resumen

1. **El problema no es la arquitectura, es la fidelidad.** Las accuracies del paper (≈77–81 % en CIFAR-10) son
   *proxy* (10 000 imágenes, 50 épocas, sin augmentation). En tu propio pipeline, Q-NAS mono-objetivo en el
   **mismo espacio Standard (11 ops)** tenía un proxy de ≈79–81 % y, tras reentrenar con el protocolo F13
   (300 épocas), llegó a **90.1–93.9 % de test** (0.5–3.5 M parámetros). Es razonable esperar que los mejores
   puntos del frente de MoQ-NAS (proxy 77–79 %, 0.9–1.7 M) queden en ~91–93 %. Eso los pone en la misma zona que
   el grupo "NAS sin celdas" de la Tabla 6 de Q-NAS (Q-NAS 92.96 %/1.6 M, CGP-CNN 93.25 %/1.52 M, MetaQNN 93.08 %/11.2 M).
2. **Qué reentrenar (por prioridad):**
   - **Tier A (imprescindible): Caso 1, CIFAR-10 tri-objetivo**: MoQ-NAS exp22 ×3, NSGA-II exp4 ×3, NSGA-III exp1 ×3.
     90 redes (`strat10`, 1 semilla) + 25 representantes (`best_acc`/`knee`/`compact`, 3 semillas).
   - **Tier B (muy recomendable): acc-FLOPs, espacio Standard**: MoQ-NAS/NSGA-II/NSGA-III ×3.
     Es la única formulación que coincide con NSGA-Net/NSGANetV1 (error + FLOPs). 90 redes + 25 representantes.
   - **Tier C (recomendable): Caso 2, MedMNIST** (4 datasets × 3 algoritmos × 3 corridas). Como mínimo los
     representantes (≈100 redes × 3 semillas); el `strat10` (360 redes) solo si sobra presupuesto.
   - **Tier D: Caso 3**: todo el frente final (≈ 115 redes) y los 5 baselines desde cero, **3 semillas cada uno**,
     en dos regímenes que se mantienen: R1 = presupuesto de búsqueda (10 k imágenes, 50 épocas) y R2 = dataset
     completo. Mismo protocolo para MoQ-NAS y baselines (hoy difieren en batch y bucle de entrenamiento).
     Fairness recalculada en FACET; requiere corregir antes el fallo 4.9.
   - **No reentrenar**: MOEA/D (frentes degenerados), `acc_flops_mixed`, `random_elite`, `noop10` (son ablaciones
     internas sin comparador en la literatura).
3. **Antes de lanzar hay que corregir cuatro cosas del pipeline de retrain** (sección 4). La más importante:
   `retrain_matrices/cifar_mo.yaml` usa `patience_retrain: 50` con cosine → el early stopping corta el
   entrenamiento con la LR todavía alta. El protocolo que dio 93–94 % (F13) usaba `patience = max_epochs`.
4. **Tabla comparativa: solo contra NAS multiobjetivo** (decisión 2026-10-03). Los comparadores se emparejan por
   objetivos: error + FLOPs (NSGA-Net, NSGANetV1) → Tier B; error + params (CARS, LEMONADE, EEEA-Net) → Tier A.
   Por eso **Tier A y Tier B tienen la misma prioridad**. Como esos papers publican puntos y no frentes, se compara el
   frente reentrenado con sus puntos (superposición, dominancia, error a complejidad igualada; §6.1). Lo
   mono-objetivo (Q-NAS, Tabla 6, MedMNIST v2) solo entra como filas de referencia etiquetadas. En MedMNIST los
   comparadores multiobjetivo son pocos y están por verificar (§6.3).

---

## 1. Diagnóstico: por qué los números del paper están "muy por debajo"

| Fase | Datos | Épocas | Augmentation | Lo que mide |
|---|---|---|---|---|
| Búsqueda (paper) | 9 000 train / 1 000 val (CIFAR-10) | 50 | no | ranking proxy (media de las últimas 5 épocas, `eval_window_agg: mean`) |
| Retrain F13 (Q-NAS interno) | 45 000 train / 5 000 val, test oficial 10 000 | 300 | TrivialAugmentWide | accuracy final |
| Literatura (NSGA-Net, LEMONADE, NSGANetV1…) | 50 000 train, test 10 000 | 600 | crop+flip+cutout (+aux head, drop-path) | accuracy final "SOTA" |

Evidencia interna (verificada en `data/qnas/experiment_v3_cifar10`, espacio Standard idéntico, 150 generaciones (300 en exp5 y exp19_r3),
reentreno F13, 3 repeticiones por red):

- Proxy del mejor individuo ≈ 79.1–81.2 % → test tras retrain 92.96 % (exp11_r1), 93.68 % (exp14_r1), 93.72 % (exp12_r1).
- Rango de las 42 redes Standard reentrenadas (14 configuraciones × 3 corridas): **90.1–93.9 %**, 0.5–3.5 M parámetros.
- MedMNIST (Q-NAS interno, `data/medmnist/qnas/experiment_*_2025`, F13): Path 89–93 %, OCT 76–79 %,
  Tissue 68–71 %, OrganA 95 %; todas por encima de ResNet-18 (28) de MedMNIST v2 en ACC con < 2 M parámetros
  (espacios algo más amplios que Standard, batch 128).

Además, los *smoke retrains* ya ejecutados en `experiment_cifar10_acc_flops/*/archive/*/retrain_parallel_1`
fueron de **5 épocas** (`--max_epochs 5`); no sirven como resultado, solo prueban que el lanzador funciona.

---

## 2. Qué experimentos reentrenar

Rutas relativas a `data/` en el Mac; en el servidor hay que mapearlas a la raíz de experimentos correspondiente
(verificar: en el servidor los legacy pueden no tener el mismo prefijo `moqnas/`/`nsga/`).

### 2.0a Carpeta de la etapa en el clúster (creada 2026-10-03)

`dualgpu1:~/MoQ-NAS/retrain_2026/` (ver su `README.md`): `protocol/` (este documento, `protocol_F13v1.yaml`, CSV de
candidatos), `tools/`, `runs/` (las 60 corridas con la misma ruta relativa que en `data/` del Mac: log, pareto y
`training_params.txt` de los 2 507 IDs; 45 MB copiados desde el Mac), `inventory/` y `logs/`.

Estado verificado en el clúster: 60/60 corridas y 2 507/2 507 IDs presentes. Las corridas que ya existían en
`~/MoQ-NAS` y `Experiments 2025` tienen `pareto_history.pkl` idéntico al del Mac. NSGA-II exp4 (Caso 1) y las 24
corridas NSGA-II/III de MedMNIST **no estaban en el clúster**; ahora están en `runs/`. Pendiente: arreglo 4.10 (las
51 corridas antiguas apuntan a `configs/<ds>.yaml`; los YAML correctos existen en `~/MoQ-NAS/dataset_configs/`).
Dataset de fairness completo: `datasets/personbin_data_96/train` = 93 594 imágenes, `val` (usado como test) = 3 908.

Ejecutar siempre desde `~/MoQ-NAS` con `--experiment_path retrain_2026/runs/<local_dir>`; las salidas quedan dentro
de `runs/` y la carpeta se descarga entera al Mac (`rsync -av --exclude '*.pth' --exclude '*.pt' ...`).

### 2.0 Inventario: qué tiene que existir en el clúster (paso 1)

60 corridas en total. Por cada corrida, `retrain_parallel.py` necesita:
- `<corrida>/log_params_evolution.txt` (secciones `train`, `QNAS`, `fn_dict`);
- `<corrida>/archive/<id>/training_params.txt` (con `net_list`) para cada ID a reentrenar;
- `<corrida>/pareto_history.pkl` (solo para reconstruir el frente o seleccionar sin `--ids`);
- el dataset en el clúster (CIFAR-10, los 4 MedMNIST, `personbin_data_96`) y su YAML en `dataset_configs/`.

| Tier | Ruta local (bajo `data/`) | Corridas | IDs en el CSV | Representantes | A reentrenar (reps + `strat10`) |
|---|---|---|---|---|---|
| A | `moqnas/experiment_cifar10_moqnas_v2/exp22_repeat_{1,2,3}` | 3 | 136 | 9 | 30 |
| A | `nsga/experiment_cifar10_nsga2/exp4_repeat_{1,2,3}` | 3 | 212 | 8 | 30 |
| A | `nsga/experiment_cifar10_nsga3/exp1_repeat_{1,2,3}` | 3 | 156 | 8 | 30 |
| B | `experiment_cifar10_acc_flops/moqnas/moqnas_repeat_{1,2,3}` | 3 | 39 | 7 | 30 |
| B | `experiment_cifar10_acc_flops/nsga2/nsga2_repeat_{1,2,3}` | 3 | 56 | 9 | 30 |
| B | `experiment_cifar10_acc_flops/nsga3/nsga3_repeat_{1,2,3}` | 3 | 52 | 9 | 30 |
| C | `medmnist/moqnas_last/experiment_{ds}_medmnist/moqnas/exp1_repeat_{1,2,3}` | 12 | 480 | 32 | 120 |
| C | `medmnist/NSGA/experiment_{ds}_nsga2/nsga2/exp1_repeat_{1,2,3}` | 12 | 619 | 34 | 120 |
| C | `medmnist/NSGA/experiment_{ds}_nsga3/nsga3/exp1_repeat_{1,2,3}` | 12 | 636 | 35 | 120 |
| D | `moqnas/experiment_personbin_qfamily/moqnas/exp{3,2}_repeat_{1,2,3}` | 6 | 121 | 9 | 121 (todo el frente) |

`{ds}` ∈ {pathmnist, octmnist, tissuemnist, organamnist}. "IDs en el CSV" incluye las filas `front` (frente
filtrado completo), que en A–C no se reentrenan; sirven por si después se amplía al frente entero.

**Verificación automática** (desde la raíz de MoQ-NAS en el clúster; puede recibir varias raíces):

```bash
python scripts/check_retrain_inventory.py --root . --root /ruta/a/resultados_antiguos
# -> retrain_inventory/runs_report.csv   (estado por corrida y ruta encontrada)
#    retrain_inventory/missing_ids.csv   (IDs sin training_params.txt)
#    retrain_inventory/upload_files.txt  (archivos a subir desde data/ del Mac)
```

Si falta algo, subir **solo los archivos necesarios** desde el Mac (no las carpetas completas):

```bash
cd "/Volumes/MacIA/Thesis/MoQ-NAS Experiments/data"
rsync -av --files-from=upload_files.txt ./ <usuario>@<cluster>:<raiz_resultados>/
```

Probado en local contra `data/`: resuelve las 60 corridas y los 2 507 IDs tienen `training_params.txt`. Además
detecta que **51 de las 60 corridas** referencian un YAML de dataset que ya no existe (`configs/<ds>.yaml`) o no
tienen `config_path_dataset` (NSGA del Caso 1) → ver arreglo 4.10. Solo las 9 corridas del Tier B están "OK".

### Tier A — Caso 1, CIFAR-10 tri-objetivo `[-Acc, Params, Time_CUDA]` (prioridad 1)

| Algoritmo | Ruta local | Frente final | Fallidos* | Únicos | ≥ 60 % proxy | Representantes |
|---|---|---|---|---|---|---|
| MoQ-NAS | `moqnas/experiment_cifar10_moqnas_v2/exp22_repeat_{1,2,3}` | 61/58/71 | 0 | 61/58/71 | 50/43/43 | 3/3/3 |
| NSGA-II | `nsga/experiment_cifar10_nsga2/exp4_repeat_{1,2,3}` | 139/123/148 | 44/15/37 | 86/91/98 | 66/72/74 | 3/2/3 |
| NSGA-III | `nsga/experiment_cifar10_nsga3/exp1_repeat_{1,2,3}` | 79/77/62 | 3/2/2 | 56/67/51 | 52/54/50 | 3/2/3 |

\* "Fallidos": registros del `pareto_history.pkl` legacy con objetivos = 0 (`accuracy: 0.0, params: 0.0`) y
carpeta de `archive/` vacía en la copia local. Son evaluaciones fallidas que quedaron "no dominadas" porque
params = 0 y tiempo = 0. **Verificar en el servidor** (`find archive -type d -empty | wc -l`). Si allí tampoco
tienen `training_params.txt`, no son reentrenables (y conviene confirmar que el análisis de HV del Caso 1 los excluye).

Representantes de MoQ-NAS (proxy):

| Corrida | `best_acc` | `knee` | `compact` |
|---|---|---|---|
| exp22_r1 | 130_14 — 77.3 %, 1.71 M | 11_2 — 71.8 %, 0.19 M | 98_1 — 73.5 %, 0.25 M |
| exp22_r2 | 66_5 — 77.4 %, 0.92 M | 142_13 — 67.8 %, 0.05 M | 139_15 — 72.8 %, 0.26 M |
| exp22_r3 | 79_19 — 78.8 %, 0.89 M | 86_16 — 72.9 %, 0.14 M | 127_14 — 75.6 %, 0.32 M |

(La lista completa para NSGA-II/III y los demás tiers está en el CSV.)

Reglas de selección usadas en el CSV (todas sobre el frente final, generación 150/149):
1. Descartar registros sin `training_params.txt` o con accuracy ≤ 0.
2. Deduplicar por arquitectura efectiva (`net_list` sin `no_op`), quedándose con el ID de mayor proxy.
3. Suelo de calidad: CIFAR-10 ≥ 60 % proxy; MedMNIST ≥ (máx. de la corrida − 10 pp). Debajo de eso son
   soluciones triviales (19–40 % de accuracy, < 50 k parámetros) que no aportan nada a una tabla de literatura.
4. `best_acc`: máxima accuracy proxy. `knee`: mínima distancia al ideal en el plano normalizado
   (accuracy, log10 params) — en el Tier B, log10 FLOPs. `compact`: el más pequeño con accuracy ≥ máx − 5 pp.
5. `strat10`: 10 redes por corrida repartidas uniformemente por rango a lo largo del eje de complejidad
   (incluye los representantes). Sirven para reconstruir un **frente reentrenado** por corrida.

Cantidad de trabajos de Tier A: 90 `strat10` × 1 semilla + 25 representantes × 2 semillas extra = **140 entrenamientos**.

#### Selección en dos etapas (incorporado 2026-10-03; reemplaza la selección de representantes por proxy)

Los roles `best_acc`/`knee`/`compact` del CSV están calculados con la accuracy **proxy**. Como el proxy puede
reordenar las redes, los representantes finales se eligen **después** del reentrenamiento:

1. **Screening (1 semilla):** reentrenar el conjunto de screening de cada corrida. Por defecto, todas las filas
   `front` + `strat10` + representantes (el frente filtrado completo: 504 redes en el Tier A y 147 en el Tier B).
   Si el presupuesto no alcanza, usar solo `strat10` (90 + 90). En cualquier caso, **el mismo criterio para los tres
   algoritmos**.
2. **Re-Pareto con la validación del retrain** (nunca con test): frentes en (error_val, params) para el Caso 1 y
   MedMNIST, y (error_val, MACs) para acc-FLOPs. Ningún caso usa la Time_CUDA de la búsqueda: no es reproducible
   porque la GPU no estaba aislada (§8.1), y tampoco sirve para comparar con otros trabajos (decisiones del
   2026-10-05). Cuantificar cuántas soluciones cambian de dominancia.
3. **Representantes por reglas fijadas antes de ver el test**, aplicadas al frente reentrenado, **iguales en los tres
   casos** (decisión del 2026-10-05):
   - `A`: menor error de validación;
   - `K`: knee = mínima distancia euclídea al ideal tras normalización min-max (lineal) dentro del frente (escribir
     la regla en el paper; no elegirlo visualmente);
   - `C` (compacto): la de menor complejidad (params; MACs en acc-FLOPs) entre las redes con accuracy de validación
     ≥ la de `A` − 5 pp.
   - **No se usan los extremos "menos params / menos MACs"** (antes `P`/`F`): son redes casi triviales (70–77 % de
     validación en CIFAR-10), y las redes reportadas deben ser útiles y competitivas frente a otras estrategias.
   - **por presupuesto**: la red de menor error de validación con params ≤ {0.25, 0.5, 1.0, 1.5} M (Tier A) y
     MACs ≤ {50, 100, 250, 500} M (Tier B). Estos presupuestos cubren el rango de los frentes de MoQ-NAS y coinciden con
     los de LightMix, LEMONADE y NSGANetV1 (0.2–1.8 M); no cambiarlos después de ver resultados.
   **Análisis que justifica la selección** (decisión 2026-10-05; lo genera el mismo script, solo con validación, en
   `<out>_report.md` y `<out>_analysis_{runs,rules}.csv`): (i) acuerdo proxy → validación (Spearman, Kendall τ_b, redes
   que quedan dominadas, si la mejor por proxy sigue siendo `A`, representantes por proxy que sobreviven);
   (ii) estabilidad ante el ruido de semilla: 500 repeticiones con N(0, 0.36 pp) en la validación de todas las redes,
   fracción en que cada regla elige la misma red y en que cada red sigue en el frente, más el margen sobre la segunda;
   tras la etapa 4 se repite con la sd medida en las semillas 11–13; (iii) sensibilidad a los parámetros fijados
   (margen de `C` de 3 y 7 pp, knee con escala logarítmica, frente con Time_CUDA donde la búsqueda la tenía).
   **Elecciones inestables (decisión 2026-10-05):** si una regla mantiene su red en < 70 % de las simulaciones, también
   se confirma la red que esa misma regla elige con más frecuencia entre las demás (etiqueta `<regla>~` en el CSV). Las
   dos se reportan; nunca se elige entre ellas con el test. Con MoQ-NAS añade 2 redes en el Caso 1 y 1 en acc-FLOPs; el
   resto de las alternativas ya eran representantes de otras reglas.
4. **Confirmación (3 semillas nuevas)** de los representantes: semillas 11, 12 y 13, distintas de la del
   screening (semilla 1), igual para los tres algoritmos. La corrida del screening no cuenta como réplica: los
   representantes se eligieron por su validación en esa semilla, y reutilizarla sesgaría la media hacia arriba (el
   mismo criterio que en R1 de fairness). Solo esta etapa alimenta la tabla del paper (media ± sd de test error).
   La semilla 1 basta en el screening porque la variación entre semillas de una misma red es pequeña (sd ≈
   0.05–0.36 pp en los F13 de Q-NAS) frente a las diferencias entre arquitecturas del frente (varios pp).

   ```bash
   # etapa 1 (screening, 1 semilla; en curso desde 2026-10-04, ver §4f)
   python launch_retrain_protocol.py --cases C1_triobj --seeds 1 --tag F13v1c --gpus 1 --jobs-per-gpu 3 --workers-per-job 2
   # pasos 2-3 (en el Mac, cuando el screening de las 9 corridas esté completo): re-Pareto con la validación
   # y representantes; se niega a escribir si falta alguna red (--partial solo para revisar antes)
   scripts/sync_retrain_results.sh                     # repo de análisis
   python scripts/select_retrain_representatives.py --case C1_triobj \
       --runs-root ../retrain_2026/cluster/dualgpu1/retrain_2026/runs --out retrain_matrices/confirm_C1_triobj_F13v1c.csv
   python scripts/select_retrain_representatives.py --case AF_std_biobj \
       --runs-root ../retrain_2026/cluster/dualgpu2/retrain_2026/runs --out retrain_matrices/confirm_AF_std_biobj_F13v1c.csv
   # paso 4 (en el servidor de cada caso, tras commitear los CSV): confirmación con 3 semillas nuevas, mismo tag
   python launch_retrain_protocol.py --candidates retrain_matrices/confirm_C1_triobj_F13v1c.csv \
       --cases C1_triobj --roles all --seeds 11 12 13 --tag F13v1c --gpus 1
   ```
   El script solo lee la accuracy de **validación** (`best_accuracy`) de la semilla 1, nunca la de test, y deja un
   informe `<out>_report.md` con el frente por corrida, cuántas redes quedan dominadas tras el retrain, el Spearman
   proxy–validación y la red elegida por cada regla. Una red elegida por varias reglas se confirma una sola vez
   (`role` = reglas unidas con `+`).

   **Decisión (2026-10-05, antes de ver el test):** el frente del Caso 1 es (error_val, params), sin la Time_CUDA de
   la búsqueda, porque esa latencia no es reproducible (§8.1). En la revisión con las 3 corridas de MoQ-NAS, frente
   a incluirla solo cambia el knee (K), en 2 de 3 corridas: con Time_CUDA, el K de `exp22_repeat_1` (53_10) quedaba en
   el frente únicamente por su latencia, aunque 11_2 lo supera en error y en params. Es el valor por defecto del
   script; `--objectives val_err params cuda_time` reproduce la variante con latencia.

Opcional (extensión de robustez, solo si el re-Pareto muestra mucha inversión de orden): añadir el rango 2 de no
dominancia de la última generación (`pareto_history.pkl[gen][2]`) al screening.

### Tier B — acc-FLOPs, espacio Standard `[-Acc, FLOPs]` (prioridad 1, igual que el Tier A)

`experiment_cifar10_acc_flops/{moqnas,nsga2,nsga3}/{algo}_repeat_{1,2,3}` (seeds 43–45). Frentes de 22–48
miembros; tras el filtro quedan 12–22 por corrida (147 en total). Mismo esquema: 90 `strat10` + 25 representantes
→ **140 entrenamientos**. Los `best_acc` coinciden con los IDs que ya se usaron en el smoke retrain (127_5, 123_5,
103_18, 107_18, 65_18, 112_1, 114_2, 29_13, 98_5), así que la selección es consistente con lo que ya habías elegido.

Por qué vale la pena aunque en el paper sea "solo" la ablación de espacio: **es el único experimento cuyo vector de
objetivos coincide con NSGA-Net y NSGANetV1 (error + FLOPs)**, que son los dos comparadores multiobjetivo
evolutivos más citados.

### Tier C — Caso 2, MedMNIST `[-Acc, Params, Time_CUDA]` (comparación con MedMNIST v2)

`medmnist/moqnas_last/experiment_{ds}_medmnist/moqnas/exp1_repeat_{1,2,3}` y
`medmnist/NSGA/experiment_{ds}_{nsga2,nsga3}/{algo}/exp1_repeat_{1,2,3}` para
`ds ∈ {pathmnist, octmnist, tissuemnist, organamnist}`. Representantes: 23–26 por dataset (≈101 en total).

**Plan revisado (2026-10-05, sustituye al del 2026-10-04): protocolo principal = el de MedMNIST v2, mismas etapas y
reglas que el Caso 1.** Los comparadores de MedMNIST (ResNet-18/50, auto-sklearn, AutoKeras, Google AutoML Vision) se
entrenaron con el protocolo del benchmark, así que la comparación más justa es usar ese mismo protocolo en todo el caso,
desde el screening (la selección y la confirmación deben hacerse con el mismo protocolo).
- **Protocolo P-Med** (perfil `medmnist_v2_adamw`, tag `PMedW`; decisión del usuario 2026-10-05): el esquema de las
  ResNet de MedMNIST v2 (100 épocas, ×0.1 en las épocas 50 y 75, batch 128, **sin augmentation**, checkpoint por mejor
  validación, splits y evaluador oficiales, ACC y AUC) con **AdamW** (lr 1e-3, weight decay 0.01, el de F13-v1), el
  optimizador de todos los demás casos y de tus trabajos previos de MedMNIST. Diferencias declaradas con MedMNIST v2:
  AdamW en lugar de Adam, y fp16. El perfil `medmnist_v2` (Adam exacto) queda solo como referencia, sin uso.
1. **Screening `strat5`, semilla 1, P-Med:** 5 redes por corrida (columna booleana `strat5` del CSV) → 36 corridas
   × 5 = **180 redes**.
   `python launch_retrain_protocol.py --cases C2_medmnist --roles strat5 --profile medmnist_v2_adamw --seeds 1 --tag PMedW --gpus 1`
2. **Re-Pareto y representantes** con `scripts/select_retrain_representatives.py --case C2_medmnist --tag PMedW`:
   frente (error_val, params), reglas `A`, `K` y `C` (sin presupuestos: los comparadores son arquitecturas fijas).
3. **Confirmación con las semillas 11, 12 y 13, P-Med**, igual que en el Caso 1 (decisión 2026-10-05). Media ± sd
   sobre semillas por red, y sobre las 3 corridas de búsqueda por algoritmo. MedMNIST v2 reporta una corrida por método.
4. **Variante opcional con augmentation**, solo si el tiempo lo permite (perfil `medmnist_v2_adamw_ta`, tag
   `PMedW_TA`): los mismos representantes y semillas, más TrivialAugmentWide, la augmentation de los F13 de
   MedMNIST. Sin flips ni cutout, por las simetrías de cada dataset (tabla del 2026-10-04, ahora en el paper):

   | Dataset | Contenido | Flip horizontal | Flip vertical | Crop con padding | Cutout |
   |---|---|---|---|---|---|
   | PathMNIST | parches de histología de colon | válido | válido | válido | aceptable, más pequeño (~8 px) |
   | TissueMNIST | células de corteza renal | válido | válido | válido | aceptable, más pequeño |
   | OCTMNIST | cortes de retina por OCT | probablemente válido | no (orientación de las capas) | desplazamiento pequeño | arriesgado (lesiones pequeñas) |
   | OrganAMNIST | cortes axiales de CT abdominal | no (cambia izquierda/derecha) | no | desplazamiento pequeño | arriesgado |

   Se reporta en filas separadas y nunca se mezcla con P-Med. Una política por dataset (flips permitidos, crop
   pequeño) necesitaría su propio piloto en los 4 datasets; no se plantea ahora.

Costo: 100 épocas en lugar de 300, así que cada entrenamiento cuesta ≈ 1/3 que en F13-v1; estimarlo con el ritmo
medido en el screening antes de lanzar la confirmación.
Limitación a declarar: en **OCTMNIST la validación y el test difieren mucho** (en los F13 previos de Q-NAS, val ≈ 95 %
frente a test ≈ 77 %). Elegir con la validación del reentrenamiento es mejor que con el proxy, pero en OCT esa elección
se traslada peor al test.
- Orden sugerido: OrganA y Path primero (más pequeños y rápidos), TissueMNIST al final (165 k imágenes de train).
- Ampliación opcional, solo si sobra tiempo: screening `strat10` (360 redes) en lugar de `strat5` (≈ +6.5 días).

### Tier D — Caso 3, fairness: frente completo × 3 semillas en dos regímenes (decisión 2026-10-03)

**Decisión:** reentrenar **todas** las soluciones del frente final con **3 semillas** y los 5 baselines desde cero
con 3 semillas, y mantener **dos comparaciones**:

| Régimen | Datos | Épocas | Qué responde |
|---|---|---|---|
| R1 — presupuesto de búsqueda | 10 000 imágenes (`limit_data_value: 10000`), 96×96 | 50 | La comparación actual del paper, ahora con 3 semillas por solución (antes: 1 evaluación de búsqueda frente a 3 semillas de baseline). |
| R2 — datos completos | `personbin_data_96/train` completo, 96×96 | a fijar en el piloto (≥ 50) | Comparación en régimen final, como en los Casos 1 y 2; permite ver si el orden de fairness de R1 se mantiene. |

Alcance: frentes finales `moqnas/experiment_personbin_qfamily/moqnas/exp3_repeat_{1,2,3}` (2 obj.: 7/10/7 únicos)
y `exp2_repeat_{1,2,3}` (3 obj.: 31/27/39 únicos; 2 por corrida sin `training_params.txt`, verificar en el servidor)
→ **≈ 115 redes únicas**. Baselines: convnext_tiny, efficientnet_v2_s, mobilenet_v3_large, resnet18, resnet50.

| Bloque | R1 | R2 |
|---|---|---|
| Frente MoQ-NAS (≈ 115 × 3 semillas) | ≈ 345 | ≈ 345 |
| Baselines desde cero (5 × 3 semillas) | 15 (ya existen; **rehacer** si se unifica el protocolo, ver abajo) | 15 |
| Total | ≈ 360 | ≈ 360 |

**Cómo se hacen las 3 semillas de R1** (decisión 2026-10-03):
- 3 entrenamientos **nuevos** por solución (semillas 1, 2, 3), para todas las ≈ 115 soluciones únicas del frente.
  La evaluación de la búsqueda **no** cuenta como semilla: las soluciones están en el frente *porque* esa
  evaluación salió buena (sesgo de selección / "winner's curse"), así que incluirla infla la media de MoQ-NAS frente
  a unos baselines que no pasaron por ninguna selección.
- Protocolo R1 = protocolo de la búsqueda: `--limit_data` con `limit_data_value: 10000`, `split_seed: 2025`
  (mismo subconjunto 9 000/1 000 que en la búsqueda, el muestreo es determinista), 50 épocas, batch 64, AdamW 1e-3,
  wd 1e-4, sin scheduler (`--lr_scheduler None`), bf16, clipping, `--patience_retrain 50`, `--keep_metrics`.
- Los 5 baselines se **reentrenan con ese mismo protocolo** y las mismas semillas (5 × 3 = 15).
- La semilla varía la inicialización y el orden de los batches; el subconjunto de datos queda fijo.
- Accuracy: el retrain guarda `validation_accuracies` de todas las épocas → calcular a posteriori las dos
  definiciones, media de las últimas 5 épocas (la de la búsqueda) y mejor val acc.
- La evaluación de la búsqueda se reporta aparte como "valor de búsqueda"; la diferencia búsqueda − media de las 3
  semillas cuantifica el optimismo del proxy en accuracy, D_group y MeanTPR.
- Alternativa más barata (no recomendada; ahorra ≈ 48 GPU-h): reutilizar la evaluación de la búsqueda como
  semilla 1 + 2 réplicas. Solo es aceptable si las réplicas y los baselines usan exactamente el protocolo de la
  búsqueda y el texto declara que una de las réplicas fue la usada para seleccionar.

#### Mismo protocolo para MoQ-NAS y baselines (requisito para que sea justo)

Hoy los dos grupos **no** se entrenan igual (verificado):

| | Búsqueda MoQ-NAS (`trainer.py`) | Baselines (`scripts/fairness_baseline/train.py`) |
|---|---|---|
| Batch | 64 | 128 |
| Optimizador | AdamW 1e-3, wd 1e-4 | AdamW 1e-3, wd 1e-4 |
| Scheduler | ninguno (en fase de búsqueda) | ninguno |
| Precisión / clipping | **fp16** (`mixed_precision: true`), sin clipping (búsqueda dic-2025 a feb-2026, antes del clipping de mayo de 2026) | **fp32** (sin autocast ni GradScaler), sin clipping |
| Valor de accuracy reportado | media de las últimas 5 épocas | mejor val acc |
| Checkpoint | mejor val acc | mejor val acc |

Para R1 y R2, fijar **un único protocolo** (batch, LR, wd, scheduler, épocas, precisión, clipping, augmentation del
branch person, split `split_seed: 2025`, `train_split: 0.9`, checkpoint por mejor val acc de COCO) y aplicarlo a
ambos grupos. Opciones, de más a menos limpia:
1. Entrenar los baselines con el mismo `BaseTrainer` (añadir los 5 modelos torchvision como `network_config`
   o una fase tipo `resnet`), así comparten el bucle, la precisión y el clipping.
2. Mantener `train.py` pero alinear todos los hiperparámetros de la tabla y documentar la diferencia de código.

**Precisión (decisión 2026-10-03): fp16 para los dos grupos y en ambos regímenes (R1 y R2)**, la misma que usó la
búsqueda de MoQ-NAS en fairness. Los baselines existentes se entrenaron en fp32, así que se **reentrenan en fp16**
(autocast fp16 + GradScaler). Con la opción 1 viene dado por el trainer; con la opción 2 hay que añadir autocast y
GradScaler a `train.py`. Vigilar NaN/overflow en ConvNeXt-T y EfficientNetV2-S en fp16: si aparecen, registrarlos
como `FAILED_NAN` (arreglo 4.11) y decidir una regla común, en lugar de cambiar la precisión solo para ese modelo.

Las semillas deben ser las mismas para ambos grupos (p. ej. 1, 2, 3), variando la inicialización y el orden de los
batches, con el split fijo (como ya hace `train.py`, líneas 95–96). Requiere el arreglo 4.3 en `retrain_parallel.py`.

#### Evaluación y análisis

- Checkpoint elegido **solo con la validación de COCO**; FACET se usa únicamente para medir accuracy, D_group y
  MeanTPR (nunca para elegir checkpoint, solución ni representante).
- Por solución: media ± sd sobre 3 semillas de accuracy, D_group y MeanTPR.
- A nivel de frente: frente de las medias en el espacio común [−Acc, D_group, −MeanTPR] y, además, un frente por
  semilla → HV_rel por semilla (distribución, no un único valor) para cada formulación (2 obj. vs. 3 obj.).
- Contra baselines: para cada baseline, cuántas soluciones de MoQ-NAS lo dominan **en media** y cuántas lo dominan
  **en las 3 semillas emparejadas**; con n = 3, tratarlo como evidencia descriptiva (como ya hace el paper).
- Entre regímenes: Kendall τ entre R1 y R2 para cada objetivo, y qué fracción del frente de R1 sigue siendo no
  dominada en R2.
- Representantes (balanced / fairness-prioritized / best-acc): elegirlos **después**, con la regla del paper
  (`W_BAL`, `W_FAIR` de `scripts/case3_revision_analysis.py`) aplicada a las medias de R2, o mantener los actuales
  y decirlo explícitamente.

#### R2 reducido (opción recomendada en una GPU)

R2 completo (todo el frente × 3 semillas con datos completos) cuesta 30–89 días en una GPU: no es viable. **R2
reducido** responde la pregunta a nivel de arquitectura con un costo acotado:
- **Qué se entrena:** los **9 representantes del Caso 3 del paper** (Tabla VI; `reports/case3_revision/05_case3_representatives.csv`):
  best_accuracy, lowest_dgroup, best_mean_tpr, balanced y fairness_prioritized de cada formulación. En la de
  2 objetivos, 20_16 es a la vez balanced y fairness_prioritized. Más los **5 baselines desde cero** (resnet18,
  resnet50, efficientnet_v2_s, convnext_tiny, mobilenet_v3_large).
- **Datos:** `personbin_data_96` completo (≈ 84k train / 9.4k val; test = val de COCO, 3 908) y FACET completo para
  fairness, a 96 px y fp16.
- **Protocolo (perfil `fairness_R2`):** optimización de F13-v1 (AdamW lr 1e-3, wd 0.01, multistep al 50/75 %), fp16, sin
  clipping, batch 64, augmentation del branch person (RandomResizedCrop + flip + TrivialAugment), checkpoint por
  validación de COCO. **Mismo protocolo para los baselines** (`run_fairness_baseline.sh` con `RUN_TAG=fairR2r
  LIMIT_DATA_VALUE= SCRATCH_EPOCHS=<E> LR_SCHEDULER=multistep WEIGHT_DECAY=0.01`; fp16 y batch 64 por defecto).
- **Semillas 1, 2, 3** en ambos grupos → 14 arquitecturas × 3 = **42 entrenamientos**.
- **Épocas:** fijarlas con un mini-piloto (el representante más pesado, 47_4 con 7.1 GFLOPs, y resnet50, ~150 épocas,
  mirando dónde se estabiliza la validación; ≈ 0.5 días). Costo de R2 reducido en una GPU (limitado por la GPU):
  **100 épocas ≈ 2.5 días**, 200 ≈ 5.1, 300 ≈ 7.6.
- Comando de los representantes:
  `python launch_retrain_protocol.py --cases C3_fairness_two C3_fairness_three --roles best_accuracy lowest_dgroup best_mean_tpr balanced fairness_prioritized --profile fairness_R2 --max-epochs <E> --seeds 1 2 3 --tag fairR2r --gpus 1`
- **Qué responde:** si, con datos completos, los representantes mantienen su posición frente a los baselines en
  accuracy, D_group y MeanTPR (afirmaciones por arquitectura, con media ± sd de 3 semillas).
- **Qué no responde:** el análisis a nivel de frente con datos completos (HV en el espacio común y estabilidad del
  orden R1 → R2). Eso queda solo en R1, que sí reentrena todo el frente.
- **Limitación a declarar:** estos representantes se eligieron en el paper con las métricas de FACET de la búsqueda,
  y FACET es también el conjunto de evaluación. R2 entrena modelos nuevos e independientes, lo que reduce el
  optimismo, pero no lo elimina. Una alternativa más estricta sería partir FACET en una mitad de selección y otra de
  evaluación, estratificadas por tono de piel; cambiaría los representantes del paper, así que no se recomienda ahora.

#### Coste y orden

- Referencia: un candidato de la búsqueda (10 k imágenes, 50 épocas) tardó ≈ 23 min, más la evaluación en FACET
  (29 945 recortes). R1 ≈ 345 × ~25 min ≈ 145 GPU-h para el frente. R2 escala ≈ (N / 9 000) × (E / 50) por red; N
  sale del log del loader (`Train loader: … samples`). Los baselines grandes (ConvNeXt-T, ResNet-50,
  EfficientNetV2-S) son lo más caro.
- Orden: smoke test (1–2 épocas, comprobar métricas de fairness y arreglo 4.9) → R1 (barato, valida el pipeline)
  → piloto de R2 con 5 redes + 1 baseline para fijar épocas y coste → R2 completo.
- **Antes de lanzar, corregir el fallo 4.9** (si no, las métricas de fairness del retrain salen de los pesos de la
  última época) y lanzar con `--keep_metrics`.

### Qué NO reentrenar

| Experimento | Motivo |
|---|---|
| MOEA/D (`acc_flops`, `acc_flops_mixed`) | Frentes de 260–1 400 registros, mayoría sin archivo y accuracy ≤ 65 %; no está en las tablas del paper. |
| `acc_flops_mixed`, `acc_flops_random_elite`, `acc_flops_std_relite_noop10` | Ablaciones internas; no hay comparador externo. |
| Ablaciones τ_u / donor del Caso 4 | Se comparan con HV proxy entre configuraciones de MoQ-NAS; reentrenar no cambia la conclusión. |

---

## 3. Protocolos de reentrenamiento

### P1 — Protocolo principal: F13 tal como se ejecutó (decisión 2026-10-03)

**Decisión:** usar el protocolo F13 *tal como se ejecutó*, reconstruido a partir del `training_params.txt` de los
retrains y del código en git de la fecha de cada retrain (no a partir de la descripción de la tesis).

**Hay dos variantes de F13 en tus datos.** Las separa la refactorización del 20/21-09-2025, que cambió la creación
del optimizador (commit `a345b47`) y el loader (commit `46162b0`):

| | **F13-v1** (hasta 10-09-2025) | F13-v2 (07-10 a 21-11-2025) |
|---|---|---|
| Retrains | Q-NAS CIFAR-10 exp0–exp15 (incluye las configuraciones Standard exp5, exp7–15) y **todos** los MedMNIST Q-NAS (`data/medmnist/qnas/*_2025`) | Q-NAS CIFAR-10 exp16–exp20 y exp23 |
| AdamW | `torch.optim.AdamW(net.parameters())` → valores por defecto de PyTorch: **lr 1e-3, weight_decay 0.01** (el `weight_decay: 1e-4` del log **no se usaba**) | lr 1e-3, **weight_decay 1e-4** (leído de la config) |
| Split de validación | `StratifiedShuffleSplit` sin semilla (semilla = `time()`) | estratificado con `split_seed` |

Común a ambas (verificado en código y logs):

```yaml
max_epochs: 300
epochs_to_eval: 300            # en retrain se evalúa en todas las épocas
patience_retrain: 300          # = max_epochs → sin early stopping (en MedMNIST la clave no existe → default = max_epochs)
lr_scheduler: multistep        # MultiStepLR, milestones 50 % y 75 % (épocas 150 y 225), gamma 0.1
optimizer: AdamW               # lr 1e-3; weight_decay según variante (arriba)
batch_size: 256                # MedMNIST: 128
eval_batch_size: 256           # MedMNIST: 128
precision: por caso            # la de la búsqueda de cada caso (ver abajo); F13 de Q-NAS usó fp16
gradient_clipping: none        # el clipping (max_norm 1.0) se añadió el 04-05-2026, después de todos los F13
data_augmentation: true        # TrivialAugmentWide(31) + ToTensor + Normalize; sin crop ni flip
train_split: 0.9               # CIFAR-10: 45 000 / 5 000, test oficial de 10 000; MedMNIST: splits oficiales
checkpoint: mejor val accuracy # el test se evalúa con ese checkpoint
reported_val: max              # el viejo trainer guardaba la mejor val accuracy
num_repetitions: 3             # por red
num_workers: 8
```

**Alcance (decisión 2026-10-03): Q-NAS no se reentrena.** El protocolo se aplica solo a los métodos
multiobjetivo: MoQ-NAS, NSGA-II y NSGA-III. Los F13 de Q-NAS que ya existen solo sirven para dos cosas: reconstruir
qué protocolo se ejecutó realmente y, si se quiere, aparecer como filas de referencia ya calculadas.

**Variante que se adopta: F13-v1.** Es la variante con la que se hizo la mayoría de los F13: 11 de las 14
configuraciones del espacio Standard y todos los de MedMNIST. Además, es la única con un regularizador explícito y
estándar (AdamW con weight decay 0.01). La única diferencia práctica con F13-v2 es el `weight_decay` (0.01 frente a
1e-4); en el espacio Standard los resultados de ambas variantes están en el mismo rango (v1: 90.1–93.9 %; v2:
92.1–93.5 %). Si alguna vez se muestran filas de referencia de Q-NAS, usar solo los exp de F13-v1, para que
compartan protocolo con lo reentrenado.

**Qué hace el protocolo robusto y justo** (esto es lo que hay que cumplir):
1. **Un solo protocolo para los tres algoritmos** y para todos los casos de CIFAR-10. MedMNIST cambia solo el
   batch, que pasa a 128.
2. **Ningún hiperparámetro heredado del log de búsqueda.** Los logs difieren entre experimentos: las corridas
   antiguas tienen `mixed_precision: true` y no tienen `eval_window_agg`, mientras que las de acc-FLOPs tienen
   `precision: bf16` y `eval_window_agg: mean`; además, las rutas de los YAML de dataset son distintas. Si el retrain
   heredara del log, MoQ-NAS del Caso 1 y del Tier B se entrenarían con precisiones distintas. Todos los valores
   salen de un único archivo de protocolo, `retrain_matrices/protocol_F13v1.yaml`, y se pasan explícitos.
3. **Mismas semillas (1, 2, 3) y mismo split** para todas las redes. El split se hace **con semilla** (`split_seed`
   fijo), aunque F13-v1 no la tenía: la proporción y la estratificación son las mismas y así el retrain es
   reproducible. Es la única desviación deliberada respecto a F13-v1; declararla.
4. **Misma regla de checkpoint** (mejor val accuracy) y el test solo se mira al final; los representantes se eligen
   con la validación (selección en dos etapas, §2).
5. **Fallos registrados** (arreglo 4.11): ninguna red se descarta en silencio, y no se cambia el batch de una red
   sin documentarlo.

**Cómo reproducir F13-v1 con el código actual** (todo como override explícito, nunca heredado del log de búsqueda):

| Parámetro | El código actual haría | Para F13-v1 | Cambio necesario |
|---|---|---|---|
| `weight_decay` | 1e-4 (del log) | **0.01** | añadir `--weight_decay` (y `--learning_rate`) a `retrain_parallel.py` |
| precisión | la del log (bf16 o fp16 según la corrida) | **la de la búsqueda de cada caso** (tabla de abajo) | añadir `--precision` como override explícito |
| gradient clipping | siempre `max_norm=1.0` (`trainer.py:199/204`) | **desactivado** | parámetro `grad_clip_norm` (default `None` en retrain) |
| early stopping | `patience_retrain` de la matriz (50 en `cifar_mo.yaml`) | **300** | pasar `--patience_retrain 300` |
| val reportada | `eval_window_agg` heredado (`mean` en corridas nuevas) | **max** | arreglo 4.2 |
| `train_split` | 0.9 por defecto | 0.9 | pasarlo explícito (`--train_split 0.9`) |
| primer paso del scheduler | se salta en la primera época de validación (`trainer.py:335`) | igual (el viejo trainer también lo hacía) | **no** corregir 4.8 para este protocolo |
| augmentation | TrivialAugmentWide + Normalize | igual | ninguno |

**Precisión por caso (decisión 2026-10-03).** Cada caso se reentrena con la precisión que usó su búsqueda, que es la
misma para los tres algoritmos dentro del caso (verificado en todos los `log_params_evolution.txt`):

| Caso | Búsqueda | Retrain |
|---|---|---|
| Caso 1, CIFAR-10 tri-objetivo (MoQ-NAS exp22, NSGA-II exp4, NSGA-III exp1) | `mixed_precision: true` | **fp16** (autocast + GradScaler) |
| Tier B, acc-FLOPs Standard (los tres algoritmos) | `precision: bf16` | **bf16** (sin GradScaler) |
| Caso 2, MedMNIST (4 datasets × 3 algoritmos) | `mixed_precision: true` | **fp16** |
| Caso 3, fairness (MoQ-NAS) | `mixed_precision: true` | **fp16** |
| Caso 3, baselines fijos | entrenados en fp32 | **fp16** (reentrenar, igual que la búsqueda de MoQ-NAS) |

Las comparaciones son siempre dentro de un caso, así que esto mantiene la igualdad entre algoritmos. Lo que no debe
hacerse es comparar números de distintos casos como si compartieran precisión.

**La tesis describe otro protocolo.** La Tabla 5.8 dice "40k train; 10k val; 10k test", "FP32" y no menciona
el weight decay. Lo que se ejecutó en F13 fue 45k/5k/10k, FP16 y AdamW con weight decay 0.01 (v1). Corregir la
tesis y describir en el paper exactamente el F13-v1 de la tabla de arriba.

**Piloto opcional de optimizador, antes del screening.** Tres arquitecturas fijadas a priori (menos params, mayor
accuracy proxy y knee proxy de MoQ-NAS exp22), mismas semillas y augmentation: F13-v1 frente a `SGD + cosine`
(lr 0.1, el que adoptó el Q-NAS de 2022 por estabilidad). Si la diferencia es marginal, mantener F13-v1. Si SGD gana
de forma clara y consistente en las tres, fijarlo para **todos** los algoritmos antes del screening y documentar el
piloto. Nunca elegir optimizador por arquitectura.

Reportar: test accuracy y **test error = 100 − test accuracy**, parámetros, **MACs = total_flops / 2** (ver §6.4),
media ± sd sobre semillas y sobre corridas.

### P-Med — MedMNIST

**Principal: F13-v1 MedMNIST** (el que produjo tus resultados Q-NAS en MedMNIST [29], [30]): igual que P1 pero con
batch y eval batch de 128, splits oficiales de MedMNIST y TrivialAugmentWide + Normalize. Así la comparación con
NSGA-II/III y con tu Q-NAS previo comparte protocolo.

Opcional, solo como fila de referencia frente a los baselines de MedMNIST v2: **P-Med-A** = el protocolo de
MedMNIST v2 (100 épocas, Adam lr 1e-3, ×0.1 en las épocas 50 y 75, batch 128, sin augmentation). Las filas de
MedMNIST v2 son mono-objetivo y de referencia, así que no hace falta igualarles el protocolo; si se usa P-Med-A,
reportarlo en filas separadas.

`MedMNIST_Metrics` ya se añade automáticamente en fase `retrain` y produce `auc_score` y `acc_medmnist` sobre el
test oficial (verificado en retrains previos de `data/medmnist/qnas`).

### P2 — Protocolo "estilo literatura" (opcional, solo representantes `best_acc`)

Solo si se quiere una fila de "techo" comparable con NSGA-Net/LEMONADE/NSGANetV1: crop 32 con padding 4 + flip
horizontal + cutout 16, SGD momentum 0.9 (nesterov), lr 0.025–0.05 cosine, weight decay 3e-4–5e-4, 600 épocas,
batch 96–128. Requiere cambios de código (§4.5). **Reportarlo siempre en filas separadas y etiquetado como P2**;
nunca mezclar P1 y P2 en la misma comparación.

---

## 4. Cambios en el código antes de lanzar

### Estado (2026-10-03): aplicados en la rama `retrain-2026` del repo MoQ-NAS

| Arreglo | Estado | Dónde |
|---|---|---|
| 4.1 early stopping | aplicado: el protocolo pasa `patience_retrain = max_epochs`; la matriz antigua `cifar_mo.yaml` también corregida | `launch_retrain_protocol.py`, `retrain_matrices/cifar_mo.yaml` |
| 4.2 val reportada | aplicado: `--eval_window_agg` (por defecto `max` en retrain) | `retrain_parallel.py` |
| 4.3 semillas | aplicado: `--seeds 1 2 3`; semilla global (Python/NumPy/Torch) y `loader_seed` por semilla, loaders nuevos por repetición; carpetas `retrain_<tag>_s<seed>`; resultados fusionados por candidato **y** semilla (las semillas nuevas no pisan las anteriores) | `retrain_parallel.py` |
| 4.4 selección legacy / por corrida | aplicado: el lanzador nuevo usa los IDs del CSV por corrida; `launch_retrain.py` traduce las claves antiguas (`accuracy`/`params`/`inference_time`) y descarta registros a cero | `launch_retrain_protocol.py`, `launch_retrain.py` |
| 4.5 crop/flip/cutout | **no aplicado** (cambia el protocolo; solo para P2) | — |
| 4.6 throughput | aplicado: `persistent_workers` + `prefetch_factor=4` solo en retrain (la búsqueda no cambia); `--workers-per-job` y `--jobs-per-gpu` en el lanzador | `core/cnn/input.py`, `launch_retrain_protocol.py` |
| 4.7 / P1 overrides | aplicado: `--learning_rate`, `--weight_decay`, `--precision`, `--grad_clip_norm none`, `--train_split`, `--split_seed`, `--limit_data_value` | `retrain_parallel.py`, `core/cnn/trainer.py` |
| 4.8 desfase del scheduler | **no corregido a propósito** (F13-v1 lo tenía); `train.py` de baselines lo imita | `scripts/fairness_baseline/train.py` |
| clipping | configurable (`grad_clip_norm`, por defecto 1.0 → la búsqueda no cambia; `None` lo desactiva) | `core/cnn/trainer.py` |
| 4.9 fairness con el modelo viejo | aplicado: tras recargar el mejor checkpoint, las métricas de post-procesado apuntan al modelo nuevo | `core/cnn/trainer.py` |
| FACET en retrain | aplicado: en retrain, `FairnessMetric` recibe `img_size` = resolución de entrenamiento (96) y la precisión del caso, igual que en la búsqueda (`core/evaluation.py`). Antes, en la ruta de retrain, caía a 224 y fp32 | `core/cnn/master.py` |
| 4.10 YAML de dataset | aplicado: `--config_path_dataset` (el protocolo lo pasa siempre) | `retrain_parallel.py`, `protocol_F13v1.yaml` |
| 4.11 fallos y registro | aplicado: estados `FAILED_OOM/NAN/EXCEPTION/NO_TEST/LOAD/WORKER` en `retrain_failures_<tag>.csv` y en los resultados; código de salida ≠ 0 si hay fallos; entorno (git, torch, CUDA, GPU, host) por resultado; `last.pt`; LR, tiempo por época, `best_acc_epoch` y `epochs_run` en los resultados; manifiesto de lanzamiento | `retrain_parallel.py`, `core/cnn/trainer.py`, `launch_retrain_protocol.py` |
| 4.12 texto del paper | pendiente (no es código) | — |
| Baselines de fairness | aplicado: `train.py` con `--precision` (autocast + GradScaler), `--weight_decay`, `--lr_scheduler`; `evaluate.py` con `--precision`; `run_fairness_baseline.sh` parametrizado (R1 por defecto: fp16, batch 64, wd 1e-4, sin scheduler, 10k) y escribe en `<dir>_<RUN_TAG>`, sin pisar los fp32 originales | `scripts/fairness_baseline/*`, `run_fairness_baseline.sh` |

Verificación hecha: compilación de todos los archivos y `--dry-run` del lanzador para los cuatro casos (precisión
fp16 / bf16 / fp16 / fp16 según el caso; fairness R2 exige `--max-epochs`). **Falta la prueba real** (smoke test en el
clúster), porque en el Mac no hay torch.

Comprobación sobre la evaluación de fairness del paper (2026-10-03): en la búsqueda, `EvalPopulation` inyectaba
`img_size = 96`, y los baselines se evaluaron con `evaluate.py --img_size 96`. La comparación del Caso 3 del paper
es simétrica en resolución de FACET. El problema de 224 existía solo en la ruta de retrain y ya está corregido.

Uso del lanzador nuevo (desde `~/MoQ-NAS`):
```bash
# ver comandos sin ejecutar
python launch_retrain_protocol.py --cases C1_triobj --roles best_acc knee compact strat10 front \
    --seeds 1 --tag F13v1 --gpus 0 1 --dry-run
# fairness R1, todo el frente, 3 semillas
python launch_retrain_protocol.py --cases C3_fairness_two C3_fairness_three --roles all \
    --profile fairness_R1 --seeds 1 2 3 --tag fairR1 --gpus 0 1
# baselines R1 con el mismo protocolo
bash run_fairness_baseline.sh --from_scratch
```

Ordenados por impacto. Los 1–4 son necesarios; 5–8 son mejoras.

### 4.1 Early stopping corta el entrenamiento (CRÍTICO)
- `retrain_matrices/cifar_mo.yaml:30` → `patience_retrain: 50` con `lr_scheduler: cosine` y `max_epochs: 200`.
- `core/cnn/trainer.py:293` y `:343`: el early stopping en retrain monitoriza **val loss** con mejora relativa
  mínima de 0.5 % (`delta_fraction`). Con AdamW, la val loss en CIFAR suele dejar de bajar (o sube por
  sobreajuste) mucho antes de que la LR decaiga, aunque la accuracy siga mejorando → se para a mitad de la
  curva cosine/multistep y se pierden puntos.
- **Acción:** `patience_retrain = max_epochs` (como F13). Si se quiere early stopping, monitorizar val accuracy
  y solo después del último escalón de LR.

### 4.2 `eval_window_agg: mean` heredado hace que la "val accuracy" del retrain no tenga sentido
- `core/cnn/trainer.py:363`: `best_accuracy` se agrega según `eval_window_agg`, que el retrain hereda de la
  config de búsqueda (`mean`). En retrain `should_evaluate` evalúa en todas las épocas, así que
  `best_accuracy` = **media de la val accuracy de todas las épocas**.
  Evidencia: en el smoke retrain, 40_5 tiene val 68.8 % y test 77.6 %; el mismo patrón (val ≈ test − 8 pp) en todos.
- La test accuracy **no** está afectada (usa el checkpoint de mejor val acc), pero cualquier número de validación
  del retrain que se reporte sí.
- **Acción:** forzar `params['eval_window_agg'] = 'max'` en `retrain_parallel.py` (añadirlo a `override_keys`,
  `retrain_parallel.py:124`, con default `max`) o en el worker cuando `phase == 'retrain'`.

### 4.3 Las repeticiones no tienen semilla
- `retrain_parallel.py:147`: el bucle de repeticiones no fija semillas; la DataLoader usa siempre el mismo
  `loader_seed` (777) y la inicialización del modelo depende del RNG global sin sembrar. Las "3 repeticiones" no son
  reproducibles ni están documentadas.
- **Acción:** añadir `--seed` y, por repetición, `torch.manual_seed(seed + rep)`, `numpy`/`random` y
  `params['loader_seed'] = seed + rep`; guardar la semilla en `retrain_results_parallel.txt`.

### 4.4 Selección de candidatos para corridas legacy y por corrida
- `launch_retrain.py:246-248`: `min_accuracy` y `weights` buscan la clave `best_accuracy`, pero los
  `pareto_history.pkl` legacy de NSGA-II/III del Caso 1 usan `accuracy`/`params`/`inference_time`
  (y contienen registros a cero). Con esas corridas, `min_accuracy` descarta todo y `weighted` falla.
- La regla `type: ids` se aplica igual a todas las repeticiones de una entrada, y los IDs difieren por corrida.
- **Acción:** usar los IDs del CSV llamando a `retrain_parallel.py` directamente por corrida (abajo), o añadir a
  `launch_retrain.py` una regla `type: csv` que lea `literature_retrain_candidates.csv` filtrando por
  `local_dir`/`role`.

```bash
# Ejemplo (una corrida, representantes con 3 semillas):
python retrain_parallel.py --experiment_path <exp_root>/exp22_repeat_1 \
  --ids 130_14 11_2 98_1 --data_path datasets/cifar10_data --dataset cifar10 \
  --max_epochs 300 --epochs_to_eval 300 --patience_retrain 300 --lr_scheduler multistep \
  --optimizer AdamW --batch_size 256 --eval_batch_size 256 --num_repetitions 3 \
  --data_augmentation --num_workers 8 --max_parallel_workers 4
```

### 4.5 Augmentation de CIFAR-10 incompleta (mejora; cambia el protocolo)
- `dataset_utils/transformations.py:47-48`: para CIFAR solo se aplica `TrivialAugmentWide`; no hay random crop
  con padding ni flip horizontal, que son el estándar en Q-NAS, CGP-CNN, CNN-GA, NSGA-Net, etc. Tampoco cutout.
- Añadir `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` antes de TrivialAugment (+ opcional
  `RandomErasing`/cutout) suele dar entre +0,5 y +1,5 pp en redes de este tamaño.
- **Recomendación:** no tocarlo para P1 (los números internos de Q-NAS se obtuvieron sin esto y se pierde la
  comparabilidad); implementarlo detrás de un flag (`augmentation_policy: ta | standard | standard+cutout`) y
  usarlo solo en P2.
- **Actualización (2026-10-04):** implementado (commit `04e2fa5`). Tras el piloto 2, `standard_cutout` pasa a ser la
  opción por defecto para CIFAR-10 en P1 (§4e). Las filas de Q-NAS F13 (solo `ta`) se mantienen como referencia y
  deben ir marcadas con su augmentation.

### 4.6 Throughput: el retrain está limitado por la CPU
- Los smoke retrains tardaron ≈ 120 s por 5 épocas (≈ 24 s/época) **independientemente** de los FLOPs
  (de 2.4 M a 1.75 G FLOPs): el cuello de botella es TrivialAugment en PIL con `num_workers: 4`.
- **Acción:** `num_workers` 8–16, añadir `persistent_workers=True` y `prefetch_factor` en
  `core/cnn/input.py:214`, y ejecutar varias redes por GPU (`--max_parallel_workers` > nº de GPUs; las redes
  de MoQ-NAS son < 2 M parámetros y no saturan una L40S). Medir el uso de CPU en el piloto antes de subir la concurrencia.

### 4.7 Hiperparámetros de optimización (mejora; cambia el protocolo)
- `core/cnn/master.py:121-122`: el código actual lee `weight_decay` del log (1e-4). F13-v1 usaba 0.01 (por defecto
  de PyTorch), así que P1 debe pasarlo explícito (ver la tabla de P1). Label smoothing no está y no se añade.
- Cualquier otro cambio de optimización queda fuera de P1; solo cabe en el piloto opcional o en P2.

### 4.8 Detalles menores
- `core/cnn/trainer.py:335`: el `continue` de la primera época de validación salta `update_scheduler`, así que
  el scheduler queda una época desfasado. El trainer de F13-v1 hacía lo mismo, así que **no corregirlo** mientras
  se use P1 (efecto pequeño e idéntico para todos los algoritmos).
- `core/cnn/trainer.py:199/204`: `clip_grad_norm_(max_norm=1.0)` siempre activo. F13-v1 **no** tenía clipping
  (se añadió el 04-05-2026) → hacerlo configurable y desactivarlo en P1.
- `MedMNIST_Metrics` es una métrica "primaria" y también se instancia para train/val, aunque su `Evaluator`
  apunta al split de test. Los retrains previos devuelven `auc_score` coherente en test; confirmarlo en el smoke
  test de MedMNIST (1–2 épocas) antes de lanzar el Tier C.

### 4.9 Fairness en retrain se calcula con los pesos de la última época (CRÍTICO para Tier D)
- `FairnessMetric` guarda una referencia al modelo que recibe al crearse (`core/cnn/metrics/fairness.py`,
  `self.model = self._init_args.get('model')`) y es una métrica de post-procesado.
- En retrain, `_run_test_phase` hace `self.model = self.reset_and_load_best_model(...)`, que construye **un objeto
  de modelo nuevo** (`core/cnn/trainer.py`, `reset_and_load_best_model`). La métrica de fairness sigue apuntando al
  objeto viejo, con los pesos de la última época, no los del mejor checkpoint. En evolución no pasa porque ahí se
  hace `load_state_dict` sobre el mismo objeto.
- **Acción:** en `_run_test_phase`, cargar el `state_dict` del mejor checkpoint en el modelo existente, o reasignar
  `metric.model = self.model` a todas las métricas de post-procesado tras recargar. Lanzar siempre con `--keep_metrics`
  (por defecto el retrain deja solo `Accuracy`) y verificar en un smoke test de 1–2 épocas que salen `per_group_tpr`,
  `fairness_spd`/`metrics_fairness` y `fairness_mean_tpr`.

### 4.10 Las corridas antiguas apuntan a un YAML de dataset que ya no existe (bloquea Tiers A, C y D)
- `core/cnn/input.py` exige `params['config_path_dataset']` apuntando a un archivo existente, y el retrain toma
  `train_spec` del `log_params_evolution.txt` de cada corrida.
- Los logs antiguos tienen `config_path_dataset: configs/<ds>.yaml` (carpeta renombrada a `dataset_configs/`) y los
  de NSGA del Caso 1 no tienen la clave. Además, el log de fairness dice `dataset: personbin`, pero el YAML es
  `dataset_configs/person_bin_96.yaml`.
- **Acción:** añadir `--config_path_dataset` a `retrain_parallel.py` (y a `override_keys`/`kv_args` de
  `launch_retrain.py`) y pasarlo siempre explícito: `dataset_configs/cifar10.yaml`, `dataset_configs/<ds>.yaml`,
  `dataset_configs/person_bin_96.yaml`. Alternativa mínima: si la ruta no existe, sustituir `configs/` por
  `dataset_configs/` y avisar en el log.
- Revisar en el smoke test que `train_spec` de los logs antiguos no traiga claves obsoletas que cambien el
  comportamiento (p. ej. `mixed_precision` en lugar de `precision`, sin `eval_window_agg`).

### 4.11 Fallos silenciosos por OOM y registro para reproducibilidad
- `core/cnn/master.py` (`retrain`): ante "out of memory" registra el error y **devuelve `None`**. En
  `retrain_parallel.py` ese `None` se guarda como resultado y el proceso principal escribe "Successfully finished
  retraining". Una red que no cupo en memoria desaparece sin aviso.
- **Acción:** devolver un estado explícito (`FAILED_OOM`, `FAILED_NAN`, `FAILED_INVALID_SHAPE`,
  `FAILED_TIMEOUT`) y escribirlo en un `failures.csv`. Si una red no cabe con batch 256, no reducir el batch solo
  para ella sin documentarlo: usar gradient accumulation (mismo batch efectivo) o marcarla como incompatible.
- Guardar por entrenamiento: commit de git, versiones de PyTorch/CUDA, GPU, semillas, split y configuración de
  augmentation/optimizador/scheduler. Las curvas ya guardan loss/acc de train y val por época; añadir la LR y el
  tiempo por época. Guardar también `last.pt` además de `best_model.pth`.
- **Latencia (solo tabla hardware-aware):** volver a medir Time_CUDA de los representantes con la GPU dedicada y un
  procedimiento fijo; ver §8.1 (la latencia de la búsqueda no es reproducible).

### 4.12 El texto del paper no describe bien el objetivo de accuracy de los experimentos acc-FLOPs
- El paper (y la tesis) dicen que el objetivo es la **mejor** val accuracy de las últimas 5 épocas. Es cierto en
  las corridas antiguas (sin `eval_window_agg` → `max`), pero las de `experiment_cifar10_acc_flops*` usan
  `eval_window_agg: mean` (**media** de las últimas 5). Corregir la descripción de la ablación de espacio en el paper.
- El paper dice que todos los experimentos usan "FP16 mixed precision". Las corridas acc-FLOPs usaron **bf16**.
  Corregir el texto (protocolo común o descripción de la ablación) e indicar la precisión del retrain por caso.

---

## 4b. Smoke tests en el Mac (2026-10-03)

Se pueden correr en el Mac (Apple M5 Pro, CPU; el código no usa MPS). Sirven para validar el camino del código, no
para medir tiempos ni accuracy.

Entorno (`conda` env `moqnas`, mismas versiones que el clúster salvo scipy/numpy):
```bash
conda create -y -n moqnas python=3.10
~/miniconda3/envs/moqnas/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 pandas==2.2.3 \
    scikit-learn==1.6.1 medmnist==3.0.2 PyYAML==6.0.1 tqdm matplotlib seaborn pymoo scikit-image pillow
~/miniconda3/envs/moqnas/bin/python -m pip install -r requirements.txt
# el wheel de scipy de PyPI no carga en este macOS (Darwin 27): usar el de conda-forge
~/miniconda3/envs/moqnas/bin/python -m pip uninstall -y scipy numpy
conda install -y -n moqnas -c conda-forge "scipy=1.15" "numpy=2.2.6"
```
Datos locales (en `MoQ-NAS/datasets` y `.cache`, ignorados por git): `cifar10_data`, `organamnist_data`,
`personbin_data_96` copiados del clúster, y un **subconjunto de FACET** de 600 filas (60 por tono, con sus recortes
en caché) marcado con `datasets/facet_data/README_SMOKE_SUBSET.txt`. Las salidas del smoke van a una copia de las
entradas, nunca a `data/`.

```bash
python launch_retrain_protocol.py --runs-root <copia>/runs --logs-dir <copia>/logs \
    --cases C1_triobj AF_std_biobj C2_medmnist --datasets cifar10 organamnist --roles all \
    --seeds 1 --tag F13v1 --smoke
python launch_retrain_protocol.py --runs-root <copia>/runs --logs-dir <copia>/logs \
    --cases C3_fairness_three --roles all --profile fairness_R1 --seeds 1 --tag fairR1 --smoke
```
`--smoke`: 2 épocas, 2 000 imágenes, el candidato más liviano (menos FLOPs) de cada caso, tag `<tag>_smoke`.

Resultado: **OK en los 4 casos** (Caso 1 fp16, acc-FLOPs bf16, MedMNIST fp16 con `auc_score`/`acc_medmnist`,
fairness R1 con FACET a 96 px y TPR por los 10 tonos) y en `train.py` + `evaluate.py` de baselines. Los parámetros
aplicados (`training_params.txt`) coinciden con el protocolo. Fallos que encontró el smoke y quedaron corregidos:
- `core/cnn/input.py`: `pin_memory=True` fijo intentaba inicializar CUDA en máquinas sin GPU → solo con CUDA.
- **`MedMNIST_Metrics` rompía todo retrain de MedMNIST**: se calculaba también en las épocas de train/val, pero
  su evaluador compara contra las etiquetas del test oficial (`AssertionError` por tamaño). Ahora es una métrica
  `test_only` y el trainer solo la usa en la fase de test.

Notas: en CPU el fp16/bf16 se emula y es muy lento con redes grandes (por eso `--smoke` elige la más liviana). En
`train.py` de baselines el autocast se desactiva en CPU, así que su ruta fp16 + GradScaler solo se prueba en el
clúster. Si se mata el lanzador, los workers de `ProcessPoolExecutor` quedan huérfanos: matarlos aparte.
Nombres en los resultados de fairness del retrain: D_group = `fairness_score` (= `spd_sum`), MeanTPR = `mean_tpr`.

## 4c. Smoke tests en el clúster, GPU 1 (2026-10-03, commit `5ab720f`)

Script: `retrain_2026/smoke/run_smoke.sh` (copia aislada de las entradas en `retrain_2026/smoke/runs`, logs en
`retrain_2026/smoke/logs`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, GPU 1 compartida con otro usuario). Duración total ≈ 25 min.

| Caso | Corrida / candidato | Precisión aplicada | Estado | Métricas extra |
|---|---|---|---|---|
| Caso 1 | exp22_r3 / 83_18 | fp16 | OK | — |
| acc-FLOPs | moqnas_r2 / 134_2 | bf16 | OK | — |
| MedMNIST Path / OCT / Tissue / OrganA | 135_4 / 118_4 / 135_0 / 143_9 | fp16 | OK (los 4) | `auc_score` en los 4 |
| Fairness R1, 3 obj. / 2 obj. | 40_16 / 77_12 | fp16 | OK | `fairness_score` (D_group), `mean_tpr` |
| Baselines resnet18, ConvNeXt-T, EfficientNetV2-S | `train.py` fp16 + multistep | fp16 | OK, sin NaN | `evaluate.py` FACET fp16: SPD 0.26 / 0.20 / 0.19 |

Todas con `weight_decay` 0.01 (0.0001 en fairness R1), sin clipping, `env.gpu = NVIDIA L40S`. Las accuracies no
significan nada (2 épocas, 2 000 imágenes).

Observación útil para el coste: la **primera época tarda 63–80 s y la segunda 0.4–2.6 s**. El arranque de los
workers del DataLoader es caro: `retrain_parallel.py` usa `spawn`, así que cada worker importa torch y recibe el
dataset serializado. Con `persistent_workers` (arreglo 4.6) se paga una vez por loader y semilla. Antes se pagaba en
cada época, lo que explica los ≈ 24 s/época de los smoke retrains de junio, independientes de los FLOPs. Así, el
coste fijo por candidato y semilla es ≈ 1–1.5 min, y el resto escala con épocas y tamaño de la red. El piloto (6 redes,
300 épocas, 45 k imágenes) dará el coste real.

## 4d. Piloto y reporte de tiempos (2026-10-03/04, GPU 1, commit `40e31f0`)

**Piloto:** 6 redes de MoQ-NAS del Caso 1 (best_acc + compact de exp22 r1–r3), protocolo F13-v1 completo (300 épocas,
45k/5k, fp16, semilla 1), las 6 en paralelo en la GPU 1, que también usaba otro usuario. Tag `F13v1`: cuentan como
screening. Inicio 21:06, fin 01:59 → **4.9 h de reloj** para 6 redes, sin fallos.

| Red | Rol | Params | GFLOPs | Proxy | **Test** | Ganancia | Mejor val (época) | h por red | s/época (mediana) |
|---|---|---|---|---|---|---|---|---|---|
| 130_14 | best_acc r1 | 1.71 M | 2.24 | 77.3 | **91.80** | +14.5 | 92.42 (288) | 4.87 | 58 |
| 66_5 | best_acc r2 | 0.92 M | 1.80 | 77.4 | **91.08** | +13.7 | 91.36 (202) | 4.80 | 58 |
| 79_19 | best_acc r3 | 0.89 M | 1.31 | 78.8 | **90.86** | +12.1 | 91.38 (226) | 3.85 | 47 |
| 127_14 | compact r3 | 0.32 M | 0.36 | 75.6 | **89.48** | +13.9 | 89.60 (205) | 2.75 | 32 |
| 98_1 | compact r1 | 0.25 M | 0.21 | 73.5 | **88.41** | +14.9 | 88.68 (282) | 2.31 | 27 |
| 139_15 | compact r2 | 0.26 M | 0.44 | 72.8 | **87.26** | +14.5 | 87.70 (219) | 3.31 | 39 |

- Ganancia proxy → test: **+12.1 a +14.9 pp**. Spearman proxy vs test = 0.77 (n = 6): el proxy ordena razonablemente,
  pero no perfectamente.
- La mejor validación llega en las épocas 202–288, después de las bajadas de LR (150, 225). Con el early stopping
  antiguo (`patience 50` sobre val loss) se habrían cortado antes: confirma el arreglo 4.1.
- Las redes de mejor accuracy quedan en 90.9–91.8 % con 0.9–1.7 M params, algo por debajo de Q-NAS 2-C1 (92.96 %,
  1.6 M) y CGP-CNN (93.25 %, 1.52 M), que usan otros protocolos. Las compactas, en 87.3–89.5 % con 0.25–0.32 M params.

**Calibración del modelo de tiempos (una GPU, compartida como ahora):**
- Rendimiento agregado de la GPU con varias redes en paralelo: **≈ 15.2 TFLOP/s** (trabajo total del piloto / 4.9 h).
  Con 2 redes al final se mantuvo en ≈ 15 → el piloto estuvo limitado por la GPU.
- Lado CPU (carga y aumentación de datos): **≥ 10.7k imágenes/s** sumando todas las redes. Una red pequeña no baja de
  ~27 s/época en CIFAR-10 completo con la máquina cargada.
- Costo fijo ≈ 2–2.5 min por red y semilla (arranque de workers, CUDA, test), amortizado al correr en paralelo.
- Correr **en paralelo** rinde mucho más que en serie: la suma de los tiempos individuales fue 21.9 h, frente a 4.9 h
  de reloj. Recomendado: 6 redes a la vez por GPU, las más pesadas primero.
- Modelo: horas ≈ máx(Σ FLOPs de entrenamiento / 15.2 TFLOP/s, Σ imágenes procesadas / 10.7k img/s) + costo fijo.

**Proyección en UNA GPU** (condiciones actuales; si la GPU queda libre, será más rápido):

| Grupo | Bloque | Entrenamientos | Días | Límite |
|---|---|---|---|---|
| **Revisor** | Caso 1 · screening reducido (`strat10` + reps, sin el piloto) | 84 | **1.8** | GPU |
| | Caso 1 · screening del frente filtrado completo (alternativa) | 497 | 16.5 | GPU |
| | Caso 1 · confirmación (≈ 25 reps × semillas 11–13) | 75 | **2.5** (≈ 5 si se suman los reps por presupuesto) | GPU |
| | acc-FLOPs · screening del frente (147) | 147 | **2.4** | CPU |
| | acc-FLOPs · confirmación (≈ 25 reps × 3) | 75 | **1.2** (≈ 2.4 con reps por presupuesto) | CPU |
| Extra | Fairness R1: frente 121 + 5 baselines, × 3, 10k img, 50 ép. | 378 | 1.7 | GPU |
| | Fairness R2 reducido: 9 reps + 5 baselines × 3, 100 ép. | 42 | 2.5 | GPU |
| | Fairness R2 completo, 100 / 300 ép. | 378 | 30 / 89 | GPU |
| | MedMNIST: solo reps de MoQ-NAS, 1 semilla | 32 | 1.2 | CPU |
| | MedMNIST: screening `strat10` + reps / confirmación × 3 | 360 / 303 | 12.9 / 12.3 | CPU / GPU |

Totales:
- **Lo que pide el revisor**, con el screening reducido del Caso 1: **≈ 8 días** (≈ 11 con los representantes por
  presupuesto). Con el screening completo del Caso 1: ≈ 23–26 días.
- Más fairness R1 + R2 reducido + MedMNIST reducido: **+ ≈ 5.4 días**.
- Fairness R2 completo y MedMNIST completo suman 1–3 meses: no son viables con una GPU.

Incertidumbres: la confirmación depende de cuántos representantes salgan del re-Pareto; el bloque acc-FLOPs y
MedMNIST están limitados por CPU, y el ritmo de CPU medido es conservador en MedMNIST (imágenes de 28×28, más
baratas de aumentar) y quizá optimista en fairness (96×96 con RandomResizedCrop); el reparto de la GPU con el otro
usuario puede cambiar.

## 4e. Augmentation de CIFAR-10: piloto 2 y brecha de protocolo con la literatura (2026-10-04)

**Decisión (2026-10-04, tras el piloto 2): se adopta la opción (c) para todos los reentrenamientos de CIFAR-10**
(Caso 1 y acc-FLOPs, los tres algoritmos). Queda fijada en `datasets.cifar10` de `protocol_F13v1.yaml`
(`augmentation_policy: standard_cutout`), así que ya no hace falta `--profile cifar_aug_c`; tag `F13v1c`.
MedMNIST y fairness siguen con `ta` (tag `F13v1`, `fairR1`/`fairR2`). Las 6 redes del piloto 2 cuentan como
screening del Caso 1: el launcher salta los candidatos que ya están OK con el mismo tag y se niega a mezclar
políticas de augmentation dentro de un tag. Por eso el tag `F13v1` en CIFAR-10 queda solo para el piloto 1.

Resultado del piloto 2 (test, semilla 1, misma GPU y concurrencia que el piloto 1):

| Red (exp22) | F13-v1 (P1) | Opción (c) (P2) | Δ test (pp) | Δ tiempo de entrenamiento |
|---|---|---|---|---|
| r1 130_14 (`best_acc`) | 91.80 | 92.89 | +1.09 | +15.0 % |
| r2 66_5 (`best_acc`) | 91.08 | 92.15 | +1.07 | +14.4 % |
| r3 79_19 (`best_acc`) | 90.86 | 91.73 | +0.87 | +13.1 % |
| r3 127_14 (`compact`) | 89.48 | 90.69 | +1.21 | +10.2 % |
| r1 98_1 (`compact`) | 88.41 | 89.54 | +1.13 | +8.9 % |
| r2 139_15 (`compact`) | 87.26 | 89.21 | +1.95 | +10.5 % |
| **Media** | 89.82 | 91.04 | **+1.22** (6/6 mejoran) | **+12.6 %** (suma de tiempos) |

Se cumplen las tres condiciones del criterio: mejora en la mayoría (6/6), mejora media ≥ 0.3 pp y tiempo
≤ +20 % (máximo +15 %). El tiempo es orientativo: la GPU se comparte con otros usuarios. Las proyecciones de §4d
hay que subirlas ≈ 13 % en los Tiers A y B.

Las dos candidatas que se compararon para el protocolo CIFAR-10 (Caso 1 y acc-FLOPs):
- **F13-v1** (`augmentation_policy: ta`): solo TrivialAugmentWide, tal como se ejecutó F13. Piloto 1, tag `F13v1`.
- **Opción (c)** (perfil `cifar_aug_c`, `augmentation_policy: standard_cutout`): RandomCrop(32, pad 4) +
  HorizontalFlip + TrivialAugmentWide + Cutout(16). Lo demás, igual que F13-v1. Piloto 2, tag `F13v1c`.

Diseño del piloto 2: mismas 6 redes, misma GPU (1), misma concurrencia (3 trabajos × 2) y misma semilla que el piloto 1 →
comparación pareada del efecto de la augmentation sobre test accuracy, y también de su costo (CPU y tiempo).
Criterio de decisión, fijado antes de ver resultados: adoptar (c) si mejora el test en la mayoría de las 6 redes y
en media, sin aumentar el tiempo de forma relevante (> 20 %). Si la mejora es marginal (< 0.3 pp de media),
quedarse con F13-v1, que tiene respaldo previo en la tesis. La decisión se aplica igual a los tres algoritmos.

**Brecha de protocolo con la literatura (para la tabla y el texto del paper).** Incluso con (c), el reentrenamiento
de este trabajo es más corto y más simple que el de los métodos de referencia:

| Método | Épocas | Augmentation | Optimizador / LR | Extras de regularización | Datos de entrenamiento | Fuente |
|---|---|---|---|---|---|---|
| **Este trabajo, F13-v1** | **300** | TrivialAugment | AdamW 1e-3, wd 0.01, multistep | ninguno | 45k (5k para validación) | código / logs ✔ |
| **Este trabajo, opción (c)** | **300** | crop + flip + TrivialAugment + cutout 16 | ídem | ninguno | 45k | código ✔ |
| NSGA-Net (micro) | 600 | cutout | (verificar) | drop-path programado, cabeza auxiliar | (verificar) | paper ✔ / TODO macro |
| NSGANetV1 | 600 (batch 96) | cutout | SGD, cosine, wd 5e-4 | drop-path programado, cabeza auxiliar (0.4) | (verificar) | paper, Tabla I y §IV-B ✔ |
| LEMONADE | 600 (batch 64) | cutout + mixup | SGD, lr 0.025, cosine, wd 5e-4 | — | (verificar) | resumen del paper, verificar |
| CNN-GA | 350 | crop + flip (96.78 % con cutout) | SGD 0.1, momentum 0.9, decaimiento en 1/149/249 | — | (verificar) | paper ✔ |
| CARS, EEEA-Net, LightMix | (verificar; habitual "estilo DARTS": 600, cutout, drop-path, aux) | | | | | TODO |

Cómo usarlo en el paper:
- Añadir a la tabla de literatura una columna o nota de **protocolo final** (épocas, cutout, cabeza auxiliar,
  drop-path) para que se vea que parte de la diferencia de error viene del presupuesto y la regularización del
  reentrenamiento, no de la búsqueda.
- Frase prudente propuesta (ajustar a los números reales): *"All retrained architectures use a single, shorter
  protocol (300 epochs, no auxiliary head or drop-path), whereas most published multi-objective NAS results are
  obtained with 600-epoch schedules and additional regularisation; the reported gaps should therefore be read as
  upper bounds on the difference attributable to the search."*
- Si se quiere cuantificar esa brecha, una prueba barata y opcional: reentrenar con 600 épocas solo los 2–3
  representantes de mejor accuracy (≈ 10 h en una GPU) y reportar la diferencia frente a 300 épocas.

**¿La opción (c) sirve para fairness? No se recomienda.**
- La rama `person` ya usa una augmentation fuerte, al estilo ImageNet: RandomResizedCrop(96, escala 0.08–1) +
  HorizontalFlip + TrivialAugmentWide. Crop y flip ya están incluidos (y con un recorte más agresivo que el padding de
  4 px).
- R1 debe reproducir el protocolo de la búsqueda de fairness; cambiar su augmentation rompe esa comparación. R2 debe
  usar la misma augmentation que R1 para que la diferencia entre ambos se deba solo a los datos.
- El cutout de 16 px es pequeño en imágenes de 96×96 (~3 % del área) y puede ocultar zonas de piel de forma aleatoria.
  Su efecto sobre las métricas por tono de piel no está estudiado: sería un factor nuevo en el análisis de fairness.
- El código lo impide: `augmentation_policy` distinto de `ta` da error para person/face/MedMNIST.

## 4f. Dos servidores: dualgpu1 y dualgpu2 (2026-10-04)

Desde el screening, la etapa corre en dos servidores (2× L40S compartidas en cada uno). Cada **caso completo** va en un
solo servidor, y siempre en la GPU 1, para que los tres algoritmos de un caso se entrenen en las mismas condiciones:

| Caso | Servidor | Lanzado | Comando (relanzarlo igual si hay un corte) |
|---|---|---|---|
| Caso 1 (C1_triobj), 84 redes + 6 del piloto 2 | dualgpu1, GPU 1 | 2026-10-04 19:40 | `launch_retrain_protocol.py --cases C1_triobj --seeds 1 --tag F13v1c --gpus 1 --jobs-per-gpu 3 --workers-per-job 2` |
| acc-FLOPs (AF_std_biobj), 147 redes | dualgpu2, GPU 1 | 2026-10-04 20:14 (3×2); **relanzado 22:01 con 12 en paralelo (4×3)** | `launch_retrain_protocol.py --cases AF_std_biobj --roles all --seeds 1 --tag F13v1c --gpus 1 --jobs-per-gpu 4 --workers-per-job 3` |
| acc-FLOPs, confirmación (paso 4): 51 representantes × semillas 11–13 = 153 | dualgpu2, GPU 1 | 2026-10-06 10:42 (4×3), con visto bueno del usuario; watchdog `AFconf` | `launch_retrain_protocol.py --cases AF_std_biobj --candidates retrain_matrices/confirm_AF_std_biobj_F13v1c.csv --roles all --seeds 11 12 13 --tag F13v1c --gpus 1 --jobs-per-gpu 4 --workers-per-job 3` |
| Caso 1, confirmación (paso 4), automática al acabar el screening (≈ 45 representantes × semillas 11–13 ≈ 135) | dualgpu1, GPU 1 | pendiente de que termine el screening; watchdog `C1conf` | `launch_retrain_protocol.py --cases C1_triobj --candidates retrain_matrices/confirm_C1_triobj_F13v1c.csv --roles all --seeds 11 12 13 --tag F13v1c --gpus 1 --jobs-per-gpu 3 --workers-per-job 2` |
| Caso 2 (C2_medmnist), screening de 180 redes | **LIRA-Server (lira-150), GPUs 0 y 1** (cambio del 2026-10-06; antes previsto en dualgpu2) | 2026-10-06 15:07 (6+6); **relanzado 15:44 con 9+9 y 4 workers**; watchdog `C2` | `launch_retrain_protocol.py --cases C2_medmnist --roles strat5 --profile medmnist_v2_adamw --seeds 1 --tag PMedW --gpus 0 1 --jobs-per-gpu 3 --workers-per-job 3` (python: `~/miniforge3/envs/moqnas/bin/python`) |

**Estado de acc-FLOPs (2026-10-06):** screening terminado a las 05:41, con 147/147 redes OK y ningún fallo. Selección
(pasos 2–3) hecha en el Mac con los datos completos: `retrain_matrices/confirm_AF_std_biobj_F13v1c.csv`, con 51
representantes (MoQ-NAS 11, NSGA-II 22, NSGA-III 18; 12 de ellas por la regla de inestabilidad), es decir, 153
entrenamientos con las semillas 11–13. El informe y el análisis están en `confirm_AF_std_biobj_F13v1c_report.md` y en
`_analysis_{runs,rules}.csv`. El usuario dio el visto bueno y la confirmación (paso 4) se lanzó el 2026-10-06 a las 10:42.

**Orden en cada servidor (decisión del usuario, 2026-10-05):** un caso se cierra por completo antes de empezar el
siguiente en la misma GPU: screening → selección (en el Mac) → confirmación con las semillas 11–13. En dualgpu2,
MedMNIST se lanza solo cuando acc-FLOPs haya terminado también su confirmación. Preparado el 2026-10-05: los cuatro
datasets de MedMNIST son idénticos en los dos servidores (mismos md5 de los `.npz`), las 36 corridas de búsqueda de
MedMNIST se copiaron de dualgpu1 a dualgpu2 (1 735 `training_params.txt`), y el dry-run en dualgpu2 da 36 trabajos y
180 redes con P-Med (100 épocas, AdamW, wd 0.01, multistep, batch 128, fp16, sin augmentation). Antes del screening,
un smoke test de MedMNIST con este perfil.

**MedMNIST pasa a LIRA-Server (decisión del usuario, 2026-10-06).** Tercer servidor (`ssh LIRA-Server`, usuario
`diego.paez`, 3× NVIDIA A30 de 24 GB compartidas, 64 CPUs, 188 GB de RAM). Se usan solo las GPUs 0 y 1. Así MedMNIST
empieza ya, en paralelo con el Caso 1 y la confirmación de acc-FLOPs, y el caso entero sigue en un solo servidor y con
un mismo modelo de GPU. En dualgpu2 ya no va MedMNIST después de acc-FLOPs.
- **Entorno replicado de dualgpu2:** miniforge en `~/miniforge3`, entorno `moqnas` con Python 3.10.16 y las mismas
  versiones (94 paquetes): torch 2.5.1 + CUDA 12.4, torchvision 0.20.1, numpy 1.26.4, scikit-learn 1.3.2, medmnist
  3.0.1, etc. Se instaló con pip sin caché y ocupa 6.7 GB. Repo clonado de GitHub (rama `retrain-2026`; LIRA accede
  a GitHub mediante el agente SSH reenviado).
- **Datos verificados:** los cuatro datasets de MedMNIST tienen el mismo md5 que en dualgpu1 y dualgpu2, y los 1 806
  archivos de las 36 corridas coinciden uno a uno con los del Mac.
- **Smoke test** con P-Med en las GPUs 0 y 1: OK (ACC y AUC del evaluador oficial; AdamW, wd 0.01, batch 128, fp16,
  sin augmentation).
- **Disco:** `/home` es compartido y estaba al 96 % (16–34 GB libres según el momento, porque otros usuarios escriben
  ahí). MedMNIST necesita ≈ 3 GB más para resultados (screening ≈ 0.5, confirmación ≈ 1.3, variante TA ≈ 1.3). Se
  vigila el espacio libre en el monitoreo horario; si baja de ≈ 3 GB, se avisa al usuario. Si hiciera falta, dejar de
  guardar `last.pt` reduciría los resultados a la mitad.
- **Concurrencia y workers en LIRA (decisión del usuario, 2026-10-06 15:44):** 9 redes por GPU (9+9) y **4 workers por
  loader** en lugar de 8 (perfiles `medmnist_v2_adamw` y `medmnist_v2_adamw_ta`, commit `3dde8f0`; CIFAR-10 sigue con 8).
  Medido con 6+6 y 8 workers: cada entrenamiento ocupaba ≈ 8.7 GB de RAM (16 workers de ≈ 730 MB casi ociosos, al 9 %
  de CPU) y ≈ 2.5 núcleos, con el proceso principal como cuello de botella (1 núcleo). Quedaban 41 GB disponibles de 188,
  así que 12+12 no cabía. Con 4 workers serían ≈ 5 GB por entrenamiento; 9+9 ≈ 90 GB y ≈ 80 % de la CPU, mientras que
  12+12 saturaría la CPU. El número de workers no cambia los resultados: P-Med no tiene augmentation aleatoria y el orden
  de los batches lo fija el muestreador del proceso principal. Se aplicó sin prueba previa, a pedido del usuario. Al
  reiniciar se perdieron las 12 redes que estaban en curso (≈ 35 min cada una); ninguna había terminado.
- **Watchdog con dos GPUs** (commit `77aa439`): ante OOM baja a 4 entrenamientos en total (1 trabajo × 2 redes en
  cada GPU). Probado en LIRA con un launcher falso.

**Verificación contra los originales del Mac (2026-10-05):** las 36 corridas de MedMNIST del plan (MoQ-NAS en
`data/medmnist/moqnas_last` y NSGA-II/III en `data/medmnist/NSGA`; 4 datasets × 3 algoritmos × 3 corridas) son
exactamente las que hay en el Mac, sin corridas de más ni de menos. Los 1 806 archivos que necesita el retrain (36
`log_params_evolution.txt`, 36 `pareto_history.pkl` y 1 734 `training_params.txt`, uno por red del CSV, incluidas las
180 del screening) tienen el mismo md5 en el Mac y en dualgpu2.

Los tiempos de entrenamiento no se comparan entre casos (servidor, carga de otros usuarios y temperatura distintos);
dentro de un caso, sí. En acc-FLOPs las primeras 10 redes se entrenaron con 6 en paralelo y el resto con 12, así que
su `training_time` tampoco es homogéneo dentro del caso: es un dato de costo, no de comparación.

**Qué GPU y cuántas redes en paralelo (decisión del usuario, 2026-10-04, tras medirlo):** solo la GPU 1 de cada
servidor. Pruebas cortas (6 épocas, mismas redes y concurrencia que las que corrían; artefactos borrados):
- **Cambiar de GPU no acelera.** En dualgpu2 las mismas 6 redes van a 5.08 s/época en la GPU 0 y a 5.1 en la GPU 1,
  aunque la GPU 0 tenga más reloj (2310 frente a 1620 MHz). El límite es compartir la GPU entre procesos, no el reloj.
  En dualgpu1 la GPU 0 está más frenada que la GPU 1 (570–855 frente a 990–1230 MHz); las dos están a 87–88 °C con
  *SW Thermal Slowdown* activo.
- **Pasar de 6 a 12 redes en la misma GPU rinde ≈ +40–55 % en total** (dualgpu2: las que corrían pasaron de 5.1 a
  ~5.5 s/época y las nuevas fueron a 8.5; dualgpu1: de 10–11 a 12.5–14 y 15.7). La memoria de GPU no limita.
- Decisión: **dualgpu1 sigue con 6** (CPU muy cargada por otros usuarios, load 80–109 de 128, y redes del Caso 1 más
  grandes: con 6 ya usa hasta 32 GB de 46) y **dualgpu2 pasa a 12**. Al reiniciar se pierden las redes en curso, así
  que se esperó a que terminaran las que estaban más avanzadas.

**Tolerancia a cortes (2026-10-04, commits `547f563` y `5b7a7e0`):** `retrain_parallel.py` guarda cada red en
`retrain_results_<tag>.txt` en cuanto termina (escritura atómica: temporal + fsync + `os.replace`), y el launcher salta
las redes ya OK con el mismo tag. Ante un corte de luz o un reinicio, basta con relanzar el mismo comando: solo se
pierden las redes que estaban entrenándose, que vuelven a la época 1. No hay reanudación a mitad de entrenamiento, a
propósito: una red reanudada no sería idéntica a una sin cortes con la misma semilla. Probado en el clúster con un
`kill -9` a mitad de trabajo. Tras un reinicio del servidor también hay que relanzar el watchdog.
**Guardado por semilla (2026-10-06, a pedido del usuario):** con varias semillas por red (confirmación 11–13), el
resultado de la red solo se escribía al terminar las tres, así que un corte hacía perder también las semillas ya
completadas. Ahora `retrain_parallel.py` guarda cada semilla en cuanto termina OK
(`archive/<id>/retrain_<tag>_s<semilla>/seed_result_<tag>.json`, escritura atómica) y, al relanzar, salta las semillas
que ya tienen ese archivo. Cada semilla empieza desde su propio estado aleatorio, así que saltar una no cambia las
demás. Probado en el Mac con un `kill -9` después de la semilla 11: al relanzar se saltó la 11 y se entrenaron la 12
y la 13. Aplica a las redes que empiezan después de actualizar el código; las que ya estaban en curso siguen con el
código anterior hasta terminar. Nota de reproducibilidad: el mismo entrenamiento con la misma semilla no es idéntico
bit a bit entre ejecuciones (diferencias desde la 4.ª decimal de la loss, 0.01–0.09 pp de test en la prueba), por
operaciones de punto flotante no deterministas; es mucho menor que la variación entre semillas (0.05–0.36 pp).

**Cómo detener un launcher sin dejar procesos sueltos:** matar el árbol de descendientes del PID del launcher
(`ps --ppid` recursivo), no `kill -- -<pgid>`: con `setsid nohup` el grupo de procesos es otro y ese kill falló el
2026-10-04 a las 22:01, de modo que llegaron a correr dos launchers durante ~1 min (se detuvieron ambos y no se dañó
ningún resultado). Desde ssh, buscar el launcher con `pgrep -f '^[^ ]*python[^ ]* launch_retrain_protocol.py'`, porque
un patrón con el texto del comando coincide con la propia sesión.

**Compatibilidad entre servidores (verificada):**
- Mismo CIFAR-10 (md5 de `cifar-10-batches-py` idéntico) y mismas torch 2.5.1 / torchvision 0.20.1 (CUDA en L40S).
- Difieren numpy (dualgpu1 2.2.6, dualgpu2 1.26.4) y sklearn (1.6.1 frente a 1.3.2). El split 45k/5k
  (`StratifiedShuffleSplit`, `split_seed 2025`) sale **idéntico** en los dos (md5 de los índices
  `67966c29c45d99cff798756623b892db`). Si se cambia el entorno de alguno, repetir esta verificación.
- Cada resultado guarda en `env` el hostname, el commit y las versiones de python, torch, torchvision, numpy y sklearn.
  El launcher imprime host y commit al empezar, y avisa si hay archivos versionados con cambios locales.

**Reglas para mantener el código sincronizado:**
- Los dos servidores corren la rama `retrain-2026` en el mismo commit que GitHub. Los cambios se hacen en el Mac,
  se llevan con `git bundle` a un servidor, se hace `git push` desde allí y el otro servidor hace `git pull`
  (dualgpu1 y dualgpu2 no se ven entre sí; los dos sí ven GitHub por SSH).
- **No cambiar el código que afecta al entrenamiento mientras haya un screening corriendo**: el launcher arranca un
  `retrain_parallel.py` nuevo por cada trabajo, y sus workers (`spawn`) vuelven a importar el código del disco, así que
  un `git pull` a mitad del screening afecta a las redes que empiezan después. Solo se pueden traer a mitad de
  camino cambios que no tocan el entrenamiento (p. ej., metadatos en `env`); en otro caso, esperar a que termine.
- No editar archivos versionados directamente en los servidores (dualgpu2 tenía `run_retrain.sh` modificado; se
  descartó el 2026-10-04 con backup en `~/MoQ-NAS_dualgpu2_local_changes_2026-10-04.tgz`).

**Vigilancia de memoria (desde 2026-10-05):** cada screening tiene un `scripts/retrain_oom_watchdog.sh` corriendo
en su servidor. Cada minuto cuenta los fallos por falta de memoria (`FAILED_OOM`, "out of memory" o un worker
terminado de golpe) en los `retrain_failures_<tag>.csv` del caso. Ante el primero detiene el launcher (todo el
árbol de procesos) y lo relanza con **4 entrenamientos en paralelo** (`--jobs-per-gpu 2 --workers-per-job 2`); las
redes ya OK se saltan y las que fallaron se reentrenan. Solo reduce una vez y sale cuando el launcher termina. Log:
`retrain_2026/watchdog/<nombre>.log`.

**Monitoreo:** además del watchdog, la sesión de trabajo revisa cada hora los dos servidores (watchdog, launcher,
procesos en la GPU 1, redes OK y filas de fallos). Si el watchdog muere con el screening en marcha, se relanza. Ante
OOM repetido con 4 en paralelo, fallos que no son OOM o un launcher que muere sin terminar, se avisa al usuario y no se
relanza nada por cuenta propia.

**Resultados:** el análisis se hace en el Mac. `scripts/sync_retrain_results.sh` (repo de análisis) baja
`retrain_2026/` y `logs/retrain.log` de cada servidor a `retrain_2026/cluster/<host>/` (un espejo por servidor,
sin borrar nada y sin pesos salvo con `--with-weights`). El Caso 1 se lee del espejo de dualgpu1 y acc-FLOPs del de
dualgpu2 (dualgpu1 también tiene una copia de las corridas de acc-FLOPs, pero sin resultados).

## 4g. Registro de decisiones (todas con fecha; mantener al día)

Toda decisión de esta etapa se registra aquí en el momento de tomarla, con la sección que la desarrolla.

| Fecha | Decisión | Dónde |
|---|---|---|
| 2026-10-03 | Protocolo de retrain = F13-v1 tal como se ejecutó (AdamW 1e-3, wd 0.01, 300 épocas, multistep 50/75 %); precisión = la de la búsqueda de cada caso; Q-NAS no se reentrena | §3 |
| 2026-10-03 | Tabla comparativa solo contra NAS multiobjetivo; lo mono-objetivo como filas de referencia | §6.1 |
| 2026-10-03 | Fairness: todo el frente × 3 semillas en R1 y R2, mismo protocolo para MoQ-NAS y baselines | §2 Tier D |
| 2026-10-04 | Data augmentation de CIFAR-10 = opción (c) (crop + flip + TrivialAugment + cutout 16) para todo CIFAR-10, tag `F13v1c`, tras el piloto 2 (+1.22 pp, 6/6) con el criterio fijado de antemano | §4e |
| 2026-10-04 | Dos servidores: Caso 1 en dualgpu1 y acc-FLOPs en dualgpu2, cada caso completo en una sola máquina y solo en la GPU 1; cambios locales de dualgpu2 descartados (con backup) | §4f |
| 2026-10-04 | Guardar cada red al terminar (tolerancia a cortes); sin reanudación a mitad de entrenamiento | §4f |
| 2026-10-04 | dualgpu1 con 6 redes en paralelo; dualgpu2 con 12 (relanzado a las 22:01) | §4f |
| 2026-10-04 | Centralizar todos los resultados en el Mac (`scripts/sync_retrain_results.sh`, un espejo por servidor) | §4f |
| 2026-10-05 | Ante OOM, bajar a 4 en paralelo (watchdog automático, una sola vez) | §4f |
| 2026-10-05 | Selección en 4 etapas para todos los casos: screening con semilla 1 → re-Pareto con validación → representantes por reglas → confirmación con semillas 11–13 | §2, selección en dos etapas |
| 2026-10-05 | Ningún frente usa la Time_CUDA de la búsqueda (no reproducible con la GPU compartida; tampoco sirve para comparar con otros trabajos) | §2 paso 2, §8.1 |
| 2026-10-05 | Reglas iguales en los tres casos: `A`, `K` lineal y `C` (menor complejidad a ≤ 5 pp de `A`), más presupuestos en el Caso 1 y acc-FLOPs; sin `P`/`F` (redes triviales: se quieren redes útiles y competitivas) | §2 paso 3 |
| 2026-10-05 | Reportar con los datos de cada caso el análisis que justifica la selección (acuerdo proxy → validación, estabilidad ante ruido de semilla, sensibilidad a los parámetros fijados) | §2 paso 3 |
| 2026-10-05 | MedMNIST: protocolo P-Med = esquema de MedMNIST v2 (100 épocas, ×0.1 en 50 y 75, batch 128, sin augmentation) con AdamW (lr 1e-3, wd 0.01), desde el screening; confirmación con semillas 11–13; variante opcional solo con TrivialAugment; comparación también con AutoML | §2 Tier C |
| 2026-10-05 | Metodología para el paper en `case of study paper/6_retraining_comparative_methodology.tex` (inglés) y `retrain_methodology_refs.bib`; cada referencia verificada en Google Scholar o, cuando Scholar bloqueó con CAPTCHA, en Crossref/DOI, arXiv, JMLR, PMLR o la web, con la fuente anotada en el `.bib` | repo de análisis |

| 2026-10-05 | Elecciones inestables: si una regla mantiene su red en < 70 % de las 500 simulaciones de ruido, se confirma también la red que esa misma regla elige con más frecuencia entre las demás (etiqueta `<regla>~`); se reportan ambas y nunca se elige entre ellas con el test. Decidido antes de ver el test | §2 paso 3 |

| 2026-10-05 | Orden por servidor: cada caso se cierra por completo (screening, selección y confirmación) antes de empezar el siguiente en la misma GPU; MedMNIST va en la GPU 1 de dualgpu2 después de cerrar acc-FLOPs | §4f |
| 2026-10-06 | Visto bueno para la confirmación de acc-FLOPs (lanzada a las 10:42) | §4f |
| 2026-10-06 | MedMNIST se ejecuta en LIRA-Server, en las GPUs 0 y 1, con un entorno replicado de dualgpu2 (sustituye a "MedMNIST en dualgpu2 después de acc-FLOPs"); screening lanzado a las 15:07 con 6 redes por GPU | §4f |
| 2026-10-06 | MedMNIST en LIRA: 9+9 redes en paralelo y 4 workers por loader (sin prueba previa), relanzado a las 15:44 | §4f |
| 2026-10-06 | Guardar cada semilla en cuanto termina y saltarla al relanzar (no perder semillas completadas si hay un corte) | §4f |
| 2026-10-07 | Caso 1: al terminar el screening se sigue **automáticamente**, sin esperar visto bueno: selección (pasos 2–3) en el Mac y confirmación con semillas 11–13 en la GPU 1 de dualgpu1 (3×2, watchdog `C1conf`), avisando al usuario de lo elegido. Solo se para si el screening tiene fallos o la selección se niega por redes faltantes. Para acc-FLOPs y MedMNIST se sigue pidiendo visto bueno | §4f |
| 2026-10-07 | Como la sesión local se cierra, la continuación automática del Caso 1 corre **en dualgpu1** con `scripts/retrain_auto_confirm.sh C1 C1_triobj 90 1 3 2`: espera a que termine el screening, comprueba 90/90 OK sin fallos, hace la selección en el servidor, la commitea y la sube a GitHub, y lanza la confirmación con el watchdog `C1conf`. Estado en `retrain_2026/auto/C1.status`. **Queda pendiente bajar los resultados al Mac** y registrarlo en este documento en la siguiente sesión | §4f |

**Decisiones pendientes:** ninguna por ahora.

## 5. Checklist en el clúster (antes de lanzar)

0. **Inventario:** correr `scripts/check_retrain_inventory.py` (§2.0) y subir con `rsync --files-from` lo que falte.
1. Sincronizar este repo (incluido `retrain_matrices/literature_retrain_candidates.csv`) y aplicar los cambios
   4.1–4.4, 4.9 y 4.10.
2. Mapear `local_dir` del CSV a las rutas del servidor (lo hace el script: columna `cluster_path`); comprobar que existe `archive/<id>/training_params.txt`
   para cada ID (`--dry-run` o un script de 10 líneas).
3. Contar carpetas vacías en `archive/` de las corridas legacy de NSGA-II/III (ver Tier A).
4. **Piloto** (≈ 1 día): 6 redes de MoQ-NAS exp22 (los 3 `best_acc` y los 3 `compact`) con P1 y 1 semilla.
   Objetivos: (a) medir horas/red y la concurrencia útil por GPU, (b) confirmar que el proxy predice el reentrenado
   (se espera ~91–93 % en `best_acc`), (c) verificar que la val accuracy reportada ya es `max`.
5. Lanzar Tier A → Tier B → Tier C (representantes) → Tier D, en ese orden.
6. Guardar por red: `test_accuracy`, `test_loss`, `best_accuracy` (val, max), semilla, épocas efectivas,
   `training_time`, GPU y concurrencia (para reportar el coste del retrain aparte del de la búsqueda).

Estimación de coste (incertidumbre alta): las 42 redes Standard de Q-NAS tardaron entre 0.3 y 9.3 h por
reentrenamiento de 300 épocas (mediana ≈ 3 h), según la concurrencia en la GPU. Tier A ≈ 140 × ~3 h ≈ 420 GPU-h
(≈ 17 GPU-días en serie; ≈ 3–4 días de reloj con 2 GPUs × 3–4 trabajos concurrentes). Tier B, similar.
Tier C con F13-v1 MedMNIST (300 épocas) cuesta por red lo mismo que CIFAR-10 o más en TissueMNIST (165 k imágenes);
por eso conviene empezar por sus representantes. El piloto fija estos números.

---

## 6. Búsqueda de métodos para la tabla comparativa

### 6.1 Criterio de búsqueda (decisión 2026-10-03: solo trabajos multiobjetivo)

**La comparación principal es solo contra NAS multiobjetivo**: métodos cuyo vector de objetivos incluye el
rendimiento predictivo (accuracy/error) y al menos un objetivo de complejidad o eficiencia (params, FLOPs/MACs,
latencia, energía), y que devuelven un frente o varios modelos de compromiso. Los trabajos mono-objetivo
(Q-NAS, CGP-CNN, MetaQNN, CNN-GA, ResNet, baselines de MedMNIST v2) **no son comparadores**; como mucho aparecen
como filas de referencia separadas y etiquetadas (§6.2-ref, §6.3-ref).

Dentro de los multiobjetivo, los comparadores se emparejan por **par de objetivos**, porque eso decide qué
experimento propio se compara con cada uno:

| Par de objetivos en la literatura | Métodos | Experimento propio que se compara |
|---|---|---|
| error + FLOPs (MACs) | NSGA-Net macro/micro [17], NSGANetV1 [37], EEEA-Net | **Tier B** (acc-FLOPs, Standard) — mismos objetivos |
| error + params | CARS, LEMONADE, EEEA-Net, NSGANetV1 (también reporta params) | **Tier A** proyectado al plano (error, params); también Tier B, porque todas las redes tienen params |
| error + latencia | DPP-Net [50], HW-PR-NAS [19] | solo cualitativo (la latencia depende del hardware) |

Consecuencia para el plan de retrain: **el Tier B sube a la misma prioridad que el Tier A**, porque es el único
experimento propio con los mismos objetivos que NSGA-Net y NSGANetV1.

Se buscó además en la bibliografía que el paper ya cita ([15], [17], [18], [19], [21], [37], [38], [49], [50]) para no
añadir citas innecesarias. Para cada método se registran objetivos, tipo de espacio, protocolo de entrenamiento final
y coste, porque esos factores deciden si la comparación es directa, parcial o solo cualitativa.

**Cómo se compara un frente con trabajos que publican puntos.** Los papers multiobjetivo casi nunca publican su
frente completo, solo 1–9 modelos seleccionados, así que no se puede calcular HV contra ellos. Se usan tres
instrumentos, todos sobre el frente **reentrenado** de MoQ-NAS (de ahí la necesidad de los `strat10`):
1. **Figura de superposición**: error de test vs. params (y vs. MACs), eje x logarítmico; frente reentrenado de
   MoQ-NAS (mediana/attainment de las 3 corridas) + puntos de la literatura con marcador según protocolo final.
2. **Dominancia por punto**: para cada modelo publicado, indicar si algún punto de MoQ-NAS lo domina, si él domina
   a MoQ-NAS o si son mutuamente no dominados.
3. **Error a complejidad igualada**: para cada modelo publicado con P parámetros (o M MACs), el mejor error de
   MoQ-NAS con params ≤ P (o MACs ≤ M); reportar Δerror. Es la misma idea que el "accuracy a FLOPs igualado" de la
   ablación de espacio.
Añadir siempre GPU-días de búsqueda y el protocolo final, porque las diferencias de protocolo (600 épocas, cutout,
aux head) explican buena parte de las diferencias de error.

"✔" = valor verificado en el paper original (fuente indicada). "TODO" = método relevante cuyo valor no se pudo
extraer; hay que leerlo del PDF antes de usarlo. **No usar ningún TODO sin verificarlo.**

### 6.2 Tabla 1 — CIFAR-10, NAS multiobjetivo

Columnas propuestas: Método | Objetivos | Tipo de espacio | Test error (%) | Params (M) | MACs (M) |
GPU-días de búsqueda | Protocolo final. La tabla de métodos multiobjetivo está más abajo
("NAS multiobjetivo evolutivo"); es la tabla principal.

#### 6.2-ref Filas de referencia mono-objetivo (opcionales, separadas y etiquetadas)

No son comparadores. Solo se justifica incluir **Q-NAS** (paper y reimplementación propia), porque MoQ-NAS es su
extensión multiobjetivo: muestra qué se gana o se pierde en el punto de mayor accuracy al pasar a un frente.
El resto de esta subsección queda como contexto por si el revisor pide el grupo "sin celdas" de la Tabla 6.

**Hand-designed** (Tabla 6 de Q-NAS ✔): ResNet [45] 93.57 % / 1.7 M; VGG (según CGP-CNN) 92.06 % / 15.2 M;
Network in Network 91.19 %; Maxout 90.70 %.

**NAS sin celdas, mono-objetivo (mismo tipo de espacio)**

| Método | Acc. (%) | Params | GPU-días | Fuente |
|---|---|---|---|---|
| Q-NAS 2-C 1 (Conv-set-1 = espacio Standard) | 92.96 | 1.6 M | 58 | Tabla 6, `QNAS_Ref.pdf` ✔ |
| Q-NAS 3-C 2 + early-stop | 93.70 | 3.8 M | 48 | ✔ |
| Q-NAS 2-Res + early-stop | 93.85 | 7.07 M | 67 | ✔ |
| CGP-CNN (ConvSet) / (ResSet) | 93.25 / 94.02 | 1.52 / 1.68 M | – / 28 | ✔ |
| MetaQNN | 93.08 | 11.18 M | 100 | ✔ |
| Large-scale Evolution | 94.60 | 5.4 M | 2 670 | ✔ |
| NASBOT | 91.31 | – | 1.67 | ✔ |
| Genetic CNN | 92.90 | – | 17 | ✔ (Tabla 6 lo pone con celdas) |
| CNN-GA (Sun et al., TCYB 2020) | 95.22 (96.78 con cutout) | 2.9 M | 35 | ✔ [CNN-GA] |
| **Q-NAS, reimplementación propia** (Standard, 150 gen., F13) | 90.1–93.9 (p. ej. exp14: 93.68/93.41/93.20) | 0.5–3.5 M | medir | `data/qnas/experiment_v3_cifar10` ✔ |

La fila de Q-NAS propia usa **resultados ya existentes; Q-NAS no se reentrena**. Si se incluye, debe ser solo con
los exp de F13-v1 (exp5, exp7–exp15 en el espacio Standard), que comparten implementación, espacio, búsqueda proxy y
protocolo de retrain con lo que se reentrena ahora. Reportar media ± sd de las 3 corridas de la configuración que
coincida con la de MoQ-NAS, con sus GPU-días medidos en este hardware.

**NAS multiobjetivo evolutivo (lo que pide el revisor)**

| Método | Objetivos | Espacio | Resultado CIFAR-10 | GPU-días | Protocolo final | Fuente |
|---|---|---|---|---|---|---|
| NSGA-Net (macro) [17] | error + FLOPs | **macro (sin celdas)** | 96.15 % (3.85 % err.), 3.3 M, 1 290 MFLOPs | 8 | TODO: confirmar el protocolo del macro (el de 600 ép. + cutout está documentado para micro) | ✔ valores [NSGA-Net] |
| NSGA-Net (micro) [17] | error + FLOPs | celdas | 97.25 % (2.75 %), 3.3 M, 535 MFLOPs | 4 | ídem | ✔ |
| NSGANetV1 A0/A1/A2/A3/A4 [37] | error + FLOPs | celdas (búsqueda en CIFAR-100) | 95.33/96.51/97.35/97.78/97.98 %; 0.2/0.5/0.9/2.2/4.0 M | 27 | 600 ép., batch 96, cutout, drop-path, aux head | ✔ [NSGANetV1, Tabla IIa] |
| LEMONADE | 5 obj. (incl. params, mult-adds, tiempo) | celdas + morfismos | 95.43/96.31/96.95/97.42 %; 0.5/1.1/4.7/13.1 M | 56–90 (según la fuente) | 600 ép., cutout + mixup | ✔ [LEMONADE] |
| CARS (A … I) | acc + params | celdas, supernet | 97.00 % (2.4 M) … 97.38 % (3.6 M) | 0.4 | estilo DARTS | ✔ [CARS] |
| EEEA-Net A/B/C | error + params + FLOPs | celdas | 96.31/97.12/97.54 %; 1.8/1.8/3.6 M | 0.34/0.36/0.52 | estilo DARTS | ✔ [EEEA-Net] |
| **LightMix** (Huang et al., *IEEE TETCI* 10(2), 2026) | accuracy + complejidad (params) | bloque mixed-scale propio, zero-cost proxy | ✔ mejor modelo: 2.52 % err., 1.78 M. TODO: LightMixNet-1/2/3 (≈ 3.48/2.92/2.67 %, 0.49/0.97/1.45 M según fuente secundaria) y MAdds | TODO (≈ 0.02 según fuente secundaria) | TODO | **misma revista del paper**; prioridad máxima; cubre justo el rango 0.5–1.8 M de MoQ-NAS |
| Bi-MOEA/D-NAS (Liang et al., *Appl. Sci.* 2024) [48] | error + params | supernet weight-sharing | TODO (fuente secundaria: 2.72–2.86 %, 2.53–3.26 M) | TODO (≈ 0.5) | TODO | **ya citado** como [48] |
| LaMOO para MO-NAS (Zhao et al., *JMLR* 25, 2024) [20] | accuracy + params (+ FLOPs/latencia) | particiones del espacio | TODO (fuente secundaria: 97.36 %, 1.62 M) | coste en nº de muestras (600), no GPU-días | TODO | **ya citado** como [20] |
| Pareto-NASH (Elsken et al., 2018) | error + params | morfismos (precursor de LEMONADE) | TODO (fuente secundaria: ≈ 4.6 % < 1 M; ≈ 3.5 % ≈ 4 M) | TODO (≈ 56) | TODO | nuevo |
| NSGANetV2 (Lu et al., ECCV 2020) | accuracy + params/FLOPs/latencia | supernet + surrogate | TODO | TODO | fine-tuning de subredes | contexto (supernet/surrogate) |
| DPP-Net [50] | error + params + latencia | celdas, progresivo | TODO | TODO | TODO | ya citado; tabla hardware-aware |
| RNSGA-Net (Tong & Du, PR 2022) [49] | error + FLOPs (punto de referencia) | NSGA-Net + residual/dense | TODO (fuente secundaria: 3.89 %, 3.00 M) | TODO (≈ 3.11) | TODO | ya citado; mismos objetivos que el Tier B |
| MOEA-PS (Xue et al., TEVC 2023) [18] | precisión + "time consumption" | TODO | TODO (fuente secundaria: ≈ 2.77 %; params inconsistentes entre fuentes) | TODO | TODO | ya citado; su objetivo temporal no es Time_CUDA |
| Lyu et al. (InfSci 2024) [15] | multiobjetivo, policy gradient | DARTS | TODO (fuente secundaria: 2.70 %, 3.2 M) | TODO (≈ 1.3, 1080Ti) | TODO | ya citado; su tabla sirve para descubrir valores de otros MO-NAS |

Los valores marcados como "fuente secundaria" vienen de un borrador externo y **no están verificados**: confirmar
cada uno en el paper primario (página y tabla) antes de usarlo. Registrar la verificación en
`literature/cifar10_monas_comparison.csv` con las columnas: `method, year, objectives, search_space_type,
test_error, params_m, compute_value, compute_unit, latency, latency_device, search_cost, search_cost_unit,
search_hardware, retrain_epochs, data_augmentation, primary_source, page_table, verified`.

Lectura honesta esperada: con P1, MoQ-NAS quedará ~3–5 pp por debajo de los métodos basados en celdas y de los que
usan 600 épocas con cutout. La comparación multiobjetivo defendible es:
- **directa** contra NSGA-II/III reentrenados con P1 (Casos 1 y Tier B: mismo espacio, objetivos, presupuesto y protocolo);
- **parcial** contra NSGA-Net macro (mismos objetivos que el Tier B y también sin celdas, pero distinto espacio y
  protocolo final) y contra LEMONADE/NSGANetV1 en la región de < 1 M parámetros (donde están las redes de MoQ-NAS);
  declarar las diferencias en la columna de protocolo y, si se quiere acercar el protocolo, usar P2 en los `best_acc`;
- **cualitativa / de contexto** contra los métodos con celdas o supernet (CARS, EEEA-Net, NSGA-Net micro,
  NSGANetV1), separándolos como hace Q-NAS al final de su Tabla 6 por el sesgo humano de la meta-arquitectura.

El argumento que sí sostienen los datos es de **eficiencia**: frentes completos en 0.69 ± 0.08 GPU-días de búsqueda
(Caso 1), con redes de 0.05–1.7 M parámetros. Es el mismo tipo de argumento que usa Q-NAS (93.85 % en 67 GPU-días
frente a 2 670 de Large-scale Evolution).

### 6.3 Tabla 2 — MedMNIST (28×28), ACC / AUC

**Decisión (2026-10-04): comparar directamente con el benchmark MedMNIST v2 [34].** Como casi no hay trabajos
multiobjetivo en MedMNIST, la tabla de MedMNIST se construye contra los 7 métodos del paper oficial: ResNet-18 y
ResNet-50 a 28 y 224, auto-sklearn, AutoKeras y Google AutoML Vision. Todos usan los splits oficiales y reportan
ACC y AUC.

Encuadre (para no sobre-afirmar):
- Esos métodos son **mono-objetivo**. No es una comparación de multiobjetivo contra multiobjetivo, sino de **nivel
  de desempeño frente a un benchmark establecido, más eficiencia**: "las redes del frente alcanzan ACC/AUC
  comparables a ResNet y AutoML con un orden de magnitud menos de parámetros" (< 2 M frente a ≈ 11 M en ResNet-18 y
  ≈ 23.5 M en ResNet-50). La comparación multiobjetivo propiamente dicha sigue siendo contra NSGA-II/III (mismo
  protocolo) y contra la literatura multiobjetivo de CIFAR-10 (Tabla 1).
- **MoQ-NAS, NSGA-II y NSGA-III aparecen los tres**, con el mismo protocolo y los mismos representantes por regla:
  la tabla muestra también la comparación interna después del reentrenamiento.
- **Dos filas por algoritmo:**
  - **F13-v1** (protocolo del estudio): fila principal, con la columna de protocolo declarada (300 épocas, AdamW,
    TrivialAugment). Tiene más presupuesto y augmentation que las ResNet de MedMNIST v2.
  - **P-Med-A** (control): protocolo idéntico al de las ResNet de MedMNIST v2 (100 épocas, Adam 1e-3, ×0.1 en las
    épocas 50 y 75, batch 128, sin augmentation, mejor checkpoint de validación). Si las redes siguen siendo
    competitivas aquí, la conclusión no depende del protocolo de reentrenamiento. Única diferencia conocida: fp16
    (la precisión del caso); MedMNIST v2 no la indica.
- Comparar con las filas **28×28** de ResNet, que es la resolución usada. Las de 224×224 son solo contexto.
- Reportar **ACC y AUC** (evaluador oficial de `medmnist`, ya integrado: `acc_medmnist`, `auc_score`).
- Nuestros valores: media ± sd sobre las 3 corridas de búsqueda por algoritmo (representantes elegidos tras el
  screening `strat5`, confirmación con 1 semilla nueva; ver Tier C).
  MedMNIST v2: una sola corrida por método. Indicarlo en la nota de la tabla.
- AutoKeras y Google AutoML Vision no reportan parámetros: "n/d".
- Filas propias de Q-NAS [29], [30]: referencia con su protocolo (F13), sin reentrenar.

**Comparadores multiobjetivo en MedMNIST (todos por verificar).** La literatura multiobjetivo en MedMNIST es escasa;
estos son los candidatos encontrados:

| Trabajo | Multiobjetivo | Objetivos / espacio | Qué verificar |
|---|---|---|---|
| Luong, Phan, Vo, Pham, Bui, "Lightweight multi-objective evolutionary NAS with low-cost proxy metrics", *Information Sciences*, 2023 | sí | accuracy + coste (params/FLOPs), zero-cost proxies; experimentos en NAS-Bench-201 y MedMNIST | qué datasets de MedMNIST, si reportan ACC/AUC de test tras entrenar, params/FLOPs, espacio (probablemente celdas NB201) |
| "An Efficient Multi-Objective Evolutionary Zero-Shot NAS Framework for Image Classification" (PubMed 36775287, 2023; mismo grupo) | sí | zero-shot, NAS-Bench-201 + MedMNIST | ídem |
| PBC-NAS / BioNAS (*Artif. Intell. Med.* 160, 2025) | **por confirmar** | reportan ACC, AUC y FLOPs; la extensión 3D se describe como "balancear accuracy y complejidad" | si la búsqueda es realmente multiobjetivo o mono-objetivo con FLOPs reportados; valores por dataset |
| WQ-NAS [31] (propio) | escalarizado (suma ponderada) | Q-NAS con objetivos ponderados | usar como fila propia de "multiobjetivo escalarizado" frente a Pareto |

Si tras verificar quedan menos de dos trabajos multiobjetivo con datasets y protocolo compatibles, la comparación
multiobjetivo externa la sostiene CIFAR-10 (Tabla 1), y en MedMNIST el paper se apoya en la comparación con
NSGA-II/III a protocolo igualado (ya está) más las filas de referencia de abajo. Decirlo explícitamente en el texto
es mejor que forzar una comparación con protocolos distintos.

#### 6.3-ref Filas de referencia (mono-objetivo, reutilizables de MedMNIST v2)

Las tablas de MedMNIST v2 se pueden reaprovechar tal cual como **referencia de nivel** (mismo split oficial, ACC y
AUC), etiquetadas como mono-objetivo. Su valor es mostrar que los modelos del frente de MoQ-NAS, con < 2 M
parámetros, alcanzan el nivel de ResNet-18/50 (≈ 11 / 23.5 M) y de AutoML; no son comparadores multiobjetivo.

Baselines oficiales de MedMNIST v2 [34] (✔ [MedMNIST v2]; AUC / ACC):

| Método | PathMNIST | OCTMNIST | TissueMNIST | OrganAMNIST |
|---|---|---|---|---|
| ResNet-18 (28) | 0.983 / 0.907 | 0.943 / 0.743 | 0.930 / 0.676 | 0.997 / 0.935 |
| ResNet-18 (224) | 0.989 / 0.909 | 0.958 / 0.763 | 0.933 / 0.681 | 0.998 / 0.951 |
| ResNet-50 (28) | 0.990 / 0.911 | 0.952 / 0.762 | 0.931 / 0.680 | 0.997 / 0.935 |
| ResNet-50 (224) | 0.989 / 0.892 | 0.958 / 0.776 | 0.932 / 0.680 | 0.998 / 0.947 |
| auto-sklearn | 0.934 / 0.716 | 0.887 / 0.601 | 0.828 / 0.532 | 0.963 / 0.762 |
| AutoKeras (NAS) | 0.959 / 0.834 | 0.955 / 0.763 | 0.941 / 0.703 | 0.994 / 0.905 |
| Google AutoML Vision | 0.944 / 0.728 | 0.963 / 0.771 | 0.924 / 0.673 | 0.990 / 0.886 |

AutoKeras y Google AutoML Vision son AutoML/NAS, pero mono-objetivo: van en el bloque de referencia. Otras filas de
referencia posibles (también mono-objetivo, salvo que la verificación diga lo contrario):
- Q-NAS de trabajos propios [29], [30] (valores de tus papers; reentrenos F13 en `data/medmnist/qnas`).
- MSTF-NAS (citado en el estudio de AIIM 2025 como el mejor en ACC media) — TODO.
- NAS evolutivo con zero-cost proxies para MedMNIST 2D/3D (Springer, 2024) y NAS robusto con evolución
  diferencial en MedMNIST (Springer, 2024) — TODO: confirmar si alguno es multiobjetivo; si lo es, sube a la tabla
  de comparadores.

Columna obligatoria: params (ResNet-18 ≈ 11 M, ResNet-50 ≈ 23.5 M frente a < 2 M de MoQ-NAS) y protocolo
(F13-v1 MedMNIST: 300 épocas, AdamW, TrivialAugment; MedMNIST v2: 100 épocas, Adam, sin augmentation). Como esas
filas son de referencia, basta con declarar la diferencia de protocolo.

### 6.4 Detalles de unidades (para que la tabla sea correcta)

- **Error vs. accuracy:** el código ya optimiza `−Acc` (minimización); el error es `100 − Acc` y produce el mismo
  orden y el mismo frente. En la tabla final: *test error* = 100 − test accuracy tras el retrain. No reportar el
  error proxy.
- **FLOPs:** `core/cnn/metrics/base_hardware.py:218` cuenta `2 × MACs` (mul + add) solo en Conv/Linear. Para
  MoQ-NAS la conversión es exacta porque conocemos el contador: `MACs = total_flops / 2`; reportar ambas.
  Para la literatura **no convertir nada**: guardar el valor y la unidad tal como los reporta cada paper (FLOPs,
  MFLOPs, MAdds, MACs) y registrar la definición si la dan. Muchos llaman "FLOPs" a multiply-adds, pero no todos.
  En la tabla, columnas `Compute` + `Unit`, o una nota metodológica.
- **Search cost:** conservar la unidad original de cada paper (GPU-días, horas, nº de muestras evaluadas en LaMOO,
  búsqueda zero-cost en LightMix) y el hardware; no normalizar artificialmente.
- **Tiempo de inferencia CUDA:** no comparable entre papers (hardware distinto). No ponerlo en la tabla de
  literatura; usar params y MACs.
- **GPU-días:** reportar la búsqueda (ya la tienes) y, aparte, el coste del retrain medido; Q-NAS hace igual.

### 6.5 Tabla 3 — Fairness (Caso 3): solo comparación cualitativa

MoENAS [38] usa FACET con disparidad de accuracy entre tonos de piel (14.09 % → 5.60 %), espacio Mixture-of-Experts
y otra métrica; Dooley et al. [55] es reconocimiento facial. No hay un comparador con el mismo objetivo, datos y
protocolo → mantener la comparación con arquitecturas fijas, ahora **simétrica y con datos completos** (Tier D), y
citar MoENAS/Dooley en una tabla de posicionamiento (objetivos, espacio, estrategia, dataset). Si al verificar MoENAS
resulta que reporta la misma tarea (clasificación de persona en FACET) con una métrica convertible a D_group o
MeanTPR, el frente reentrenado con datos completos permitiría añadirlo como punto de comparación parcial.

---

## 7. Qué hacer con los resultados del retrain

1. **Frente reentrenado por algoritmo (Caso 1 y Tier B):** con los `strat10`, recalcular el frente no dominado en
   (test error, params[, MACs]) y su HV con el mismo punto de referencia para los tres algoritmos.
   Pregunta que responde: ¿se mantiene la conclusión del paper (sin diferencias significativas en HV, MoQ-NAS más
   barato) cuando la accuracy es la final y no la proxy?
2. **Fidelidad del proxy:** Kendall τ / Spearman entre proxy y test reentrenado, por algoritmo. Responde
   directamente a la limitación que el propio paper declara en la sección "Efficiency Mechanisms".
3. **Tabla de literatura multiobjetivo:** MoQ-NAS con 2–3 puntos (`compact`, `knee`, `best_acc`), media ± sd sobre
   corridas (y sd entre semillas), más NSGA-II/III reentrenados en las mismas condiciones, junto a la figura de
   superposición, la dominancia por punto y el error a complejidad igualada de §6.1.
4. **Afirmación prudente propuesta** (ajustar a los números reales): "Under a matched retraining protocol, MoQ-NAS
   recovers a front of compact architectures (0.05–1.7 M parameters) in a fraction of the search cost reported by
   other multi-objective evolutionary NAS methods; in the low-parameter region its retrained solutions are
   [non-dominated by / within X pp of] the reported models of NSGA-Net and LEMONADE, whereas cell-based methods
   trained with longer, more heavily regularized schedules remain more accurate."

---

## 8. Riesgos y limitaciones a declarar

- El proxy puede ordenar mal las redes grandes (el paper ya lo dice); el análisis de §7.2 lo cuantifica, pero puede
  mostrar que parte del frente proxy no es frente tras el retrain.
- Una sola búsqueda por configuración y 3 corridas: las diferencias entre algoritmos tras el retrain deben tratarse
  con el mismo cuidado estadístico que el HV (Welch + Holm), o de forma descriptiva.
- Los GPU-días de la búsqueda dependen del entorno de ejecución (ver la nota del 92.2 % de la ablación); no
  comparar GPU-días entre papers como si fueran del mismo hardware.
- Valores marcados como TODO en §6: verificar en el PDF antes de citarlos.
- **La latencia (Time_CUDA) medida durante la búsqueda no es reproducible** (ver §8.1). Afecta al objetivo del
  Caso 1 (tri-objetivo) y del Caso 2 (MedMNIST).

### 8.1 Limitación: medición de la latencia (Time_CUDA) en GPU compartida

**Cómo se midió en la búsqueda.** `HardwareMetrics` mide la latencia al terminar el entrenamiento de cada
candidato: lote de 10 imágenes, 5 iteraciones de calentamiento, media de 10 repeticiones con
`torch.cuda.synchronize()` (`core/cnn/metrics/hardware.py`, `core/cnn/metrics/base_hardware.py`). Las corridas del
Caso 1 tenían `threads: 20` en 2 GPUs, es decir ~10 candidatos entrenándose a la vez en cada GPU
(`core/evaluation.py::_num_workers`). Cada latencia se midió mientras otros candidatos entrenaban en la misma GPU, y
posiblemente con procesos de otros usuarios en el clúster.

**Evidencia de que no es reproducible** (`reports/ablation_search_space/`): las mismas 307 arquitecturas, medidas en
dos ejecuciones de la misma búsqueda:

| Objetivo | Valores idénticos | Cambio entre ejecuciones (mediana [Q1–Q3], máx.) | Spearman |
|---|---|---|---|
| Accuracy, FLOPs, parámetros | 100 % | ×1 | 1.00 |
| **Latencia CUDA** | **0 %** | **×4.3 [×2.7–×10.4], máx. ×139** | **0.30** |

El frente no dominado en (accuracy, latencia) cambia casi por completo entre ejecuciones (Jaccard 0.10–0.33 por
corrida; archivo 58), mientras que en (accuracy, FLOPs) es idéntico (Jaccard 1.0).

**Qué implica.**
- El objetivo Time_CUDA de los Casos 1 y 2 tiene **mucho ruido de medición**, que depende de la carga de la GPU en el
  momento de medir y no solo de la arquitectura. Como afectó por igual a MoQ-NAS, NSGA-II y NSGA-III (mismo
  protocolo y misma concurrencia), la comparación entre algoritmos no queda sesgada en una dirección conocida. Pero
  el HV y los frentes que incluyen la latencia heredan ese ruido, y los valores individuales de latencia no deben
  interpretarse como la latencia real de cada red.
- Declararlo en el paper como limitación: la latencia se midió bajo ejecución concurrente en GPU compartida y es
  poco reproducible. FLOPs y parámetros son medidas deterministas y sirven como proxies de complejidad robustos.
  Mencionarlo también como motivo para usar FLOPs en la ablación de espacio.
- **La latencia no entra en la tabla de comparación con la literatura** (además, depende del hardware de cada
  paper). Se usan parámetros y MACs.
- **Los resultados de accuracy/AUC/fairness no se ven afectados** por compartir la GPU: el cálculo es el mismo, solo
  cambia el tiempo. Por eso los reentrenamientos sí pueden ejecutarse en paralelo.

**Si se reporta latencia de los representantes reentrenados** (solo en una tabla hardware-aware aparte), volver a
medirla con la GPU **dedicada**:
1. comprobar con `nvidia-smi` que no hay ningún otro proceso en esa GPU antes y durante la medición (el clúster es
   compartido: buscar o coordinar una ventana libre);
2. `model.eval()` + `torch.inference_mode()`, precisión del caso y lote fijos (reportar el tamaño de lote);
3. calentamiento suficiente (≥ 50 iteraciones), `torch.cuda.synchronize()` antes y después de cada medida;
4. muchas repeticiones (≥ 200) y reportar **mediana** y p95, no la media;
5. misma GPU (L40S), mismo entorno (versión de torch/CUDA) y misma resolución para todos los algoritmos;
6. repetir la medición en al menos dos momentos distintos para comprobar la estabilidad.

---

## Fuentes consultadas (literatura)

- Q-NAS: Szwarcman, Civitarese, Vellasco, *Applied Soft Computing* 120 (2022), Tabla 6 (`case of study paper/QNAS_Ref.pdf`).
- NSGA-Net: https://arxiv.org/abs/1810.03522
- NSGANetV1: https://arxiv.org/abs/1912.01369 (Tabla IIa; PDF: https://www.cs.cityu.edu.hk/~zhichalu/assets/paper/20_TEVC_NSGANetV1.pdf)
- LEMONADE: https://arxiv.org/abs/1804.09081
- CARS: https://arxiv.org/abs/1909.04977
- EEEA-Net: https://arxiv.org/abs/2108.06156
- CNN-GA: https://arxiv.org/abs/1808.03818
- DPP-Net: https://arxiv.org/abs/1806.08198
- RNSGA-Net: https://doi.org/10.1016/j.patcog.2022.108962
- Lyu et al. 2024: https://opus.lib.uts.edu.au/handle/10453/176264
- MedMNIST v2: https://arxiv.org/abs/2110.14795
- BioNAS / PBC-NAS (AIIM 2025): https://research.itu.edu.tr/en/publications/neural-architecture-search-for-biomedical-image-classification-a-/
- NAS evolutivo 2D/3D MedMNIST: https://link.springer.com/chapter/10.1007/978-3-031-63751-3_9
- NAS robusto con DE en imágenes médicas: https://link.springer.com/chapter/10.1007/978-3-031-56855-8_10
- MoENAS: https://arxiv.org/abs/2502.07422
- LightMix (IEEE TETCI 10(2), 2026): https://ieeexplore.ieee.org/document/11023228/
- Bi-MOEA/D-NAS (Applied Sciences 2024, ref. [48]): https://doi.org/10.3390/app14146143
- LaMOO / MO-NAS por particiones (JMLR 25, 2024, ref. [20]): https://jmlr.org/papers/v25/23-1013.html
