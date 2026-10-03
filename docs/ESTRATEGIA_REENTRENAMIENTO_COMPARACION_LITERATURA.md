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
2. **Re-Pareto con la validación del retrain** (nunca con test): frentes en (error_val, params), (error_val, MACs)
   y, para el Caso 1, (error_val, params, Time_CUDA). Cuantificar cuántas soluciones cambian de dominancia.
3. **Representantes por reglas fijadas antes de ver el test**, aplicadas al frente reentrenado:
   - `A`: menor error de validación;
   - `P`: menos parámetros; `F`: menos MACs;
   - `K`: knee = mínima distancia euclídea al ideal tras normalización min-max dentro del frente (escribir la regla
     en el paper; no elegirlo visualmente);
   - **por presupuesto**: la red de menor error de validación con params ≤ {0.25, 0.5, 1.0, 1.5} M (Tier A) y
     MACs ≤ {50, 100, 250, 500} M (Tier B). Estos presupuestos cubren el rango de los frentes de MoQ-NAS y coinciden con
     los de LightMix, LEMONADE y NSGANetV1 (0.2–1.8 M); no cambiarlos después de ver resultados.
4. **Confirmación (3 semillas nuevas)** de los representantes: semillas 11, 12 y 13, distintas de la del
   screening (semilla 1), igual para los tres algoritmos. La corrida del screening no cuenta como réplica: los
   representantes se eligieron por su validación en esa semilla, y reutilizarla sesgaría la media hacia arriba (el
   mismo criterio que en R1 de fairness). Solo esta etapa alimenta la tabla del paper (media ± sd de test error).
   La semilla 1 basta en el screening porque la variación entre semillas de una misma red es pequeña (sd ≈
   0.05–0.36 pp en los F13 de Q-NAS) frente a las diferencias entre arquitecturas del frente (varios pp).

   ```bash
   # etapa 1 (screening) y etapa 2 (confirmación, tras elegir representantes con la validación)
   python launch_retrain_protocol.py --cases C1_triobj --roles all --seeds 1 --tag F13v1 --gpus 0 1
   python launch_retrain_protocol.py --cases C1_triobj --runs <run> --ids <ids elegidos> \
       --roles all --seeds 11 12 13 --tag F13v1 --gpus 0 1
   ```

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

### Tier C — Caso 2, MedMNIST `[-Acc, Params, Time_CUDA]` (prioridad 3)

`medmnist/moqnas_last/experiment_{ds}_medmnist/moqnas/exp1_repeat_{1,2,3}` y
`medmnist/NSGA/experiment_{ds}_{nsga2,nsga3}/{algo}/exp1_repeat_{1,2,3}` para
`ds ∈ {pathmnist, octmnist, tissuemnist, organamnist}`. Representantes: 23–26 por dataset (≈101 en total).

- Mínimo: representantes × 3 semillas ≈ **300 entrenamientos** (cortos con el protocolo MedMNIST de 100 épocas).
- Ampliado: `strat10` (360 redes) × 1 semilla, solo si el presupuesto lo permite.
- Orden sugerido: OrganA y Path primero (más pequeños/rápidos), TissueMNIST al final (165 k imágenes de train).

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
- **Latencia (solo tabla hardware-aware):** volver a medir Time_CUDA de los representantes con un procedimiento
  fijo (`eval()`, `inference_mode`, precisión y batch fijos, warm-up, `torch.cuda.synchronize()`, mediana de
  varias repeticiones, misma L40S).

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
