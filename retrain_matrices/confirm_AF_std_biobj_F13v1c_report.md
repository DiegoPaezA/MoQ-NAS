# Representantes AF_std_biobj (tag F13v1c, validación de la semilla 1)

Frente: val_err, macs. Compacto (C): menor complejidad a <= 5 pp de A. Presupuestos: 50, 100, 250, 500 M MACs.

| Corrida | Algoritmo | Cribadas | En el frente (val) | Dominadas tras el retrain | Spearman proxy–val | Representantes |
|---|---|---|---|---|---|---|
| moqnas_repeat_1 | moqnas | 14 | 7 | 7 | 0.82 | 4 |
| moqnas_repeat_2 | moqnas | 12 | 7 | 5 | 0.59 | 3 |
| moqnas_repeat_3 | moqnas | 13 | 8 | 5 | 0.84 | 3 |
| nsga2_repeat_1 | nsga2 | 22 | 10 | 12 | 0.95 | 5 |
| nsga2_repeat_2 | nsga2 | 15 | 14 | 1 | 1.00 | 6 |
| nsga2_repeat_3 | nsga2 | 19 | 15 | 4 | 0.99 | 5 |
| nsga3_repeat_1 | nsga3 | 17 | 13 | 4 | 0.98 | 4 |
| nsga3_repeat_2 | nsga3 | 17 | 9 | 8 | 0.96 | 3 |
| nsga3_repeat_3 | nsga3 | 18 | 13 | 5 | 0.98 | 6 |

## experiment_cifar10_acc_flops/moqnas/moqnas_repeat_1

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 48_5 | 94.62 | 1.869 M | 536.1 |
| K | 40_5 | 93.22 | 1.020 M | 86.1 |
| C | 75_14 | 91.02 | 0.523 M | 64.1 |
| B50 | 92_8 | 88.20 | 0.480 M | 37.2 |
| B100 | 40_5 | 93.22 | 1.020 M | 86.1 |
| B250 | 40_5 | 93.22 | 1.020 M | 86.1 |
| B500 | 40_5 | 93.22 | 1.020 M | 86.1 |

## experiment_cifar10_acc_flops/moqnas/moqnas_repeat_2

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 91_14 | 92.16 | 0.648 M | 80.9 |
| K | 30_16 | 86.08 | 0.346 M | 9.1 |
| C | 132_0 | 89.00 | 0.454 M | 45.5 |
| B50 | 132_0 | 89.00 | 0.454 M | 45.5 |
| B100 | 91_14 | 92.16 | 0.648 M | 80.9 |
| B250 | 91_14 | 92.16 | 0.648 M | 80.9 |
| B500 | 91_14 | 92.16 | 0.648 M | 80.9 |

## experiment_cifar10_acc_flops/moqnas/moqnas_repeat_3

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 118_2 | 92.96 | 0.825 M | 134.9 |
| K | 93_13 | 86.14 | 0.129 M | 18.2 |
| C | 71_3 | 92.76 | 0.494 M | 95.2 |
| B50 | 93_13 | 86.14 | 0.129 M | 18.2 |
| B100 | 71_3 | 92.76 | 0.494 M | 95.2 |
| B250 | 118_2 | 92.96 | 0.825 M | 134.9 |
| B500 | 118_2 | 92.96 | 0.825 M | 134.9 |

## experiment_cifar10_acc_flops/nsga2/nsga2_repeat_1

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 148_11 | 95.68 | 1.569 M | 448.5 |
| K | 126_15 | 93.98 | 1.273 M | 110.1 |
| C | 89_2 | 91.16 | 1.603 M | 73.6 |
| B50 | 111_19 | 89.92 | 1.324 M | 32.7 |
| B100 | 128_3 | 93.26 | 0.975 M | 91.2 |
| B250 | 126_15 | 93.98 | 1.273 M | 110.1 |
| B500 | 148_11 | 95.68 | 1.569 M | 448.5 |

## experiment_cifar10_acc_flops/nsga2/nsga2_repeat_2

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 65_18 | 94.30 | 0.959 M | 271.0 |
| K | 143_8 | 92.34 | 0.717 M | 66.7 |
| C | 117_5 | 89.92 | 0.974 M | 34.7 |
| B50 | 147_8 | 90.92 | 1.352 M | 43.2 |
| B100 | 137_16 | 92.88 | 0.974 M | 96.5 |
| B250 | 115_9 | 93.54 | 1.450 M | 132.0 |
| B500 | 65_18 | 94.30 | 0.959 M | 271.0 |

## experiment_cifar10_acc_flops/nsga2/nsga2_repeat_3

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 112_1 | 93.78 | 0.770 M | 423.3 |
| K | 129_16 | 90.64 | 0.508 M | 43.8 |
| C | 136_3 | 89.14 | 0.386 M | 22.4 |
| B50 | 129_16 | 90.64 | 0.508 M | 43.8 |
| B100 | 93_10 | 91.80 | 0.358 M | 78.7 |
| B250 | 127_1 | 93.36 | 0.511 M | 159.0 |
| B500 | 112_1 | 93.78 | 0.770 M | 423.3 |

## experiment_cifar10_acc_flops/nsga3/nsga3_repeat_1

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 120_18 | 93.98 | 1.439 M | 104.9 |
| K | 112_13 | 89.74 | 0.554 M | 25.0 |
| C | 112_13 | 89.74 | 0.554 M | 25.0 |
| B50 | 129_3 | 92.30 | 1.274 M | 48.4 |
| B100 | 97_8 | 93.10 | 0.789 M | 94.5 |
| B250 | 120_18 | 93.98 | 1.439 M | 104.9 |
| B500 | 120_18 | 93.98 | 1.439 M | 104.9 |

## experiment_cifar10_acc_flops/nsga3/nsga3_repeat_2

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 29_13 | 92.68 | 0.564 M | 231.7 |
| K | 58_7 | 90.30 | 0.272 M | 27.3 |
| C | 83_2 | 89.86 | 0.252 M | 20.8 |
| B50 | 58_7 | 90.30 | 0.272 M | 27.3 |
| B100 | 58_7 | 90.30 | 0.272 M | 27.3 |
| B250 | 29_13 | 92.68 | 0.564 M | 231.7 |
| B500 | 29_13 | 92.68 | 0.564 M | 231.7 |

## experiment_cifar10_acc_flops/nsga3/nsga3_repeat_3

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 98_5 | 94.94 | 1.194 M | 390.0 |
| K | 147_19 | 92.30 | 0.406 M | 65.6 |
| C | 115_18 | 90.28 | 0.575 M | 18.4 |
| B50 | 149_4 | 91.06 | 0.517 M | 40.5 |
| B100 | 127_1 | 93.20 | 0.448 M | 93.2 |
| B250 | 129_16 | 94.44 | 1.030 M | 223.4 |
| B500 | 98_5 | 94.94 | 1.194 M | 390.0 |

# Análisis de la selección (solo validación)

Ruido de referencia entre semillas: 0.36 pp (sd máxima entre semillas de una misma red en los reentrenamientos previos). "Frontera": redes dominadas que entrarían al frente con +ruido (fuera) o miembros que saldrían con −ruido (dentro).

## Proxy → validación por algoritmo

| Algoritmo | Corridas | ρ Spearman (media) | τ Kendall (media) | Dominadas / cribadas | Frontera fuera / dentro | Mejor proxy = A | Representantes proxy conservados |
|---|---|---|---|---|---|---|---|
| moqnas | 3 | 0.75 | 0.62 | 17 / 39 | 0 / 2 | 0 / 3 | 2 / 7 |
| nsga2 | 3 | 0.98 | 0.91 | 17 / 56 | 5 / 9 | 2 / 3 | 5 / 9 |
| nsga3 | 3 | 0.97 | 0.89 | 17 / 52 | 8 / 9 | 2 / 3 | 5 / 9 |

## Estabilidad de cada elección

Estabilidad: fracción de 500 simulaciones con ruido N(0, ruido) en la accuracy de validación de todas las redes en las que la regla elige la misma red. Margen: pp de accuracy sobre la siguiente (A, presupuestos), pp sobre el umbral A−5 (C) o diferencia de distancia normalizada al ideal (K).

Si la estabilidad es < 70%, también se confirma la alternativa más frecuente de la misma regla en las simulaciones (etiqueta `<regla>~`).

| Corrida | Regla | id | val acc | Margen | Estabilidad | Alternativa más frecuente | ¿Se confirma también? | Detalle |
|---|---|---|---|---|---|---|---|---|
| moqnas moqnas_repeat_1 | A | 48_5 | 94.62 | 1.400 | 99% | 40_5 (1%) | no | runner-up 40_5 at 1.40 pp |
| moqnas moqnas_repeat_1 | K | 40_5 | 93.22 | 0.059 | 99% | 65_11 (1%) | no | distance 0.174; runner-up 75_14 at 0.233 |
| moqnas moqnas_repeat_1 | C | 75_14 | 91.02 | 1.400 | 99% | 40_5 (1%) | no | 1.40 pp above the A-5 threshold; next: 40_5 |
| moqnas moqnas_repeat_1 | B50 | 92_8 | 88.20 | 2.180 | 100% | — | no | runner-up 65_11 at 2.18 pp |
| moqnas moqnas_repeat_1 | B100 | 40_5 | 93.22 | 2.200 | 100% | — | no | runner-up 75_14 at 2.20 pp |
| moqnas moqnas_repeat_1 | B250 | 40_5 | 93.22 | 2.200 | 98% | 98_12 (2%) | no | runner-up 75_14 at 2.20 pp |
| moqnas moqnas_repeat_1 | B500 | 40_5 | 93.22 | 2.200 | 98% | 98_12 (2%) | no | runner-up 75_14 at 2.20 pp |
| moqnas moqnas_repeat_2 | A | 91_14 | 92.16 | 0.740 | 91% | 14_2 (5%) | no | runner-up 14_2 at 0.74 pp |
| moqnas moqnas_repeat_2 | K | 30_16 | 86.08 | 0.039 | 97% | 7_15 (3%) | no | distance 0.295; runner-up 7_15 at 0.334 |
| moqnas moqnas_repeat_2 | C | 132_0 | 89.00 | 1.840 | 67% | 7_15 (32%) | sí | 1.84 pp above the A-5 threshold; next: 112_5 |
| moqnas moqnas_repeat_2 | B50 | 132_0 | 89.00 | 2.000 | 100% | — | no | runner-up 7_15 at 2.00 pp |
| moqnas moqnas_repeat_2 | B100 | 91_14 | 92.16 | 0.740 | 92% | 14_2 (5%) | no | runner-up 14_2 at 0.74 pp |
| moqnas moqnas_repeat_2 | B250 | 91_14 | 92.16 | 0.740 | 91% | 14_2 (5%) | no | runner-up 14_2 at 0.74 pp |
| moqnas moqnas_repeat_2 | B500 | 91_14 | 92.16 | 0.740 | 91% | 14_2 (5%) | no | runner-up 14_2 at 0.74 pp |
| moqnas moqnas_repeat_3 | A | 118_2 | 92.96 | 0.200 | 65% | 71_3 (34%) | sí | runner-up 71_3 at 0.20 pp |
| moqnas moqnas_repeat_3 | K | 93_13 | 86.14 | 0.032 | 83% | 11_18 (15%) | no | distance 0.423; runner-up 11_18 at 0.455 |
| moqnas moqnas_repeat_3 | C | 71_3 | 92.76 | 4.800 | 85% | 141_18 (15%) | no | 4.80 pp above the A-5 threshold; next: 118_2 |
| moqnas moqnas_repeat_3 | B50 | 93_13 | 86.14 | 0.680 | 83% | 91_9 (10%) | no | runner-up 11_18 at 0.68 pp |
| moqnas moqnas_repeat_3 | B100 | 71_3 | 92.76 | 5.140 | 100% | — | no | runner-up 141_18 at 5.14 pp |
| moqnas moqnas_repeat_3 | B250 | 118_2 | 92.96 | 0.200 | 66% | 71_3 (34%) | sí | runner-up 71_3 at 0.20 pp |
| moqnas moqnas_repeat_3 | B500 | 118_2 | 92.96 | 0.200 | 66% | 71_3 (34%) | sí | runner-up 71_3 at 0.20 pp |
| nsga2 nsga2_repeat_1 | A | 148_11 | 95.68 | 1.700 | 65% | 74_4 (18%) | sí | runner-up 126_15 at 1.70 pp |
| nsga2 nsga2_repeat_1 | K | 126_15 | 93.98 | 0.023 | 74% | 128_3 (25%) | no | distance 0.288; runner-up 128_3 at 0.311 |
| nsga2 nsga2_repeat_1 | C | 89_2 | 91.16 | 0.480 | 76% | 128_3 (21%) | no | 0.48 pp above the A-5 threshold; next: 128_3 |
| nsga2 nsga2_repeat_1 | B50 | 111_19 | 89.92 | 1.000 | 96% | 148_15 (2%) | no | runner-up 148_15 at 1.00 pp |
| nsga2 nsga2_repeat_1 | B100 | 128_3 | 93.26 | 2.100 | 100% | — | no | runner-up 89_2 at 2.10 pp |
| nsga2 nsga2_repeat_1 | B250 | 126_15 | 93.98 | 0.720 | 47% | 86_13 (43%) | sí | runner-up 128_3 at 0.72 pp |
| nsga2 nsga2_repeat_1 | B500 | 148_11 | 95.68 | 1.700 | 76% | 74_4 (24%) | no | runner-up 126_15 at 1.70 pp |
| nsga2 nsga2_repeat_2 | A | 65_18 | 94.30 | 0.760 | 91% | 115_9 (5%) | no | runner-up 115_9 at 0.76 pp |
| nsga2 nsga2_repeat_2 | K | 143_8 | 92.34 | 0.014 | 59% | 124_14 (25%) | sí | distance 0.289; runner-up 124_14 at 0.303 |
| nsga2 nsga2_repeat_2 | C | 117_5 | 89.92 | 0.620 | 70% | 141_2 (17%) | no | 0.62 pp above the A-5 threshold; next: 88_5 |
| nsga2 nsga2_repeat_2 | B50 | 147_8 | 90.92 | 0.040 | 56% | 88_5 (43%) | sí | runner-up 88_5 at 0.04 pp |
| nsga2 nsga2_repeat_2 | B100 | 137_16 | 92.88 | 0.500 | 73% | 124_14 (14%) | no | runner-up 124_14 at 0.50 pp |
| nsga2 nsga2_repeat_2 | B250 | 115_9 | 93.54 | 0.080 | 50% | 144_3 (36%) | sí | runner-up 144_3 at 0.08 pp |
| nsga2 nsga2_repeat_2 | B500 | 65_18 | 94.30 | 0.760 | 91% | 115_9 (5%) | no | runner-up 115_9 at 0.76 pp |
| nsga2 nsga2_repeat_3 | A | 112_1 | 93.78 | 0.420 | 77% | 127_1 (22%) | no | runner-up 127_1 at 0.42 pp |
| nsga2 nsga2_repeat_3 | K | 129_16 | 90.64 | 0.003 | 34% | 148_5 (22%) | sí | distance 0.204; runner-up 148_5 at 0.207 |
| nsga2 nsga2_repeat_3 | C | 136_3 | 89.14 | 0.360 | 70% | 113_9 (26%) | no | 0.36 pp above the A-5 threshold; next: 113_9 |
| nsga2 nsga2_repeat_3 | B50 | 129_16 | 90.64 | 0.260 | 72% | 113_9 (28%) | no | runner-up 113_9 at 0.26 pp |
| nsga2 nsga2_repeat_3 | B100 | 93_10 | 91.80 | 0.460 | 80% | 148_5 (19%) | no | runner-up 148_5 at 0.46 pp |
| nsga2 nsga2_repeat_3 | B250 | 127_1 | 93.36 | 0.700 | 86% | 137_4 (7%) | no | runner-up 137_4 at 0.70 pp |
| nsga2 nsga2_repeat_3 | B500 | 112_1 | 93.78 | 0.420 | 77% | 127_1 (22%) | no | runner-up 127_1 at 0.42 pp |
| nsga3 nsga3_repeat_1 | A | 120_18 | 93.98 | 0.880 | 57% | 114_2 (35%) | sí | runner-up 97_8 at 0.88 pp |
| nsga3 nsga3_repeat_1 | K | 112_13 | 89.74 | 0.027 | 62% | 129_3 (36%) | sí | distance 0.428; runner-up 129_3 at 0.454 |
| nsga3 nsga3_repeat_1 | C | 112_13 | 89.74 | 0.760 | 69% | 116_4 (16%) | sí | 0.76 pp above the A-5 threshold; next: 129_3 |
| nsga3 nsga3_repeat_1 | B50 | 129_3 | 92.30 | 2.560 | 100% | — | no | runner-up 112_13 at 2.56 pp |
| nsga3 nsga3_repeat_1 | B100 | 97_8 | 93.10 | 0.400 | 71% | 141_5 (20%) | no | runner-up 141_5 at 0.40 pp |
| nsga3 nsga3_repeat_1 | B250 | 120_18 | 93.98 | 0.880 | 57% | 114_2 (35%) | sí | runner-up 97_8 at 0.88 pp |
| nsga3 nsga3_repeat_1 | B500 | 120_18 | 93.98 | 0.880 | 57% | 114_2 (35%) | sí | runner-up 97_8 at 0.88 pp |
| nsga3 nsga3_repeat_2 | A | 29_13 | 92.68 | 0.300 | 67% | 102_9 (27%) | sí | runner-up 102_9 at 0.30 pp |
| nsga3 nsga3_repeat_2 | K | 58_7 | 90.30 | 0.024 | 66% | 83_2 (28%) | sí | distance 0.222; runner-up 83_2 at 0.245 |
| nsga3 nsga3_repeat_2 | C | 83_2 | 89.86 | 2.180 | 100% | — | no | 2.18 pp above the A-5 threshold; next: 58_7 |
| nsga3 nsga3_repeat_2 | B50 | 58_7 | 90.30 | 0.440 | 76% | 83_2 (18%) | no | runner-up 83_2 at 0.44 pp |
| nsga3 nsga3_repeat_2 | B100 | 58_7 | 90.30 | 0.440 | 76% | 83_2 (18%) | no | runner-up 83_2 at 0.44 pp |
| nsga3 nsga3_repeat_2 | B250 | 29_13 | 92.68 | 0.300 | 67% | 102_9 (27%) | sí | runner-up 102_9 at 0.30 pp |
| nsga3 nsga3_repeat_2 | B500 | 29_13 | 92.68 | 0.300 | 67% | 102_9 (27%) | sí | runner-up 102_9 at 0.30 pp |
| nsga3 nsga3_repeat_3 | A | 98_5 | 94.94 | 0.500 | 72% | 129_16 (14%) | no | runner-up 129_16 at 0.50 pp |
| nsga3 nsga3_repeat_3 | K | 147_19 | 92.30 | 0.007 | 36% | 122_18 (24%) | sí | distance 0.222; runner-up 122_18 at 0.228 |
| nsga3 nsga3_repeat_3 | C | 115_18 | 90.28 | 0.340 | 74% | 149_4 (18%) | no | 0.34 pp above the A-5 threshold; next: 149_4 |
| nsga3 nsga3_repeat_3 | B50 | 149_4 | 91.06 | 0.780 | 94% | 115_18 (6%) | no | runner-up 115_18 at 0.78 pp |
| nsga3 nsga3_repeat_3 | B100 | 127_1 | 93.20 | 0.900 | 92% | 80_8 (4%) | no | runner-up 147_19 at 0.90 pp |
| nsga3 nsga3_repeat_3 | B250 | 129_16 | 94.44 | 0.120 | 56% | 97_0 (37%) | sí | runner-up 97_0 at 0.12 pp |
| nsga3 nsga3_repeat_3 | B500 | 98_5 | 94.94 | 0.500 | 72% | 129_16 (14%) | no | runner-up 129_16 at 0.50 pp |

## Sensibilidad a los parámetros fijados

Red que elige cada regla afectada con cada variante (— = ninguna cambia) y Jaccard del frente frente al de referencia.

| Corrida | C 3 pp | C 7 pp | K log |
|---|---|---|---|
| moqnas moqnas_repeat_1 | C: 75_14→40_5 (J=1.00) | C: 75_14→92_8 (J=1.00) | K: 40_5→92_8 (J=1.00) |
| moqnas moqnas_repeat_2 | C: 132_0→112_5 (J=1.00) | C: 132_0→30_16 (J=1.00) | — (J=1.00) |
| moqnas moqnas_repeat_3 | — (J=1.00) | C: 71_3→93_13 (J=1.00) | K: 93_13→11_18 (J=1.00) |
| nsga2 nsga2_repeat_1 | C: 89_2→128_3 (J=1.00) | C: 89_2→129_19 (J=1.00) | K: 126_15→128_3 (J=1.00) |
| nsga2 nsga2_repeat_2 | C: 117_5→143_8 (J=1.00) | C: 117_5→104_6 (J=1.00) | K: 143_8→88_5 (J=1.00) |
| nsga2 nsga2_repeat_3 | C: 136_3→148_5 (J=1.00) | C: 136_3→107_7 (J=1.00) | K: 129_16→107_7 (J=1.00) |
| nsga3 nsga3_repeat_1 | C: 112_13→129_3 (J=1.00) | C: 112_13→124_16 (J=1.00) | — (J=1.00) |
| nsga3 nsga3_repeat_2 | — (J=1.00) | C: 83_2→108_0 (J=1.00) | K: 58_7→83_2 (J=1.00) |
| nsga3 nsga3_repeat_3 | C: 115_18→147_19 (J=1.00) | — (J=1.00) | K: 147_19→115_18 (J=1.00) |

Estabilidad media por regla: A 76%, K 68%, C 79%, B50 86%, B100 87%, B250 69%, B500 77%.

Resumen: C 3 pp: cambia alguna regla en 7/9 corridas; C 7 pp: cambia alguna regla en 8/9 corridas; K log: cambia alguna regla en 7/9 corridas; elecciones dentro del ruido: 12/63.
