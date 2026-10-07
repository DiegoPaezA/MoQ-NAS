# Representantes C1_triobj (tag F13v1c, validación de la semilla 1)

Frente: val_err, params. Compacto (C): menor complejidad a <= 5 pp de A. Presupuestos: 0.25, 0.5, 1, 1.5 M params.

| Corrida | Algoritmo | Cribadas | En el frente (val) | Dominadas tras el retrain | Spearman proxy–val | Representantes |
|---|---|---|---|---|---|---|
| exp22_repeat_1 | moqnas | 10 | 9 | 1 | 0.95 | 4 |
| exp22_repeat_2 | moqnas | 10 | 9 | 1 | 0.93 | 3 |
| exp22_repeat_3 | moqnas | 10 | 8 | 2 | 0.93 | 4 |
| exp4_repeat_1 | nsga2 | 10 | 7 | 3 | 0.90 | 5 |
| exp4_repeat_2 | nsga2 | 10 | 9 | 1 | 0.71 | 6 |
| exp4_repeat_3 | nsga2 | 10 | 8 | 2 | 0.85 | 4 |
| exp1_repeat_1 | nsga3 | 10 | 9 | 1 | 0.70 | 5 |
| exp1_repeat_2 | nsga3 | 10 | 9 | 1 | 0.96 | 4 |
| exp1_repeat_3 | nsga3 | 10 | 9 | 1 | 0.82 | 3 |

## moqnas/experiment_cifar10_moqnas_v2/exp22_repeat_1

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 130_14 | 93.60 | 1.714 M | 1117.7 |
| K | 98_1 | 90.12 | 0.255 M | 105.1 |
| C | 141_2 | 89.48 | 0.254 M | 86.7 |
| B0.25 | 11_2 | 88.46 | 0.190 M | 124.7 |
| B0.5 | 98_1 | 90.12 | 0.255 M | 105.1 |
| B1 | 98_1 | 90.12 | 0.255 M | 105.1 |
| B1.5 | 98_1 | 90.12 | 0.255 M | 105.1 |

## moqnas/experiment_cifar10_moqnas_v2/exp22_repeat_2

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 66_5 | 92.68 | 0.916 M | 899.3 |
| K | 53_15 | 88.58 | 0.223 M | 221.6 |
| C | 53_15 | 88.58 | 0.223 M | 221.6 |
| B0.25 | 53_15 | 88.58 | 0.223 M | 221.6 |
| B0.5 | 116_19 | 90.28 | 0.373 M | 196.0 |
| B1 | 66_5 | 92.68 | 0.916 M | 899.3 |
| B1.5 | 66_5 | 92.68 | 0.916 M | 899.3 |

## moqnas/experiment_cifar10_moqnas_v2/exp22_repeat_3

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 79_19 | 92.26 | 0.894 M | 657.1 |
| K | 86_16 | 86.98 | 0.137 M | 100.9 |
| C | 143_6 | 88.10 | 0.216 M | 198.9 |
| B0.25 | 143_6 | 88.10 | 0.216 M | 198.9 |
| B0.5 | 127_14 | 91.28 | 0.318 M | 181.1 |
| B1 | 79_19 | 92.26 | 0.894 M | 657.1 |
| B1.5 | 79_19 | 92.26 | 0.894 M | 657.1 |

## nsga/experiment_cifar10_nsga2/exp4_repeat_1

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 144_0 | 95.24 | 1.494 M | 944.2 |
| K | 25_9 | 92.44 | 0.516 M | 440.1 |
| C | 25_9 | 92.44 | 0.516 M | 440.1 |
| B0.25 | 144_17 | 83.48 | 0.075 M | 16.2 |
| B0.5 | 133_11 | 88.56 | 0.277 M | 21.4 |
| B1 | 59_7 | 93.78 | 0.914 M | 651.8 |
| B1.5 | 144_0 | 95.24 | 1.494 M | 944.2 |

## nsga/experiment_cifar10_nsga2/exp4_repeat_2

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 103_12 | 95.02 | 2.147 M | 1609.9 |
| K | 90_9 | 92.62 | 0.384 M | 121.3 |
| C | 38_19 | 91.24 | 0.351 M | 221.9 |
| B0.25 | 13_18 | 87.12 | 0.182 M | 62.0 |
| B0.5 | 90_9 | 92.62 | 0.384 M | 121.3 |
| B1 | 16_8 | 93.76 | 0.808 M | 435.7 |
| B1.5 | 1_7 | 94.50 | 1.226 M | 879.4 |

## nsga/experiment_cifar10_nsga2/exp4_repeat_3

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 113_1 | 94.86 | 2.728 M | 1219.4 |
| K | 1_4 | 91.96 | 0.470 M | 319.1 |
| C | 42_5 | 90.02 | 0.177 M | 76.9 |
| B0.25 | 42_5 | 90.02 | 0.177 M | 76.9 |
| B0.5 | 1_4 | 91.96 | 0.470 M | 319.1 |
| B1 | 137_1 | 93.32 | 0.808 M | 797.2 |
| B1.5 | 137_1 | 93.32 | 0.808 M | 797.2 |

## nsga/experiment_cifar10_nsga3/exp1_repeat_1

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 116_19 | 93.02 | 0.610 M | 490.7 |
| K | 140_4 | 89.42 | 0.153 M | 83.0 |
| C | 135_11 | 88.72 | 0.149 M | 83.0 |
| B0.25 | 76_16 | 90.98 | 0.230 M | 39.3 |
| B0.5 | 23_14 | 92.10 | 0.334 M | 173.0 |
| B1 | 116_19 | 93.02 | 0.610 M | 490.7 |
| B1.5 | 116_19 | 93.02 | 0.610 M | 490.7 |

## nsga/experiment_cifar10_nsga3/exp1_repeat_2

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 90_3 | 93.60 | 0.978 M | 774.7 |
| K | 118_8 | 90.06 | 0.183 M | 121.0 |
| C | 109_10 | 88.62 | 0.150 M | 118.6 |
| B0.25 | 118_8 | 90.06 | 0.183 M | 121.0 |
| B0.5 | 148_7 | 90.88 | 0.273 M | 197.5 |
| B1 | 90_3 | 93.60 | 0.978 M | 774.7 |
| B1.5 | 90_3 | 93.60 | 0.978 M | 774.7 |

## nsga/experiment_cifar10_nsga3/exp1_repeat_3

| Regla | id | val acc | params | MACs (M) |
|---|---|---|---|---|
| A | 45_0 | 93.14 | 1.268 M | 1111.5 |
| K | 132_13 | 90.24 | 0.198 M | 180.4 |
| C | 132_13 | 90.24 | 0.198 M | 180.4 |
| B0.25 | 132_13 | 90.24 | 0.198 M | 180.4 |
| B0.5 | 131_18 | 92.50 | 0.416 M | 135.9 |
| B1 | 131_18 | 92.50 | 0.416 M | 135.9 |
| B1.5 | 45_0 | 93.14 | 1.268 M | 1111.5 |

# Análisis de la selección (solo validación)

Ruido de referencia entre semillas: 0.36 pp (sd máxima entre semillas de una misma red en los reentrenamientos previos). "Frontera": redes dominadas que entrarían al frente con +ruido (fuera) o miembros que saldrían con −ruido (dentro).

## Proxy → validación por algoritmo

| Algoritmo | Corridas | ρ Spearman (media) | τ Kendall (media) | Dominadas / cribadas | Frontera fuera / dentro | Mejor proxy = A | Representantes proxy conservados |
|---|---|---|---|---|---|---|---|
| moqnas | 3 | 0.94 | 0.84 | 4 / 30 | 0 / 2 | 3 / 3 | 7 / 9 |
| nsga2 | 3 | 0.82 | 0.68 | 6 / 30 | 3 / 3 | 2 / 3 | 7 / 8 |
| nsga3 | 3 | 0.83 | 0.72 | 3 / 30 | 0 / 3 | 3 / 3 | 5 / 8 |

## Estabilidad de cada elección

Estabilidad: fracción de 500 simulaciones con ruido N(0, ruido) en la accuracy de validación de todas las redes en las que la regla elige la misma red. Margen: pp de accuracy sobre la siguiente (A, presupuestos), pp sobre el umbral A−5 (C) o diferencia de distancia normalizada al ideal (K).

Si la estabilidad es < 70%, también se confirma la alternativa más frecuente de la misma regla en las simulaciones (etiqueta `<regla>~`).

| Corrida | Regla | id | val acc | Margen | Estabilidad | Alternativa más frecuente | ¿Se confirma también? | Detalle |
|---|---|---|---|---|---|---|---|---|
| moqnas exp22_repeat_1 | A | 130_14 | 93.60 | 3.480 | 100% | — | no | runner-up 98_1 at 3.48 pp |
| moqnas exp22_repeat_1 | K | 98_1 | 90.12 | 0.023 | 88% | 141_2 (11%) | no | distance 0.217; runner-up 141_2 at 0.240 |
| moqnas exp22_repeat_1 | C | 141_2 | 89.48 | 0.880 | 60% | 11_2 (36%) | sí | 0.88 pp above the A-5 threshold; next: 98_1 |
| moqnas exp22_repeat_1 | B0.25 | 11_2 | 88.46 | 1.760 | 100% | — | no | runner-up 135_13 at 1.76 pp |
| moqnas exp22_repeat_1 | B0.5 | 98_1 | 90.12 | 0.640 | 89% | 141_2 (11%) | no | runner-up 141_2 at 0.64 pp |
| moqnas exp22_repeat_1 | B1 | 98_1 | 90.12 | 0.640 | 89% | 141_2 (11%) | no | runner-up 141_2 at 0.64 pp |
| moqnas exp22_repeat_1 | B1.5 | 98_1 | 90.12 | 0.640 | 89% | 141_2 (11%) | no | runner-up 141_2 at 0.64 pp |
| moqnas exp22_repeat_2 | A | 66_5 | 92.68 | 2.400 | 100% | — | no | runner-up 116_19 at 2.40 pp |
| moqnas exp22_repeat_2 | K | 53_15 | 88.58 | 0.002 | 47% | 139_15 (38%) | sí | distance 0.307; runner-up 139_15 at 0.308 |
| moqnas exp22_repeat_2 | C | 53_15 | 88.58 | 0.900 | 96% | 139_15 (4%) | no | 0.90 pp above the A-5 threshold; next: 139_15 |
| moqnas exp22_repeat_2 | B0.25 | 53_15 | 88.58 | 2.400 | 100% | — | no | runner-up 10_3 at 2.40 pp |
| moqnas exp22_repeat_2 | B0.5 | 116_19 | 90.28 | 0.220 | 62% | 46_12 (33%) | sí | runner-up 46_12 at 0.22 pp |
| moqnas exp22_repeat_2 | B1 | 66_5 | 92.68 | 2.400 | 100% | — | no | runner-up 116_19 at 2.40 pp |
| moqnas exp22_repeat_2 | B1.5 | 66_5 | 92.68 | 2.400 | 100% | — | no | runner-up 116_19 at 2.40 pp |
| moqnas exp22_repeat_3 | A | 79_19 | 92.26 | 0.980 | 97% | 127_14 (3%) | no | runner-up 127_14 at 0.98 pp |
| moqnas exp22_repeat_3 | K | 86_16 | 86.98 | 0.000 | 46% | 143_6 (43%) | sí | distance 0.326; runner-up 143_6 at 0.327 |
| moqnas exp22_repeat_3 | C | 143_6 | 88.10 | 0.840 | 64% | 86_16 (24%) | sí | 0.84 pp above the A-5 threshold; next: 127_14 |
| moqnas exp22_repeat_3 | B0.25 | 143_6 | 88.10 | 1.120 | 99% | 86_16 (1%) | no | runner-up 86_16 at 1.12 pp |
| moqnas exp22_repeat_3 | B0.5 | 127_14 | 91.28 | 3.180 | 88% | 143_11 (12%) | no | runner-up 143_6 at 3.18 pp |
| moqnas exp22_repeat_3 | B1 | 79_19 | 92.26 | 0.980 | 97% | 127_14 (3%) | no | runner-up 127_14 at 0.98 pp |
| moqnas exp22_repeat_3 | B1.5 | 79_19 | 92.26 | 0.980 | 97% | 127_14 (3%) | no | runner-up 127_14 at 0.98 pp |
| nsga2 exp4_repeat_1 | A | 144_0 | 95.24 | 1.060 | 51% | 82_0 (38%) | sí | runner-up 121_19 at 1.06 pp |
| nsga2 exp4_repeat_1 | K | 25_9 | 92.44 | 0.128 | 84% | 23_2 (12%) | no | distance 0.392; runner-up 23_2 at 0.520 |
| nsga2 exp4_repeat_1 | C | 25_9 | 92.44 | 2.200 | 100% | — | no | 2.20 pp above the A-5 threshold; next: 23_2 |
| nsga2 exp4_repeat_1 | B0.25 | 144_17 | 83.48 | nan | 100% | — | no | only network within the budget |
| nsga2 exp4_repeat_1 | B0.5 | 133_11 | 88.56 | 5.080 | 100% | — | no | runner-up 144_17 at 5.08 pp |
| nsga2 exp4_repeat_1 | B1 | 59_7 | 93.78 | 0.520 | 88% | 23_2 (12%) | no | runner-up 23_2 at 0.52 pp |
| nsga2 exp4_repeat_1 | B1.5 | 144_0 | 95.24 | 1.060 | 98% | 121_19 (1%) | no | runner-up 121_19 at 1.06 pp |
| nsga2 exp4_repeat_2 | A | 103_12 | 95.02 | 0.260 | 67% | 75_15 (25%) | sí | runner-up 75_15 at 0.26 pp |
| nsga2 exp4_repeat_2 | K | 90_9 | 92.62 | 0.018 | 81% | 97_11 (18%) | no | distance 0.248; runner-up 97_11 at 0.266 |
| nsga2 exp4_repeat_2 | C | 38_19 | 91.24 | 1.220 | 99% | 90_9 (1%) | no | 1.22 pp above the A-5 threshold; next: 90_9 |
| nsga2 exp4_repeat_2 | B0.25 | 13_18 | 87.12 | 4.600 | 100% | — | no | runner-up 57_19 at 4.60 pp |
| nsga2 exp4_repeat_2 | B0.5 | 90_9 | 92.62 | 1.380 | 100% | — | no | runner-up 38_19 at 1.38 pp |
| nsga2 exp4_repeat_2 | B1 | 16_8 | 93.76 | 0.400 | 77% | 97_11 (22%) | no | runner-up 97_11 at 0.40 pp |
| nsga2 exp4_repeat_2 | B1.5 | 1_7 | 94.50 | 0.740 | 91% | 16_8 (7%) | no | runner-up 16_8 at 0.74 pp |
| nsga2 exp4_repeat_3 | A | 113_1 | 94.86 | 0.920 | 97% | 31_8 (3%) | no | runner-up 31_8 at 0.92 pp |
| nsga2 exp4_repeat_3 | K | 1_4 | 91.96 | 0.050 | 96% | 42_5 (4%) | no | distance 0.230; runner-up 42_5 at 0.280 |
| nsga2 exp4_repeat_3 | C | 42_5 | 90.02 | 0.160 | 56% | 1_4 (39%) | sí | 0.16 pp above the A-5 threshold; next: 1_4 |
| nsga2 exp4_repeat_3 | B0.25 | 42_5 | 90.02 | 1.040 | 97% | 83_5 (3%) | no | runner-up 83_5 at 1.04 pp |
| nsga2 exp4_repeat_3 | B0.5 | 1_4 | 91.96 | 1.940 | 100% | — | no | runner-up 42_5 at 1.94 pp |
| nsga2 exp4_repeat_3 | B1 | 137_1 | 93.32 | 1.120 | 62% | 69_16 (38%) | sí | runner-up 29_11 at 1.12 pp |
| nsga2 exp4_repeat_3 | B1.5 | 137_1 | 93.32 | 1.120 | 47% | 69_16 (28%) | sí | runner-up 29_11 at 1.12 pp |
| nsga3 exp1_repeat_1 | A | 116_19 | 93.02 | 0.920 | 97% | 23_14 (3%) | no | runner-up 23_14 at 0.92 pp |
| nsga3 exp1_repeat_1 | K | 140_4 | 89.42 | 0.026 | 79% | 132_3 (12%) | no | distance 0.305; runner-up 132_3 at 0.331 |
| nsga3 exp1_repeat_1 | C | 135_11 | 88.72 | 0.700 | 54% | 122_19 (38%) | sí | 0.70 pp above the A-5 threshold; next: 132_3 |
| nsga3 exp1_repeat_1 | B0.25 | 76_16 | 90.98 | 1.560 | 100% | 140_4 (0%) | no | runner-up 140_4 at 1.56 pp |
| nsga3 exp1_repeat_1 | B0.5 | 23_14 | 92.10 | 1.120 | 98% | 76_16 (2%) | no | runner-up 76_16 at 1.12 pp |
| nsga3 exp1_repeat_1 | B1 | 116_19 | 93.02 | 0.920 | 97% | 23_14 (3%) | no | runner-up 23_14 at 0.92 pp |
| nsga3 exp1_repeat_1 | B1.5 | 116_19 | 93.02 | 0.920 | 97% | 23_14 (3%) | no | runner-up 23_14 at 0.92 pp |
| nsga3 exp1_repeat_2 | A | 90_3 | 93.60 | 2.720 | 100% | — | no | runner-up 148_7 at 2.72 pp |
| nsga3 exp1_repeat_2 | K | 118_8 | 90.06 | 0.027 | 93% | 109_10 (7%) | no | distance 0.239; runner-up 109_10 at 0.266 |
| nsga3 exp1_repeat_2 | C | 109_10 | 88.62 | 0.020 | 51% | 118_8 (47%) | sí | 0.02 pp above the A-5 threshold; next: 118_8 |
| nsga3 exp1_repeat_2 | B0.25 | 118_8 | 90.06 | 1.440 | 100% | 109_10 (0%) | no | runner-up 109_10 at 1.44 pp |
| nsga3 exp1_repeat_2 | B0.5 | 148_7 | 90.88 | 0.820 | 93% | 118_8 (7%) | no | runner-up 118_8 at 0.82 pp |
| nsga3 exp1_repeat_2 | B1 | 90_3 | 93.60 | 2.720 | 100% | — | no | runner-up 148_7 at 2.72 pp |
| nsga3 exp1_repeat_2 | B1.5 | 90_3 | 93.60 | 2.720 | 100% | — | no | runner-up 148_7 at 2.72 pp |
| nsga3 exp1_repeat_3 | A | 45_0 | 93.14 | 0.640 | 90% | 131_18 (10%) | no | runner-up 131_18 at 0.64 pp |
| nsga3 exp1_repeat_3 | K | 132_13 | 90.24 | 0.002 | 44% | 109_10 (39%) | sí | distance 0.223; runner-up 109_10 at 0.224 |
| nsga3 exp1_repeat_3 | C | 132_13 | 90.24 | 2.100 | 54% | 130_4 (30%) | sí | 2.10 pp above the A-5 threshold; next: 101_16 |
| nsga3 exp1_repeat_3 | B0.25 | 132_13 | 90.24 | 2.240 | 100% | — | no | runner-up 64_14 at 2.24 pp |
| nsga3 exp1_repeat_3 | B0.5 | 131_18 | 92.50 | 0.560 | 86% | 109_10 (14%) | no | runner-up 109_10 at 0.56 pp |
| nsga3 exp1_repeat_3 | B1 | 131_18 | 92.50 | 0.560 | 86% | 109_10 (14%) | no | runner-up 109_10 at 0.56 pp |
| nsga3 exp1_repeat_3 | B1.5 | 45_0 | 93.14 | 0.640 | 90% | 131_18 (10%) | no | runner-up 131_18 at 0.64 pp |

## Sensibilidad a los parámetros fijados

Red que elige cada regla afectada con cada variante (— = ninguna cambia) y Jaccard del frente frente al de referencia.

| Corrida | C 3 pp | C 7 pp | K log | +Time_CUDA |
|---|---|---|---|---|
| moqnas exp22_repeat_1 | C: 141_2→130_14 (J=1.00) | C: 141_2→135_13 (J=1.00) | K: 98_1→11_2 (J=1.00) | K: 98_1→53_10 (J=0.90) |
| moqnas exp22_repeat_2 | C: 53_15→46_12 (J=1.00) | C: 53_15→10_3 (J=1.00) | K: 53_15→10_3 (J=1.00) | — (J=1.00) |
| moqnas exp22_repeat_3 | C: 143_6→127_14 (J=1.00) | C: 143_6→37_17 (J=1.00) | K: 86_16→37_17 (J=1.00) | K: 86_16→143_6 (J=0.80) |
| nsga2 exp4_repeat_1 | — (J=1.00) | C: 25_9→133_11 (J=1.00) | — (J=1.00) | K: 25_9→23_2 (J=0.70) |
| nsga2 exp4_repeat_2 | C: 38_19→90_9 (J=1.00) | — (J=1.00) | — (J=1.00) | K: 90_9→2_17 (J=0.90) |
| nsga2 exp4_repeat_3 | C: 42_5→1_4 (J=1.00) | C: 42_5→83_5 (J=1.00) | K: 1_4→42_5 (J=1.00) | — (J=0.80) |
| nsga3 exp1_repeat_1 | C: 135_11→76_16 (J=1.00) | C: 135_11→97_12 (J=1.00) | K: 140_4→97_12 (J=1.00) | K: 140_4→76_16 (J=0.90) |
| nsga3 exp1_repeat_2 | C: 109_10→148_7 (J=1.00) | C: 109_10→141_12 (J=1.00) | K: 118_8→48_16 (J=1.00) | K: 118_8→48_16 (J=0.90) |
| nsga3 exp1_repeat_3 | — (J=1.00) | C: 132_13→116_9 (J=1.00) | K: 132_13→130_4 (J=1.00) | K: 132_13→101_16 (J=0.90) |

Estabilidad media por regla: A 89%, K 73%, C 71%, B0.25 99%, B0.5 91%, B1 89%, B1.5 90%.

Resumen: C 3 pp: cambia alguna regla en 7/9 corridas; C 7 pp: cambia alguna regla en 8/9 corridas; K log: cambia alguna regla en 7/9 corridas; +Time_CUDA: cambia alguna regla en 7/9 corridas; elecciones dentro del ruido: 4/63.
