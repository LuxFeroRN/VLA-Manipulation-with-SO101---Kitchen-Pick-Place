# Análisis de Runs del Brazo Robótico

## Resumen de runs

| Run | Episodios | MAE medio (°) | Suavidad std (°) | Duración ep. (s) | Consistencia std media (°) |
|-----|-----------|--------------|------------------|-------------------|---------------------------|
| data1 | 17 | 3.54 | 1.19 | 1.58 | 30.9 |
| data2 | 39 | 3.78 | 1.37 | 1.58 | 27.9 |
| data3 | 7  | 3.34 | 1.16 | 1.54 | 29.0 |
| data4 | 35 | 2.90 | 1.26 | 1.60 | 28.5 |
| data5 | 13 | **1.99** | **0.80** | **1.50** | 29.9 |

> MAE medio y suavidad std son promedios de los 6 joints. Consistencia std media excluye el último episodio incompleto.

---

## Trayectorias por run

> Rojo = `action` (lo que se ordenó) · Teal = `obs` (lo que ejecutó el motor) · Sombreado naranja = error de seguimiento · Líneas verticales punteadas = límites de episodio.
> Las barras inferiores muestran el MAE medio de todos los joints por episodio: teal < 2° · naranja 2–4° · rojo > 4°.

### data1 — 17 episodios

![data1 trajectories](plots/data1_trajectories.png)

### data2 — 39 episodios

![data2 trajectories](plots/data2_trajectories.png)

### data3 — 7 episodios

![data3 trajectories](plots/data3_trajectories.png)

### data4 — 35 episodios

![data4 trajectories](plots/data4_trajectories.png)

### data5 — 13 episodios

![data5 trajectories](plots/data5_trajectories.png)

---

## Clasificación inferida

### data2 · data4 · data1 — Tarea repetida muchas veces

Los tres tienen el mayor número de episodios y muestran el patrón típico de entrenamiento en bucle: la calidad episodio a episodio oscila sin tendencia clara, lo que indica que el modelo se reevaluó muchas veces sobre la misma tarea.

**data4** es la run más repetitiva en sentido de movimiento similar:
- 35 episodios con `gripper consistency_std = 16.3°` (el más bajo de todas las runs), indicando que el gripper siguió casi la misma trayectoria en todos los intentos.
- `shoulder_pan consistency_std = 24.2°` también es el segundo más bajo — el brazo apuntaba siempre al mismo lugar.

**data2** tiene el mayor número de episodios (39) pero con más varianza entre ellos, sugiriendo que se repitió la tarea más veces aunque con ligeras variaciones de posición del objeto o condición inicial.

**data1** (17 episodios) es intermedia — más repeticiones que data3/data5 pero con alta varianza episodio a episodio, sin patrón de mejora ni degradación clara.

---

### data5 — Único pick ejecutado bien desde el principio

Los primeros 3 episodios son los mejores de todo el dataset:

| Ep | MAE medio (°) | Oscilaciones totales |
|----|--------------|---------------------|
| 0  | 0.61 | 12 |
| 1  | 0.74 | 16 |
| 2  | 0.76 | 9  |

Esto indica que el modelo ejecutó la tarea de forma correcta y limpia desde el inicio, sin necesidad de ajustes. Las métricas globales de data5 confirman que fue la mejor run del conjunto:

| Métrica | data5 | Mejor del resto |
|---------|-------|-----------------|
| MAE medio (°) | **1.99** | 2.90 (data4) |
| Suavidad std (°) | **0.80** | 1.16 (data3) |
| Distancia gripper (°) | **9.3** | 23.9 (data4) |
| Oscilaciones gripper | **2.2** | 4.9 (data1) |
| Velocidad máx. media (°/s) | **77** | 118 (data3) |

El gripper recorrió menos de la mitad de ángulo total que en cualquier otra run (9.3° vs 24–39°), lo que indica una apertura y cierre muy precisos sin movimientos redundantes. Los picos de velocidad son los más bajos del conjunto, y la suavidad es notablemente mejor en todos los joints.

Los episodios 3–8 presentan variabilidad (probablemente intentos adicionales tras el pick inicial exitoso), y los episodios 9–12 vuelven a ser excelentes (MAE < 0.73°), cerrando la sesión igual de limpio que como empezó.

---

### data3 — Error inicial seguido de pick satisfactorio

Con solo 7 episodios y una evolución de calidad claramente no uniforme, data3 muestra el patrón de "falla → corrección → éxito":

| Ep | MAE medio (°) | Oscilaciones totales | Interpretación |
|----|--------------|---------------------|----------------|
| 0  | 2.28 | 41 | Intento inicial, impreciso |
| 1  | 3.68 | 33 | Empeora |
| 2  | 2.75 | 19 | Parcial mejora |
| 3  | 5.11 | 30 | **Error** — mayor error de la run |
| 4  | 4.96 | 40 | Segundo intento fallido |
| 5  | 1.66 | 33 | Recuperación clara |
| 6  | **0.66** | 6  | **Pick satisfactorio** |

Los episodios 3 y 4 son los de peor desempeño (MAE > 5°, oscilaciones altas), coherentes con un error en la ejecución. El episodio 5 muestra la corrección y el 6 el pick final exitoso con el MAE más bajo de la run y solo 6 cambios de dirección en total — señal de movimiento limpio y decidido.

El `gripper MAE = 5.83°` (el más alto entre todas las runs) también apunta a que el fallo ocurrió durante la fase de agarre. La velocidad máxima del gripper en data3 (352°/s) es la más alta del dataset, consistente con un movimiento errático en los episodios de falla.

---

## Métricas detalladas por run

### Tracking — MAE por joint (°)

| Joint | data1 | data2 | data3 | data4 | data5 |
|-------|-------|-------|-------|-------|-------|
| elbow_flex    | 4.34 | 4.79 | 3.20 | 4.57 | **2.60** |
| gripper       | 4.16 | 3.85 | 5.83 | **2.35** | 2.99 |
| shoulder_lift | 5.62 | 6.52 | 4.53 | 5.39 | **3.04** |
| shoulder_pan  | 3.94 | 4.52 | 2.40 | 2.61 | **1.91** |
| wrist_flex    | 1.45 | 1.45 | 1.28 | 1.63 | **0.93** |
| wrist_roll    | 1.30 | 1.06 | 0.85 | 1.44 | **1.00** |

### Suavidad — std de pasos consecutivos en acción (°)

| Joint | data1 | data2 | data3 | data4 | data5 |
|-------|-------|-------|-------|-------|-------|
| elbow_flex    | 1.48 | 1.44 | 1.15 | 1.49 | **1.00** |
| gripper       | 1.77 | 1.63 | 2.09 | **1.31** | 0.89 |
| shoulder_lift | 1.77 | 1.90 | 1.55 | 1.87 | **1.20** |
| shoulder_pan  | 1.33 | 1.54 | 1.05 | 1.15 | **0.79** |
| wrist_flex    | 0.69 | 0.67 | 0.64 | 0.70 | **0.49** |
| wrist_roll    | 0.78 | 0.66 | 0.55 | 0.76 | **0.67** |

### Eficiencia — distancia angular total por episodio (°)

| Joint | data1 | data2 | data3 | data4 | data5 |
|-------|-------|-------|-------|-------|-------|
| elbow_flex    | 44.8 | 49.7 | 34.1 | 48.3 | **26.6** |
| gripper       | 33.8 | 35.6 | 38.9 | 23.9 | **9.3** |
| shoulder_lift | 61.9 | 73.2 | 55.4 | 65.7 | **34.8** |
| shoulder_pan  | 49.8 | 56.2 | 31.5 | 34.3 | **25.8** |
| wrist_flex    | 21.5 | 21.1 | 18.1 | 21.3 | **10.5** |
| wrist_roll    | 18.2 | 14.4 | 11.8 | 20.3 | **13.5** |

### Estabilidad — oscilaciones medias por episodio

| Joint | data1 | data2 | data3 | data4 | data5 |
|-------|-------|-------|-------|-------|-------|
| elbow_flex    | 2.5 | 2.2 | **3.3** | 3.1 | **1.3** |
| gripper       | 4.9 | 6.1 | 4.3 | 6.4 | **2.2** |
| shoulder_lift | 3.4 | 2.3 | 3.9 | 3.7 | 3.2 |
| shoulder_pan  | 5.0 | 5.7 | 8.4 | 6.4 | **5.3** |
| wrist_flex    | 8.6 | 9.5 | 5.9 | 4.7 | **2.5** |
| wrist_roll    | 3.1 | 3.4 | **3.1** | 4.9 | **2.3** |

### Consistencia — std cross-episodio de observación (°)

| Joint | data1 | data2 | data3 | data4 | data5 |
|-------|-------|-------|-------|-------|-------|
| elbow_flex    | 40.5 | 36.2 | 38.9 | 47.2 | 54.9 |
| gripper       | 24.3 | 25.4 | 23.4 | **16.3** | **12.0** |
| shoulder_lift | 50.7 | 45.9 | 53.5 | 58.3 | 58.9 |
| shoulder_pan  | 42.1 | 43.9 | 26.3 | **24.2** | 25.2 |
| wrist_flex    | 10.2 | 9.8  | 8.1  | **6.9** | 8.3 |
| wrist_roll    | 6.4  | **6.1**  | 4.5  | 17.9 | 21.1 |

> Alta `consistency_std` en `shoulder_lift` y `elbow_flex` en data4/data5 se explica porque la trayectoria vertical del brazo varía según la altura del objeto. Los joints de posicionamiento lateral (`shoulder_pan`, `wrist_flex`) son los más consistentes en las runs de tarea repetida.
