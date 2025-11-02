## Machine Learning II

**Estudiante:** Yudith Diana Chalco Cerezo  
**Estudiante:** Sara Cristine Ocon Tovar
**Institución:** Escuela Nacional de Estadística e Informática (ENEI – INEI)  
**Curso:** Machine Learning II  
**Entrega:** Parte I y Parte II  

---

## Parte I – PCA y Reconocimiento Facial (Eigenfaces)

### Objetivo
Aplicar el Análisis de Componentes Principales (PCA) para la reducción de dimensionalidad en imágenes faciales, calcular el rostro promedio y representar las principales componentes (eigenfaces).  
Además, utilizar PCA para la reconstrucción de rostros y la clasificación mediante Regresión Logística.

---

### Metodología
1. **Normalización:**  
   Escalado de píxeles al rango [0,1] y centrado de los datos restando la media (rostro promedio).  

2. **Rostro promedio:**  
   Cálculo del promedio de todos los rostros del conjunto de entrenamiento y visualización en una matriz de 50×50.  

3. **PCA y Eigenfaces:**  
   Se calcularon 200 componentes principales con la opción `whiten=True`.  
   Las 10 primeras eigenfaces fueron visualizadas, mostrando los patrones dominantes de variabilidad facial.  

4. **Varianza explicada:**  
   La curva de varianza acumulada mostró que con aproximadamente 40 componentes se captura más del 95% de la varianza total.  

5. **Clasificación:**  
   Se entrenó una Regresión Logística usando los componentes PCA como entrada.  
   Se determinó el número óptimo de componentes r=42, con una precisión máxima de 0.99.  

6. **Reconstrucción:**  
   Se midió el error de reconstrucción (Frobenius) al variar r, observándose una disminución progresiva con más componentes.  

---

### Resultados
| Métrica | Valor |
|----------|--------|
| Componentes óptimos (r) | 42 |
| Precisión máxima | 0.99 |
| Error de reconstrucción mínimo | 0.09 |
| Varianza acumulada (r=42) | ≈95% |

---

### Conclusiones
- El PCA permite representar rostros con un número reducido de variables sin pérdida significativa de información.  
- Con pocas componentes (alrededor de 40), el modelo logra una alta precisión (99%) en el reconocimiento facial.  
- Aumentar el número de componentes mejora marginalmente la precisión pero incrementa el costo computacional.  
- Las eigenfaces representan las direcciones principales de variabilidad en el conjunto de rostros.  
- El error de reconstrucción disminuye conforme se agregan componentes, confirmando la efectividad del PCA.  

---

## Parte II – Convolutional Neural Networks (CNN) con MNIST

### Objetivo
Implementar, entrenar y comparar redes neuronales convolucionales (CNN) de distintas profundidades para la clasificación del conjunto de datos MNIST de dígitos manuscritos.  
Analizar sus curvas de convergencia, rendimiento y complejidad.

---

### Metodología
1. **Dataset:**  
   Conjunto MNIST original (`datasets.MNIST`) con 60,000 imágenes de entrenamiento y 10,000 de prueba.  
   Se utilizó la transformación `transforms.ToTensor()` para normalizar los datos en el rango [0,1].

2. **Modelos diseñados:**  
   - CNN_Simple: 1 capa convolucional y 1 capa densa.  
   - CNN_Medium: 2 capas convolucionales y 2 densas.  
   - CNN_Deep: 3 capas convolucionales, dropout y 2 densas.  

3. **Entrenamiento:**  
   - Épocas: 10  
   - Tamaño de lote: 64  
   - Optimizador: Adam (lr = 0.001)  
   - Métricas: pérdida de entrenamiento, pérdida de prueba y precisión.  

4. **Visualización:**  
   Curvas de pérdida (train/test), curvas de precisión y comparación final entre modelos.  

---

### Resultados experimentales
| Modelo | Pérdida final (train/test) | Precisión test | Tiempo por época |
|---------|-----------------------------|----------------|------------------|
| CNN_Simple | 0.05 / 0.06 | 0.9805 | ~16 s |
| CNN_Medium | 0.013 / 0.028 | 0.9915 | ~37 s |
| CNN_Deep | 0.021 / 0.024 | 0.9916 | ~45 s |

---

### Análisis
- Las curvas de pérdida muestran una convergencia rápida sin signos de sobreajuste.  
- La precisión mejora al aumentar la profundidad, aunque la diferencia entre los modelos Medium y Deep es marginal.  
- El modelo CNN_Deep presenta el mejor equilibrio entre precisión y estabilidad, con un costo computacional ligeramente mayor.  
- Los tres modelos alcanzan precisiones superiores al 98%, demostrando la eficacia de las CNN en la tarea de reconocimiento de dígitos.  

---

### Conclusiones
- Las redes convolucionales son altamente eficaces para tareas de reconocimiento visual.  
- Aumentar la profundidad de la red incrementa la capacidad de representación, aunque con rendimientos decrecientes.  
- Las curvas de convergencia evidencian un aprendizaje estable y buena generalización.  
- Los resultados obtenidos (alrededor del 99%) son consistentes con los valores de referencia para MNIST.  
- La práctica permitió consolidar la comprensión de las operaciones de convolución, pooling y dropout en redes profundas.  

---
