# ASL Hand Detection & Training System

Sistema completo para detección de manos ASL y entrenamiento de modelos con MobileNetV2 para inferencia en tiempo real (30 FPS).

## 🚀 Inicio Rápido

### 1. Activar entorno
```bash
source venv/bin/activate
```

### 2. Recopilar datos
```bash
python -m src.asl_cam.collect
```
**Controles principales:**
- `S` - Guardar detección actual
- `L` - Cambiar etiqueta (letra ASL)
- `B` - Activar/desactivar eliminación de fondo
- `C` - Capturar y visualizar datos
- `Q` - Salir

### 3. Entrenar modelo
```bash
# Ver estimaciones de tiempo
python -m src.asl_cam.train --dry-run

# Entrenamiento rápido (10 épocas)
python -m src.asl_cam.train --epochs 10

# Entrenamiento completo (25 épocas)
python -m src.asl_cam.train --epochs 25

# Modelo ligero para mayor velocidad
python -m src.asl_cam.train --model mobilenetv2_lite --epochs 15
```

### 4. Usar modelo entrenado
```bash
python -m src.asl_cam.infer
```

## 📁 Estructura del Proyecto

```
CV-asl/
├── src/asl_cam/
│   ├── collect.py      # 📹 Recopilación de datos
│   ├── train.py        # 🧠 Entrenamiento MobileNetV2
│   ├── infer.py        # 🚀 Inferencia en tiempo real
│   └── vision/         # 👁️  Algoritmos de visión
├── data/raw/           # 💾 Datos recopilados
├── models/             # 🎯 Modelos entrenados
└── logs/               # 📊 Logs de entrenamiento
```

## 🎯 Workflow Completo

1. **Recopilar datos** por letra ASL
2. **Entrenar modelo** MobileNetV2 optimizado
3. **Evaluar rendimiento** (precisión + FPS)
4. **Usar en tiempo real** para reconocimiento ASL

## 🔧 Características Principales

### Detección de Manos
- ✅ Detección por color de piel + movimiento
- ✅ Eliminación inteligente del fondo
- ✅ Tracking persistente de manos
- ✅ Control de calidad automático

### Entrenamiento
- ✅ MobileNetV2 optimizado para 30 FPS
- ✅ Monitoreo de progreso en tiempo real
- ✅ TensorBoard para visualización
- ✅ Benchmark automático de velocidad

### Inferencia
- ✅ Clasificación en tiempo real
- ✅ Optimizado para 30+ FPS
- ✅ Suavizado temporal
- ✅ Estadísticas de rendimiento

## 📊 Modelos Disponibles

| Modelo | Precisión | FPS | Parámetros | Uso |
|--------|-----------|-----|------------|-----|
| MobileNetV2 | ~92% | 30-50 | 2.3M | Balanced |
| MobileNetV2 Lite | ~88% | 50-80 | 1.3M | Speed |

## 🎮 Controles del Sistema

### Recopilación de Datos
- `S` - Guardar muestra actual
- `L` - Cambiar etiqueta ASL
- `B` - Toggle eliminación de fondo  
- `N` - Cambiar método de fondo
- `C` - Capturar y analizar
- `X` - Toggle detección de movimiento
- `M/K` - Ver máscaras de detección

### Entrenamiento
```bash
python -m src.asl_cam.train --help
```

### Inferencia
- `SPACE` - Modo debug
- `S` - Mostrar estadísticas
- `Q` - Salir

## 🔍 Troubleshooting

### Problema: "command not found: python"
```bash
# Verificar que el entorno virtual esté activo
source venv/bin/activate
which python  # Debe mostrar la ruta del venv
```

### Problema: No se detectan manos
1. Verificar iluminación (luz uniforme)
2. Activar detección de movimiento (`X`)
3. Ajustar tolerancia de piel (`T`)
4. Resetear tracker (`R`)

### Problema: Entrenamiento lento
1. Usar `mobilenetv2_lite`
2. Reducir epochs: `--epochs 10`
3. Usar batch size menor: `--batch-size 16`

## 📈 Optimización de Rendimiento

### Para Recopilación:
- Buena iluminación
- Fondo contrastante
- Movimiento suave de manos

### Para Entrenamiento:
- GPU recomendada (pero funciona en CPU)
- Mínimo 100 muestras por letra
- Datos balanceados entre clases

### Para Inferencia:
- Cámara de buena calidad
- Procesador moderno
- Iluminación estable

## 🎯 Próximos Pasos

1. Recopilar más datos ASL
2. Entrenar con dataset completo de Kaggle
3. Optimizar para deployment móvil
4. Añadir más letras del alfabeto ASL

---

**Sistema optimizado para 30 FPS real-time ASL recognition! 🚀**
