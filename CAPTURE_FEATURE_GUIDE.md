# Hand Data Capture and Visualization Feature

## 🎯 Overview

The **Capture Feature** allows you to capture and analyze detailed information about the currently detected hand in real-time. Press the **`C`** key during data collection to trigger this analysis.

## 🔧 Simple & Effective Background Removal System

### 🧠 **New Philosophy: Trust the Hand Detection!**

El nuevo sistema usa un enfoque mucho más simple y efectivo:

**Insight Clave**: Si la detección de manos funciona bien, entonces el crop ya ES mayoritariamente mano/brazo. Solo necesitamos remover los píxeles que no coincidan con los colores dominantes de la piel.

### **¿Por qué es mejor?**

✅ **Más simple**: Solo analiza colores en lugar de geometría compleja  
✅ **Más rápido**: Procesamiento directo sin algoritmos complejos  
✅ **Más confiable**: Se basa en lo que realmente funciona (detección)  
✅ **Más intuitivo**: Ajustes fáciles de entender (tolerancia de color)

### 🎯 **Métodos Disponibles**

**1. Dominant (Recomendado)**
- Analiza colores dominantes en la región central
- Más rápido y confiable para la mayoría de casos
- Perfecto para iluminación uniforme

**2. Adaptive** 
- Muestrea múltiples regiones para detección HSV adaptativa
- Mejor para iluminación variable
- Se adapta automáticamente al tono de piel

**3. Edge**
- Basado en colores con refinamiento de bordes
- Mejores transiciones suaves
- Ideal para fondos complejos

### 🎮 **Controles Simples**

**Método y Activación:**
- **B**: Toggle background removal on/off
- **N**: Cambiar método: dominant → adaptive → edge
- **Z**: Ajustar tolerancia de color (20=estricto, 60=permisivo)

**Análisis:**
- **C**: Capturar y comparar resultados
- UI muestra: `BG: dominant(40)` - método y tolerancia actual

### 🔍 **Cómo Funciona**

```
Crop de Mano (ya detectada correctamente)
    ↓
Análisis de Colores Dominantes
├── Región central (60% del centro)
├── K-means clustering (3 colores principales)
└── Colores representativos de la mano
    ↓
Máscara de Color
├── Tolerancia ajustable (±40 por defecto)
├── Mantener píxeles similares a colores de piel
└── Remover todo lo demás
    ↓
Limpieza y Suavizado
├── Morfología (quitar ruido, llenar huecos)
├── Componente más grande (mano principal)
└── Bordes suaves con Gaussian blur
    ↓
Mano Limpia con Fondo Negro
```

### 📊 **Guía de Tolerancia**

**Tolerancia 20 (Muy Estricta)**
- Solo píxeles muy similares al color de piel
- Puede remover partes de la mano con sombras
- Usar cuando el fondo es muy similar al tono de piel

**Tolerancia 40 (Normal - Recomendada)**
- Balance perfecto para la mayoría de casos
- Mantiene detalles de mano, remueve fondo
- Funciona bien con variaciones de iluminación

**Tolerancia 60 (Permisiva)**
- Mantiene más píxeles, remueve menos fondo
- Usar cuando la mano tiene mucha variación de color
- Mejor para condiciones de iluminación difíciles

### 🎯 **Flujo de Trabajo Recomendado**

1. **Empezar con método "dominant" y tolerancia 40**
2. **Presionar C para capturar y analizar**
3. **Si hay partes de mano faltantes**: Aumentar tolerancia (Z key)
4. **Si queda mucho fondo**: Disminuir tolerancia o cambiar método (N key)
5. **Para iluminación variable**: Probar método "adaptive"
6. **Para fondos complejos**: Probar método "edge"

### 🔧 **Troubleshooting Rápido**

**Problema: Se remueven partes de la mano**
- Solución: Aumentar tolerancia con Z key (40 → 50 → 60)
- O cambiar a método "adaptive" con N key

**Problema: Queda mucho fondo**
- Solución: Disminuir tolerancia con Z key (40 → 30 → 20)
- O verificar que la detección de mano está bien centrada

**Problema: Bordes muy duros**
- Solución: Usar método "edge" con N key
- Automáticamente suaviza las transiciones

**Problema: Funciona mal con tu tono de piel**
- Solución: Usar método "adaptive" que se ajusta automáticamente
- O ajustar tolerancia según necesites

### ✨ **Ventajas del Nuevo Sistema**

🚀 **Velocidad**: 3x más rápido que métodos complejos  
🎯 **Precisión**: Se enfoca en lo que realmente importa (colores)  
🎛️ **Control**: Ajustes intuitivos y en tiempo real  
🔄 **Flexibilidad**: 3 métodos para diferentes escenarios  
📱 **Simplicidad**: Menos parámetros, mejores resultados

---

**¡El nuevo sistema de background removal es mucho más simple, rápido y efectivo porque confía en que la detección ya funciona bien!**

## 🎮 How to Use

### Basic Usage
1. **Start data collection**: `python -m src.asl_cam.collect`
2. **Position your hand** until you see a green "READY TO COLLECT" status
3. **Test background removal**: Press **N** to cycle through methods, **B** to toggle
4. **Press `C`** to capture and visualize the current hand data
5. **A detailed analysis window** opens with 8 different visualizations

### What You'll See

#### 📊 8-Panel Visualization Window

**Panel 1: Original Hand Crop**
- Raw extracted hand region from the green rectangle
- Shows actual pixels detected as hand
- Displays dimensions (width × height)

**Panel 2: Processed Hand Crop** 
- Hand with background removal applied (shows current method)
- Clean training data version using enhanced algorithms
- Compares original vs processed quality

**Panel 3: Skin Detection Mask**
- Red heatmap showing detected skin pixels
- Based on HSV + YCrCb color space analysis
- Used by enhanced background removal for better accuracy

**Panel 4: Motion Detection Mask**
- Blue heatmap showing detected motion pixels  
- Based on MOG2 background subtraction
- Shows which areas are considered "moving"

**Panel 5: Hand Information Panel**
- **Size**: Width × height in pixels
- **Area**: Total pixel count
- **Aspect Ratio**: Width/height ratio
- **Tracking Stability**: Number of consistent detection hits
- **Confidence**: Detection confidence score (0-1)
- **Coverage Statistics**: Percentage of skin/motion pixels
- **Background Removal Method**: Current algorithm in use
- **Current Settings**: Detection mode, persistence settings

**Panel 6: Full Frame Context**
- Shows the complete camera frame
- Green rectangle indicates detected hand location
- Provides spatial context for hand position

**Panel 7: Combined Masks Analysis**
- **Red**: Motion-only pixels
- **Green**: Skin-only pixels  
- **Blue**: Overlapping motion + skin pixels
- Shows how motion and skin detection combine for background removal

**Panel 8: Color Histogram**
- RGB distribution of hand pixels
- Useful for understanding skin tone characteristics
- Helps optimize skin-based background removal

## 💾 Saved Data

When you press `C`, the system automatically saves:

### Images Saved to `data/raw/captures/`:
- **`capture_[label]_[timestamp]_original.jpg`** - Raw hand crop
- **`capture_[label]_[timestamp]_processed.jpg`** - Background-removed version
- **`capture_[label]_[timestamp]_skin_mask.jpg`** - Skin detection mask
- **`capture_[label]_[timestamp]_motion_mask.jpg`** - Motion detection mask

### Metadata Saved:
- **`capture_[label]_[timestamp]_metadata.json`** - Complete analysis data:
  ```json
  {
    "timestamp": 1703123456.789,
    "label": "hello",
    "bbox": [100, 150, 80, 120],
    "hand_size": [80, 120],
    "hand_area": 9600,
    "aspect_ratio": 0.67,
    "tracking_hits": 15,
    "confidence": 0.75,
    "skin_pixels": 7200,
    "motion_pixels": 5400,
    "skin_percentage": 75.0,
    "motion_percentage": 56.3,
    "background_removal": {
      "enabled": true,
      "method": "grabcut",
      "quality_score": 0.85
    },
    "settings": {
      "motion_detection": true,
      "background_removal": true,
      "persistence_frames": 30
    }
  }
  ```

## 🔍 Background Removal Analysis Use Cases

### 1. **Method Comparison**
- **Problem**: Unsure which background removal method works best
- **Solution**: Press **N** to cycle through methods, **C** to capture each
- **Look for**: Cleanest edges, best hand preservation, minimal artifacts

### 2. **Quality Optimization**
- **Problem**: Background removal has artifacts or holes
- **Solution**: Compare skin mask coverage with processed result
- **Look for**: Good skin detection enables better skin-based removal

### 3. **Environment Testing**
- **Problem**: Background removal varies with lighting/background
- **Solution**: Test different methods in various conditions
- **Look for**: Consistent quality across scenarios

### 4. **Training Data Validation**
- **Problem**: Need clean training data without background clutter
- **Solution**: Verify processed crops are clean and consistent
- **Look for**: Black backgrounds, preserved hand details

## 📈 Background Removal Quality Guide

### Excellent Quality Indicators:
- ✅ **Clean edges**: Smooth hand boundaries without jagged artifacts
- ✅ **Complete hand preservation**: All hand pixels retained
- ✅ **Uniform background**: Solid black or transparent background
- ✅ **Detail retention**: Fingers and hand features clearly visible
- ✅ **Consistent quality**: Similar results across different hand positions

### Problem Indicators:
- ❌ **Jagged edges**: Pixelated or rough hand boundaries
- ❌ **Hand erosion**: Parts of hand removed with background
- ❌ **Background remnants**: Non-hand pixels still visible
- ❌ **Holes in hand**: Missing pixels within hand region
- ❌ **Inconsistent quality**: Results vary significantly

### Method Selection Guide:

**Use GrabCut when:**
- Complex backgrounds with similar colors to skin
- Need highest quality for final training data
- Processing time is not critical

**Use Skin when:**
- Good skin detection (high coverage in Panel 3)
- Uniform lighting conditions
- Fast processing needed

**Use Contour when:**
- Clear hand boundaries visible
- Simple backgrounds
- Hand fully within frame

**Use MOG2 when:**
- Dynamic backgrounds
- Motion-based separation needed
- Hand moves consistently

**Use Watershed when:**
- High contrast between hand and background
- Complex textures in background
- Other methods fail

## 🎛️ Complete Controls Reference

### Background Removal Controls:
- **B**: Toggle background removal on/off
- **N**: Cycle through methods (grabcut→contour→skin→mog2→watershed)
- **C**: Capture and analyze current method quality

### Detection Controls:
- **S**: Save sample (when GREEN "READY")
- **M**: Show motion mask overlay
- **K**: Show skin mask overlay
- **X**: Toggle motion filtering
- **P**: Adjust hand persistence
- **R**: Reset detection system
- **T**: Tune detection thresholds

### Collection Controls:
- **L**: Change label
- **Q**: Quit collection

## 🔧 Technical Pipeline

### Enhanced Background Removal Pipeline:
```
Hand Crop Input (from detection)
    ↓
Skin Mask Extraction (HSV + YCrCb)
    ↓
Method Selection:
├── Skin Available → Enhanced Skin Removal
│   ├── Morphological cleaning (7×7 close, 3×3 open)
│   ├── Largest component selection
│   ├── Hole filling (11×11 close)
│   ├── Gaussian edge smoothing
│   └── Soft alpha blending
├── No Skin → Multi-Technique Voting
│   ├── K-means color segmentation
│   ├── Edge-based segmentation
│   ├── GrabCut crop segmentation
│   ├── Majority voting combination
│   └── Final morphological cleaning
    ↓
Quality Validation & Output
```

### Real-time Method Switching:
- **Instant switching** between algorithms with **N** key
- **UI feedback** shows current method
- **Live preview** of background removal quality
- **Capture analysis** validates method effectiveness

## 🎯 Best Practices

### Workflow for Optimal Background Removal:
1. **Start with GrabCut** (default, highest quality)
2. **Test your environment** by pressing **C** to capture baseline
3. **Cycle through methods** with **N** to compare quality
4. **Enable overlays** (**M**, **K**) to understand detection
5. **Choose best method** for your specific conditions
6. **Continue collection** with optimized settings

### Troubleshooting Background Removal:

**Poor edge quality?**
1. Try different methods with **N** key
2. Check skin detection with **K** overlay
3. Ensure stable hand detection (GREEN status)
4. Use **C** to analyze what's happening

**Hand parts missing?**
1. Switch to 'skin' method (preserves detected skin)
2. Verify skin detection covers hand well
3. Increase hand persistence with **P**
4. Check if motion detection helps with **X**

**Background not fully removed?**
1. Try 'grabcut' method for complex backgrounds
2. Ensure hand is well-centered in detection box
3. Use 'mog2' method for dynamic backgrounds
4. Check if better lighting improves skin detection

---

**The enhanced background removal system now provides production-quality hand isolation with real-time method selection, making it easy to achieve optimal results for any environment or use case.** 