# Hand Data Capture and Visualization Feature

## 🎯 Overview

The **Capture Feature** allows you to capture and analyze detailed information about the currently detected hand in real-time. Press the **`C`** key during data collection to trigger this analysis.

## 🎮 How to Use

### Basic Usage
1. **Start data collection**: `python -m src.asl_cam.collect`
2. **Position your hand** until you see a green "READY TO COLLECT" status
3. **Press `C`** to capture and visualize the current hand data
4. **A detailed analysis window** opens with 8 different visualizations

### What You'll See

#### 📊 8-Panel Visualization Window

**Panel 1: Original Hand Crop**
- Raw extracted hand region from the green rectangle
- Shows actual pixels detected as hand
- Displays dimensions (width × height)

**Panel 2: Processed Hand Crop** 
- Hand with background removal applied (if enabled)
- Shows clean training data version
- Compares original vs processed

**Panel 3: Skin Detection Mask**
- Red heatmap showing detected skin pixels
- Based on HSV + YCrCb color space analysis
- Higher intensity = stronger skin detection

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
- **Current Settings**: Detection mode, background removal, persistence

**Panel 6: Full Frame Context**
- Shows the complete camera frame
- Green rectangle indicates detected hand location
- Provides spatial context for hand position

**Panel 7: Combined Masks Analysis**
- **Red**: Motion-only pixels
- **Green**: Skin-only pixels  
- **Blue**: Overlapping motion + skin pixels
- Shows how motion and skin detection combine

**Panel 8: Color Histogram**
- RGB distribution of hand pixels
- Useful for understanding skin tone characteristics
- Helps debug color-based detection issues

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
    "settings": {
      "motion_detection": true,
      "background_removal": true,
      "persistence_frames": 30
    }
  }
  ```

## 🔍 Analysis Use Cases

### 1. **Debug Detection Issues**
- **Problem**: Hand not being detected consistently
- **Solution**: Check skin/motion mask coverage
- **Look for**: Low skin percentage, insufficient motion

### 2. **Optimize Settings**
- **Problem**: False torso detection  
- **Solution**: Verify motion detection is working
- **Look for**: Large motion areas indicating static regions

### 3. **Quality Assessment**
- **Problem**: Inconsistent data quality
- **Solution**: Check tracking stability and confidence
- **Look for**: Low hit counts, poor aspect ratios

### 4. **Background Removal Tuning**
- **Problem**: Background removal not working well
- **Solution**: Compare original vs processed crops
- **Look for**: Artifacts, incomplete removal

### 5. **Training Data Validation**
- **Problem**: Need to verify collected data quality
- **Solution**: Analyze multiple captures across sessions
- **Look for**: Consistent hand characteristics

## 📈 Interpretation Guide

### Good Detection Indicators:
- ✅ **High skin coverage**: 60-90% of hand region
- ✅ **Moderate motion coverage**: 30-70% (shows hand is moving)
- ✅ **High tracking hits**: 10+ hits (stable detection)
- ✅ **Good aspect ratio**: 0.5-2.0 (hand-like proportions)
- ✅ **Clean background removal**: Clear hand separation

### Problem Indicators:
- ❌ **Low skin coverage**: <40% (poor skin detection)
- ❌ **Very high motion**: >90% (camera shake, too much movement)
- ❌ **Very low motion**: <20% (hand too static, becoming background)
- ❌ **Low tracking hits**: <5 hits (unstable detection)
- ❌ **Poor aspect ratio**: <0.3 or >3.0 (non-hand shapes)

## 🎛️ Integration with Controls

The capture feature works seamlessly with all other controls:

- **Before Capture**: Adjust settings (X, P, B, M, K)
- **During Capture**: Analysis reflects current settings
- **After Capture**: Continue collection or adjust based on findings

### Recommended Workflow:
1. **Start collection** with default settings
2. **Press `C`** to capture baseline data
3. **Adjust settings** based on analysis
4. **Press `C`** again to compare improvements
5. **Continue collection** with optimized settings

## 🔧 Technical Details

### Real-time Analysis:
- Non-blocking visualization (collection continues)
- Instant feedback on detection quality
- Live parameter adjustment based on findings

### Data Pipeline Integration:
- Uses same detection algorithms as collection
- Consistent with saved training data
- Validates actual collection quality

### Performance:
- Minimal impact on collection speed
- Analysis runs in separate thread
- Saves data locally for later review

## 🎯 Best Practices

### When to Use Capture:
- **New session start**: Baseline analysis
- **Detection problems**: Debug issues
- **Setting changes**: Verify improvements
- **Quality checks**: Validate data quality
- **Different conditions**: Lighting changes, background changes

### Analysis Tips:
- **Compare multiple captures** across time
- **Check different hand positions** (center, edges, close, far)
- **Test different lighting conditions**
- **Verify motion detection** by staying still vs moving
- **Validate background removal** effectiveness

---

**The capture feature transforms data collection from a "black box" process into a transparent, analyzable pipeline where you can understand exactly what the system is detecting and why.** 