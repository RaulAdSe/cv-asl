"""Hand detection utilities using OpenCV."""
import cv2
import numpy as np
from typing import Tuple, Optional, NamedTuple

class HandDetection(NamedTuple):
    """Container for hand detection results."""
    contour: np.ndarray
    hull: np.ndarray
    confidence: float
    defects: np.ndarray

def detect_skin(frame: np.ndarray) -> np.ndarray:
    """
    Detect skin pixels using YCrCb color space (more robust than HSV).
    Returns a binary mask where white pixels represent skin.
    """
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Define range for skin color in YCrCb
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)
    
    # Create binary mask
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Apply Gaussian blur to smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask

def calculate_hand_confidence(contour: np.ndarray, hull: np.ndarray, defects: np.ndarray, frame_area: float) -> float:
    """
    Calculate a confidence score (0-1) that the contour is a hand based on various features.
    """
    # Get contour features
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Relative area (shouldn't be too small or too large)
    area_ratio = area / frame_area
    if not (0.02 < area_ratio < 0.2):  # Hand should be 2-20% of frame
        return 0.0
    
    # Circularity (hand is not very circular)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity > 0.8:  # Too circular, probably a face
        return 0.0
    
    # Convexity defects (fingers create defects)
    if len(defects) < 3:  # Need at least 3 defects for a hand
        return 0.0
    
    # Calculate average depth of defects
    depths = []
    for defect in defects:
        start, end, far, d = defect[0]
        depths.append(d)
    avg_depth = np.mean(depths)
    
    # Combine metrics into confidence score
    confidence = 0.0
    
    # Good number of defects (4 for fingers)
    defect_score = min(len(defects) / 5.0, 1.0)
    confidence += 0.4 * defect_score
    
    # Good area ratio
    area_score = 1.0 - abs(area_ratio - 0.1) / 0.1  # Ideal ratio around 10%
    confidence += 0.3 * max(0, area_score)
    
    # Good defect depth
    depth_score = min(avg_depth / 1000.0, 1.0)
    confidence += 0.3 * depth_score
    
    return min(max(confidence, 0.0), 1.0)

def get_hand_contour(mask: np.ndarray, min_confidence: float = 0.6) -> Optional[HandDetection]:
    """
    Find the largest contour in the mask that looks like a hand.
    Returns HandDetection if confidence exceeds threshold, None otherwise.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    frame_area = mask.shape[0] * mask.shape[1]
    
    # Check each contour until we find a good hand candidate
    for contour in contours[:3]:  # Check top 3 largest contours
        # Filter out small contours
        if cv2.contourArea(contour) < 3000:
            continue
        
        # Get convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        
        try:
            # Get convexity defects
            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                continue
            
            # Calculate confidence
            confidence = calculate_hand_confidence(contour, hull, defects, frame_area)
            
            if confidence >= min_confidence:
                return HandDetection(
                    contour=contour,
                    hull=cv2.convexHull(contour),  # Get hull with points for drawing
                    confidence=confidence,
                    defects=defects
                )
        except Exception:
            continue
    
    return None

def draw_hand_detection(
    frame: np.ndarray,
    detection: Optional[HandDetection]
) -> np.ndarray:
    """Draw hand detection visualization on the frame."""
    if detection is None:
        return frame
    
    # Draw the contour and hull
    cv2.drawContours(frame, [detection.contour], -1, (0, 255, 0), 2)
    cv2.drawContours(frame, [detection.hull], -1, (255, 0, 0), 2)
    
    # Draw convexity defects (finger separations)
    for defect in detection.defects:
        start, end, far, _ = defect[0]
        start_point = tuple(detection.contour[start][0])
        end_point = tuple(detection.contour[end][0])
        far_point = tuple(detection.contour[far][0])
        
        # Draw points and connections
        cv2.circle(frame, far_point, 5, (0, 0, 255), -1)
        cv2.line(frame, start_point, end_point, (0, 255, 0), 1)
    
    # Draw bounding rectangle
    x, y, w, h = cv2.boundingRect(detection.contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    # Add confidence score
    cv2.putText(
        frame,
        f"Hand Detected ({detection.confidence:.2f})",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )
    
    return frame

def get_hand_roi(
    frame: np.ndarray,
    detection: Optional[HandDetection]
) -> Optional[np.ndarray]:
    """Extract the region of interest (ROI) containing the hand."""
    if detection is None:
        return None
    
    x, y, w, h = cv2.boundingRect(detection.contour)
    # Add padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2 * padding)
    h = min(frame.shape[0] - y, h + 2 * padding)
    
    return frame[y:y+h, x:x+w] 