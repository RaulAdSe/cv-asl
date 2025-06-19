"""Hand detection utilities using OpenCV."""
import cv2
import numpy as np
from typing import Tuple, Optional

def detect_skin(frame: np.ndarray) -> np.ndarray:
    """
    Detect skin pixels using HSV color space.
    Returns a binary mask where white pixels represent skin.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create binary mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Add a second range for skin color (handles wrap-around in Hue)
    lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    
    # Combine masks
    mask = cv2.add(mask, mask2)
    
    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask

def get_hand_contour(mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find the largest contour in the mask (presumably the hand).
    Returns the contour and its convex hull.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None, None
    
    # Get the largest contour by area
    hand_contour = max(contours, key=cv2.contourArea)
    
    # Filter out small contours that are unlikely to be hands
    if cv2.contourArea(hand_contour) < 3000:  # Adjust threshold as needed
        return None, None
    
    # Get the convex hull
    hull = cv2.convexHull(hand_contour)
    
    return hand_contour, hull

def draw_hand_detection(
    frame: np.ndarray,
    hand_contour: Optional[np.ndarray],
    hull: Optional[np.ndarray]
) -> np.ndarray:
    """Draw hand detection visualization on the frame."""
    if hand_contour is None or hull is None:
        return frame
    
    # Draw the contour and hull
    cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
    cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
    
    # Draw bounding rectangle
    x, y, w, h = cv2.boundingRect(hand_contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    # Add text indicating hand detected
    cv2.putText(
        frame,
        "Hand Detected",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )
    
    return frame

def get_hand_roi(
    frame: np.ndarray,
    hand_contour: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Extract the region of interest (ROI) containing the hand."""
    if hand_contour is None:
        return None
    
    x, y, w, h = cv2.boundingRect(hand_contour)
    # Add padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2 * padding)
    h = min(frame.shape[0] - y, h + 2 * padding)
    
    return frame[y:y+h, x:x+w] 