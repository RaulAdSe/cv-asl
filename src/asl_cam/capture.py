"""
Basic webcam capture module for ASL recognition.
"""
import argparse
import logging
import os
from datetime import datetime
from typing import NoReturn, Optional, Tuple

import cv2
import numpy as np

from asl_cam.utils.hand_detection import (
    detect_skin,
    get_hand_contour,
    draw_hand_detection,
    get_hand_roi,
    HandDetection
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_camera(device_id: int = 0) -> cv2.VideoCapture:
    """Initialize the webcam."""
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device_id}")
    
    # Set resolution to 640x480 for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def process_frame(
    frame: np.ndarray,
    detect_hands: bool = True,
    min_confidence: float = 0.6
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[HandDetection]]:
    """
    Process a single frame.
    Returns:
        - Processed frame
        - Hand ROI if detected
        - HandDetection object if detected
    """
    # Mirror the frame
    frame = cv2.flip(frame, 1)
    
    if not detect_hands:
        return frame, None, None
    
    # Detect hands
    skin_mask = detect_skin(frame)
    detection = get_hand_contour(skin_mask, min_confidence=min_confidence)
    
    # Draw detection visualization
    frame = draw_hand_detection(frame, detection)
    
    # Get hand ROI
    hand_roi = get_hand_roi(frame, detection)
    
    return frame, hand_roi, detection

def save_hand_image(hand_roi: np.ndarray, save_dir: str, label: str) -> None:
    """Save the hand ROI image with timestamp."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    cv2.imwrite(filepath, hand_roi)
    logger.info(f"Saved hand image: {filepath}")

def capture_loop(
    cap: cv2.VideoCapture,
    save_dir: Optional[str] = None,
    min_confidence: float = 0.6
) -> NoReturn:
    """Main capture loop."""
    current_label = None
    show_skin_mask = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                break

            # Process frame
            frame, hand_roi, detection = process_frame(
                frame,
                detect_hands=True,
                min_confidence=min_confidence
            )
            
            # Get skin mask for debugging
            if show_skin_mask:
                skin_mask = detect_skin(frame)
                cv2.imshow("Skin Mask", skin_mask)
            
            # Add FPS counter
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Add current label if in collection mode
            if save_dir and current_label:
                cv2.putText(
                    frame,
                    f"Collecting: {current_label}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            
            # Add controls help
            cv2.putText(
                frame,
                "Controls: 's'-save, 'n'-new label, 'm'-toggle mask, 'q'-quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow("ASL Camera", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and save_dir and hand_roi is not None and detection and detection.confidence >= min_confidence:
                # Ask for label if not set
                if current_label is None:
                    cv2.putText(
                        frame,
                        "Enter label in terminal",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    cv2.imshow("ASL Camera", frame)
                    current_label = input("Enter label for the hand sign: ")
                
                save_hand_image(hand_roi, save_dir, current_label)
            elif key == ord('n'):
                # Change label
                current_label = input("Enter new label for the hand sign: ")
            elif key == ord('m'):
                # Toggle skin mask display
                show_skin_mask = not show_skin_mask
                if not show_skin_mask:
                    cv2.destroyWindow("Skin Mask")

    finally:
        cap.release()
        cv2.destroyAllWindows()

def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="ASL Camera capture module")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save captured hand images"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold for hand detection (default: 0.6)"
    )
    args = parser.parse_args()

    try:
        cap = setup_camera(args.device)
        logger.info("Camera initialized successfully")
        capture_loop(cap, args.save_dir, args.min_confidence)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
