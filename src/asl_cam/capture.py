"""
Basic webcam capture module for ASL recognition.
"""
import argparse
import logging
import os
from datetime import datetime
from typing import NoReturn, Optional

import cv2
import numpy as np

from asl_cam.utils.hand_detection import (
    detect_skin,
    get_hand_contour,
    draw_hand_detection,
    get_hand_roi
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

def process_frame(frame: np.ndarray, detect_hands: bool = True) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Process a single frame.
    Returns the processed frame and hand ROI if detected.
    """
    # Mirror the frame
    frame = cv2.flip(frame, 1)
    
    if not detect_hands:
        return frame, None
    
    # Detect hands
    skin_mask = detect_skin(frame)
    hand_contour, hull = get_hand_contour(skin_mask)
    
    # Draw detection visualization
    frame = draw_hand_detection(frame, hand_contour, hull)
    
    # Get hand ROI
    hand_roi = get_hand_roi(frame, hand_contour)
    
    return frame, hand_roi

def save_hand_image(hand_roi: np.ndarray, save_dir: str, label: str) -> None:
    """Save the hand ROI image with timestamp."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    cv2.imwrite(filepath, hand_roi)
    logger.info(f"Saved hand image: {filepath}")

def capture_loop(cap: cv2.VideoCapture, save_dir: Optional[str] = None) -> NoReturn:
    """Main capture loop."""
    current_label = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                break

            # Process frame
            frame, hand_roi = process_frame(frame, detect_hands=True)
            
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

            cv2.imshow("ASL Camera", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and save_dir and hand_roi is not None:
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
    args = parser.parse_args()

    try:
        cap = setup_camera(args.device)
        logger.info("Camera initialized successfully")
        capture_loop(cap, args.save_dir)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
