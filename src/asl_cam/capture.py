"""
Basic webcam capture module for ASL recognition.
"""
import argparse
import logging
from typing import NoReturn

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_camera(device_id: int = 0) -> cv2.VideoCapture:
    """Initialize the webcam."""
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device_id}")
    return cap

def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Process a single frame.
    Currently just flips the image horizontally for mirror effect.
    """
    return cv2.flip(frame, 1)

def capture_loop(cap: cv2.VideoCapture) -> NoReturn:
    """Main capture loop."""
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                break

            frame = process_frame(frame)
            
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

            cv2.imshow("ASL Camera", frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
    args = parser.parse_args()

    try:
        cap = setup_camera(args.device)
        logger.info("Camera initialized successfully")
        capture_loop(cap)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
