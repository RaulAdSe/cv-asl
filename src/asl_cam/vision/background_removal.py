"""
Advanced Background Remover for ASL Recognition

This module provides a robust background removal system based on learning
the scene's background and subtracting it to isolate the user's hand.
This is more reliable than simple skin color detection, especially in
variable lighting conditions.
"""
import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class BackgroundRemover:
    """
    A class to handle background learning and subtraction.
    """
    def __init__(self,
                 learning_rate: float = -1, # Let OpenCV manage the learning rate during the learning phase
                 history: int = 300,
                 threshold: float = 25.0):
        """
        Initializes the background remover.

        Args:
            learning_rate: How quickly the model adapts to changes.
                           -1 means it's determined automatically during the learning phase.
            history: Number of frames used to build the background model.
            threshold: Threshold for detecting changes. Higher values reduce sensitivity.
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=threshold,
            detectShadows=False
        )
        self.learning_rate = learning_rate
        self.bg_model_learned = False
        self.frames_learned = 0
        self.history = history
        self.background_image: Optional[np.ndarray] = None

        # Kernels for cleaning up the foreground mask
        self.open_kernel = np.ones((5, 5), np.uint8)
        self.close_kernel = np.ones((7, 7), np.uint8)

        logger.info("Advanced BackgroundRemover initialized.")

    def learn_background(self, frame: np.ndarray):
        """
        Learns the background from a given frame. Call this repeatedly on a
        static scene without the user's hand in view.

        Args:
            frame: A frame containing only the background.
        """
        if self.bg_model_learned:
            return

        # Apply the frame to the model to learn
        self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        self.frames_learned += 1

        # Consider the model learned after enough frames
        if self.frames_learned >= self.history:
            self.bg_model_learned = True
            # CRITICAL: Capture the learned background image
            self.background_image = self.bg_subtractor.getBackgroundImage()
            logger.info("✅ Background model learned and static background image captured.")

    def get_progress(self) -> float:
        """Returns the background learning progress as a percentage."""
        return min(1.0, self.frames_learned / self.history)

    def remove_background(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes the learned background from a frame.

        Args:
            frame: The input frame with a foreground object (e.g., a hand).

        Returns:
            A tuple containing:
            - The foreground object isolated against a black background.
            - The binary foreground mask.
        """
        if not self.bg_model_learned:
            # If still learning, don't return a mask yet
            return frame, np.zeros(frame.shape[:2], dtype=np.uint8)

        # Get the foreground mask with a learning rate of 0 to prevent the hand from being learned.
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0)

        # Apply morphological operations to clean up the mask and remove noise
        # Using a smaller kernel and fewer iterations to be less aggressive
        # and preserve more of the hand's shape.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Opening removes small noise spots
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Closing fills small holes in the main object
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 3. Find the largest contour, assuming it's the hand/arm.
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        final_mask = np.zeros_like(fg_mask)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Only keep contours above a certain size to filter out small noise.
            if cv2.contourArea(largest_contour) > 1000:
                cv2.drawContours(final_mask, [largest_contour], -1, (255), cv2.FILLED)

        # Apply the final mask to the original frame
        foreground_img = cv2.bitwise_and(frame, frame, mask=final_mask)

        return foreground_img, final_mask

    def reset(self):
        """Resets the background model without creating a new MOG2 instance."""
        # Store current settings
        current_threshold = self.bg_subtractor.getVarThreshold()
        
        # Reset the learning state
        self.bg_model_learned = False
        self.frames_learned = 0
        self.background_image = None
        
        # Clear the background model but keep the same MOG2 instance
        # This is safer than creating a new instance which can cause tracking instability
        self.bg_subtractor.clear()
        
        logger.info("BackgroundRemover has been reset.") 