import time

class FPSTracker:
    """A simple class to track and calculate frames per second."""
    def __init__(self):
        self._start_time = None
        self._frame_count = 0
        self.fps = 0.0

    def start(self):
        """Start the FPS timer."""
        self._start_time = time.time()
        self._frame_count = 0

    def update(self):
        """Update the tracker with a new frame."""
        if self._start_time is None:
            self.start()
        
        self._frame_count += 1
        elapsed_time = time.time() - self._start_time
        
        if elapsed_time > 1.0: # Update FPS calculation every second
            self.fps = self._frame_count / elapsed_time
            self._start_time = time.time()
            self._frame_count = 0

    def get_fps(self) -> float:
        """Get the current calculated FPS."""
        return self.fps 