"""
System Performance Benchmarks.

Tests for measuring and validating system performance characteristics.
"""
import pytest
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from asl_cam.live_asl import LiveASLRecognizer
from asl_cam.vision.asl_hand_detector import ASLHandDetector

class TestSystemBenchmarks:
    """System-level performance benchmarks."""
    
    @pytest.fixture
    def live_recognizer(self):
        """Create a live ASL recognizer."""
        return LiveASLRecognizer()
    
    @pytest.fixture
    def asl_detector(self):
        """Create an ASL hand detector."""
        return ASLHandDetector()
    
    def test_model_inference_speed_benchmark(self, live_recognizer):
        """
        BENCHMARK: Model inference speed.
        
        WHY: Critical for real-time performance - should be fast enough
        for 30 FPS video processing.
        
        MEASURES: Inference time per prediction, throughput.
        """
        # Warm up the model
        warmup_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(5):
            live_recognizer.predict_hand_sign(warmup_crop)
        
        # Benchmark different batch sizes
        batch_sizes = [1, 5, 10]
        results = {}
        
        for batch_size in batch_sizes:
            crops = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) 
                    for _ in range(batch_size)]
            
            start_time = time.time()
            
            for crop in crops:
                prediction, confidence = live_recognizer.predict_hand_sign(crop)
                assert prediction in ['A', 'B', 'C']
            
            total_time = time.time() - start_time
            avg_time_per_prediction = total_time / batch_size
            
            results[batch_size] = {
                'total_time': total_time,
                'avg_per_prediction': avg_time_per_prediction,
                'fps_potential': 1.0 / avg_time_per_prediction if avg_time_per_prediction > 0 else float('inf')
            }
        
        # Performance requirements
        for batch_size, metrics in results.items():
            # Should be fast enough for real-time (at least 10 FPS potential)
            assert metrics['fps_potential'] >= 10.0, f"Too slow for batch size {batch_size}: {metrics['fps_potential']:.1f} FPS potential"
            
            # Individual predictions should be under 100ms for good responsiveness
            assert metrics['avg_per_prediction'] < 0.1, f"Individual prediction too slow: {metrics['avg_per_prediction']:.3f}s"
        
        print(f"🚀 Model inference benchmark:")
        for batch_size, metrics in results.items():
            print(f"  Batch {batch_size}: {metrics['avg_per_prediction']*1000:.1f}ms/pred, {metrics['fps_potential']:.1f} FPS potential")
    
    def test_hand_detection_speed_benchmark(self, asl_detector):
        """
        BENCHMARK: Hand detection speed.
        
        WHY: Hand detection is often the bottleneck in real-time systems.
        
        MEASURES: Detection time per frame, throughput.
        """
        # Test different frame sizes
        frame_sizes = [
            (240, 320),   # Small
            (480, 640),   # Medium (VGA)
            (720, 1280),  # Large (HD)
        ]
        
        results = {}
        
        for height, width in frame_sizes:
            # Create test frames with hand-like regions
            frames = []
            for i in range(10):
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                # Add hand-like rectangular region
                h_start, w_start = height//4, width//4
                h_end, w_end = 3*height//4, 3*width//4
                frame[h_start:h_end, w_start:w_end] = 255
                frames.append(frame)
            
            # Benchmark detection
            start_time = time.time()
            
            total_hands_detected = 0
            for frame in frames:
                hands = asl_detector.detect_hands_asl(frame)
                total_hands_detected += len(hands)
            
            total_time = time.time() - start_time
            avg_time_per_frame = total_time / len(frames)
            
            results[(height, width)] = {
                'total_time': total_time,
                'avg_per_frame': avg_time_per_frame,
                'fps_potential': 1.0 / avg_time_per_frame if avg_time_per_frame > 0 else float('inf'),
                'hands_detected': total_hands_detected
            }
        
        # Performance requirements
        for frame_size, metrics in results.items():
            # Should handle at least 15 FPS for real-time
            assert metrics['fps_potential'] >= 15.0, f"Hand detection too slow for {frame_size}: {metrics['fps_potential']:.1f} FPS"
        
        print(f"🤚 Hand detection benchmark:")
        for frame_size, metrics in results.items():
            print(f"  {frame_size}: {metrics['avg_per_frame']*1000:.1f}ms/frame, {metrics['fps_potential']:.1f} FPS, {metrics['hands_detected']} hands")
    
    def test_complete_pipeline_benchmark(self, live_recognizer, asl_detector):
        """
        BENCHMARK: Complete pipeline performance.
        
        WHY: The complete pipeline (detection + recognition) must be
        fast enough for real-time use.
        
        MEASURES: End-to-end latency, total throughput.
        """
        # Create realistic test scenario
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add multiple hand-like regions
        import cv2
        cv2.rectangle(frame, (150, 100), (300, 250), (255, 255, 255), -1)
        cv2.rectangle(frame, (400, 200), (550, 350), (255, 255, 255), -1)
        
        num_iterations = 20
        pipeline_times = []
        total_predictions = 0
        
        for i in range(num_iterations):
            start_time = time.time()
            
            # Step 1: Hand detection
            hands = asl_detector.detect_hands_asl(frame)
            
            # Step 2: Process each hand
            for hand in hands:
                if 'bbox' in hand:
                    x, y, w, h = hand['bbox']
                    if w > 0 and h > 0:
                        # Extract and resize crop
                        hand_crop = frame[y:y+h, x:x+w]
                        if hand_crop.size > 0:
                            hand_crop_resized = cv2.resize(hand_crop, (224, 224))
                            
                            # Step 3: Predict sign
                            prediction, confidence = live_recognizer.predict_hand_sign(hand_crop_resized)
                            total_predictions += 1
            
            pipeline_time = time.time() - start_time
            pipeline_times.append(pipeline_time)
        
        # Analyze performance
        avg_pipeline_time = sum(pipeline_times) / len(pipeline_times)
        max_pipeline_time = max(pipeline_times)
        min_pipeline_time = min(pipeline_times)
        
        fps_potential = 1.0 / avg_pipeline_time if avg_pipeline_time > 0 else float('inf')
        
        # Performance requirements
        assert fps_potential >= 15.0, f"Complete pipeline too slow: {fps_potential:.1f} FPS"
        assert max_pipeline_time < 0.2, f"Pipeline has slow outliers: {max_pipeline_time:.3f}s max"
        assert total_predictions > 0, "No predictions were made"
        
        print(f"🔗 Complete pipeline benchmark:")
        print(f"  Average: {avg_pipeline_time*1000:.1f}ms ({fps_potential:.1f} FPS)")
        print(f"  Range: {min_pipeline_time*1000:.1f}ms - {max_pipeline_time*1000:.1f}ms")
        print(f"  Predictions made: {total_predictions}")
    
    def test_memory_usage_benchmark(self, live_recognizer, asl_detector):
        """
        BENCHMARK: Memory usage characteristics.
        
        WHY: Long-running applications should not have memory leaks
        or excessive memory usage.
        
        MEASURES: Memory stability over time.
        """
        import gc
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run sustained operations
        num_iterations = 50
        memory_samples = []
        
        for i in range(num_iterations):
            # Create frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (200, 150), (400, 350), (255, 255, 255), -1)
            
            # Process frame
            hands = asl_detector.detect_hands_asl(frame)
            
            for hand in hands:
                if 'bbox' in hand:
                    x, y, w, h = hand['bbox']
                    if w > 0 and h > 0:
                        hand_crop = frame[y:y+h, x:x+w]
                        if hand_crop.size > 0:
                            hand_crop_resized = cv2.resize(hand_crop, (224, 224))
                            prediction, confidence = live_recognizer.predict_hand_sign(hand_crop_resized)
            
            # Sample memory every 10 iterations
            if i % 10 == 0:
                gc.collect()  # Force garbage collection
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory)
        
        final_memory = memory_samples[-1]
        max_memory = max(memory_samples)
        memory_growth = final_memory - initial_memory
        
        # Memory requirements
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f} MB"
        assert max_memory < initial_memory + 200, f"Peak memory too high: {max_memory:.1f} MB"
        
        print(f"💾 Memory usage benchmark:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Growth: {memory_growth:.1f} MB")
        print(f"  Peak: {max_memory:.1f} MB")
    
    def test_concurrent_processing_benchmark(self, live_recognizer):
        """
        BENCHMARK: Concurrent processing capabilities.
        
        WHY: System should handle multiple requests efficiently.
        
        MEASURES: Throughput under concurrent load.
        """
        import threading
        import queue
        
        # Create test data
        test_crops = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) 
                     for _ in range(20)]
        
        results_queue = queue.Queue()
        
        def worker(crops, worker_id):
            """Worker function for concurrent processing."""
            worker_results = []
            for i, crop in enumerate(crops):
                start_time = time.time()
                prediction, confidence = live_recognizer.predict_hand_sign(crop)
                processing_time = time.time() - start_time
                
                worker_results.append({
                    'worker_id': worker_id,
                    'crop_id': i,
                    'prediction': prediction,
                    'confidence': confidence,
                    'processing_time': processing_time
                })
            
            results_queue.put(worker_results)
        
        # Test sequential processing
        start_time = time.time()
        for crop in test_crops:
            prediction, confidence = live_recognizer.predict_hand_sign(crop)
        sequential_time = time.time() - start_time
        
        # Test concurrent processing (simulated - PyTorch models typically don't benefit from threading)
        num_workers = 2
        crops_per_worker = len(test_crops) // num_workers
        
        start_time = time.time()
        threads = []
        
        for worker_id in range(num_workers):
            start_idx = worker_id * crops_per_worker
            end_idx = start_idx + crops_per_worker
            worker_crops = test_crops[start_idx:end_idx]
            
            thread = threading.Thread(target=worker, args=(worker_crops, worker_id))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        concurrent_time = time.time() - start_time
        
        # Collect results
        all_results = []
        while not results_queue.empty():
            worker_results = results_queue.get()
            all_results.extend(worker_results)
        
        # Analyze results
        total_predictions = len(all_results)
        sequential_throughput = len(test_crops) / sequential_time
        concurrent_throughput = total_predictions / concurrent_time
        
        # Verify all predictions are valid
        for result in all_results:
            assert result['prediction'] in ['A', 'B', 'C']
            assert 0.0 <= result['confidence'] <= 1.0
        
        print(f"⚡ Concurrent processing benchmark:")
        print(f"  Sequential: {sequential_time:.3f}s ({sequential_throughput:.1f} pred/s)")
        print(f"  Concurrent: {concurrent_time:.3f}s ({concurrent_throughput:.1f} pred/s)")
        print(f"  Predictions: {total_predictions}")
        
        # Note: Due to GIL, concurrent might not be faster, but should still work correctly
        assert total_predictions >= len(test_crops) - 2  # Allow for some rounding in division
    
    def test_startup_time_benchmark(self):
        """
        BENCHMARK: System startup time.
        
        WHY: Users want the system to be ready quickly.
        
        MEASURES: Time to initialize all components.
        """
        # Test individual component startup times
        components = {}
        
        # ASL Detector startup
        start_time = time.time()
        detector = ASLHandDetector()
        components['ASL Detector'] = time.time() - start_time
        
        # Live Recognizer startup (includes model loading)
        start_time = time.time()
        recognizer = LiveASLRecognizer()
        components['Live Recognizer'] = time.time() - start_time
        
        # Test first prediction (includes model warmup)
        dummy_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        start_time = time.time()
        prediction, confidence = recognizer.predict_hand_sign(dummy_crop)
        components['First Prediction'] = time.time() - start_time
        
        total_startup_time = sum(components.values())
        
        # Startup requirements
        assert total_startup_time < 10.0, f"Total startup too slow: {total_startup_time:.1f}s"
        assert components['First Prediction'] < 2.0, f"First prediction too slow: {components['First Prediction']:.1f}s"
        
        print(f"🚀 Startup time benchmark:")
        for component, startup_time in components.items():
            print(f"  {component}: {startup_time:.3f}s")
        print(f"  Total: {total_startup_time:.3f}s") 