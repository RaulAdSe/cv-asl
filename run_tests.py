#!/usr/bin/env python3
"""
CV-ASL Comprehensive Test Suite

Organized test runner with separated concerns:
- Unit tests for individual components
- Integration tests between modules
- System tests for end-to-end functionality
"""

import subprocess
import sys
import os
from pathlib import Path
import numpy as np

# --- Path Correction ---
# Add the 'src' directory to the Python path to allow imports of 'asl_cam' and 'asl_dl'
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# ---------------------

# ANSI escape codes for colors
GREEN = "\033[92m"
RED = "\033[91m"

# Get project paths
project_root = Path(__file__).parent
src_path = project_root / "src"

def run_pytest_tests():
    """Run the complete pytest suite."""
    print("\n🧪 Running pytest tests...")
    print("=" * 60)
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = str(src_path)
        
        # Run pytest on the entire tests directory
        cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
        
        result = subprocess.run(cmd, env=env, cwd=project_root, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        return False

def run_unit_tests():
    """Run unit tests by category."""
    print("\n🔬 Running Unit Tests...")
    print("=" * 60)
    
    test_categories = {
        "ASL Camera Vision": "tests/unit/asl_cam/vision/",
        "ASL Camera Utils": "tests/unit/asl_cam/utils/", 
        "ASL Camera Core": "tests/unit/asl_cam/",
        "ASL DL Models": "tests/unit/asl_dl/models/",
        "ASL DL Data": "tests/unit/asl_dl/data/",
    }
    
    results = {}
    
    for category, test_path in test_categories.items():
        test_dir = project_root / test_path
        if not test_dir.exists():
            print(f"⚠️  Test category not found: {category}")
            results[category] = "MISSING"
            continue
            
        print(f"\n📋 Running {category}...")
        
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = str(src_path)
        
        cmd = [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"]
        
        try:
            result = subprocess.run(cmd, env=env, cwd=project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {category} - PASSED")
                results[category] = "PASSED"
            else:
                print(f"❌ {category} - FAILED")
                print("Error output:")
                print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                results[category] = "FAILED"
                
        except Exception as e:
            print(f"❌ {category} - ERROR: {e}")
            results[category] = "ERROR"
    
    return results

def run_integration_tests():
    """Run integration tests."""
    print("\n🔗 Running Integration Tests...")
    print("=" * 60)
    
    integration_tests = {
        "Camera-DL Integration": "tests/integration/cam_dl_integration/",
        "End-to-End Integration": "tests/integration/end_to_end/",
    }
    
    results = {}
    
    for test_name, test_path in integration_tests.items():
        test_dir = project_root / test_path
        if not test_dir.exists() or not any(test_dir.glob("test_*.py")):
            print(f"⚠️  {test_name} - NO TESTS FOUND")
            results[test_name] = "MISSING"
            continue
            
        print(f"\n📋 Running {test_name}...")
        
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = str(src_path)
        
        cmd = [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"]
        
        try:
            result = subprocess.run(cmd, env=env, cwd=project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {test_name} - PASSED")
                results[test_name] = "PASSED"
            else:
                print(f"❌ {test_name} - FAILED")
                print("Error output:")
                print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                results[test_name] = "FAILED"
                
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            results[test_name] = "ERROR"
    
    return results

def run_live_asl_integration_test():
    """Run live ASL integration test."""
    try:
        from asl_cam.live_asl import LiveASLRecognizer
        from asl_cam.vision.asl_hand_detector import ASLHandDetector
        
        print("✅ Live ASL imports successful")
        
        # Initialize detector
        detector = ASLHandDetector(min_detection_confidence=0.5)
        
        # Test detection on a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Manually set the background model to 'learned' for this test
        detector.bg_remover.bg_model_learned = True
        
        # Use the new, correct method
        processed_hand, hand_info = detector.detect_and_process_hand(dummy_frame, 224)
        
        # We expect no hand to be found in an empty frame
        assert processed_hand is None, "Processed hand should be None for an empty frame"
        assert hand_info is None, "Hand info should be None for an empty frame"
        
        print("✅ Live ASL integration test passed")
        return True
    except Exception as e:
        print(f"❌ Live ASL integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_asl_dl_tests():
    """Run tests for the ASL deep learning components."""
    print("\n🧠 Running ASL Deep Learning Component Tests...")
    print("=" * 60)
    
    try:
        # Test model imports
        from asl_dl.models.mobilenet import MobileNetV2ASL
        print("✅ MobileNet model import successful")
        
        # Test model instantiation
        model = MobileNetV2ASL(num_classes=3)
        print(f"✅ MobileNet model created - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test data loading
        from asl_dl.data.kaggle_downloader import download_kaggle_asl
        print("✅ Kaggle downloader import successful")
        
        # Test visualization
        from asl_dl.visualization.training_plots import TrainingVisualizer
        visualizer = TrainingVisualizer()
        print("✅ Training visualizer import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ ASL DL component tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_report(pytest_passed, unit_results, integration_results, live_asl_passed, dl_passed):
    """Create a comprehensive test report."""
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    
    total_tests = 0
    passed_tests = 0
    
    # Pytest results
    print(f"\n📊 Complete PyTest Suite: {'✅ PASSED' if pytest_passed else '❌ FAILED'}")
    if pytest_passed:
        passed_tests += 1
    total_tests += 1
    
    # Unit test results
    print(f"\n🔬 Unit Tests:")
    for test_category, result in unit_results.items():
        status_icon = "✅" if result == "PASSED" else "❌" if result == "FAILED" else "⚠️"
        print(f"  {status_icon} {test_category}: {result}")
        if result == "PASSED":
            passed_tests += 1
        total_tests += 1
    
    # Integration test results
    print(f"\n🔗 Integration Tests:")
    for test_name, result in integration_results.items():
        status_icon = "✅" if result == "PASSED" else "❌" if result == "FAILED" else "⚠️"
        print(f"  {status_icon} {test_name}: {result}")
        if result == "PASSED":
            passed_tests += 1
        total_tests += 1
    
    # System tests
    print(f"\n🎥 Live ASL System: {'✅ PASSED' if live_asl_passed else '❌ FAILED'}")
    if live_asl_passed:
        passed_tests += 1
    total_tests += 1
    
    print(f"\n🧠 ASL DL Components: {'✅ PASSED' if dl_passed else '❌ FAILED'}")
    if dl_passed:
        passed_tests += 1
    total_tests += 1
    
    # Summary
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"\n📈 OVERALL RESULTS:")
    print(f"  Total Test Categories: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {total_tests - passed_tests}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    
    if pass_rate >= 90:
        print(f"\n🎉 EXCELLENT! Test suite is in excellent shape!")
    elif pass_rate >= 80:
        print(f"\n👍 GOOD! Most tests are passing, minor fixes needed.")
    elif pass_rate >= 60:
        print(f"\n⚠️  FAIR! Some test failures need attention.")
    else:
        print(f"\n🚨 NEEDS WORK! Multiple test failures need immediate attention.")
    
    return pass_rate >= 80

def main():
    """Main test runner function."""
    print("🚀 CV-ASL Organized Test Suite")
    print("=" * 80)
    print("Testing all components with organized structure...")
    print("📁 Unit Tests: Individual component testing")
    print("🔗 Integration Tests: Module interaction testing")  
    print("🎥 System Tests: End-to-end functionality testing")
    
    # Run all test suites
    pytest_passed = run_pytest_tests()
    unit_results = run_unit_tests()
    integration_results = run_integration_tests()
    live_asl_passed = run_live_asl_integration_test()
    dl_passed = run_asl_dl_tests()
    
    # Create final report
    overall_success = create_test_report(pytest_passed, unit_results, integration_results, live_asl_passed, dl_passed)
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)

if __name__ == "__main__":
    main() 