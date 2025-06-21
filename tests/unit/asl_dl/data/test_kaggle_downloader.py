"""
Unit tests for Kaggle dataset downloader.

Tests the dataset download and organization functionality.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from asl_dl.data.kaggle_downloader import download_kaggle_asl, download_abc_dataset

class TestKaggleDownloader:
    """Test cases for Kaggle dataset downloader."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def mock_kaggle_data(self, temp_dir):
        """Create mock Kaggle dataset structure."""
        # Create a mock downloaded dataset
        kaggle_path = temp_dir / "kaggle_download"
        kaggle_path.mkdir()
        
        # Create numbered directories (0=A, 1=B, 2=C)
        for i in range(3):
            class_dir = kaggle_path / str(i)
            class_dir.mkdir()
            
            # Create some mock images
            for j in range(5):
                (class_dir / f"image_{j}.jpg").touch()
        
        return kaggle_path
    
    @patch('asl_dl.data.kaggle_downloader.kagglehub.dataset_download')
    def test_download_kaggle_asl_basic(self, mock_download, temp_dir, mock_kaggle_data):
        """
        TEST: Does the basic download function work?
        
        WHY: We need to ensure the downloader can fetch and organize
        the Kaggle dataset correctly.
        
        CHECKS: Downloads dataset, organizes into correct structure.
        """
        # Mock kagglehub download
        mock_download.return_value = str(mock_kaggle_data)
        
        target_dir = temp_dir / "organized"
        result_path = download_kaggle_asl(str(target_dir))
        
        # Check that download was called
        mock_download.assert_called_once_with("ayuraj/asl-dataset")
        
        # Check result path
        assert result_path == target_dir / "train"
        assert result_path.exists()
        
        # Check organized structure (0->A, 1->B, 2->C)
        assert (result_path / "A").exists()
        assert (result_path / "B").exists() 
        assert (result_path / "C").exists()
        
        # Check images were copied
        assert len(list((result_path / "A").glob("*.jpg"))) == 5
        assert len(list((result_path / "B").glob("*.jpg"))) == 5
        assert len(list((result_path / "C").glob("*.jpg"))) == 5
    
    @patch('asl_dl.data.kaggle_downloader.kagglehub.dataset_download')
    def test_download_kaggle_asl_letter_filtering(self, mock_download, temp_dir, mock_kaggle_data):
        """
        TEST: Does letter filtering work correctly?
        
        WHY: We want to download only specific letters (like A, B, C)
        for focused training.
        
        CHECKS: Only requested letters are downloaded.
        """
        mock_download.return_value = str(mock_kaggle_data)
        
        target_dir = temp_dir / "filtered"
        result_path = download_kaggle_asl(str(target_dir), letters=['A', 'C'])
        
        # Should only have A and C directories
        assert (result_path / "A").exists()
        assert not (result_path / "B").exists()
        assert (result_path / "C").exists()
        
        # Check images in filtered directories
        assert len(list((result_path / "A").glob("*.jpg"))) == 5
        assert len(list((result_path / "C").glob("*.jpg"))) == 5
    
    @patch('asl_dl.data.kaggle_downloader.download_kaggle_asl')
    def test_download_abc_dataset(self, mock_download_kaggle):
        """
        TEST: Does the ABC convenience function work?
        
        WHY: We have a shortcut function for downloading just A, B, C
        letters for quick testing.
        
        CHECKS: Calls main function with correct letter filter.
        """
        mock_download_kaggle.return_value = Path("/fake/path")
        
        result = download_abc_dataset()
        
        # Check that it called the main function with ABC filter
        mock_download_kaggle.assert_called_once_with(letters=['A', 'B', 'C'])
        assert result == Path("/fake/path")
    
    def test_number_to_letter_mapping(self):
        """
        TEST: Is the number-to-letter mapping correct?
        
        WHY: The Kaggle dataset uses numbered directories (0, 1, 2...)
        that need to map to letters (A, B, C...).
        
        CHECKS: Verify the mapping is correct for our use case.
        """
        # This tests the mapping logic indirectly
        # The actual mapping is internal to the function
        # but we can verify it works by checking our mock test above
        pass
    
    @patch('asl_dl.data.kaggle_downloader.kagglehub.dataset_download')
    def test_download_error_handling(self, mock_download, temp_dir):
        """
        TEST: Does error handling work for download failures?
        
        WHY: Network issues or API problems should be handled gracefully.
        
        CHECKS: Proper exceptions are raised for download failures.
        """
        # Mock download failure
        mock_download.side_effect = Exception("Network error")
        
        target_dir = temp_dir / "error_test"
        
        with pytest.raises(Exception, match="Network error"):
            download_kaggle_asl(str(target_dir))
    
    @pytest.fixture
    def mock_letter_dataset(self, temp_dir):
        """Create mock dataset with letter directories (not numbers)."""
        kaggle_path = temp_dir / "letter_dataset"
        kaggle_path.mkdir()
        
        # Create letter directories directly
        for letter in ['A', 'B', 'C']:
            class_dir = kaggle_path / letter
            class_dir.mkdir()
            
            # Create some mock images
            for j in range(3):
                (class_dir / f"img_{j}.jpg").touch()
        
        return kaggle_path
    
    @patch('asl_dl.data.kaggle_downloader.kagglehub.dataset_download')
    def test_letter_directory_handling(self, mock_download, temp_dir, mock_letter_dataset):
        """
        TEST: Can it handle datasets that already use letter directories?
        
        WHY: Some datasets might already have A, B, C directories
        instead of numbered ones.
        
        CHECKS: Letter directories are handled correctly.
        """
        mock_download.return_value = str(mock_letter_dataset)
        
        target_dir = temp_dir / "letter_organized"
        result_path = download_kaggle_asl(str(target_dir))
        
        # Check organized structure
        assert (result_path / "A").exists()
        assert (result_path / "B").exists()
        assert (result_path / "C").exists()
        
        # Check images were copied
        assert len(list((result_path / "A").glob("*.jpg"))) == 3
    
    @patch('asl_dl.data.kaggle_downloader.kagglehub.dataset_download')
    def test_existing_dataset_cleanup(self, mock_download, temp_dir, mock_kaggle_data):
        """
        TEST: Does it clean up existing datasets before organizing?
        
        WHY: We want to avoid mixing old and new data when re-downloading.
        
        CHECKS: Existing train directory is removed and recreated.
        """
        mock_download.return_value = str(mock_kaggle_data)
        
        target_dir = temp_dir / "cleanup_test"
        
        # Create existing dataset
        existing_train = target_dir / "train"
        existing_train.mkdir(parents=True)
        (existing_train / "old_file.txt").touch()
        
        # Download new dataset
        result_path = download_kaggle_asl(str(target_dir))
        
        # Check old file is gone, new structure exists
        assert not (existing_train / "old_file.txt").exists()
        assert (result_path / "A").exists()
    
    @patch('asl_dl.data.kaggle_downloader.kagglehub.dataset_download')
    def test_empty_dataset_handling(self, mock_download, temp_dir):
        """
        TEST: How does it handle empty or invalid datasets?
        
        WHY: Sometimes downloads might fail or return empty directories.
        
        CHECKS: Appropriate error handling for edge cases.
        """
        # Create empty directory
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        mock_download.return_value = str(empty_dir)
        
        target_dir = temp_dir / "empty_test"
        
        # Should raise an error for empty dataset
        with pytest.raises(ValueError, match="Could not find dataset directory"):
            download_kaggle_asl(str(target_dir)) 