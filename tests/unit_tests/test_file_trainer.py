from urnai.trainers.filetrainer import FileTrainer
import pytest

def test_file_format_not_supported_error_file_trainer():
    with pytest.raises(IsADirectoryError) as ffnse:
        trainer = FileTrainer('.')
