"""Tests for the Dataset Preprocessor."""
from pathlib import Path
from app.datasets.preprocessor import DatasetPreprocessor

def test_normalization():
    from app.datasets.preprocessor import normalize_text
    assert normalize_text("  This is a   TEST.  ") == "this is a test."
    assert normalize_text("Patient has high bp") == "patient has high blood pressure"

def test_preprocessor_sample_generation(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    
    prep = DatasetPreprocessor(raw_dir=raw_dir, processed_dir=processed_dir)
    # The first run should auto-generate samples since raw is empty
    results = prep.process_all()
    
    assert raw_dir.exists()
    assert (raw_dir / "disease_sample.csv").exists()
    # The first call creates them, but doesn't immediately process them if they weren't there at the start of the loop
    # Let's run it again to process the newly generated samples
    results = prep.process_all()
    
    assert results["disease_sample.csv"] > 0
    assert (processed_dir / "diseases.json").exists()
