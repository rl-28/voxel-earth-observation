import pytest
import sys
from unittest.mock import MagicMock

def test_streamlit_app_imports():
    """
    Test that app.py can be imported safely and backend functions are exposed.
    We mock streamlit to prevent it from attempting to render the UI during pytest.
    """
    mock_st = MagicMock()
    sys.modules["streamlit"] = mock_st
    
    try:
        import app
        
        assert hasattr(app, "process_bands")
        assert callable(app.process_bands)
        
    finally:
        if "app" in sys.modules:
            del sys.modules["app"]
        if "streamlit" in sys.modules:
            del sys.modules["streamlit"]
