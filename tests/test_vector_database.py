import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vector_database import _sanitize_filename


def test_sanitize_filename_rejects_path_traversal():
    try:
        _sanitize_filename("../../evil.pdf")
    except ValueError:
        assert True
    else:
        assert False, "Path traversal filenames should be rejected"


def test_sanitize_filename_allows_safe_name():
    assert _sanitize_filename("sample-01.pdf") == "sample-01.pdf"
