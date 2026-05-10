from __future__ import annotations

import subprocess
import sys


def test_main_help_lists_expected_subcommands():
    result = subprocess.run([sys.executable, "main.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "smoke" in result.stdout
    assert "train" in result.stdout
    assert "batch" in result.stdout
    assert "resume" in result.stdout
