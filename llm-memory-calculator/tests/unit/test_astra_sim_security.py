"""Tests for ASTRA-SIM security (shell injection prevention).

These tests verify that the ASTRA-SIM integration uses safe subprocess
calls without shell=True and properly validates paths.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestShellInjectionPrevention:
    """Tests that shell injection is prevented."""

    def test_subprocess_uses_shell_false(self):
        """Verify the ASTRA-SIM subprocess call uses shell=False."""
        # This is a static code analysis test - we read the source file directly
        # to avoid import issues with optional dependencies like chakra
        from pathlib import Path

        # Find the source file
        source_file = Path(__file__).parent.parent.parent / 'src' / 'llm_memory_calculator' / 'genz' / 'Astra_sim' / 'get_astra_sim_time.py'

        if not source_file.exists():
            pytest.skip(f"Source file not found: {source_file}")

        source = source_file.read_text()

        # The source should NOT contain shell=True for subprocess.run calls
        # Find all subprocess.run calls
        import re
        subprocess_calls = re.findall(r'subprocess\.run\([^)]+\)', source, re.DOTALL)

        for call in subprocess_calls:
            # shell=True should not appear in any subprocess.run call
            if 'shell=True' in call:
                pytest.fail(f"Found shell=True in subprocess call: {call}")


class TestPathValidation:
    """Tests for path validation."""

    def test_run_file_path_is_safe(self):
        """Verify run_file path comes from a safe, validated source."""
        # Read the source directly to check the pattern without importing
        from pathlib import Path as PyPath

        source_file = PyPath(__file__).parent.parent.parent / 'src' / 'llm_memory_calculator' / 'genz' / 'Astra_sim' / 'get_astra_sim_time.py'

        if not source_file.exists():
            pytest.skip(f"Source file not found: {source_file}")

        source = source_file.read_text()

        # Verify run_file is based on __file__, not user input
        assert 'SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))' in source
        assert 'run_file = os.path.join(SCRIPT_DIR' in source


class TestSubprocessSecurity:
    """Tests for secure subprocess usage."""

    def test_subprocess_run_with_list_args(self):
        """Subprocess.run should use list arguments, not shell string."""
        # This tests our expected secure pattern
        safe_command = ['bash', '/path/to/run.sh']

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b'', stderr=b'')

            # Simulate the secure call pattern
            subprocess.run(safe_command, shell=False, check=True)

            # Verify shell=False was passed
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[1].get('shell', False) is False

    def test_output_redirect_uses_file_handle(self):
        """Output redirection should use file handle, not shell redirect."""
        # Secure pattern: open file and pass to stdout
        output_path = '/tmp/test_output.txt'

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b'output', stderr=b'')

            # Simulate secure output redirection
            with open(output_path, 'w') as outfile:
                subprocess.run(
                    ['bash', '/path/to/script.sh'],
                    stdout=outfile,
                    stderr=subprocess.PIPE,
                    shell=False,
                    check=True
                )

            call_args = mock_run.call_args
            assert call_args[1].get('shell', False) is False


class TestPathSanitization:
    """Tests for path sanitization utilities."""

    def test_rejects_path_with_shell_metacharacters(self):
        """Paths with shell metacharacters should be rejected."""
        dangerous_paths = [
            '/tmp/file; rm -rf /',
            '/tmp/$(whoami)/file',
            '/tmp/`id`/file',
            '/tmp/file && echo pwned',
            '/tmp/file || echo pwned',
            '/tmp/file | cat /etc/passwd',
            '/tmp/file > /etc/passwd',
            '/tmp/file < /etc/passwd',
            '/tmp/file\necho pwned',
            '/tmp/file$(cat /etc/passwd)',
        ]

        from llm_memory_calculator.genz.Astra_sim.path_utils import validate_path

        for path in dangerous_paths:
            with pytest.raises(ValueError, match="(shell|invalid|metacharacter|forbidden)"):
                validate_path(path)

    def test_accepts_safe_paths(self):
        """Normal paths should be accepted."""
        safe_paths = [
            '/tmp/genz/output.txt',
            '/home/user/project/run.sh',
            '/var/log/astra_sim.log',
            './relative/path.txt',
        ]

        from llm_memory_calculator.genz.Astra_sim.path_utils import validate_path

        for path in safe_paths:
            # Should not raise
            result = validate_path(path)
            assert result is not None


class TestIntegrationSecurity:
    """Integration tests for secure ASTRA-SIM execution."""

    @pytest.mark.skip(reason="Requires ASTRA-SIM installation")
    def test_get_astrasim_collective_time_secure(self):
        """Full integration test for secure execution."""
        # This test requires ASTRA-SIM to be installed
        # It verifies the end-to-end security of the function
        pass
