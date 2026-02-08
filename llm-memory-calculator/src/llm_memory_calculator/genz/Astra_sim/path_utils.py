"""Path validation utilities for secure subprocess execution.

This module provides functions to validate and sanitize paths before
using them in subprocess calls to prevent shell injection attacks.
"""

import re
from pathlib import Path
from typing import Union


# Shell metacharacters that could be used for injection
SHELL_METACHARACTERS = frozenset([
    ';', '&', '|', '>', '<', '`', '$', '(', ')', '{', '}',
    '\n', '\r', '\0'
])

# Regex pattern to detect shell command injection attempts
INJECTION_PATTERN = re.compile(
    r'[;&|><`$(){}\n\r\x00]|'  # Individual metacharacters
    r'\$\(|'                    # Command substitution $(...)
    r'`[^`]*`|'                 # Backtick command substitution
    r'&&|'                      # Logical AND
    r'\|\||'                    # Logical OR
    r'>>'                       # Append redirect
)


class PathValidationError(ValueError):
    """Raised when a path fails security validation."""
    pass


def validate_path(path: Union[str, Path]) -> Path:
    """Validate a path for safe use in subprocess calls.

    Checks that the path does not contain shell metacharacters or
    patterns that could be used for command injection.

    Args:
        path: The path to validate.

    Returns:
        The validated path as a Path object.

    Raises:
        PathValidationError: If the path contains forbidden characters.
    """
    path_str = str(path)

    # Check for shell metacharacters
    if INJECTION_PATTERN.search(path_str):
        raise PathValidationError(
            f"Path contains forbidden shell metacharacters: {path_str!r}"
        )

    # Check for any individual metacharacters
    found_chars = set(path_str) & SHELL_METACHARACTERS
    if found_chars:
        raise PathValidationError(
            f"Path contains shell metacharacters {found_chars}: {path_str!r}"
        )

    return Path(path_str)


def validate_script_path(path: Union[str, Path], allowed_base: Path = None) -> Path:
    """Validate a script path for execution.

    Performs additional checks beyond basic path validation:
    - Resolves the path to eliminate symlink attacks
    - Optionally checks that the path is under an allowed base directory

    Args:
        path: The script path to validate.
        allowed_base: Optional base directory the script must be under.

    Returns:
        The validated, resolved path.

    Raises:
        PathValidationError: If the path fails validation.
    """
    validated = validate_path(path)

    # Resolve to catch symlink attacks
    try:
        resolved = validated.resolve()
    except (OSError, RuntimeError) as e:
        raise PathValidationError(f"Cannot resolve path {path}: {e}")

    # Check if under allowed base
    if allowed_base is not None:
        allowed_resolved = Path(allowed_base).resolve()
        try:
            resolved.relative_to(allowed_resolved)
        except ValueError:
            raise PathValidationError(
                f"Path {resolved} is not under allowed directory {allowed_resolved}"
            )

    return resolved
