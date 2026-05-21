"""Minimal setup.py shim — all metadata lives in pyproject.toml.

This file exists only for compatibility with tools that invoke setup.py
directly (e.g. some editable-install paths in older pip versions).
All package metadata, dependencies, and build configuration are declared
in pyproject.toml per PEP 517/621. setup.py must not duplicate or
override them.
"""

from setuptools import setup

setup()
