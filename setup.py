#!/usr/bin/env python
"""
This module contains setup instructions for sllim.
Based on pytube setup.py
"""
import codecs
import os

from setuptools import setup

with codecs.open(
    os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8"
) as fh:
    long_description = "\n" + fh.read()

setup(
    name="sllim",
    version="0.1.11",
    author="Kaiser Pister",
    author_email="kaiser@pister.dev",
    packages=["sllim"],
    package_data={
        "": ["LICENSE"],
    },
    url="https://github.com/kpister/sllim",
    license="MIT",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python",
        "Topic :: Terminals",
        "Topic :: Utilities",
    ],
    description=("A fixed api for openai."),
    include_package_data=True,
    long_description_content_type="text/markdown",
    long_description=long_description,
    zip_safe=True,
    python_requires=">=3.9",
    keywords=["gpt-4", "llm"],
)
