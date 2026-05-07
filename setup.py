from setuptools import setup
from Cython.Build import cythonize
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

setup(
    ext_modules=cythonize(
        os.path.join(BASE_DIR, "semantic_norm.pyx")
    )
)
