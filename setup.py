from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "CalcModule",
        ["CalcModule.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)