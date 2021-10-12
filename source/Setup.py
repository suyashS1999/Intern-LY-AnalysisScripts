from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules = cythonize('linearAdvec_mat.pyx', compiler_directives = {'language_level' : '3'}))
