# lv17analysis

# Stephan Kuschel, 2022

# from .other import *
# from .sphere_size import *
# from .cross_sections import *


from . import _version
__version__ = _version.get_versions()['version']


# Print version on every import so the version of the analysis
# hopefully sticks to some log file in case you need to reproduce old results
print('imported lv17analysis version: {}'.format(__version__))
