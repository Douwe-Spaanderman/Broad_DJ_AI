from .version import __version__
from .main import main

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
    'main',
]