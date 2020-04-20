import os
os.environ['THEANO_FLAGS'] = "device=cpu"
McmcTools_path = os.path.dirname(os.path.abspath( __file__ ))

from . import tools
from . import MonoSearch
from . import MonoFit
from .stellar import starpars

