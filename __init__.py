import theano
#This forces it to use CPUs:
theano.config.floatX = 'float32' 
theano.config.set_device = 'cpu' 
theano.config.force_device = 'True' 

from . import tools
from . import MonoSearch
from . import MonoFit
from .stellar import starpars

import os
McmcTools_path = os.path.dirname(os.path.abspath( __file__ ))

