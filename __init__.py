import os
os.environ['THEANO_FLAGS'] = "device=cpu"
MonoTools_tablepath = os.path.join(os.path.dirname(os.path.abspath( __file__ )),'data','table')
if os.environ.get('MONOTOOLSPATH') is None:
    MonoData_savepath = os.environ.get('MONOTOOLSPATH')
else:
    MonoData_savepath = os.path.join(os.path.dirname(os.path.abspath( __file__ )),'data')

#from . import tools
#from . import MonoSearch
#from . import MonoFit
#from . import starpars

