import os

DATADIR = os.environ['ISOCLASSIFY']

from .grid import *
from .direct import *
from . import pipeline
