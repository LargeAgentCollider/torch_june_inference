import logging 
logger = logging.getLogger("initialization")
from .base import InferenceEngine
from .gradient_descent import GradientDescent
try:
    from .multinest import MultiNest
except:
    logger.warning("Failed to import MultiNest, not installed?")
from .ultranest import UltraNest
from .pyro import Pyro
from .svi import SVI
