import sys
import os

# Add root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR+'/../'))
from models.univnet.generator import Generator
import constants