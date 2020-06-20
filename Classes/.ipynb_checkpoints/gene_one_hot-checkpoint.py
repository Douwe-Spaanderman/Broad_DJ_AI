### Author Douwe Spaanderman - 16 June 2020 ###

from dataclasses import dataclass, field
import numpy as np

@dataclass
class one_hot:
    counts: np.array
    levels: list
        
    def flatten(self):
        
        
    def 