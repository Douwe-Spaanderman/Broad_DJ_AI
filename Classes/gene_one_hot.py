### Author Douwe Spaanderman - 16 June 2020 ###

from dataclasses import dataclass
import numpy as np
from collections import OrderedDict

@dataclass
class one_hot:
    counts: np.array
    levels: np.array
        
    def sanity(self):
        if len(self.counts.shape) == 1:
            assert self.counts.shape[0] == len(self.levels)
        else:
            assert self.counts.shape[0] == len(self.levels[0])
            assert self.counts.shape[1] == len(self.levels[1])
        
    def flatten(self):
        counts = np.array([item for sublist in self.counts for item in sublist])
        levels = np.array([str(x) + ":::" + str(y) for x in self.levels[0] for y in self.levels[1]])
        return one_hot(counts, levels)
        
    def make_2D(self, size):
        counts = self.counts.reshape(int(len(self.counts)/size), size)
        xlist,ylist = zip(*[x.split(":::") for x in self.levels])
        levels = np.array([list(OrderedDict.fromkeys(xlist)), list(OrderedDict.fromkeys(ylist))])
        return one_hot(counts, levels)