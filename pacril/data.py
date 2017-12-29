# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats

def get_period(year):
    periods = np.array([[0, 1, 1900], [1, 1900, 1930], [2, 1930, 1960],
                        [3, 1960, 1985], [4, 1985, 10000]])
    period = periods[(periods[:, 1]<=year) & (year<periods[:, 2])][0][0]
    return period

def Locomotive:
    def __init__(self, xp, P, fuel, uic_class):
        self.year = year
        self.xp  = np.asfarray(xp)
        self.p = np.asfarray(p)
        self.fuel = fuel
        self.uic_class = uic_class

locomotives = dict()

locomotives[1] = Locomotive()



if __name__ == "__main__":
    print get_period(1899)