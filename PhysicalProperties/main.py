"""
version: 1.7.4
time: 2025/03/23
Tips:
1, All code formatting has been adjusted
2, Adjusted visualisation subgraphs
3, Optional density calculation method set in the main function
4, Added the PR and SRK equation for calculating density
5, Added setting of whether data is exported or not
6, Fixed an issue with incorrect thermal conductivity
7, Fixed thermal capacity error
8, Fixed enthalpy error
9, Add content related to PR and SRK function coefficients
10, Add some equations about BWR function, but some coefficents in BWR_eos still need to concern
11, Add one colorful gui
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

from propfunc import *
from utils import *

if __name__ == '__main__':

    N = 60
    component = 19
    Pout = 3.45e6
    # Pout = 0.1e6
    Tin = 473
    To = 900
    Ti = 300
    Xi0 = np.array([1] + [0] * (component - 1))

    # 选择密度计算方法
    density_method = 'RK_PR'  # choose 'RK_PR' or 'PR' or 'SRK' or "BWR" (at low pressure situation like 0.1e6 Pa)

    properties = PhysicalProperties(Tin, Pout, component, density_method)
    
    # figures(To, Ti, Pout, N, component, Xi0, density_method, export_to_excel=False, export_path='RK_PR.xlsx')
    loop()
