import numpy as np
from .parameter_all import acp

def cp_idear(component, T):
    """
    计算理想气体的热容
    :param component: 组分数
    :param T: 温度 (K)
    :return: 理想气体的热容 (J/(mol*K))
    """
    a = acp
    Cpi = np.zeros(component)
    for i in range(component):
        Cpi[i] = (a[i, 0] + a[i, 1] * T + a[i, 2] * T ** 2 + a[i, 3] * T ** 3 + a[i, 4] * T ** 4) * 8.3145
    return Cpi