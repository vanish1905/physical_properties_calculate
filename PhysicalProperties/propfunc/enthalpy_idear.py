import numpy as np
from .parameter_all import acp

def enthalpy_idear(component, T):
    """
    计算理想气体的焓
    :param component: 组分数
    :param T: 温度 (K)
    :return: 理想气体的焓 (J/(mol*K))
    """
    Tref = 298.15
    a = acp
    Han_idear = np.zeros(component)
    for i in range(component):
        Han_idear[i] = a[i, 0] * (T - Tref) + 0.5 * a[i, 1] * (T ** 2 - Tref ** 2) + \
                        (1.0 / 3.0) * a[i, 2] * (T ** 3 - Tref ** 3) + 0.25 * a[i, 3] * (T ** 4 - Tref ** 4) + \
                        0.2 * a[i, 4] * (T ** 5 - Tref ** 5)
        Han_idear[i] = Han_idear[i] * 8.3145  # 单位：J/mol/K
    return Han_idear