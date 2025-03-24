import numpy as np

def mix_parameters(Xi, substance_parameters):
    """
    Source: 《流体热物性学》 P69
    混合多物质参数
    :param Xi: 各物质的摩尔分数
    :param substance_parameters: 各物质的 BWR 参数矩阵
    :return: 混合后的 BWR 参数
    """
    mixed = np.zeros(substance_parameters.shape[1])
    for param_idx in range(substance_parameters.shape[1]):
        if param_idx in [0, 2, 4, 6]:  # 三次根混合: a, b, c, alpha
            sum_term = sum(
                Xi[i] * np.sign(p) * np.abs(p) ** (1 / 3)
                for i, p in enumerate(substance_parameters[:, param_idx])
            )
            mixed[param_idx] = sum_term ** 3
        elif param_idx in [1, 5, 7]:  # 平方根混合: A0, C0, gamma
            sum_term = sum(
                Xi[i] * np.sign(p) * np.abs(p) ** (1 / 2)
                for i, p in enumerate(substance_parameters[:, param_idx])
            )
            mixed[param_idx] = sum_term ** 2
        elif param_idx == 3:  # 线性混合: B0
            mixed[param_idx] = np.dot(Xi[:substance_parameters.shape[0]], substance_parameters[:, param_idx])
    return mixed 