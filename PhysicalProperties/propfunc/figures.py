import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .Phys_Prop import *

def figures(To, Ti, P, N, component, Xi, density_method, export_to_excel=False, export_path=None):
    """
    计算物理属性并绘制图表
    :param N: 等分数
    :param component: 组分数
    :param Xi: 摩尔分数
    :param export_to_excel: 是否导出数据为 Excel 文件，默认为 False
    :param export_path: Excel 文件的保存路径，默认为当前路径
    """

    P = P
    T_ = np.zeros(N)
    rou_ = np.zeros(N)
    eta_ = np.zeros(N)
    lambda_ = np.zeros(N)
    Cp_ = np.zeros(N)
    Tin = 473
    
    properties = PhysicalProperties(Tin, P, component, density_method)
    
    for i in range(N):
        T_[i] = (To - Ti) / (N - 1) * (i - 1) + Ti
        rou_[i] = properties.density(T_[i], P, component, Xi)
        eta_[i] = properties.viscosity_mixture(T_[i], rou_[i], component, Xi)
        lambda_[i] = properties.thermal_mixture(T_[i], rou_[i], component, Xi)
        Cp_[i] = properties.heat_capacity(T_[i], rou_[i], component, Xi)

    # 创建一个2行2列的子图
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    
    # 绘制density图
    axs[0, 0].plot(T_, rou_)
    axs[0, 0].set_title('Density')
    axs[0, 0].set_xlabel('Temperature (K)')
    axs[0, 0].set_ylabel('Density (kg/m^3)')
    axs[0, 0].grid(True, linestyle='--')  # 打开网格线并设置为虚线
    
    # 绘制viscosity图
    axs[0, 1].plot(T_, eta_)
    axs[0, 1].set_title('Viscosity')
    axs[0, 1].set_xlabel('Temperature (K)')
    axs[0, 1].set_ylabel('Viscosity (Pa·s)')
    axs[0, 1].grid(True, linestyle='--')  # 打开网格线并设置为虚线
    
    # 绘制lambda图
    axs[1, 0].plot(T_, lambda_)
    axs[1, 0].set_title('Thermal Conductivity')
    axs[1, 0].set_xlabel('Temperature (K)')
    axs[1, 0].set_ylabel('Thermal Conductivity (W/m·K)')
    axs[1, 0].grid(True, linestyle='--')  # 打开网格线并设置为虚线

    # 绘制heat capacity图
    axs[1, 1].plot(T_, Cp_)
    axs[1, 1].set_title('Heat Capacity')
    axs[1, 1].set_xlabel('Temperature (K)')
    axs[1, 1].set_ylabel('Heat Capacity (J/kg·K)')
    axs[1, 1].grid(True, linestyle='--')  # 打开网格线并设置为虚线
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 显示图形
    # plt.show()
    
    # 导出数据为 Excel 文件
    if export_to_excel:
        # 将数据保存为 DataFrame
        data = {
            'Temperature (K)': T_,
            'Density (kg/m^3)': rou_,
            'Viscosity (Pa·s)': eta_,
            'Thermal Conductivity (W/m·K)': lambda_,
            'Heat Capacity (J/kg·K)': Cp_
        }
        df = pd.DataFrame(data)
        # 设置保存路径
        if export_path is None:
            export_path = 'physical_properties.xlsx'  # 默认保存路径
        else:
            if not export_path.endswith('.xlsx'):
                export_path += '.xlsx'  # 确保文件扩展名为 .xlsx

        # 保存为 Excel 文件
        df.to_excel(export_path, index=False)
        print(f"Data has been exported to: {export_path}")
        
    return fig