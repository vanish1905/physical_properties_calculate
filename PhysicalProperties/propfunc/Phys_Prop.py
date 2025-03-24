import numpy as np
from scipy.optimize import brentq
from .parameter_all import *
from .mix_parameters import *
from .cp_idear import *
from .enthalpy_idear import *

class PhysicalProperties:
    def __init__(self, Tin, Pout, component, density_method):
        """
        初始化物理属性类
        :param Tin: 输入温度 (K)
        :param Pout: 输出压力 (Pa)
        :param component: 组分数
        :param density_method: 计算方法
        """
        self.Tin = Tin
        self.Pout = Pout
        self.component = component
        self.density_method = density_method  # 将 density_method 赋值给实例属性
        self.Xi0 = np.array([1] + [0] * (component - 1))  # 初始摩尔分数
        self.Yiθ = np.array([1] + [0] * (component - 1))  # 初始质量分数
        self.Cp = np.zeros(component)  # 热容
        self.Han = np.zeros(component)  # 焓
        self.eta = np.zeros(component)  # 粘度
        self.lambdam = np.zeros(component)  # 导热系数

        # RK-PR 状态方程参数
        self.taod = taod
        self.kA1 = kA1
        self.kA0 = kA0
        self.kB1 = kB1
        self.kB0 = kB0
        self.kC1 = kC1
        self.kC0 = kC0

        # 临界参数
        self.Tci = Tci
        self.Pci = Pci
        self.Vci = Vci
        self.biasnu = biasnu
        self.Zc = Zc
        self.Mi = Mi
        self.Href = Href

        # 状态方程系数
        self.taoi1 = np.zeros(component)
        self.taoi2 = np.zeros(component)
        self.alphaki = np.zeros(component)
        self.aii = np.zeros(component)
        self.bii = np.zeros(component)
        self.alphahi = np.zeros(component)
        self.aS = np.zeros(component)

        # 混合规则参数
        self.aij = np.zeros([component, component])
        self.am1 = 0
        self.bm1 = 0
        self.taomp1 = 0
        self.taomp2 = 0
        self.uu = 0
        self.ww = 0
        
        # BWR混合相关参数
        self.mix = np.zeros(8)

        # 温度相关参数
        self.Tri = np.zeros(component)
        self.alphai = np.zeros(component)
        self.palpti1 = np.zeros(component)
        self.palpti2 = np.zeros(component)
        self.aami = np.zeros([component, component])
        self.bbm = 0
        self.aam = 0
        self.palphaij_pt = np.zeros([component, component])
        self.palphaij_pt2 = np.zeros([component, component])
        self.palphat1 = 0
        self.palphat2 = 0

    def pre_eoscom(self):
        """
        计算状态方程的系数
        article paper: S.-K. Kim et al. / Combustion and Flame 159 (2012) 1351-1365  Table 2
        """
        omega = self.biasnu
        Pcc = self.Pci
        Tcc = self.Tci
        Zcc = self.Zc        
        if self.density_method == 'RK_PR':
            for i in range(self.component):
                if Zcc[i] > 0.29:
                    paraa = np.sqrt(2.0) - 1.0
                else:
                    paraa = self.taod[0] + self.taod[1] * (self.taod[2] - 1.168 * Zcc[i]) ** self.taod[3] + \
                            self.taod[4] * (self.taod[2] - 1.168 * Zcc[i]) ** self.taod[5]

                self.taoi1[i] = paraa
                self.taoi2[i] = (1.0 - paraa) / (1.0 + paraa)
                ad = (1.0 + paraa ** 2.0) / (1.0 + paraa)
                ay = 1.0 + (2.0 * (1.0 + paraa)) ** (1.0 / 3.0) + (4.0 / (1.0 + paraa)) ** (1.0 / 3.0)
                self.alphaki[i] = (1.168 * Zcc[i] * self.kA1 + self.kA0) * omega[i] ** 2.0 + \
                                (1.168 * Zcc[i] * self.kB1 + self.kB0) * omega[i] + \
                                (1.168 * Zcc[i] * self.kC1 + self.kC0)
                self.aii[i] = ((3.0 * ay * ay + 3.0 * ay * ad + ad * ad + ad - 1.0) / (3.0 * ay + ad - 1.0) ** 2.0) * \
                            (8.3145 ** 2 * Tcc[i] ** 2) / Pcc[i]
                self.bii[i] = 1.0 / (3.0 * ay + ad - 1.0) * (8.3145 * Tcc[i] / Pcc[i])
    
        elif self.density_method == 'PR':
            for i in range(self.component):
                self.taoi1[i] = 1 + np.sqrt(2)
                self.taoi2[i] = 1 - np.sqrt(2)
                self.aS[i] = 0.37464 + 1.54226 * omega[i] - 0.26992 * omega[i] ** 2
                self.aii[i] = 0.45724 * (8.3145 ** 2 * Tcc[i] ** 2) / Pcc[i]
                self.bii[i] = 0.07780 * ((8.3145 * Tcc[i]) / Pcc[i])
                
        elif self.density_method == 'SRK':
            for i in range(self.component):
                self.taoi1[i] = 1
                self.taoi2[i] = 0
                self.aS[i] = 0.48508 + 1.55171 * omega[i] - 0.15613 * omega[i] ** 2
                self.aii[i] = 0.42747 * (8.3145 ** 2 * Tcc[i] ** 2) / Pcc[i]
                self.bii[i] = 0.08664 * ((8.3145 * Tcc[i]) / Pcc[i])
    
        # 关于BWR的气体状态方程系数相关还需要查询相关论文
        elif self.density_method == 'BWR': 
            for i in range(self.component):
                if Zcc[i] > 0.29:
                    paraa = np.sqrt(2.0) - 1.0
                else:
                    paraa = self.taod[0] + self.taod[1] * (self.taod[2] - 1.168 * Zcc[i]) ** self.taod[3] + \
                            self.taod[4] * (self.taod[2] - 1.168 * Zcc[i]) ** self.taod[5]

                self.taoi1[i] = paraa
                self.taoi2[i] = (1.0 - paraa) / (1.0 + paraa)
                ad = (1.0 + paraa ** 2.0) / (1.0 + paraa)
                ay = 1.0 + (2.0 * (1.0 + paraa)) ** (1.0 / 3.0) + (4.0 / (1.0 + paraa)) ** (1.0 / 3.0)
                self.alphaki[i] = (1.168 * Zcc[i] * self.kA1 + self.kA0) * omega[i] ** 2.0 + \
                                (1.168 * Zcc[i] * self.kB1 + self.kB0) * omega[i] + \
                                (1.168 * Zcc[i] * self.kC1 + self.kC0)
                self.aii[i] = ((3.0 * ay * ay + 3.0 * ay * ad + ad * ad + ad - 1.0) / (3.0 * ay + ad - 1.0) ** 2.0) * \
                            (8.3145 ** 2 * Tcc[i] ** 2) / Pcc[i]
                self.bii[i] = 1.0 / (3.0 * ay + ad - 1.0) * (8.3145 * Tcc[i] / Pcc[i])          
        else:
            raise ValueError(f"Unknown density method: {self.density_method}")

    def eos_mix(self, T, component, Xi):
        """
        计算混合物状态参数所需要的混合规则
        :param T: 温度 (K)
        :param component: 组分数
        :param Xi: 摩尔分数
        """
        ai = self.aii
        bi = self.bii
        tao1 = self.taoi1
        tao2 = self.taoi2
        aS = self.aS
        
        if self.density_method == 'RK_PR':
            for i in range(component):
                self.alphahi[i] = (3.0 / (2.0 + T / self.Tci[i])) ** self.alphaki[i]
        
        elif self.density_method == "BWR":   
            for i in range(component):
                self.alphahi[i] = (3.0 / (2.0 + T / self.Tci[i])) ** self.alphaki[i]
            bwr_parameters = self.bwr_parameters
            self.mix = mix_parameters(Xi, bwr_parameters) 
                
        elif self.density_method == 'PR':
            for i in range(component):
                self.alphahi[i] = (1.0 + aS[i] * (1.0 - np.sqrt(T / self.Tci[i]))) ** 2
        
        elif self.density_method == "SRK":
            for i in range(component):
                self.alphahi[i] = (1.0 + aS[i] * (1.0 - np.sqrt(T / self.Tci[i]))) ** 2
            
        else:
            raise ValueError(f"Unknown density method: {self.density_method}")
  
        for i in range(component):
            for j in range(component):
                self.aij[i, j] = np.sqrt(ai[i] * ai[j] * self.alphahi[i] * self.alphahi[j])

        self.am1 = 0.0
        self.bm1 = 0.0
        self.taomp1 = 0.0
        self.taomp2 = 0.0
        for i in range(component):
            self.bm1 += Xi[i] * bi[i]
            self.taomp1 += Xi[i] * tao1[i]
            self.taomp2 += Xi[i] * tao2[i]
            for j in range(component):
                self.am1 += Xi[i] * Xi[j] * self.aij[i, j]

        self.uu = self.taomp1 + self.taomp2
        self.ww = self.taomp1 * self.taomp2

    def eos_combine(self, T, component, Xi):
        """
        计算混合物状态参数所需要的组合规则
        :param T: 温度 (K)
        :param component: 组分数
        :param Xi: 摩尔分数
        """
        omegai = self.biasnu
        Tci = self.Tci
        Vci = self.Vci
        Mi = 1000 * self.Mi
        Mw = np.dot(Xi, Mi)  # 平均摩尔质量 (kg/mol)
        aphk = self.alphaki
        ai = self.aii
        bi = self.bii
        tao1 = self.taoi1
        tao2 = self.taoi2

        self.taomp1 = 0.0
        self.taomp2 = 0.0
        for i in range(component):
            self.taomp1 += Xi[i] * tao1[i]
            self.taomp2 += Xi[i] * tao2[i]
            self.Tri[i] = T / Tci[i]
            self.alphai[i] = (3.0 / (2.0 + self.Tri[i])) ** aphk[i]
            self.palpti1[i] = -3.0 ** aphk[i] * aphk[i] / Tci[i] / (2.0 + self.Tri[i]) ** (aphk[i] + 1.0)
            self.palpti2[i] = 3.0 ** aphk[i] * aphk[i] * (aphk[i] + 1.0) / Tci[i] ** 2 / (2.0 + self.Tri[i]) ** (aphk[i] + 2.0)

        alphami = self.alphai
        for i in range(component):
            for j in range(component):
                self.aami[i, j] = np.sqrt(ai[i] * ai[j] * alphami[i] * alphami[j])

        self.bbm = 0
        self.aam = 0
        for i in range(component):
            self.bbm += Xi[i] * bi[i]
            for j in range(component):
                self.aam += Xi[i] * Xi[j] * self.aami[i, j]

        palptmi1 = self.palpti1
        palptmi2 = self.palpti2
        for i in range(component):
            for j in range(component):
                self.palphaij_pt[i, j] = np.sqrt(ai[i] * ai[j]) * 0.5 * (np.sqrt(alphami[i] / alphami[j]) * palptmi1[j] + \
                                                                         np.sqrt(alphami[j] / alphami[i]) * palptmi1[i])
                self.palphaij_pt2[i, j] = np.sqrt(ai[i] * ai[j]) * (0.5 / np.sqrt(alphami[i] * alphami[j]) * palptmi1[i] * palptmi1[j] - \
                                                                    0.25 * np.sqrt(alphami[i] / alphami[j] ** 3) * palptmi1[j] ** 2 - \
                                                                    0.25 * np.sqrt(alphami[j] / alphami[i] ** 3) * palptmi1[i] ** 2 + \
                                                                    0.5 * np.sqrt(alphami[i] / alphami[j]) * palptmi2[j] + \
                                                                    0.5 * np.sqrt(alphami[j] / alphami[i]) * palptmi2[i])

        self.palphat1 = 0
        self.palphat2 = 0
        for i in range(component):
            for j in range(component):
                self.palphat1 += Xi[i] * Xi[j] * self.palphaij_pt[i, j]
                self.palphat2 += Xi[i] * Xi[j] * self.palphaij_pt2[i, j]

    def density_cubic(self, T, P, component, Xi):
        """
        T:输入温度K, P:输入压力pa
        Tci:输入临界温度K, Pci:输入压力Pa
        Mi:各组分摩尔质量kg/mol, Xi:摩尔分数, Yi:质量分数
        biasnu:偏心因子
        盛金公式求解循环
        from: 论文_超临界碳氢燃料蒸汽重整过程的热沉分布特征研究_刘玉娜_2.4_物性计算方法
        """
        self.eos_mix(T, component, Xi)

        Mw = np.dot(Xi, self.Mi)
        am = self.am1
        bm = self.bm1
        if self.density_method == 'RK_PR':
            ww = self.ww
            uu = self.uu
        elif self.density_method == 'PR':
            ww = -1.
            uu = 2. 
        elif self.density_method == 'SRK':  
            ww = 0.
            uu = 1.             
        # T = self.Tin
        # P = self.Pout

        aa = am * bm + ww * bm * bm * 8.3145 * T + P * ww * bm ** 3.0
        bb = 8.3145 * T * uu * Mw * bm - am * Mw + P * bm * bm * uu * Mw - P * Mw * ww * bm ** 2.0
        cc = 8.3145 * T * Mw * Mw + P * bm * Mw * Mw - P * uu * Mw * Mw * bm
        dd = -P * Mw ** 3.0
        A = bb ** 2 - 3.0 * aa * cc
        B = bb * cc - 9.0 * aa * dd
        C = cc ** 2 - 3.0 * bb * dd
        delta = B ** 2 - 4.0 * A * C

        if (A == 0.0) and (B == 0.0):
            X1 = -bb / 3.0 / aa
            X2 = X1
            X3 = X1
        elif delta > 0.0:
            Y1 = A * bb + 3.0 * aa * (-B + np.sqrt(B * B - 4.0 * A * C)) * 0.5
            Y2 = A * bb + 3.0 * aa * (-B - np.sqrt(B * B - 4.0 * A * C)) * 0.5
            if Y1 < 0.0:
                if Y2 < 0.0:
                    X1 = (-bb - (-abs(Y1) ** (1.0 / 3.0) - abs(Y2) ** (1.0 / 3.0))) / 3.0 / aa
                    X2 = (-2 * bb - abs(Y1) ** (1 / 3) - abs(Y2) ** (1 / 3) + (
                                3 ** (1 / 2) * (-abs(Y1) ** (1 / 3) + abs(Y2) ** (1 / 3))) * 1j) / (6 * aa)
                    X3 = (-2 * bb - abs(Y1) ** (1 / 3) - abs(Y2) ** (1 / 3) - (
                                3 ** (1 / 2) * (-abs(Y1) ** (1 / 3) + abs(Y2) ** (1 / 3))) * 1j) / (6 * aa)
                else:
                    X1 = (-bb - (-abs(Y1) ** (1.0 / 3.0) + abs(Y2) ** (1.0 / 3.0))) / 3.0 / aa
                    X2 = (-2 * bb - abs(Y1) ** (1 / 3) + abs(Y2) ** (1 / 3) + (
                                3 ** (1 / 2) * (-abs(Y1) ** (1 / 3) - abs(Y2) ** (1 / 3))) * 1j) / (6 * aa)
                    X3 = (-2 * bb - abs(Y1) ** (1 / 3) + abs(Y2) ** (1 / 3) - (
                                3 ** (1 / 2) * (-abs(Y1) ** (1 / 3) - abs(Y2) ** (1 / 3))) * 1j) / (6 * aa)
            else:
                if Y2 < 0.0:
                    X1 = (-bb - (abs(Y1) ** (1.0 / 3.0) - abs(Y2) ** (1.0 / 3.0))) / 3.0 / aa
                    X2 = (-2 * bb + abs(Y1) ** (1 / 3) - abs(Y2) ** (1 / 3) + (
                                3 ** (1 / 2) * (abs(Y1) ** (1 / 3) + abs(Y2) ** (1 / 3))) * 1j) / (6 * aa)
                    X3 = (-2 * bb + abs(Y1) ** (1 / 3) - abs(Y2) ** (1 / 3) - (
                                3 ** (1 / 2) * (abs(Y1) ** (1 / 3) + abs(Y2) ** (1 / 3))) * 1j) / (6 * aa)
                else:
                    X1 = (-bb - (Y1 ** (1.0 / 3.0) + Y2 ** (1.0 / 3.0))) / 3.0 / aa
                    X2 = (-2 * bb + (Y1) ** (1 / 3) + (Y2) ** (1 / 3) + (
                                3 ** (1 / 2) * ((Y1) ** (1 / 3) - (Y2) ** (1 / 3))) * 1j) / (6 * aa)
                    X3 = (-2 * bb + (Y1) ** (1 / 3) + (Y2) ** (1 / 3) - (
                                3 ** (1 / 2) * ((Y1) ** (1 / 3) - (Y2) ** (1 / 3))) * 1j) / (6 * aa)
        elif delta == 0.0:
            X1 = -bb / aa + B / A
            X2 = -B / A * 0.5
            X3 = X2
        elif delta < 0.0:
            salta = np.arccos((2.0 * A * bb - 3.0 * aa * B) * 0.5 / np.sqrt(A ** 3))
            X1 = (-bb - 2.0 * np.sqrt(A) * np.cos(salta / 3.0)) / 3.0 / aa
            X2 = (-bb + np.sqrt(A) * (np.cos(salta / 3.0) + np.sqrt(3.0) * np.sin(salta / 3.0))) / 3.0 / aa
            X3 = (-bb + np.sqrt(A) * (np.cos(salta / 3.0) - np.sqrt(3.0) * np.sin(salta / 3.0))) / 3.0 / aa

        if np.imag(X1) == 0:
            if np.imag(X2) == 0:
                test = min(X1, X2)
                rou = min(test, X3)
            else:
                rou = X1
        else:
            if np.imag(X2) == 0:
                rou = X2
            else:
                rou = X3

        return rou

    def bwr_pressure(self, rho, T, R=8.314, **params):
        """
        Source: 《流体热物性学》 P38
        BWR 方程计算压力
        :param rho: 密度 (mol/L)
        :param T: 温度 (K)
        :param R: 气体常数 (J/(mol·K))
        :param params: BWR 方程参数
        :return: 压力 (atm)
        """
        term1 = (rho * R * T) / 101.325
        term2 = (params['B0'] * R * T / 101.325 - params['A0'] - params['C0'] / T ** 2) * rho ** 2
        term3 = (params['b'] * R * T / 101.325 - params['a']) * rho ** 3
        term4 = params['a'] * params['alpha'] * rho ** 6
        exp_term = (params['c'] * rho ** 3 / T ** 2) * (1 + params['gamma'] * rho ** 2) * np.exp(-params['gamma'] * rho ** 2)
        return term1 + term2 + term3 + term4 + exp_term


    def solve_bwr_density(self, P_target, T, **params):
        """
        求解 BWR 密度
        :param P_target: 目标压力 (Pa)
        :param T: 温度 (K)
        :param params: BWR 方程参数
        :return: 密度 (mol/L)
        """
        def f(rho):
            return self.bwr_pressure(rho, T, **params) - P_target / 101325

        rho_initial = 0.001 * P_target / (8.314 * T)  # 理想气体初始猜测
        return brentq(f, 1e-3 * rho_initial, 3000, maxiter=100)


    def density_BWR(self, T, P_target, component, Xi):
        """
        主计算函数
        :param P_target: 目标压力 (Pa)
        :param T_range: 温度范围 (K)
        :param Xi: 各物质的摩尔分数
        :param export_path: 数据保存路径
        :return: 温度数组和密度数组
        """
        # 参数混合
        bwr_parameters = self.bwr_parameters
        self.eos_mix(T, component, Xi)
        mixed = self.mix
        params = {
            'a': mixed[0], 'A0': mixed[1], 'b': mixed[2],
            'B0': mixed[3], 'c': mixed[4], 'C0': mixed[5],
            'alpha': mixed[6], 'gamma': mixed[7]
        }
        Mi = np.array([142.29, 16.04, 17.03, 58.12, 100.21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # g/mol
        # 计算混合摩尔质量
        M_mix = np.dot(Xi, Mi)

        # 密度计算
        densities_molL = self.solve_bwr_density(P_target, T, **params)  # Unit: mol/L

        # 单位转换: mol/L → kg/m³
        densities_kgm3 = densities_molL* M_mix

        return densities_kgm3

    def density(self, T, P, component, Xi):
        """
        根据 density_method 选择密度计算方法
        :param T: 温度 (K)
        :param P: 压力 (Pa)
        :param component: 组分数
        :param Xi: 摩尔分数
        :return: 密度 (kg/m^3)
        """
        self.pre_eoscom()
        if self.density_method == 'RK_PR':       
            return self.density_cubic(T, P, component, Xi)
        elif self.density_method == 'PR':
            return self.density_cubic(T, P, component, Xi)
        elif self.density_method == 'SRK':
            return self.density_cubic(T, P, component, Xi)
        elif self.density_method == 'BWR':
            return self.density_BWR(T, P, component, Xi)
        else:
            raise ValueError(f"Unknown density method: {self.density_method}")

    def viscosity(self, T, rou):
        """
        计算粘度
        :param T: 温度 (K)
        :param rou: 密度 (kg/m^3)
        :return: 粘度 (Pa*s)
        """
        omega = self.biasnu
        Tc = self.Tci
        Vc = self.Vci
        Mi = self.Mi
        M = 1000 * Mi

        V = 1000 * M / rou  # V: cm3/mol
        k = Tc / 1.2593
        Tx = T / k
        psiv = 1.16145 * (Tx) ** (-0.14874) + 0.52487 * np.exp(-0.77320 * Tx) + 2.16178 * np.exp(-2.43787 * Tx)

        E1 = 6.324 + 50.412 * omega
        E2 = 1.21e-3 - 1.154e-3 * omega
        E3 = 5.283 + 254.209 * omega
        E4 = 6.623 + 38.096 * omega
        E5 = 19.745 + 7.630 * omega
        E6 = -1.9 - 12.537 * omega
        E7 = 24.275 + 3.45 * omega
        E8 = 0.7972 + 1.117 * omega
        E9 = (-0.2382) + 0.0677 * omega
        E10 = 0.06863 + 0.3479 * omega

        y = Vc / V / 6.0
        G1 = (1 - 0.5 * y) / ((1 - y) ** 3)
        G2 = E1 * ((1 - np.exp(-E4 * y)) / y) + E2 * G1 * np.exp(E5 * y) + E3 * G1
        G2 = G2 / (E1 * E4 + E2 + E3)
        Fc = 1 - 0.2756 * omega

        etaxx = E7 * (y ** 2) * G2 * np.exp(E8 + E9 / Tx + E10 * (Tx ** (-2)))
        etax = (Tx ** 0.5) * (Fc * (1 / G2 + E6 * y)) / psiv + etaxx

        eta = etax * 36.344 * (M * Tc) ** 0.5
        eta = eta / (Vc ** (2.0 / 3.0))  # 单位:微 P(微泊) 1P = 100cP, 1cP = 1mPa*s
        eta = eta / 1e7  # 单位:Pa*s

        return eta

    def viscosity_mixture(self, T, rou, component, Xi):
        """
        计算混合物的粘度
        :param T: 温度 (K)
        :param rou: 密度 (kg/m^3)
        :param component: 组分数
        :param Xi: 摩尔分数
        :return: 混合物的粘度 (Pa*s)
        """
        omegai = self.biasnu
        Tci = self.Tci
        Vci = self.Vci
        Mi = 1000 * self.Mi
        sigmai = np.zeros(component)
        epslionki = np.zeros(component)

        for i in range(component):
            sigmai[i] = 0.809 * Vci[i] ** (1 / 3)
            epslionki[i] = Tci[i] / 1.2593

        sigmaij = np.zeros((component, component))
        epslionkij = np.zeros((component, component))
        omegaij = np.zeros((component, component))
        Mij = np.zeros((component, component))

        for i in range(component):
            for j in range(component):
                sigmaij[i, j] = np.sqrt(sigmai[i] * sigmai[j])
                epslionkij[i, j] = np.sqrt(epslionki[i] * epslionki[j])
                omegaij[i, j] = 0.5 * (omegai[i] + omegai[j])
                Mij[i, j] = 2 * Mi[i] * Mi[j] / (Mi[i] + Mi[j])

        sigmam3 = 0.0
        epslionkm = 0.0
        omegam = 0.0
        Mm = 0.0

        for i in range(component):
            for j in range(component):
                sigmam3 += Xi[i] * Xi[j] * sigmaij[i, j] ** 3

        for i in range(component):
            for j in range(component):
                epslionkm += Xi[i] * Xi[j] * epslionkij[i, j] * sigmaij[i, j] ** 3

        epslionkm = epslionkm / sigmam3

        for i in range(component):
            for j in range(component):
                omegam += Xi[i] * Xi[j] * omegaij[i, j] * sigmaij[i, j] ** 3
                Mm += Xi[i] * Xi[j] * epslionkij[i, j] * sigmaij[i, j] * sigmaij[i, j] * np.sqrt(Mij[i, j])

        omegam = omegam / sigmam3
        Mm = (Mm / epslionkm / sigmam3 ** (2.0 / 3.0)) ** 2

        Tcm = 1.2593 * epslionkm
        Vcm = sigmam3 / 0.809 ** 3
        Tmx = T / epslionkm
        psiv = 1.16145 * (Tmx) ** (-0.14874) + 0.52487 * np.exp(-0.77320 * Tmx) + 2.16178 * np.exp(-2.43787 * Tmx)
        Fcm = 1 - 0.2756 * omegam

        etam0 = 40.785 * Fcm * (Mm * T) ** 0.5 / (Vcm ** (2.0 / 3.0)) / psiv  # 单位:微 P
        etam0 = etam0 / 1e7  # 单位:Pa*s

        E1 = 6.324 + 50.412 * omegam
        E2 = 1.21e-3 - 1.154e-3 * omegam
        E3 = 5.283 + 254.209 * omegam
        E4 = 6.623 + 38.096 * omegam
        E5 = 19.745 + 7.630 * omegam
        E6 = -1.9 - 12.537 * omegam
        E7 = 24.275 + 3.45 * omegam
        E8 = 0.7972 + 1.117 * omegam
        E9 = (-0.2382) + 0.0677 * omegam
        E10 = 0.06863 + 0.3479 * omegam

        V = 1000 * Mm / rou  # V: cm3/mol
        y = Vcm / V / 6.0
        G1 = (1 - 0.5 * y) / ((1 - y) ** 3)
        G2 = E1 * ((1 - np.exp(-E4 * y)) / y) + E2 * G1 * np.exp(E5 * y) + E3 * G1
        G2 = G2 / (E1 * E4 + E2 + E3)
        Fc = 1 - 0.2756 * omegam

        etaxx = E7 * (y ** 2) * G2 * np.exp(E8 + E9 / Tmx + E10 * (Tmx ** (-2)))
        etax = (Tmx ** 0.5) * (Fc * (1 / G2 + E6 * y)) / psiv + etaxx

        eta = etax * 36.344 * (Mm * Tcm) ** 0.5
        eta = eta / (Vcm ** (2.0 / 3.0))  # 单位:微 P(微泊) 1P = 100cP, 1cP = 1mPa*s
        eta = eta / 1e7  # 单位:Pa*s

        return eta

    def thermal(self, component, T, rou):
        """
        计算单一物质的导热系数
        :param component: 组分数
        :param T: 温度 (K)
        :param rou: 密度 (kg/m^3)
        :return: 导热系数 (W/(m*K))
        """
        Cpi = cp_idear(component, T)
        Cp = Cpi

        omega = self.biasnu
        Tc = self.Tci
        Vc = self.Vci
        M = 1000 * self.Mi
        V = 1000 * M / rou

        Tx = 1.2593 * T / Tc
        omega_v = 1.16145 * (Tx) ** (-0.14874) + 0.52487 * np.exp(-0.77320 * Tx) + 2.16178 * np.exp(-2.43787 * Tx)
        Fc = 1 - 0.2756 * omega
        eta0 = 40.785 * Fc * (M * T) ** 0.5 / (Vc ** (2.0 / 3.0)) / omega_v  # 单位:微 P
        eta0 = eta0 / 1e7  # 单位:Pa*s

        Tr = T / Tc
        Z = 2 + 10.5 * Tr ** 2
        Cv = Cp - 8.3145  # 理想气体的定容热容
        alpha = (Cv / 8.3145) - 3 / 2
        beta = 0.7862 - 0.7109 * omega + 1.3168 * omega ** 2
        psi = 1 + alpha * ((0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * Z) / (0.6366
                                                                                     + beta * Z + 1.061 * alpha * beta))

        y = Vc / V / 6.0

        B1 = 2.4166 + 7.4824e-1 * omega
        B2 = -5.0924e-1 - 1.5094 * omega
        B3 = 6.6107 + 5.6207 * omega
        B4 = 1.4543e1 - 8.9139 * omega
        B5 = 7.9274e-1 + 8.2019e-1 * omega
        B6 = -5.8634 + 1.2801e1 * omega
        B7 = 9.1089e1 + 1.2811e2 * omega

        G1 = (1 - 0.5 * y) / (1 - y) ** 3
        G2 = (B1 / y) * (1 - np.exp(-B4 * y)) + B2 * G1 * np.exp(B5 * y) + B3 * G1
        G2 = G2 / (B1 * B4 + B2 + B3)
        q = 3.586e-3 * (Tc / (M / 1000)) ** 0.5 / (Vc ** (2 / 3))  # M/1000:单位转换为 kg/mol

        lambda_val = (31.2 * eta0 * psi / (M / 1000) * (1 / G2 + B6 * y) + q * B7 * y ** 2 * Tr ** 0.5 * G2)

        return lambda_val

    def thermal_mixture(self, T, rou, component, Xi):
        """
        计算混合物的导热系数
        :param T: 温度 (K)
        :param rou: 密度 (kg/m^3)
        :param component: 组分数
        :param Xi: 摩尔分数
        :return: 混合物的导热系数 (W/(m*K))
        """
        Cpi = cp_idear(component, T)

        omegai = self.biasnu
        Tci = self.Tci
        Vci = self.Vci
        Mi = 1000 * self.Mi

        sigmai = np.zeros(component)
        epslionki = np.zeros(component)
        for i in range(component):
            sigmai[i] = 0.809 * Vci[i] ** (1 / 3)
            epslionki[i] = Tci[i] / 1.2593

        sigmaij = np.zeros((component, component))
        epslionkij = np.zeros((component, component))
        omegaij = np.zeros((component, component))
        Mij = np.zeros((component, component))
        for i in range(component):
            for j in range(component):
                sigmaij[i, j] = np.sqrt(sigmai[i] * sigmai[j])
                epslionkij[i, j] = np.sqrt(epslionki[i] * epslionki[j])
                omegaij[i, j] = 0.5 * (omegai[i] + omegai[j])
                Mij[i, j] = 2 * Mi[i] * Mi[j] / (Mi[i] + Mi[j])

        sigmam3 = 0.0
        epslionkm = 0.0
        omegam = 0.0
        Mm = 0.0

        for i in range(component):
            for j in range(component):
                sigmam3 += Xi[i] * Xi[j] * sigmaij[i, j] ** 3

        for i in range(component):
            for j in range(component):
                epslionkm += Xi[i] * Xi[j] * epslionkij[i, j] * sigmaij[i, j] ** 3
        epslionkm = epslionkm / sigmam3

        for i in range(component):
            for j in range(component):
                omegam += Xi[i] * Xi[j] * omegaij[i, j] * sigmaij[i, j] ** 3
                Mm += Xi[i] * Xi[j] * epslionkij[i, j] * sigmaij[i, j] * sigmaij[i, j] * np.sqrt(Mij[i, j])
        omegam = omegam / sigmam3
        Mm = (Mm / epslionkm / sigmam3 ** (2.0 / 3.0)) ** 2

        Tcm = 1.2593 * epslionkm
        Vcm = sigmam3 / 0.809 ** 3

        Tmx = T / epslionkm
        omega_v = 1.16145 * (Tmx) ** (-0.14874) + 0.52487 * np.exp(-0.77320 * Tmx) + 2.16178 * np.exp(-2.43787 * Tmx)
        Fcm = 1 - 0.2756 * omegam

        etam0 = 40.785 * Fcm * (Mm * T) ** 0.5 / (Vcm ** (2.0 / 3.0)) / omega_v  # 单位:微 P
        etam0 = etam0 / 1e7  # 单位:Pa*s

        Tr = T / Tcm
        Z = 2 + 10.5 * Tr ** 2
        Cvi = Cpi - np.tile(8.3145, component)  # 理想气体的定容热容
        Cvm = np.dot(Cvi, Xi)
        alpham = (Cvm / 8.3145) - 3 / 2
        betam = 0.7862 - 0.7109 * omegam + 1.3168 * omegam ** 2
        psi = 1 + alpham * ((0.215 + 0.28288 * alpham - 1.061 * betam + 0.26665 * Z) /
                            (0.6366 + betam * Z + 1.061 * alpham * betam))

        V = 1000 * Mm / rou  # V: cm3/mol
        ym = Vcm / V / 6.0

        B1 = 2.4166 + 7.4824e-1 * omegam
        B2 = -5.0924e-1 - 1.5094 * omegam
        B3 = 6.6107 + 5.6207 * omegam
        B4 = 1.4543e1 - 8.9139 * omegam
        B5 = 7.9274e-1 + 8.2019e-1 * omegam
        B6 = -5.8634 + 1.2801e1 * omegam
        B7 = 9.1089e1 + 1.2811e2 * omegam

        G1 = (1 - 0.5 * ym) / (1 - ym) ** 3
        G2 = (B1 / ym) * (1 - np.exp(-B4 * ym)) + B2 * G1 * np.exp(B5 * ym) + B3 * G1
        G2 = G2 / (B1 * B4 + B2 + B3)
        q = 3.586e-3 * (Tcm / (Mm / 1000)) ** 0.5 / (Vcm ** (2 / 3))  # Mm/1000:单位转换为 kg/mol

        lambdam = (31.2 * etam0 * psi / (Mm / 1000) * (1 / G2 + B6 * ym) + q * B7 * ym ** 2 * Tr ** 0.5 * G2)

        return lambdam

    def heat_capacity(self, T, rou, component, Xi):
        """
        计算热容
        :param T: 温度 (K)
        :param rou: 密度 (kg/m^3)
        :param component: 组分数
        :param Xi: 摩尔分数
        :return: 热容 (J/kg/K)
        """
        Cpi = cp_idear(component, T)
        Mi = self.Mi
        Mw = np.dot(Xi, Mi)  # 平均摩尔质量 (kg/mol)

        self.eos_combine(T, component, Xi)
        aam = self.aam
        bbm = self.bbm
        taomp1 = self.taomp1
        taomp2 = self.taomp2
        palphat1 = self.palphat1
        palphat2 = self.palphat2

        Cvi = np.zeros(component)
        pppt = 8.3145 * rou / (Mw - bbm * rou) - \
               palphat1 * rou * rou / (Mw + taomp1 * bbm * rou) / (Mw + taomp2 * bbm * rou)
        ppprou = 8.3145 * T * Mw / (Mw - bbm * rou) ** 2\
                 - aam * rou * Mw * (2 * Mw + (taomp1 + taomp2) * bbm * rou) / (Mw + taomp1 * bbm * rou)\
                 ** 2 / (Mw + taomp2 * bbm * rou) ** 2
        for i in range(component):
            Cvi[i] = Cpi[i] - 8.3145
        Cvm = np.dot(Cvi, Xi)
        Cvm = Cvm / Mw  # J/mol/K 转化为 J/kg/K
        Cv = Cvm + T / ((taomp1 - taomp2) * bbm * Mw) * palphat2 \
             * np.log((Mw + taomp1 * bbm * rou) / (Mw + taomp2 * bbm * rou))
        Cp = Cv + T * pppt * pppt / (rou * rou) / ppprou  # J/kg/K

        return Cp

    def enthalpy(self, T, P, rou, component, Xi, Yi, Yi0):
        """
        计算气体的焓
        :param T: 温度 (K)
        :param P: 压力 (Pa)
        :param rou: 密度 (kg/m^3)
        :param component: 组分数
        :param Xi: 摩尔分数
        :param Yi: 质量分数
        :param Yi0: 初始质量分数
        :return: 焓 (J/kg)
        """
        Tci = self.Tci
        Mi = self.Mi
        Href = self.Href
        Mw = np.dot(Xi, Mi)  # 平均摩尔质量 (kg/mol)
        aphk = self.alphaki
        ai = self.aii

        self.eos_combine(T, component, Xi)
        aam = self.aam
        bbm = self.bbm
        taomp1 = self.taomp1
        taomp2 = self.taomp2
        palphat1 = self.palphat1

        Han_idear = enthalpy_idear(component, T)
        Han = (np.dot(Han_idear, Xi) - 8.3145 * T) / Mw  # J/mol 转化为 J/kg
        para = 1.0 / ((taomp1 - taomp2) * bbm * Mw) * (T * palphat1 - aam)\
               * np.log((Mw + taomp1 * bbm * rou) / (Mw + taomp2 * bbm * rou))
        Han = Han + para + P / rou

        Tref = 298.15
        Pref = 101325
        Tri = np.zeros(component)
        alphairef = np.zeros(component)
        palpti1ref = np.zeros(component)

        rouref = self.density(Tref, Pref, component, Xi)
        for i in range(component):
            Tri[i] = Tref / Tci[i]
            alphairef[i] = (3.0 / (2.0 + Tri[i])) ** aphk[i]
            palpti1ref[i] = -3.0 ** aphk[i] * aphk[i] / Tci[i] / (2.0 + Tri[i]) ** (aphk[i] + 1.0)

        alphamiref = alphairef
        amiref = np.zeros([component, component])
        for i in range(component):
            for j in range(component):
                amiref[i, j] = np.sqrt(ai[i] * ai[j] * alphamiref[i] * alphamiref[j])

        amref = 0.0
        for i in range(component):
            for j in range(component):
                amref = amref + Xi[i] * Xi[j] * amiref[i, j]

        palptmi1ref = palpti1ref
        palphaij_ptref = np.zeros([component, component])
        for i in range(component):
            for j in range(component):
                palphaij_ptref[i, j] = np.sqrt(ai[i] * ai[j]) * 0.5 * (np.sqrt(alphamiref[i] / alphamiref[j]) * palptmi1ref[j] +
                                                                         np.sqrt(alphamiref[j] / alphamiref[i]) * palptmi1ref[i])

        palphat1ref = 0.0
        for i in range(component):
            for j in range(component):
                palphat1ref = palphat1ref + Xi[i] * Xi[j] * palphaij_ptref[i, j]

        Yirea = np.zeros(component)
        for i in range(component):
            Yirea[i] = Yi[i] - Yi0[i]
        Href = Href / Mi

        Hanref = -8.3145 * Tref / Mw
        Hanref = Hanref\
                 + 1.0 / ((taomp1 - taomp2) * bbm * Mw) * (Tref * palphat1ref - amref) *\
                 np.log((Mw + taomp1 * bbm * rouref) / (Mw + taomp2 * bbm * rouref))\
                 + Pref / rouref
        Han = Han - Hanref + np.dot(Href, Yi)

        return Han


if __name__ == '__main__':

    component = 19
    Pout = 3.45e6
    Uin = 0.0424
    # Pout = 0.1e6
    Tin = 473

    Xi0 = np.array([1] + [0] * (component - 1))
    Yiθ = np.array([1] + [0] * (component - 1))

    # 选择密度计算方法
    density_method = 'RK_PR'  # choose 'RK_PR' or 'PR' or 'SRK' or "BWR" (at low pressure situation like 0.1e6 Pa)

    properties = PhysicalProperties(Tin, Pout, component, density_method)
    rou0 = properties.density(Tin, Pout, component, Xi0)
    rouu0 = Uin * rou0
    Han0 = properties.enthalpy(Tin, Pout, rou0, component, Xi0, Yiθ, Yiθ)
    rouh0 = rou0 * Han0
    rouY10 = rou0 * 1
    
    print(rou0)