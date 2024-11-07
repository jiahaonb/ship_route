# -*- coding: utf-8 -*-
"""
船舶航线规划和调度的两阶段随机规划模型
使用 Gurobi 求解，包含二次约束和非凸约束
"""
from gurobipy import Model, GRB, quicksum
import numpy as np
import  pandas as pd
import sys

# 数据输入地址
csv_pm_path = "均值油价PM.csv"
csv_ph_path = "均值油价PH.csv"
# 数据输出地址
file = open("../py/1106均值为油价，计算teta 路程更改 1.txt", "w", encoding="utf-8")

# 保存原始的sys.stdout
original_stdout = sys.stdout

# 重定向print到文件和控制台
class Tee:
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()

# 使用Tee类来输出到文件和控制台
sys.stdout = Tee(file, original_stdout)


# 港口数量
N = 6  # 示例中的港口数量

# 集合
P = range(1, N + 1)  # 港口集合 [1,2,3,4,5,6]
P_plus = range(1, N + 2)  # 包含最后一个虚拟港口 N+1
Omega = [1]

# 时间窗  是30个
K = [1,2,3,4]

# 定义 P_，用于那些从港口 p 到 p+1 的距离已知的情况
P_ = range(1, N)  # [1,2,3,4,5]

# 参数

# 时间窗口和处理效率
a = {
    (1, 1): 0,   (1, 2): 10,  (1, 3): 18,  (1, 4): 26,
    (2, 1): 10,  (2, 2): 18,  (2, 3): 26,  (2, 4): 35,
    (3, 1): 52,  (3, 2): 62,  (3, 3): 72,  (3, 4): 84,
    (4, 1): 92,  (4, 2): 100,  (4, 3): 108,  (4, 4): 117,
    (5, 1):115 ,  (5, 2): 125, (5, 3): 135, (5, 4): 143,
    (6, 1):125, (6, 2): 136, (6, 3): 146, (6, 4): 156
}

# 时间窗的结束时间
b = {
    (1, 1): 6,   (1, 2): 16,  (1, 3): 24,  (1, 4): 32,
    (2, 1): 16,  (2, 2): 24,  (2, 3): 32,  (2, 4): 41,
    (3, 1): 58,  (3, 2): 68,  (3, 3): 78,  (3, 4): 90,
    (4, 1): 98,  (4, 2): 106,  (4, 3): 114, (4, 4): 125,
    (5, 1): 121, (5, 2): 132, (5, 3): 141, (5, 4): 150,
    (6, 1): 132, (6, 2): 142, (6, 3): 152, (6, 4): 165
}

h = {
    (1, 1): 240, (1, 2): 210, (1, 3): 245, (1, 4): 174,
    (2, 1): 209, (2, 2): 241, (2, 3): 206, (2, 4): 210,
    (3, 1): 159, (3, 2): 237, (3, 3): 157, (3, 4): 235,
    (4, 1): 224, (4, 2): 231, (4, 3): 188, (4, 4): 212,
    (5, 1): 245, (5, 2): 183, (5, 3): 205, (5, 4): 195,
    (6, 1): 170, (6, 2): 186, (6, 3): 239, (6, 4): 209
}

c = {
    (1,1): 201, (1,2): 169, (1,3): 212, (1,4): 215,
    (2,1): 171, (2,2): 159, (2,3): 227, (2,4): 234,
    (3,1): 231, (3,2): 178, (3,3): 214, (3,4): 219,
    (4,1): 235, (4,2): 217, (4,3): 198, (4,4): 197,
    (5,1): 194, (5,2): 248, (5,3): 239, (5,4): 195,
    (6,1): 164, (6,2): 211, (6,3): 163, (6,4): 174
}

# 成本
C0 = 625  # 固定运营成本（$/h）
C1 = 5  # 库存成本（$/TEU·h）
C_late = {1:500, 2:255, 3:260,4:300,5:320,6:290}  # 在港口 p 的迟到惩罚成本（$/h）
F_q = 1000  # 加油的固定成本（$）

# 燃油参数
# 低硫燃油价格（$/吨）
loaded_data = pd.read_csv(csv_pm_path, header=None)
P_M = {
    (col + 1, row + 1): float(loaded_data.iloc[row, col])
    for col in range(loaded_data.shape[1])
    for row in range(loaded_data.shape[0])
}

# 高硫燃油价格（$/吨）
loaded_data = pd.read_csv(csv_ph_path, header=None)
P_H = {
    (col + 1, row + 1): float(loaded_data.iloc[row, col])
    for col in range(loaded_data.shape[1])
    for row in range(loaded_data.shape[0])
}

# 输出测试
# for i in range(100):
#         print(f"{P_M[1, i+1]}, {P_M[2, i+1]}, {P_M[3, i+1]}, {P_M[4, i+1]}, {P_M[5, i+1]}, {P_M[6, i+1]}")
# print(f"下面是PH")
# for i in range(100):
#         print(f"{P_H[1, i+1]}, {P_H[2, i+1]}, {P_H[3, i+1]}, {P_H[4, i+1]}, {P_H[5, i+1]}, {P_H[6, i+1]}")

alpha_M = 3.081
alpha_H = 3.0
F = 1.71  # 燃油消耗系数
v_max = 25  # 最大航速（节）
v_min = 10  # 最小航速（节）

# 距离参数
l_p = {1:90.89, 2:400.95, 3:166.94,4:180.12,5:21.75}  # 从港口 p 到 p+1 的不绕行距离（海里）
d_p = {1:12, 2:12, 3:12,4:12,5:12,6:12}  # 到达ECA边界的垂直距离（海里）
L_p = {1:78.29, 2:364.47, 3:143.09,4:162.53,5:15.66}  # 从港口 p 到 p+1 的不绕行距离（海里

# 货物和库存
W_p1 = {1: 217, 2: 173, 3: 8, 4: 9, 5: 10, 6: 83}
W_p2 = {1: 217, 2: 173, 3: 8, 4: 9, 5: 10, 6: 83}
W_sp = {1: 1000, 2: 900, 3: 1000, 4: 1000, 5: 1000, 6: 1000}


# 燃油容量
L_M = 20  # 低硫燃油最大容量（吨）
L_H = 20  # 高硫燃油最大容量（吨）
S_M = 3  # 低硫燃油最小容量（吨）
S_H = 3  # 高硫燃油最小容量（吨）
B_M1 =3  # 到达港口 1 时的初始低硫燃油量（吨）
B_H1 = 3  # 到达港口 1 时的初始高硫燃油量（吨）

# 大 M 值
M_big = 1e7

# 情景概率
p_omega = {omega: 1 / len(Omega) for omega in Omega}  # 假设等概率

# PHA 参数
# 初始罚参数
rho_t = 10.0  # 对 t_eta 的初始罚参数
rho_z = 10.0  # 对 z 的初始罚参数

# 罚参数的调整步长和上下限
rho_step = 1.0    # 罚参数调整的步长
rho_min = 1.0     # 罚参数的最小值
rho_max = 100.0   # 罚参数的最大值

max_iter = 200  # 最大迭代次数
epsilon_t = 1e-3  # 收敛阈值
epsilon_x = 1e-3  # 收敛阈值

prev_convergence_t = float('inf')
prev_convergence_x = float('inf')

# 初始化协调变量和拉格朗日乘子
t_eta_bar = {p: 0.0 for p in P}
z_bar = {(p, k): 1.0 / len(K) for p in P for k in K}  # 均匀分配初值

# 初始化拉格朗日乘子
mu_t = {omega: {p: 0.0 for p in P} for omega in Omega}
mu_z = {omega: {(p, k): 0.0 for p in P for k in K} for omega in Omega}

# 初始化情景变量的存储
scenario_solutions = {}

# 初始出发时间（港口 1 的 ETA），可以由用户设定
initial_departure_time = 8.0  # 例如，设置为 8:00

# 定义一个非常小的正数 epsilon
epsilon = 1e-6

# 循环开始迭代
for n in range(max_iter):
    print(f"\n--- PHA 迭代 {n+1} ---")
    # 对于每个情景，建立并求解子问题
    scenario_objs = []
    for omega in Omega:
        # 创建 Gurobi 模型 参数如下：
        model = Model(f"TwoStageShipRouting_Scenario_{omega}")
        model.Params.OutputFlag = 0
        model.Params.NonConvex = 2  # 允许非凸二次约束
        model.Params.MIPGap = 0.1
        model.Params.Threads = 28

        # 一阶段变量（针对每个情景）
        t_eta = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_eta")
        # 将 x[p,k] 定义为连续变量
        x = model.addVars(P, K, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
        # 新增二进制变量 z[p,k]
        z = model.addVars(P, K, vtype=GRB.BINARY, name="z")

        # 新增：t_eta_mod 和 n_days
        t_eta_mod = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, ub=168, name="t_eta_mod")
        n_days = model.addVars(P, vtype=GRB.INTEGER, lb=0, name="n_days")

        # 二阶段变量（针对每个情景）
        t_arr = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_arr")
        t_dep = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_dep")
        t_wait = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_wait")
        t_late = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_late")
        t_sail = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_sail")
        t_stay = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_stay")

        # 新增变量
        t_arr_mod = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, ub=168, name="t_arr_mod")
        n_days_arrival = model.addVars(P, vtype=GRB.INTEGER, lb=0, name="n_days_arrival")
        t_port_entry = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_port_entry")
        t_port_entry_mod = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, ub=168, name="t_port_entry_mod")
        # 新的 n_days_port_entry[p,k]
        n_days_port_entry = model.addVars(P, K, vtype=GRB.INTEGER, lb=0, name="n_days_port_entry")

        v_eca = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=v_min, ub=v_max, name="v_eca")
        v_neca = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=v_min, ub=v_max, name="v_neca")
        y = model.addVars(P_, vtype=GRB.BINARY, name="y")  # 保持为二进制变量
        x_b = model.addVars(P, vtype=GRB.BINARY, name="x_b")  # 保持为二进制变量

        Q_M = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="Q_M")
        Q_H = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="Q_H")

        q_M_arr = model.addVars(P_plus, vtype=GRB.CONTINUOUS, lb=0, name="q_M_arr")
        q_H_arr = model.addVars(P_plus, vtype=GRB.CONTINUOUS, lb=0, name="q_H_arr")
        q_M_dep = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="q_M_dep")
        q_H_dep = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="q_H_dep")

        # 燃油消耗变量
        R_M = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name="R_M")
        R_H = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name="R_H")
        z_p_M = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name=f"z_p_M")
        z_p_H = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name=f"z_p_H")

        # 航行距离变量
        l_e = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name="l_e")  # ECA区域内航行距离
        l_ne = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name="l_ne")  # 非ECA区域航行距离
        theta = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0.01, ub=(np.pi / 2) - 0.01, name="theta")  # 绕行角度

        # 辅助变量：cos(theta) 和 tan(theta)
        cos_theta = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="cos_theta")
        tan_theta = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name="tan_theta")
        sin_theta = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="sin_theta")
        cot_theta = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name="cot_theta")

        # 一阶段目标函数（加入罚项和拉格朗日乘子）
        FirstStageCost = quicksum(c[p, k] * z[p, k] * (W_p1[p] + W_p2[p]) for p in P for k in K)

        penalty_t = quicksum(
            rho_t / 2 * (t_eta[p] - t_eta_bar[p])**2 + mu_t[omega][p] * (t_eta[p] - t_eta_bar[p]) for p in P
        )

        penalty_z = quicksum(
            rho_z / 2 * (z[p, k] - z_bar[p, k])**2 + mu_z[omega][p, k] * (z[p, k] - z_bar[p, k]) for p in P for k in K
        )


        # 二阶段目标函数
        SecondStageCost = (
            quicksum(
                C0 * t_sail[p] +
                C1 * W_sp[p] * (t_sail[p] + t_wait[p]) +
                F_q * x_b[p] +
                P_M[p, omega] * Q_M[p] +
                P_H[p, omega] * Q_H[p] +
                C_late[p] * t_late[p]
                for p in P)
        )

        model.setObjective(
            FirstStageCost + SecondStageCost + penalty_t + penalty_z,
            GRB.MINIMIZE
        )


        # 一阶段约束
        # 固定初始出发时间
        model.addConstr(t_eta[1] == initial_departure_time, name="InitialETA")
        # t_eta_mod 和 n_days 的关系
        model.addConstrs((t_eta[p] == 168 * n_days[p] + t_eta_mod[p] for p in P), name="ETA_Modulo")
        model.addConstrs((t_eta_mod[p] >= 0 for p in P), name="ETA_Modulo_NonNegative")
        model.addConstrs((t_eta_mod[p] <= 168 for p in P), name="ETA_Modulo_Max")

        for p in P:
            # 时间窗口权重之和为 1
            model.addConstr(quicksum(x[p, k] for k in K) == 1, name=f"TimeWindowWeight_{p}")
            # 只能选择一个时间窗口
            model.addConstr(quicksum(z[p, k] for k in K) == 1, name=f"TimeWindowSelection_{p}")
            for k in K:
                # x[p,k] 与 z[p,k] 的关系
                model.addConstr(x[p, k] <= z[p, k], name=f"X_Z_UpperBound_{p}_{k}")
                model.addConstr(x[p, k] >= epsilon * z[p, k], name=f"X_Z_LowerBound_{p}_{k}")
                # x[p,k] 的取值范围
                model.addConstr(x[p, k] >= 0, name=f"X_NonNegative_{p}_{k}")
                model.addConstr(x[p, k] <= 1, name=f"X_UpperBound_{p}_{k}")
            # z[p,k] 为二进制变量，已经在变量定义中指定


        # 二阶段约束
        for p in P:
            if p == 1:
                # 港口 1 的实际到达时间等于预计到达时间
                model.addConstr(t_arr[1] == t_eta[1], name=f"ActualArrivalTime_{1}")
                # 等待时间为零
                model.addConstr(t_wait[1] == 0, name=f"WaitingTime_{1}")
                # 迟到时间为零
                model.addConstr(t_late[1] == 0, name=f"LateTime_{1}")
                # 离开时间
                model.addConstr(t_dep[1] == t_arr[1] + t_stay[1], name=f"DepartureTime_{1}")
                # 在港停留时间
                model.addConstr(t_stay[1] == quicksum((W_p1[1] + W_p2[1]) / h[1, k] * x[1, k] for k in K), name=f"StayTime_{1}")
            else:
                # 实际到达时间计算
                model.addConstr(t_arr[p] == t_dep[p - 1] + t_sail[p - 1], name=f"ActualArrivalTime_{p}")

                # t_arr_mod 和 n_days_arrival 的关系
                model.addConstr(t_arr[p] == 168 * n_days_arrival[p] + t_arr_mod[p], name=f"ArrivalTimeModulo_{p}")
                model.addConstr(t_arr_mod[p] >= 0, name=f"ArrivalModNonNegative_{p}")
                model.addConstr(t_arr_mod[p] <= 168, name=f"ArrivalModMax_{p}")

                # 等待时间
                model.addConstr(t_wait[p] >= 0, name=f"WaitingTimeNonNegative_{p}")

                # 进港时间计算
                model.addConstr( t_wait[p] ==t_port_entry[p] - t_arr[p], name=f"PortEntryTime_{p}")

                # 迟到时间
                model.addConstr(t_late[p] >= t_port_entry[p] - t_eta[p], name=f"LateTime_{p}")
                model.addConstr(t_late[p] >= 0, name=f"LateTimeNonNegative_{p}")

                # 离开时间
                model.addConstr(t_dep[p] == t_port_entry[p] + t_stay[p], name=f"DepartureTime_{p}")
                # 在港停留时间
                model.addConstr(t_stay[p] == quicksum((W_p1[p] + W_p2[p]) / h[p, k] * x[p, k] for k in K), name=f"StayTime_{p}")

                # 进港时间约束
                for k in K:
                    # 进港时间必须在选择的时间窗口内

                    model.addConstr(
                        t_port_entry[p] >= 168 * n_days_port_entry[p, k] + a[p, k]*z[p,k] - M_big * (1 - z[p, k]),
                        name=f"PortEntryStartWindow_{p}_{k}"
                    )
                    model.addConstr(
                        t_port_entry[p] <= 168 * n_days_port_entry[p, k] + b[p, k]*z[p,k] + M_big * (1 - z[p, k]),
                        name=f"PortEntryEndWindow_{p}_{k}"
                    )
                    # 预计到达时间必须在选择的时间窗口内
                    model.addConstr(
                        t_eta[p] >= 168 * n_days[p] + a[p, k]*z[p,k] - M_big * (1 - z[p, k]),
                        name=f"tETAStartWindow_{p}_{k}"
                    )
                    model.addConstr(
                        t_eta[p] <= 168 * n_days[p] + b[p, k]*z[p,k] + M_big * (1 - z[p, k]),
                        name=f"tETAEndWindow_{p}_{k}"
                    )
                    # 进港日期不早于到达日期
                    model.addConstr(
                        n_days_port_entry[p, k] >= n_days_arrival[p],
                        name=f"PortEntryDays_{p}_{k}"
                    )
                # 进港时间不早于到达时间
                model.addConstr(t_port_entry[p] >= t_arr[p], name=f"PortEntryAfterArrival_{p}")

        # 航行时间计算（对于 p in P_）
        for p in P_:
            # 定义辅助变量
            t_sail1 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_sail1_{p}")
            t_sail2 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_sail2_{p}")
            t_sail3 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_sail3_{p}")
            # 角度限制
            model.addConstr(theta[p] >= 0.1, name=f"ThetaMin_{p}")
            model.addConstr(theta[p] <= (np.pi / 2) - 0.1, name=f"ThetaMax_{p}")

            # 定义theta的取值点和对应的cos(theta)、tan(theta)值，用于PWL近似
            theta_breakpoints = np.linspace(0.1, (np.pi / 2) - 0.1, 100)
            cos_theta_values = np.cos(theta_breakpoints)
            tan_theta_values = np.tan(theta_breakpoints)
            sin_theta_values = np.sin(theta_breakpoints)
            cot_theta_values = 1 / tan_theta_values

            # 添加PWL约束，定义cos_theta和tan_theta
            model.addGenConstrPWL(theta[p], cos_theta[p],
                                  theta_breakpoints.tolist(), cos_theta_values.tolist(),
                                  name=f"CosTheta_{p}")
            model.addGenConstrPWL(theta[p], tan_theta[p],
                                  theta_breakpoints.tolist(), tan_theta_values.tolist(),
                                  name=f"TanTheta_{p}")
            model.addGenConstrPWL(theta[p], sin_theta[p],
                                  theta_breakpoints.tolist(), sin_theta_values.tolist(),
                                  name=f"SinTheta_{p}")
            model.addGenConstrPWL(theta[p], cot_theta[p],
                                  theta_breakpoints.tolist(), cot_theta_values.tolist(),
                                  name=f"CotTheta_{p}")

            # 航线距离计算
            d_p_total = d_p[p] + d_p.get(p + 1, 0)
            # l_e[p] * sin_theta[p] == d_p_total * y[p]
            model.addConstr(l_e[p] * sin_theta[p] == d_p_total * y[p],
                            name=f"DistanceECA_{p}")
            # l_ne[p] == (L_p[p] - d_p_total * cot_theta[p]) * y[p]
            model.addConstr(l_ne[p] == (L_p[p] - d_p_total * cot_theta[p]) * y[p],
                            name=f"DistanceNECA_{p}")

            # 航行时间与速度的关系
            model.addConstr(t_sail1 * v_eca[p] == l_p[p] * (1 - y[p]), name=f"SailTime1_{p}")
            model.addConstr(t_sail2 * v_eca[p] == l_e[p] * y[p], name=f"SailTime2_{p}")
            model.addConstr(t_sail3 * v_neca[p] == l_ne[p] * y[p], name=f"SailTime3_{p}")

            # 总航行时间
            model.addConstr(t_sail[p] == t_sail1 + t_sail2 + t_sail3, name=f"TotalSailTime_{p}")

            # 定义燃油消耗距离

            # 计算 z_p_M 和 z_p_H # 公式里面的l2e就是这个了
            model.addConstr(z_p_M[p] == l_p[p] * (1 - y[p]) + l_e[p] * y[p], name=f"Z_p_M_{p}")
            model.addConstr(z_p_H[p] == l_ne[p] * y[p], name=f"Z_p_H_{p}")

            F = 0.013 / 24  # 燃油消耗系数
            # 定义速度平方的辅助变量
            v_eca_sq = model.addVars(P_, vtype=GRB.CONTINUOUS, name="v_eca_sq")
            v_neca_sq = model.addVars(P_, vtype=GRB.CONTINUOUS, name="v_neca_sq")

            # 添加辅助变量与原始变量之间的关系
            model.addConstrs((v_eca_sq[p] == v_eca[p] * v_eca[p] for p in P_), name="v_eca_squared")
            model.addConstrs((v_neca_sq[p] == v_neca[p] * v_neca[p] for p in P_), name="v_neca_squared")

            # 重写燃油消耗的约束
            model.addConstrs((R_M[p] == F * z_p_M[p] * v_eca_sq[p] for p in P_), name="FuelConsumptionM")
            model.addConstrs((R_H[p] == F * z_p_H[p] * v_neca_sq[p] for p in P_), name="FuelConsumptionH")

        # 燃油库存平衡
        for p in P:
            if p == 1:
                model.addConstr(q_M_arr[p] == B_M1, name=f"FuelInventoryArrivalM_{p}")
                model.addConstr(q_H_arr[p] == B_H1, name=f"FuelInventoryArrivalH_{p}")
            else:
                model.addConstr(q_M_arr[p] == q_M_dep[p - 1]- R_M[p-1], name=f"FuelInventoryArrivalM_{p}")
                model.addConstr(q_H_arr[p] == q_H_dep[p - 1]- R_H[p-1], name=f"FuelInventoryArrivalH_{p}")
            # 燃油消耗在 P_ 中定义，需区分 p 是否在 P_
            if p in P_:
                model.addConstr(q_M_dep[p] == q_M_arr[p] + x_b[p] * Q_M[p], name=f"FuelInventoryDepartureM_{p}")
                model.addConstr(q_H_dep[p] == q_H_arr[p] + x_b[p] * Q_H[p], name=f"FuelInventoryDepartureH_{p}")
            else:
                # 对于 p = N，需要调整燃油消耗
                model.addConstr(q_M_dep[p] == q_M_arr[p] + x_b[p] * Q_M[p], name=f"FuelInventoryDepartureM_{p}")
                model.addConstr(q_H_dep[p] == q_H_arr[p] + x_b[p] * Q_H[p], name=f"FuelInventoryDepartureH_{p}")
            # 燃油容量限制
            model.addConstr(q_M_dep[p] >= S_M, name=f"FuelCapacityM_Min_{p}")
            model.addConstr(q_M_dep[p] <= L_M, name=f"FuelCapacityM_Max_{p}")
            model.addConstr(q_H_dep[p] >= S_H, name=f"FuelCapacityH_Min_{p}")
            model.addConstr(q_H_dep[p] <= L_H, name=f"FuelCapacityH_Max_{p}")

        # 最后一个虚拟港口的燃油库存限制
        model.addConstr(q_M_arr[N + 1] >= S_M, name=f"FinalFuelInventoryM")
        model.addConstr(q_H_arr[N + 1] >= S_H, name=f"FinalFuelInventoryH")

        model.Params.IntFeasTol = 1e-8  # 将整数容差调整为更小的值


        # def my_callback(model, where):
        #     if where == GRB.Callback.MIPSOL:
        #         # 输出当前的 z[p,k] 变量值
        #         print("\n当前的解：")
        #         for p in P_:
        #             theta_value = theta[p].X
        #             sin = sin_theta_values[p].X
        #             cos = cos_theta_values[p].X
        #             cot = cot_theta_values[p].X
        #             print(f"港口 {p} 最优theta值: {theta_value:.4f} radians")
        #             print(
        #                 f"sin(theta) = {sin:.4f}, cos(theta) = {cos:.4f}, cot(theta) = {cot:.4f}")


        # 求解模型
        model.optimize()

        print("\n当前的解：")
        for p in P_:
            theta_value = theta[p].X
            sin = sin_theta[p].X
            cos = cos_theta[p].X
            cot = cot_theta[p].X
            print(f"港口 {p} 最优theta值: {theta_value:.4f} radians")
            print(
                f"sin(theta) = {sin:.4f}, cos(theta) = {cos:.4f}, cot(theta) = {cot:.4f}")


        if model.status == GRB.OPTIMAL:
            # 存储情景解
            scenario_solutions[omega] = {
                't_eta': {p: t_eta[p].X for p in P},
                't_eta_mod': {p: t_eta_mod[p].X for p in P},
                'n_days': {p: n_days[p].X for p in P},
                'x': {(p, k): x[p, k].X for p in P for k in K},
                'z': {(p, k): z[p, k].X for p in P for k in K},
                'mu_t': mu_t[omega].copy(),
                'mu_z': mu_z[omega].copy(),
                'obj': model.ObjVal,  # 这个就是set_O
                # 存储二阶段变量以便输出
                't_arr': {p: t_arr[p].X for p in P},
                't_arr_mod': {p: t_arr_mod[p].X if p != 1 else t_eta_mod[p].X for p in P},
                'n_days_arrival': {p: n_days_arrival[p].X if p != 1 else n_days[p].X for p in P},
                't_port_entry': {p: t_port_entry[p].X for p in P},
                't_wait': {p: t_wait[p].X for p in P},
                't_late': {p: t_late[p].X for p in P},
                't_dep': {p: t_dep[p].X for p in P},
                't_sail': {p: t_sail[p].X if p in P_ else 0.0 for p in P},
                't_stay': {p: t_stay[p].X for p in P},
                'v_eca': {p: v_eca[p].X if p in P_ else None for p in P},
                'v_neca': {p: v_neca[p].X if p in P_ else None for p in P},
                'l_ne': {p: l_ne[p].X if p in P_ else 0.0 for p in P},
                'l_e': {p: l_e[p].X if p in P_ else 0.0 for p in P},
                'y': {p: y[p].X if p in P_ else None for p in P},
                'x_b': {p: x_b[p].X for p in P},
                'Q_M': {p: Q_M[p].X for p in P},
                'Q_H': {p: Q_H[p].X for p in P},
                'q_M_arr': {p: q_M_arr[p].X for p in P_plus},
                'q_H_arr': {p: q_H_arr[p].X for p in P_plus},
                'q_M_dep': {p: q_M_dep[p].X for p in P},
                'q_H_dep': {p: q_H_dep[p].X for p in P},
                'R_M': {p: R_M[p].X if p in P_ else 0.0 for p in P},
                'R_H': {p: R_H[p].X if p in P_ else 0.0 for p in P},
                'z_p_M': {p: z_p_M[p].X if p in P_ else 0.0 for p in P},
                'z_p_H': {p: z_p_H[p].X if p in P_ else 0.0 for p in P},

            }
            scenario_objs.append(model.ObjVal)

            # 输出每次迭代的结果
            print(f"\n情景 {omega} 的结果：")
            for p in P:
                print(f"港口 {p}:")
                # 加油策略
                refuel_strategy = '加油' if scenario_solutions[omega]['x_b'][p] > 0.5 else '不加油'
                print(f"  加油策略: {refuel_strategy}")
                print(f"    低硫燃油消耗: {scenario_solutions[omega]['R_M'][p]:.6f}")
                print(f"    高硫燃油消耗: {scenario_solutions[omega]['R_H'][p]:.6f}")
                print(f"    低硫路程: {scenario_solutions[omega]['z_p_M'][p]:.6f}")
                print(f"    高硫路程: {scenario_solutions[omega]['z_p_H'][p]:.6f}")
                print(f"    低硫: {scenario_solutions[omega]['l_e'][p]:.6f}")
                print(f"    高硫: {scenario_solutions[omega]['l_ne'][p]:.6f}")
                # 绕行策略（仅对 p in P_）
                if p in P_:
                    detour_strategy = '绕行' if scenario_solutions[omega]['y'][p] > 0.5 else '直接航线'
                    print(f"  从港口 {p} 到港口 {p + 1} 的绕行策略: {detour_strategy}")
                    # 最优速度
                    print(f"  从港口 {p} 到港口 {p + 1} 的 ECA 航速: {scenario_solutions[omega]['v_eca'][p]:.6f} 节")
                    print(f"  从港口 {p} 到港口 {p + 1} 的非 ECA 航速: {scenario_solutions[omega]['v_neca'][p]:.6f} 节")
                else:
                    print(f"  没有后续航段。")
                # 加油量
                if scenario_solutions[omega]['x_b'][p] > 0.5:
                    print(f"  加油量（低硫燃油）: {scenario_solutions[omega]['Q_M'][p]:.6f} 吨")
                    print(f"  加油量（高硫燃油）: {scenario_solutions[omega]['Q_H'][p]:.6f} 吨")
                else:
                    print(f"  加油量（低硫燃油）: 0.00 吨")
                    print(f"  加油量（高硫燃油）: 0.00 吨")
                # ETA
                print(f"  ETA（预计到达时间）: {scenario_solutions[omega]['t_eta'][p]:.6f} 小时")
                # 实际到达时间
                print(f"  实际到达时间: {scenario_solutions[omega]['t_arr'][p]:.6f} 小时")
                # 选择的时间窗
                print(f"  {scenario_solutions[omega]['z'][p, 1]}, {scenario_solutions[omega]['z'][p, 2]}, {scenario_solutions[omega]['z'][p, 3]}, {scenario_solutions[omega]['z'][p, 4]}")
                # 进港时间
                print(f"  进港时间: {scenario_solutions[omega]['t_port_entry'][p]:.6f} 小时")
                # 等待时间
                print(f"  等待时间: {scenario_solutions[omega]['t_wait'][p]:.6f} 小时")

                # 时间窗选择
                selected_window = max(scenario_solutions[omega]['z'][p, k] for k in K)
                for k in K:
                    if scenario_solutions[omega]['z'][p, k] == selected_window:
                        print(f"  选择的时间窗口: {k}")
                        break
                # 到达时间是当日具体时间
                if p ==1:
                    print(f"  当日到达时间（15d）: {scenario_solutions[omega]['t_eta_mod'][p]:.6f} 小时")
                else:
                    print(f"  当日到达时间（15d）: {scenario_solutions[omega]['t_arr_mod'][p]:.6f} 小时")
        else:
            print(f"情景 {omega} 求解未找到最优解。")
            break

    # 计算完成所有的情景，开始迭代处理
    if n == 0:
        # 更新协调变量
        for p in P:
            t_eta_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['t_eta'][p] for omega in Omega)
        for p in P:
            for k in K:
                z_bar[p, k] = sum(p_omega[omega] * scenario_solutions[omega]['z'][p, k] for omega in Omega)

        # 更新拉格朗日乘子
        for omega in Omega:
            for p in P:
                mu_t[omega][p] += rho_t * (scenario_solutions[omega]['t_eta'][p] - t_eta_bar[p])
            for p in P:
                for k in K:
                    mu_z[omega][p, k] += rho_z * (scenario_solutions[omega]['z'][p, k] - z_bar[p, k])

    else:
        # 在更新协调变量之前计算收敛值
        convergence_t = sum(
            (scenario_solutions[omega]['t_eta'][p] - t_eta_bar[p])**2 for omega in Omega for p in P
        )
        convergence_x = sum(
            (scenario_solutions[omega]['z'][p, k] - z_bar[p, k])**2 for omega in Omega for p in P for k in K
        )

        print(f"\nt的收敛值：{convergence_t}")
        print(f"x的收敛值：{convergence_x}")

        # 动态调整罚参数
        # 如果收敛值大于前一轮，则增大罚参数
        if convergence_t > prev_convergence_t:
            rho_t = min(rho_t + rho_step, rho_max)
            print(f"增大 rho_t 到 {rho_t}")
        elif convergence_t < prev_convergence_t:
            rho_t = max(rho_t - rho_step, rho_min)
            print(f"减小 rho_t 到 {rho_t}")

        if convergence_x > prev_convergence_x:
            rho_z = min(rho_z + rho_step, rho_max)
            print(f"增大 rho_z 到 {rho_z}")
        elif convergence_x < prev_convergence_x:
            rho_z = max(rho_z - rho_step, rho_min)
            print(f"减小 rho_z 到 {rho_z}")

        # 更新前一轮的收敛值
        prev_convergence_t = convergence_t
        prev_convergence_x = convergence_x

        # 检查收敛条件
        if convergence_t < epsilon_t and convergence_x < epsilon_x:
            print("收敛条件满足，停止迭代。")
            break






    # 更新协调变量（在计算收敛值之后）
    for p in P:
        t_eta_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['t_eta'][p] for omega in Omega)
    for p in P:
        for k in K:
            z_bar[p, k] = sum(p_omega[omega] * scenario_solutions[omega]['z'][p, k] for omega in Omega)

    # 更新拉格朗日乘子
    for omega in Omega:
        for p in P:
            mu_t[omega][p] += rho_t * (scenario_solutions[omega]['t_eta'][p] - t_eta_bar[p])
        for p in P:
            for k in K:
                mu_z[omega][p, k] += rho_z * (scenario_solutions[omega]['z'][p, k] - z_bar[p, k])


        # 继续下一次迭代

# 输出最终结果
print("\n最终结果：")
# 输出最终协调后的一阶段决策变量
print("协调后的一阶段决策变量：")
for p in P:
    print(f"  港口 {p}:")
    print(f"    ETA（预计到达时间）: {t_eta_bar[p]:.2f}")
    for k in K:
        print(f"    时间窗口 {k}: 权重 = {z_bar[p, k]:.2f}")

# 输出每个情景的解
for omega in Omega:
    print(f"\n情景 {omega} 的一阶段决策变量：")
    for p in P:
        print(f"  港口 {p}:")
        print(f"    ETA（预计到达时间）: {scenario_solutions[omega]['t_eta'][p]:.2f}")
        for k in K:
            print(f"    时间窗口 {k}: 权重 = {scenario_solutions[omega]['x'][p, k]:.2f}")
            print(f"    时间窗口 {k}: 选择 = {scenario_solutions[omega]['z'][p, k]:.0f}")
        print(f"    加油量（低硫燃油）: {scenario_solutions[omega]['Q_M'][p]:.2f}")
        print(f"    加油量（高硫燃油）: {scenario_solutions[omega]['Q_H'][p]:.2f}")
        if p in P_:
            print(f"    从港口 {p} 到港口 {p+1} 的 ECA 航速: {scenario_solutions[omega]['v_eca'][p]:.2f}")
            print(f"    从港口 {p} 到港口 {p+1} 的非 ECA 航速: {scenario_solutions[omega]['v_neca'][p]:.2f}")
            # 绕行策略和加油策略
            print(f"    绕行策略 (y): {scenario_solutions[omega]['y'][p]}")
        else:
            print(f"    没有后续航段。")
        print(f"    加油策略 (x_b): {scenario_solutions[omega]['x_b'][p]}")
    print("\n二阶段决策变量：")
    for p in P:
        print(f"  港口 {p}:")
        print(f"    实际到达时间: {scenario_solutions[omega]['t_arr'][p]:.2f}")
        print(f"    预计到达时间: {scenario_solutions[omega]['t_eta'][p]:.2f}")
        print(f"    进港时间: {scenario_solutions[omega]['t_port_entry'][p]:.2f}")
        print(f"    离开时间: {scenario_solutions[omega]['t_dep'][p]:.2f}")
        print(f"    等待时间: {scenario_solutions[omega]['t_wait'][p]:.2f}")
        print(f"    迟到时间: {scenario_solutions[omega]['t_late'][p]:.2f}")
        print(f"    航行时间: {scenario_solutions[omega]['t_sail'][p]:.2f}")
        print(f"    停留时间: {scenario_solutions[omega]['t_stay'][p]:.2f}")
        if p in P_:
            print(f"    从港口 {p} 到港口 {p+1} 的 ECA 航速: {scenario_solutions[omega]['v_eca'][p]:.2f}")
            print(f"    从港口 {p} 到港口 {p+1} 的非 ECA 航速: {scenario_solutions[omega]['v_neca'][p]:.2f}")
            print(f"    绕行决策 (y): {scenario_solutions[omega]['y'][p]}")
            # 绕行和加油策略输出
            detour_strategy = '绕行' if scenario_solutions[omega]['y'][p] > 0.5 else '直接航线'
            print(f"    绕行策略: {detour_strategy}")
        else:
            print(f"    没有后续航段。")
        print(f"    加油决策 (x_b): {scenario_solutions[omega]['x_b'][p]}")
        refuel_strategy = '加油' if scenario_solutions[omega]['x_b'][p] > 0.5 else '不加油'
        print(f"    加油策略: {refuel_strategy}")
        if scenario_solutions[omega]['x_b'][p] > 0.5:
            print(f"      补充的低硫燃油量: {scenario_solutions[omega]['Q_M'][p]:.2f}")
            print(f"      补充的高硫燃油量: {scenario_solutions[omega]['Q_H'][p]:.2f}")
        print(f"    到达时的低硫燃油量: {scenario_solutions[omega]['q_M_arr'][p]:.2f}")
        print(f"    到达时的高硫燃油量: {scenario_solutions[omega]['q_H_arr'][p]:.2f}")
        print(f"    离开时的低硫燃油量: {scenario_solutions[omega]['q_M_dep'][p]:.2f}")
        print(f"    离开时的高硫燃油量: {scenario_solutions[omega]['q_H_dep'][p]:.2f}")
        print(f"    低硫燃油消耗: {scenario_solutions[omega]['R_M'][p]:.2f}")
        print(f"    高硫燃油消耗: {scenario_solutions[omega]['R_H'][p]:.2f}")
        # 输出当日具体到达时间
        if p ==1:
            print(f"    当日到达时间（15d）: {scenario_solutions[omega]['t_eta_mod'][p]:.2f} 小时")
        else:
            print(f"    当日到达时间（15d）: {scenario_solutions[omega]['t_arr_mod'][p]:.2f} 小时")
    print(f"  在港口 {N + 1} 的最终低硫燃油量: {scenario_solutions[omega]['q_M_arr'][N+1]:.2f}")
    print(f"  在港口 {N + 1} 的最终高硫燃油量: {scenario_solutions[omega]['q_H_arr'][N+1]:.2f}")

    print(f"\n情景 {omega} 的目标函数值：{scenario_solutions[omega]['obj']}")

