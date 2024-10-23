# -*- coding: utf-8 -*-
"""
船舶航线规划和调度的两阶段随机规划模型
使用 Gurobi 求解，包含二次约束和非凸约束
"""
from gurobipy import Model, GRB, quicksum
import numpy as np

# 数据输入

# 港口数量
N = 6  # 示例中的港口数量

# 集合
P = range(1, N + 1)  # 港口集合
P_plus = range(1, N + 2)  # 包含最后一个虚拟港口 N+1
Omega = [1, 2,3,4,5,6,7]  # 情景集合
K = [1, 2, 3, 4]  # 每个港口的时间窗口集合

# 参数

# 时间窗口和处理效率

a = {
    (1,1):0, (1,2):6, (1,3):12, (1,4):18,
    (2,1):0, (2,2):6, (2,3):12, (2,4):18,
    (3,1):0, (3,2):6, (3,3):12, (3,4):18,
    (4,1):0, (4,2):6, (4,3):12, (4,4):18,
    (5,1):0, (5,2):6, (5,3):12, (5,4):18,
    (6,1):0, (6,2):6, (6,3):12, (6,4):18
}

b = {
    (1,1):6, (1,2):12, (1,3):18, (1,4):24,
    (2,1):6, (2,2):12, (2,3):18, (2,4):24,
    (3,1):6, (3,2):12, (3,3):18, (3,4):24,
    (4,1):6, (4,2):12, (4,3):18, (4,4):24,
    (5,1):6, (5,2):12, (5,3):18, (5,4):24,
    (6,1):6, (6,2):12, (6,3):18, (6,4):24
}

h = {
    (1,1):100, (1,2):120, (1,3):140, (1,4):160,
    (2,1):110, (2,2):130, (2,3):150, (2,4):170,
    (3,1):105, (3,2):125, (3,3):145, (3,4):165,
    (4,1):115, (4,2):135, (4,3):155, (4,4):175,
    (5,1):120, (5,2):140, (5,3):160, (5,4):180,
    (6,1):125, (6,2):145, (6,3):165, (6,4):185
}

c = {
    (1,1):50, (1,2):60, (1,3):70, (1,4):80,
    (2,1):55, (2,2):65, (2,3):75, (2,4):85,
    (3,1):53, (3,2):63, (3,3):73, (3,4):83,
    (4,1):58, (4,2):68, (4,3):78, (4,4):88,
    (5,1):60, (5,2):70, (5,3):80, (5,4):90,
    (6,1):62, (6,2):72, (6,3):82, (6,4):92
}

# 成本
C0 = 625  # 固定运营成本（$/h）
C1 = 0.5  # 库存成本（$/TEU·h）
C_late = {1:500, 2:2550, 3:2600,4:3000,5:3200,6:2900}  # 在港口 p 的迟到惩罚成本（$/h）
F_q = 1000  # 加油的固定成本（$）

# 燃油参数
# 低硫燃油价格（$/吨）
P_M = {
    (1, 1): 60, (1, 2): 2620, (1, 3): 70, (1, 4): 80, (1, 5): 90, (1, 6): 100, (1, 7): 110,
    (2, 1): 61, (2, 2): 6630, (2, 3): 72, (2, 4): 82, (2, 5): 92, (2, 6): 102, (2, 7): 112,
    (3, 1): 65, (3, 2): 3, (3, 3): 75, (3, 4): 85, (3, 5): 95, (3, 6): 105, (3, 7): 115,
    (4, 1): 66, (4, 2): 3, (4, 3): 76, (4, 4): 86, (4, 5): 96, (4, 6): 106, (4, 7): 116,
    (5, 1): 67, (5, 2): 3, (5, 3): 77, (5, 4): 87, (5, 5): 97, (5, 6): 107, (5, 7): 117,
    (6, 1): 68, (6, 2): 3, (6, 3): 78, (6, 4): 88, (6, 5): 98, (6, 6): 108, (6, 7): 118
}

# 高硫燃油价格（$/吨）
P_H = {
    (1, 1): 40, (1, 2): 4420, (1, 3): 45, (1, 4): 50, (1, 5): 55, (1, 6): 60, (1, 7): 65,
    (2, 1): 410, (2, 2): 4430, (2, 3): 415, (2, 4): 420, (2, 5): 425, (2, 6): 430, (2, 7): 435,
    (3, 1): 405, (3, 2): 2, (3, 3): 410, (3, 4): 415, (3, 5): 420, (3, 6): 425, (3, 7): 430,
    (4, 1): 406, (4, 2): 2, (4, 3): 411, (4, 4): 416, (4, 5): 421, (4, 6): 426, (4, 7): 431,
    (5, 1): 407, (5, 2): 2, (5, 3): 412, (5, 4): 417, (5, 5): 422, (5, 6): 427, (5, 7): 432,
    (6, 1): 408, (6, 2): 2, (6, 3): 413, (6, 4): 418, (6, 5): 423, (6, 6): 428, (6, 7): 433
}

alpha_M = 3.081
alpha_H = 3.0
F = 1.71  # 燃油消耗系数
v_max = 25  # 最大航速（节）
v_min = 15  # 最小航速（节）

# 距离参数
l_p = {1:193.89, 2:448.95, 3:186.94,4:221.12,5:28.75}  # 从港口 p 到 p+1 的直线距离（海里）
d_p = {1:20, 2:25, 3:24,4:3,5:12,6:15}  # 到达ECA边界的垂直距离（海里）

# 货物和库存
W_p1 = {1:200, 2:250, 3:240,4:330,5:320,6:290}  # 在港口 p 卸载的货物量（TEU）
W_p2 = {1:200, 2:250, 3:240,4:320,5:310,6:290}  # 在港口 p 装载的货物量（TEU）
W_sp = {1:1000, 2:1050, 3:1100,4:1200,5:1300,6:1100}  # 在港口 p 和 p+1 之间的库存量（TEU）

# 燃油容量
L_M = 5000  # 低硫燃油最大容量（吨）
L_H = 5000  # 高硫燃油最大容量（吨）
S_M = 500  # 低硫燃油最小容量（吨），设置为 0 以便测试
S_H = 500  # 高硫燃油最小容量（吨），设置为 0 以便测试
B_M1 = 600  # 到达港口 1 时的初始低硫燃油量（吨）
B_H1 = 500  # 到达港口 1 时的初始高硫燃油量（吨）

# 大 M 值
M_big = 1e6

# 情景概率
p_omega = {omega: 1 / len(Omega) for omega in Omega}  # 假设等概率

# PHA 参数
rho_t = 10.0  # 对 t_eta 的罚参数
rho_x = 10.0  # 对 x 的罚参数
rho_Q_M = 10.0  # 对 Q_M 的罚参数
rho_Q_H = 10.0  # 对 Q_H 的罚参数
rho_v_eca = 10.0  # 对 v_eca 的罚参数
rho_v_neca = 10.0  # 对 v_neca 的罚参数
rho_y = 10.0  # 对 y 的罚参数
rho_x_b = 10.0  # 对 x_b 的罚参数
max_iter = 500  # 最大迭代次数
epsilon = 1e-5  # 收敛阈值

# 初始化协调变量和拉格朗日乘子
t_eta_bar = {p: 0.0 for p in P}
x_bar = {(p, k): 1.0 / len(K) for p in P for k in K}  # 均匀分配初值

# 新增协调变量
Q_M_bar = {p: 0.0 for p in P}
Q_H_bar = {p: 0.0 for p in P}
v_eca_bar = {p: (v_min + v_max) / 2 for p in P}
v_neca_bar = {p: (v_min + v_max) / 2 for p in P}
y_bar = {p: 0.0 for p in P}  # 初始化为 0
x_b_bar = {p: 0.0 for p in P}  # 初始化为 0

# 初始化拉格朗日乘子
mu_t = {omega: {p: 0.0 for p in P} for omega in Omega}
mu_x = {omega: {(p, k): 0.0 for p in P for k in K} for omega in Omega}

# 新增拉格朗日乘子
mu_Q_M = {omega: {p: 0.0 for p in P} for omega in Omega}
mu_Q_H = {omega: {p: 0.0 for p in P} for omega in Omega}
mu_v_eca = {omega: {p: 0.0 for p in P} for omega in Omega}
mu_v_neca = {omega: {p: 0.0 for p in P} for omega in Omega}
mu_y = {omega: {p: 0.0 for p in P} for omega in Omega}
mu_x_b = {omega: {p: 0.0 for p in P} for omega in Omega}

# 初始化情景变量的存储
scenario_solutions = {}

for n in range(max_iter):
    print(f"\n--- PHA 迭代 {n+1} ---")
    # 对于每个情景，建立并求解子问题
    scenario_objs = []
    for omega in Omega:
        # 创建 Gurobi 模型
        model = Model(f"TwoStageShipRouting_Scenario_{omega}")
        model.Params.OutputFlag = 0
        model.Params.NonConvex = 2  # 允许非凸二次约束

        # 一阶段变量（针对每个情景）
        t_eta = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_eta")
        # 将 x[p,k] 放松为连续变量
        x = model.addVars(P, K, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")

        # 二阶段变量（针对每个情景）
        t_arr = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_arr")
        t_dep = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_dep")
        t_wait = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_wait")
        t_late = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_late")
        t_sail = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_sail")
        t_stay = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_stay")

        v_eca = model.addVars(P, vtype=GRB.CONTINUOUS, lb=v_min, ub=v_max, name="v_eca")
        v_neca = model.addVars(P, vtype=GRB.CONTINUOUS, lb=v_min, ub=v_max, name="v_neca")
        y = model.addVars(P, vtype=GRB.BINARY, name="y")  # 保持为二进制变量
        x_b = model.addVars(P, vtype=GRB.BINARY, name="x_b")  # 保持为二进制变量

        Q_M = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="Q_M")
        Q_H = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="Q_H")

        q_M_arr = model.addVars(P_plus, vtype=GRB.CONTINUOUS, lb=0, name="q_M_arr")
        q_H_arr = model.addVars(P_plus, vtype=GRB.CONTINUOUS, lb=0, name="q_H_arr")
        q_M_dep = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="q_M_dep")
        q_H_dep = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="q_H_dep")

        # 燃油消耗变量
        R_M = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="R_M")
        R_H = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="R_H")

        # 航行距离变量
        l_e: None = model.addVars(P, Omega, vtype=GRB.CONTINUOUS, lb=0, name="l_e")  # ECA区域内航行距离
        l_ne = model.addVars(P, Omega, vtype=GRB.CONTINUOUS, lb=0, name="l_ne")  # 非ECA区域航行距离
        theta = model.addVars(P, Omega, vtype=GRB.CONTINUOUS, lb=0.01, ub=(np.pi / 2) - 0.01, name="theta")  # 绕行角度

        # 辅助变量：cos(theta) 和 tan(theta)
        cos_theta = model.addVars(P, Omega, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="cos_theta")
        tan_theta = model.addVars(P, Omega, vtype=GRB.CONTINUOUS, lb=0, name="tan_theta")

        # 一阶段目标函数（加入罚项和拉格朗日乘子）
        FirstStageCost = quicksum(c[p, k] * x[p, k] * (W_p1[p] + W_p2[p]) for p in P for k in K)

        penalty_t = quicksum(
            rho_t / 2 * (t_eta[p] - t_eta_bar[p])**2 + mu_t[omega][p] * (t_eta[p] - t_eta_bar[p]) for p in P
        )

        penalty_x = quicksum(
            rho_x / 2 * (x[p, k] - x_bar[p, k])**2 + mu_x[omega][p, k] * (x[p, k] - x_bar[p, k]) for p in P for k in K
        )

        # 新增罚项
        penalty_Q_M = quicksum(
            rho_Q_M / 2 * (Q_M[p] - Q_M_bar[p])**2 + mu_Q_M[omega][p] * (Q_M[p] - Q_M_bar[p]) for p in P
        )

        penalty_Q_H = quicksum(
            rho_Q_H / 2 * (Q_H[p] - Q_H_bar[p])**2 + mu_Q_H[omega][p] * (Q_H[p] - Q_H_bar[p]) for p in P
        )

        penalty_v_eca = quicksum(
            rho_v_eca / 2 * (v_eca[p] - v_eca_bar[p])**2 + mu_v_eca[omega][p] * (v_eca[p] - v_eca_bar[p]) for p in P
        )

        penalty_v_neca = quicksum(
            rho_v_neca / 2 * (v_neca[p] - v_neca_bar[p])**2 + mu_v_neca[omega][p] * (v_neca[p] - v_neca_bar[p]) for p in P
        )

        penalty_y = quicksum(
            rho_y / 2 * (y[p] - y_bar[p])**2 + mu_y[omega][p] * (y[p] - y_bar[p]) for p in P
        )

        penalty_x_b = quicksum(
            rho_x_b / 2 * (x_b[p] - x_b_bar[p])**2 + mu_x_b[omega][p] * (x_b[p] - x_b_bar[p]) for p in P
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

        # 总目标函数
        model.setObjective(
            FirstStageCost + SecondStageCost + penalty_t + penalty_x + penalty_Q_M + penalty_Q_H +
            penalty_v_eca + penalty_v_neca + penalty_y + penalty_x_b,
            GRB.MINIMIZE
        )

        # 一阶段约束
        for p in P:
            # 时间窗口选择
            model.addConstr(quicksum(x[p, k] for k in K) == 1, name=f"TimeWindowSelection_{p}")
            for k in K:
                # 预计到达时间在选择的时间窗口内
                model.addConstr(t_eta[p] >= a[p, k] - M_big * (1 - x[p, k]), name=f"ETAStartWindow_{p}_{k}")
                model.addConstr(t_eta[p] <= b[p, k] + M_big * (1 - x[p, k]), name=f"ETAEndWindow_{p}_{k}")

        # 二阶段约束
        for p in P:
            # 实际到达时间计算
            if p == 1:
                model.addConstr(t_arr[p] == t_sail[p], name=f"ActualArrivalTime_{p}")
            else:
                model.addConstr(t_arr[p] == t_dep[p - 1] + t_sail[p], name=f"ActualArrivalTime_{p}")
            # 等待时间和迟到时间
            model.addConstr(t_wait[p] >= t_eta[p] - t_arr[p], name=f"WaitTime_{p}")
            model.addConstr(t_late[p] >= t_arr[p] - t_eta[p], name=f"LateTime_{p}")
            # 离开时间
            model.addConstr(t_dep[p] == t_arr[p] + t_wait[p] + t_stay[p], name=f"DepartureTime_{p}")
            # 在港停留时间
            model.addConstr(t_stay[p] == quicksum((W_p1[p] + W_p2[p]) / h[p, k] * x[p, k] for k in K), name=f"StayTime_{p}")

            # 航行时间计算
            # 定义辅助变量
            t_sail1 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_sail1_{p}")
            t_sail2 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_sail2_{p}")
            t_sail3 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_sail3_{p}")
            # 角度限制
            model.addConstr(theta[p, omega] >= 0.01, name=f"ThetaMin_{p}_{omega}")
            model.addConstr(theta[p, omega] <= (np.pi / 2) - 0.01, name=f"ThetaMax_{p}_{omega}")

            # 定义theta的取值点和对应的cos(theta)、tan(theta)值，用于PWL近似
            theta_breakpoints = np.linspace(0.01, (np.pi / 2) - 0.01, 100)
            cos_theta_values = np.cos(theta_breakpoints)
            tan_theta_values = np.tan(theta_breakpoints)

            # 添加PWL约束，定义cos_theta和tan_theta
            model.addGenConstrPWL(theta[p, omega], cos_theta[p, omega],
                                  theta_breakpoints.tolist(), cos_theta_values.tolist(),
                                  name=f"CosTheta_{p}_{omega}")
            model.addGenConstrPWL(theta[p, omega], tan_theta[p, omega],
                                  theta_breakpoints.tolist(), tan_theta_values.tolist(),
                                  name=f"TanTheta_{p}_{omega}")

            # 航线距离计算
            d_p_total = d_p[p] + d_p.get(p + 1, 0)
            # l_e[p, omega] * cos_theta[p, omega] == d_p_total * y[p, omega]
            model.addConstr(l_e[p, omega] * cos_theta[p, omega] == d_p_total * y[p, omega],
                            name=f"DistanceECA_{p}_{omega}")
            # l_ne[p, omega] == (l_p[p] - d_p_total * tan_theta[p, omega]) * y[p, omega] + l_p[p] * (1 - y[p, omega])
            model.addConstr(l_ne[p, omega] == (l_p[p] - d_p_total * tan_theta[p, omega]) * y[p, omega],
                            name=f"DistanceNECA_{p}_{omega}")

            # 航行时间与速度的关系
            model.addConstr(t_sail1 * v_eca[p] == l_p[p] * (1 - y[p]), name=f"SailTime1_{p}")
            model.addConstr(t_sail2 * v_eca[p] == l_e[p] * y[p], name=f"SailTime2_{p}")
            model.addConstr(t_sail3 * v_neca[p] == l_ne[p] * y[p], name=f"SailTime3_{p}")

            # 总航行时间
            model.addConstr(t_sail[p] == t_sail1 + t_sail2 + t_sail3, name=f"TotalSailTime_{p}")

            # 定义速度比和幂次约束
            v_ratio_M = model.addVar(vtype=GRB.CONTINUOUS, lb=v_min / v_max, ub=1, name=f"v_ratio_M_{p}")
            v_ratio_H = model.addVar(vtype=GRB.CONTINUOUS, lb=v_min / v_max, ub=1, name=f"v_ratio_H_{p}")
            model.addConstr(v_ratio_M * v_max == v_eca[p], name=f"VRatioM_{p}")
            model.addConstr(v_ratio_H * v_max == v_neca[p], name=f"VRatioH_{p}")

            # 定义速度比的幂次变量
            v_ratio_M_alpha = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"v_ratio_M_alpha_{p}")
            v_ratio_H_alpha = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"v_ratio_H_alpha_{p}")
            model.addGenConstrPow(v_ratio_M, v_ratio_M_alpha, alpha_M, name=f"VRatioAlphaM_{p}")
            model.addGenConstrPow(v_ratio_H, v_ratio_H_alpha, alpha_H, name=f"VRatioAlphaH_{p}")

            # 定义燃油消耗距离
            z_p_M = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_p_M_{p}")
            z_p_H = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_p_H_{p}")

            # 计算 z_p_M 和 z_p_H
            model.addConstr(z_p_M == l_p[p] * (1 - y[p]) + l_e[p] * y[p], name=f"Z_p_M_{p}")
            model.addConstr(z_p_H == l_ne[p] * y[p], name=f"Z_p_H_{p}")

            # 定义燃油消耗辅助变量
            s_M = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"s_M_{p}")
            s_H = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"s_H_{p}")

            # 构建燃油消耗约束
            model.addConstr(s_M == z_p_M * v_ratio_M_alpha, name=f"SM_{p}")
            model.addConstr(s_H == z_p_H * v_ratio_H_alpha, name=f"SH_{p}")

            # 总燃油消耗公式
            model.addConstr(R_M[p] == F * s_M, name=f"FuelConsumptionM_{p}")
            model.addConstr(R_H[p] == F * s_H, name=f"FuelConsumptionH_{p}")

            # 燃油库存平衡
            if p == 1:
                model.addConstr(q_M_arr[p] == B_M1, name=f"FuelInventoryArrivalM_{p}")
                model.addConstr(q_H_arr[p] == B_H1, name=f"FuelInventoryArrivalH_{p}")
            else:
                model.addConstr(q_M_arr[p] == q_M_dep[p - 1], name=f"FuelInventoryArrivalM_{p}")
                model.addConstr(q_H_arr[p] == q_H_dep[p - 1], name=f"FuelInventoryArrivalH_{p}")
            model.addConstr(q_M_dep[p] == q_M_arr[p] + x_b[p] * Q_M[p] - R_M[p], name=f"FuelInventoryDepartureM_{p}")
            model.addConstr(q_H_dep[p] == q_H_arr[p] + x_b[p] * Q_H[p] - R_H[p], name=f"FuelInventoryDepartureH_{p}")
            # 燃油容量限制
            model.addConstr(q_M_dep[p] >= S_M, name=f"FuelCapacityM_Min_{p}")
            model.addConstr(q_M_dep[p] <= L_M, name=f"FuelCapacityM_Max_{p}")
            model.addConstr(q_H_dep[p] >= S_H, name=f"FuelCapacityH_Min_{p}")
            model.addConstr(q_H_dep[p] <= L_H, name=f"FuelCapacityH_Max_{p}")

        # 最后一个虚拟港口的燃油库存限制
        model.addConstr(q_M_arr[N + 1] >= S_M, name=f"FinalFuelInventoryM")
        model.addConstr(q_H_arr[N + 1] >= S_H, name=f"FinalFuelInventoryH")

        # 求解模型
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # 存储情景解
            scenario_solutions[omega] = {
                't_eta': {p: t_eta[p].X for p in P},
                'x': {(p, k): x[p, k].X for p in P for k in K},
                'mu_t': mu_t[omega].copy(),
                'mu_x': mu_x[omega].copy(),
                'obj': model.ObjVal,
                # 存储二阶段变量以便输出
                't_arr': {p: t_arr[p].X for p in P},
                't_dep': {p: t_dep[p].X for p in P},
                't_wait': {p: t_wait[p].X for p in P},
                't_late': {p: t_late[p].X for p in P},
                't_sail': {p: t_sail[p].X for p in P},
                't_stay': {p: t_stay[p].X for p in P},
                'v_eca': {p: v_eca[p].X for p in P},
                'v_neca': {p: v_neca[p].X for p in P},
                'y': {p: y[p].X for p in P},
                'x_b': {p: x_b[p].X for p in P},
                'Q_M': {p: Q_M[p].X for p in P},
                'Q_H': {p: Q_H[p].X for p in P},
                'q_M_arr': {p: q_M_arr[p].X for p in P_plus},
                'q_H_arr': {p: q_H_arr[p].X for p in P_plus},
                'q_M_dep': {p: q_M_dep[p].X for p in P},
                'q_H_dep': {p: q_H_dep[p].X for p in P},
                'R_M': {p: R_M[p].X for p in P},
                'R_H': {p: R_H[p].X for p in P},
                # 新增拉格朗日乘子
                'mu_Q_M': mu_Q_M[omega].copy(),
                'mu_Q_H': mu_Q_H[omega].copy(),
                'mu_v_eca': mu_v_eca[omega].copy(),
                'mu_v_neca': mu_v_neca[omega].copy(),
                'mu_y': mu_y[omega].copy(),
                'mu_x_b': mu_x_b[omega].copy(),
            }
            scenario_objs.append(model.ObjVal)
        else:
            print(f"情景 {omega} 求解未找到最优解。")
            break

    else:
        # 在更新协调变量之前计算收敛值
        convergence = sum(
            (scenario_solutions[omega]['t_eta'][p] - t_eta_bar[p])**2 for p in P for omega in Omega
        )
        convergence += sum(
            (scenario_solutions[omega]['x'][p, k] - x_bar[p, k])**2 for p in P for k in K for omega in Omega
        )
        # 加入新的变量
        convergence += sum(
            (scenario_solutions[omega]['Q_M'][p] - Q_M_bar[p])**2 for p in P for omega in Omega
        )
        convergence += sum(
            (scenario_solutions[omega]['Q_H'][p] - Q_H_bar[p])**2 for p in P for omega in Omega
        )
        convergence += sum(
            (scenario_solutions[omega]['v_eca'][p] - v_eca_bar[p])**2 for p in P for omega in Omega
        )
        convergence += sum(
            (scenario_solutions[omega]['v_neca'][p] - v_neca_bar[p])**2 for p in P for omega in Omega
        )
        convergence += sum(
            (scenario_solutions[omega]['y'][p] - y_bar[p])**2 for p in P for omega in Omega
        )
        convergence += sum(
            (scenario_solutions[omega]['x_b'][p] - x_b_bar[p])**2 for p in P for omega in Omega
        )
        print(f"收敛值：{convergence}")

        # 检查收敛条件
        if convergence < epsilon:
            print("收敛条件满足，停止迭代。")
            break

        # 更新协调变量（在计算收敛值之后）
        for p in P:
            t_eta_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['t_eta'][p] for omega in Omega)
        for p in P:
            for k in K:
                x_bar[p, k] = sum(p_omega[omega] * scenario_solutions[omega]['x'][p, k] for omega in Omega)

        # 新增变量的协调值更新
        for p in P:
            Q_M_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['Q_M'][p] for omega in Omega)
            Q_H_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['Q_H'][p] for omega in Omega)
            v_eca_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['v_eca'][p] for omega in Omega)
            v_neca_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['v_neca'][p] for omega in Omega)
            y_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['y'][p] for omega in Omega)
            x_b_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['x_b'][p] for omega in Omega)

        # 更新拉格朗日乘子
        for omega in Omega:
            for p in P:
                mu_t[omega][p] += rho_t * (scenario_solutions[omega]['t_eta'][p] - t_eta_bar[p])
            for p in P:
                for k in K:
                    mu_x[omega][p, k] += rho_x * (scenario_solutions[omega]['x'][p, k] - x_bar[p, k])

            # 新增变量的拉格朗日乘子更新
            for p in P:
                mu_Q_M[omega][p] += rho_Q_M * (scenario_solutions[omega]['Q_M'][p] - Q_M_bar[p])
                mu_Q_H[omega][p] += rho_Q_H * (scenario_solutions[omega]['Q_H'][p] - Q_H_bar[p])
                mu_v_eca[omega][p] += rho_v_eca * (scenario_solutions[omega]['v_eca'][p] - v_eca_bar[p])
                mu_v_neca[omega][p] += rho_v_neca * (scenario_solutions[omega]['v_neca'][p] - v_neca_bar[p])
                mu_y[omega][p] += rho_y * (scenario_solutions[omega]['y'][p] - y_bar[p])
                mu_x_b[omega][p] += rho_x_b * (scenario_solutions[omega]['x_b'][p] - x_b_bar[p])

        # 继续下一次迭代

# 输出最终结果
print("\n最终结果：")
# 输出最终协调后的一阶段决策变量
print("协调后的一阶段决策变量：")
for p in P:
    print(f"  港口 {p}:")
    print(f"    ETA（预计到达时间）: {t_eta_bar[p]:.2f}")
    for k in K:
        print(f"    时间窗口 {k}: 选择 = {x_bar[p, k]:.2f}")
    # 输出新增变量
    print(f"    加油量（低硫燃油）: {Q_M_bar[p]:.2f}")
    print(f"    加油量（高硫燃油）: {Q_H_bar[p]:.2f}")
    print(f"    ECA航速: {v_eca_bar[p]:.2f}")
    print(f"    非ECA航速: {v_neca_bar[p]:.2f}")
    # 绕行策略和加油策略
    print(f"    绕行策略 (y): {y_bar[p]:.2f}")
    print(f"    加油策略 (x_b): {x_b_bar[p]:.2f}")

# 输出每个情景的解
for omega in Omega:
    print(f"\n情景 {omega} 的一阶段决策变量：")
    for p in P:
        print(f"  港口 {p}:")
        print(f"    ETA（预计到达时间）: {scenario_solutions[omega]['t_eta'][p]:.2f}")
        for k in K:
            print(f"    时间窗口 {k}: 选择 = {scenario_solutions[omega]['x'][p, k]:.2f}")
        print(f"    加油量（低硫燃油）: {scenario_solutions[omega]['Q_M'][p]:.2f}")
        print(f"    加油量（高硫燃油）: {scenario_solutions[omega]['Q_H'][p]:.2f}")
        print(f"    ECA航速: {scenario_solutions[omega]['v_eca'][p]:.2f}")
        print(f"    非ECA航速: {scenario_solutions[omega]['v_neca'][p]:.2f}")
        # 绕行策略和加油策略
        print(f"    绕行策略 (y): {scenario_solutions[omega]['y'][p]}")
        print(f"    加油策略 (x_b): {scenario_solutions[omega]['x_b'][p]}")
    print("\n二阶段决策变量：")
    for p in P:
        print(f"  港口 {p}:")
        print(f"    实际到达时间: {scenario_solutions[omega]['t_arr'][p]:.2f}")
        print(f"    预计到达时间: {scenario_solutions[omega]['t_eta'][p]:.2f}")
        print(f"    离开时间: {scenario_solutions[omega]['t_dep'][p]:.2f}")
        print(f"    等待时间: {scenario_solutions[omega]['t_wait'][p]:.2f}")
        print(f"    迟到时间: {scenario_solutions[omega]['t_late'][p]:.2f}")
        print(f"    航行时间: {scenario_solutions[omega]['t_sail'][p]:.2f}")
        print(f"    停留时间: {scenario_solutions[omega]['t_stay'][p]:.2f}")
        print(f"    ECA航速: {scenario_solutions[omega]['v_eca'][p]:.2f}")
        print(f"    非ECA航速: {scenario_solutions[omega]['v_neca'][p]:.2f}")
        print(f"    绕行决策 (y): {scenario_solutions[omega]['y'][p]}")
        print(f"    加油决策 (x_b): {scenario_solutions[omega]['x_b'][p]}")
        # 绕行和加油策略输出
        detour_strategy = '绕行' if scenario_solutions[omega]['y'][p] > 0.5 else '直接航线'
        refuel_strategy = '加油' if scenario_solutions[omega]['x_b'][p] > 0.5 else '不加油'
        print(f"    绕行策略: {detour_strategy}")
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
    print(f"  在港口 {N + 1} 的最终低硫燃油量: {scenario_solutions[omega]['q_M_arr'][N+1]:.2f}")
    print(f"  在港口 {N + 1} 的最终高硫燃油量: {scenario_solutions[omega]['q_H_arr'][N+1]:.2f}")

    print(f"\n情景 {omega} 的目标函数值：{scenario_solutions[omega]['obj']}")

# 计算平均目标函数值
average_obj = sum(p_omega[omega] * scenario_solutions[omega]['obj'] for omega in Omega)
print(f"\n平均目标函数值：{average_obj}")
