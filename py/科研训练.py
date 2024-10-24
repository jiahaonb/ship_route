# -*- coding: utf-8 -*-
"""
单场景船舶航线规划和调度的混合整数非线性规划模型
使用 Gurobi 求解，包含二次约束和非凸约束
"""
from gurobipy import Model, GRB, quicksum
import numpy as np

# 数据输入

# 港口数量
N = 6  # 示例中的港口数量

# 集合
P = range(1, N + 1)  # 港口集合 [1,2,3,4,5,6]
P_plus = range(1, N + 2)  # 包含最后一个虚拟港口 N+1
# 时间窗
K = [1, 2, 3, 4]

# 定义 P_，用于那些从港口 p 到 p+1 的距离已知的情况
P_ = range(1, N)  # [1,2,3,4,5]

# 参数

# 时间窗口和处理效率
a = {
    (1, 1): 0,   (1, 2): 10,  (1, 3): 18,  (1, 4): 26,
    (2, 1): 10,  (2, 2): 18,  (2, 3): 26,  (2, 4): 35,
    (3, 1): 52,  (3, 2): 62,  (3, 3): 72,  (3, 4): 84,
    (4, 1): 72,  (4, 2): 84,  (4, 3): 94,  (4, 4): 106,
    (5, 1): 94,  (5, 2): 106, (5, 3): 118, (5, 4): 128,
    (6, 1): 110, (6, 2): 125, (6, 3): 140, (6, 4): 155
}

# 时间窗口的结束时间
b = {
    (1, 1): 6,   (1, 2): 16,  (1, 3): 24,  (1, 4): 32,
    (2, 1): 16,  (2, 2): 24,  (2, 3): 32,  (2, 4): 41,
    (3, 1): 58,  (3, 2): 68,  (3, 3): 78,  (3, 4): 90,
    (4, 1): 78,  (4, 2): 90,  (4, 3): 102, (4, 4): 114,
    (5, 1): 102, (5, 2): 114, (5, 3): 125, (5, 4): 132,
    (6, 1): 118, (6, 2): 134, (6, 3): 149, (6, 4): 165
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
    (1, 1): 201, (1, 2): 169, (1, 3): 212, (1, 4): 215,
    (2, 1): 171, (2, 2): 159, (2, 3): 227, (2, 4): 234,
    (3, 1): 231, (3, 2): 178, (3, 3): 214, (3, 4): 219,
    (4, 1): 235, (4, 2): 217, (4, 3): 198, (4, 4): 197,
    (5, 1): 194, (5, 2): 248, (5, 3): 239, (5, 4): 195,
    (6, 1): 164, (6, 2): 211, (6, 3): 163, (6, 4): 174
}

# 成本
C0 = 625  # 固定运营成本（$/h）
C1 = 0.5  # 库存成本（$/TEU·h）
C_late = {1: 500, 2: 255, 3: 260, 4: 300, 5: 320, 6: 290}  # 在港口 p 的迟到惩罚成本（$/h）
F_q = 1000  # 加油的固定成本（$）

# 燃油参数
# 低硫燃油价格（$/吨）
P_M = {
    1: 705,
    2: 721,
    3: 715,
    4: 736,
    5: 667,
    6: 705
}

# 高硫燃油价格（$/吨）
P_H = {
    1: 504,
    2: 507,
    3: 508,
    4: 505,
    5: 491,
    6: 496
}

alpha_M = 3.081
alpha_H = 3.0
F = 1.71  # 燃油消耗系数
v_max = 25  # 最大航速（节）
v_min = 10  # 最小航速（节）

# 距离参数
l_p = {1: 193.89, 2: 448.95, 3: 186.94, 4: 221.12, 5: 28.75}  # 从港口 p 到 p+1 的不绕行距离（海里）
d_p = {1: 12, 2: 12, 3: 12, 4: 12, 5: 12, 6: 12}  # 到达ECA边界的垂直距离（海里）
L_p = {1: 78.29, 2: 364.47, 3: 143.09, 4: 162.53, 5: 15.66}  # 绕行距离参数

# 货物和库存
W_p1 = {1: 1027, 2: 750, 3: 30, 4: 13, 5: 670, 6: 350}
W_p2 = {1: 800, 2: 900, 3: 45, 4: 15, 5: 740, 6: 320}
W_sp = {1: 3000, 2: 2773, 3: 2923, 4: 2898, 5: 2896, 6: 2926}

# 燃油容量
L_M = 100  # 低硫燃油最大容量（吨）
L_H = 100  # 高硫燃油最大容量（吨）
S_M = 30  # 低硫燃油最小容量（吨）
S_H = 30  # 高硫燃油最小容量（吨）
B_M1 = 40  # 到达港口 1 时的初始低硫燃油量（吨）
B_H1 = 45  # 到达港口 1 时的初始高硫燃油量（吨）

# 大 M 值
M_big = 1e7

# 定义一个非常小的正数 epsilon
epsilon = 1e-6

# 已知的到达和离开时间（示例数据，可以根据需要修改）
T_arr = {1: 8.0, 2: 24.0, 3: 40.0, 4: 56.0, 5: 72.0, 6: 88.0}
T_dep_N = 100.0  # 港口 N 的已知离开时间
Q = 1.0  # 参数 Q
T_sail =  {1: 5, 2: 35, 3: 15, 4: 7, 5: 2}

# 创建 Gurobi 模型
model = Model("SingleScenarioShipRouting")
model.Params.OutputFlag = 1
model.Params.NonConvex = 2  # 允许非凸二次约束
model.Params.MIPGap = 0.1
model.Params.Threads = 6

# 变量定义
# 时间窗口变量
x = model.addVars(P, K, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
# 二进制变量 z[p,k]
z = model.addVars(P, K, vtype=GRB.BINARY, name="z")

# 二阶段变量
t_arr = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_arr")
t_dep = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_dep")
t_wait = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_wait")
t_late = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_late")
t_sail = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name="t_sail")
t_stay = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_stay")

# 新增变量
t_arr_mod = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, ub=372.6, name="t_arr_mod")
n_days_arrival = model.addVars(P, vtype=GRB.INTEGER, lb=0, name="n_days_arrival")
t_port_entry = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_port_entry")
t_port_entry_mod = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, ub=372.6, name="t_port_entry_mod")
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

# 航行距离变量
l_e = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name="l_e")  # ECA区域内航行距离
l_ne = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name="l_ne")  # 非ECA区域航行距离
theta = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0.01, ub=(np.pi / 2) - 0.01, name="theta")  # 绕行角度

# 辅助变量：cos(theta) 和 tan(theta)
cos_theta = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="cos_theta")
tan_theta = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name="tan_theta")
sin_theta = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="sin_theta")
cot_theta = model.addVars(P_, vtype=GRB.CONTINUOUS, lb=0, name="cot_theta")

# 固定初始到达时间
model.addConstr(t_arr[1] == T_arr[1] - Q, name="InitialArrivalTime")

# t_late[p] = t_arr[p] - T_arr[p]
model.addConstrs((t_late[p] == t_arr[p] - T_arr[p] for p in P), name="LateTimeDefinition")

# 时间窗口约束
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
        # 港口 1 的到达时间已知
        model.addConstr(t_arr[1] == T_arr[1] - Q, name=f"ArrivalTime_{1}")
        # 等待时间为零
        model.addConstr(t_wait[1] == 0, name=f"WaitingTime_{1}")
        # 离开时间
        model.addConstr(t_dep[1] == t_arr[1] + t_stay[1], name=f"DepartureTime_{1}")
        # 在港停留时间
        model.addConstr(t_stay[1] == quicksum((W_p1[1] + W_p2[1]) / h[1, k] * x[1, k] for k in K), name=f"StayTime_{1}")
    else:
        # 实际到达时间计算
        model.addConstr(t_arr[p] == t_dep[p - 1] + t_sail[p - 1], name=f"ActualArrivalTime_{p}")

        # t_arr_mod 和 n_days_arrival 的关系
        model.addConstr(t_arr[p] == 372.6 * n_days_arrival[p] + t_arr_mod[p], name=f"ArrivalTimeModulo_{p}")
        model.addConstr(t_arr_mod[p] >= 0, name=f"ArrivalModNonNegative_{p}")
        model.addConstr(t_arr_mod[p] <= 372.6, name=f"ArrivalModMax_{p}")

        # 等待时间
        model.addConstr(t_wait[p] >= 0, name=f"WaitingTimeNonNegative_{p}")

        # 进港时间计算
        model.addConstr(t_wait[p] == t_port_entry[p] - t_arr[p], name=f"PortEntryTime_{p}")

        # 离开时间
        model.addConstr(t_dep[p] == t_port_entry[p] + t_stay[p], name=f"DepartureTime_{p}")
        # 在港停留时间
        model.addConstr(t_stay[p] == quicksum((W_p1[p] + W_p2[p]) / h[p, k] * x[p, k] for k in K), name=f"StayTime_{p}")

        # 进港时间约束
        for k in K:
            # 进港时间必须在选择的时间窗口内

            model.addConstr(
                t_port_entry[p] >= 372.6 * n_days_port_entry[p, k] + a[p, k] - M_big * (1 - z[p, k]),
                name=f"PortEntryStartWindow_{p}_{k}"
            )
            model.addConstr(
                t_port_entry[p] <= 372.6 * n_days_port_entry[p, k] + b[p, k] + M_big * (1 - z[p, k]),
                name=f"PortEntryEndWindow_{p}_{k}"
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
    theta_breakpoints = np.linspace(0.1, (np.pi / 2) - 0.1, 10)
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

    # 定义速度比和幂次约束 v_ratio就是v2e/v_max
    v_ratio_M = model.addVar(vtype=GRB.CONTINUOUS, lb=v_min / v_max, ub=1, name=f"v_ratio_M_{p}")
    v_ratio_H = model.addVar(vtype=GRB.CONTINUOUS, lb=v_min / v_max, ub=1, name=f"v_ratio_H_{p}")
    model.addConstr(v_ratio_M * v_max == v_eca[p], name=f"VRatioM_{p}")
    model.addConstr(v_ratio_H * v_max == v_neca[p], name=f"VRatioH_{p}")

    # 定义速度比的幂次变量, 这就是比值的alpha次方
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

    # 引入辅助变量 inverse_v_ratio_M 和 inverse_v_ratio_H
    inverse_v_eca = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"inverse_v_eca_{p}")
    inverse_v_neca = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"inverse_v_neca_{p}")

    # 添加约束，定义这两个辅助变量为速度比的倒数
    model.addConstr(inverse_v_eca * v_eca[p] == 1, name=f"InverseV_M_{p}")
    model.addConstr(inverse_v_neca * v_neca[p] == 1, name=f"InverseV_H_{p}")

    # 总燃油消耗公式
    model.addConstr(R_M[p] == F * s_M * inverse_v_eca, name=f"FuelConsumptionM_{p}")
    model.addConstr(R_H[p] == F * s_H * inverse_v_eca, name=f"FuelConsumptionH_{p}")

# 燃油库存平衡
for p in P:
    if p == 1:
        model.addConstr(q_M_arr[p] == B_M1, name=f"FuelInventoryArrivalM_{p}")
        model.addConstr(q_H_arr[p] == B_H1, name=f"FuelInventoryArrivalH_{p}")
    else:
        model.addConstr(q_M_arr[p] == q_M_dep[p - 1], name=f"FuelInventoryArrivalM_{p}")
        model.addConstr(q_H_arr[p] == q_H_dep[p - 1], name=f"FuelInventoryArrivalH_{p}")
    # 燃油消耗在 P_ 中定义，需区分 p 是否在 P_
    if p in P_:
        model.addConstr(q_M_dep[p] == q_M_arr[p] + x_b[p] * Q_M[p] - R_M[p], name=f"FuelInventoryDepartureM_{p}")
        model.addConstr(q_H_dep[p] == q_H_arr[p] + x_b[p] * Q_H[p] - R_H[p], name=f"FuelInventoryDepartureH_{p}")
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

# 增加约束 t_dep[N] <= Q + T_dep[N]
model.addConstr(t_dep[N] <= Q + T_dep_N, name="FinalDepartureTimeConstraint")

# 目标函数
Objective = (
    quicksum(C0 * t_sail[p] for p in P_) +
    quicksum(F_q * x_b[p] for p in P) +
    quicksum(P_M[p] * Q_M[p] + P_H[p] * Q_H[p] for p in P) +
    quicksum(C1 * W_sp[p] * (t_stay[p] + t_wait[p] + (t_sail[p - 1] if p != 1 else 0)) for p in P) +
    quicksum(C_late[p] * t_late[p] for p in P)
)

model.setObjective(Objective, GRB.MINIMIZE)

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print("\n模型求解成功，最优目标值为：", model.ObjVal)
    # 初始化延误列表 E_p
    E_p = {}
    # 初始化恢复时间列表 r_p
    r_p = {}
    for p in P:
        print(f"\n港口 {p}:")
        print(f"  到达时间 t_arr: {t_arr[p].X:.2f}")
        print(f"  离开时间 t_dep: {t_dep[p].X:.2f}")
        print(f"  等待时间 t_wait: {t_wait[p].X:.2f}")
        print(f"  迟到时间 t_late: {t_late[p].X:.2f}")
        print(f"  停留时间 t_stay: {t_stay[p].X:.2f}")
        # 时间窗口选择
        selected_window = max(z[p, k].X for k in K)
        for k in K:
            if z[p, k].X == selected_window:
                print(f"  选择的时间窗口: {k}")
                break
        # 加油策略
        refuel_strategy = '加油' if x_b[p].X > 0.5 else '不加油'
        print(f"  加油策略: {refuel_strategy}")
        if x_b[p].X > 0.5:
            print(f"    补充的低硫燃油量 Q_M: {Q_M[p].X:.2f}")
            print(f"    补充的高硫燃油量 Q_H: {Q_H[p].X:.2f}")
        print(f"  到达时的低硫燃油量 q_M_arr: {q_M_arr[p].X:.2f}")
        print(f"  到达时的高硫燃油量 q_H_arr: {q_H_arr[p].X:.2f}")
        print(f"  离开时的低硫燃油量 q_M_dep: {q_M_dep[p].X:.2f}")
        print(f"  离开时的高硫燃油量 q_H_dep: {q_H_dep[p].X:.2f}")
        if p in P_:
            print(f"  航行时间 t_sail: {t_sail[p].X:.2f}")
            print(f"  从港口 {p} 到港口 {p+1} 的 ECA 航速 v_eca: {v_eca[p].X:.2f}")
            print(f"  从港口 {p} 到港口 {p+1} 的非 ECA 航速 v_neca: {v_neca[p].X:.2f}")
            detour_strategy = '绕行' if y[p].X > 0.5 else '直接航线'
            print(f"  绕行策略: {detour_strategy}")
            print(f"    低硫燃油消耗 R_M: {R_M[p].X:.2f}")
            print(f"    高硫燃油消耗 R_H: {R_H[p].X:.2f}")
            # 计算延误时间 E_p
            if p == 1:
                E_p[p] = Q + t_sail[p].X - T_sail[p]
            else:
                E_p[p] = t_sail[p].X - T_sail[p]
        else:
            print(f"  没有后续航段。")
            E_p[p] = 0  # 最后一个港口没有航行时间
    # 计算每一段的恢复时间 r_p
    total_recovery = 0
    for p in P_:
        if p == 1:
            r_p[p] = 0  # 第一段没有恢复
        else:
            r_p[p] = (E_p[p - 1] - E_p[p])
            total_recovery += r_p[p]
    # 输出恢复时间
    print("\n每一段的船期恢复时间（r_p）：")
    for p in P_:
        print(f"  第 {p} 段（从港口 {p} 到港口 {p+1}）：恢复时间 r_{p} = {r_p[p]:.2f} 小时")
    print(f"\n总的恢复时间：{total_recovery:.2f} 小时，初始延误 Q = {Q:.2f} 小时")
    if abs(total_recovery - Q) < 1e-2:
        print("总恢复时间等于初始延误时间，船舶已完全恢复延误。")
    else:
        print("总恢复时间不等于初始延误时间，船舶未能完全恢复延误。")
else:
    print("模型未找到最优解。")
