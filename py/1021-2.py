# -*- coding: utf-8 -*-
"""
船舶航线规划和调度的两阶段随机规划模型
使用 Gurobi 求解，包含二次约束和非凸约束
"""
from gurobipy import Model, GRB, quicksum
import numpy as np
import math
from scipy.optimize import minimize
# 数据输入

# 港口数量
N = 6  # 示例中的港口数量

# 集合
P = range(1, N + 1)  # 港口集合 [1,2,3,4,5,6]
P_plus = range(1, N + 2)  # 包含最后一个虚拟港口 N+1
Omega = [1, 2, 3, 4, 5, 6, 7]  # 情景集合
# 时间窗  是30个
K = [0] * 30
for i in range(len(K)):
    K[i] = i + 1

# 定义 P_，用于那些从港口 p 到 p+1 的距离已知的情况
P_ = range(1, N)  # [1,2,3,4,5]

# 参数

# 时间窗口和处理效率

# 定义初始值和差值
initial_values_a = {
    1: 2.50,
    2: 6.00,
    3: 0.00,
    4: 6.20,
    5: 4.19,
    6: 6.00
}
initial_values_b = {
    1: 8.71,
    2: 12.21,
    3: 6.21,
    4: 12.41,
    5: 10.40,
    6: 12.21
}
diff = 12.42
num_elements = 30

# 计算最优的theta值
optimal_theta = {}
sin_theta_values = {}
cos_theta_values = {}
tan_theta_values = {}
cot_theta_values = {}

# 创建字典 a
a = {} # 时间窗起始时间
b = {} # 时间窗结束时间

# 使用循环生成各个组别的值
for group in range(1, 7):  # 遍历第1组到第6组
    for i in range(1, num_elements + 1):  # 每组有30个元素
        a[(group, i)] = initial_values_a[group] + (i - 1) * diff

for group in range(1, 7):  # 遍历第1组到第6组
    for i in range(1, num_elements + 1):  # 每组有30个元素
        b[(group, i)] = initial_values_b[group] + (i - 1) * diff

print(a)
print(b)


h = {
    (1, 1): 240, (1, 2): 210, (1, 3): 245, (1, 4): 174, (1, 5): 202, (1, 6): 196, (1, 7): 247, (1, 8): 228, (1, 9): 202,
    (1, 10): 207,
    (1, 11): 225, (1, 12): 188, (1, 13): 229, (1, 14): 240, (1, 15): 243, (1, 16): 211, (1, 17): 160, (1, 18): 189,
    (1, 19): 230, (1, 20): 201,
    (1, 21): 248, (1, 22): 178, (1, 23): 182, (1, 24): 240, (1, 25): 161, (1, 26): 159, (1, 27): 201, (1, 28): 200,
    (1, 29): 242, (1, 30): 226,

    (2, 1): 209, (2, 2): 241, (2, 3): 206, (2, 4): 210, (2, 5): 168, (2, 6): 188, (2, 7): 204, (2, 8): 205, (2, 9): 182,
    (2, 10): 208,
    (2, 11): 169, (2, 12): 240, (2, 13): 178, (2, 14): 230, (2, 15): 189, (2, 16): 164, (2, 17): 223, (2, 18): 219,
    (2, 19): 208, (2, 20): 214,
    (2, 21): 220, (2, 22): 186, (2, 23): 210, (2, 24): 159, (2, 25): 211, (2, 26): 240, (2, 27): 178, (2, 28): 220,
    (2, 29): 192, (2, 30): 241,

    (3, 1): 159, (3, 2): 237, (3, 3): 157, (3, 4): 235, (3, 5): 197, (3, 6): 204, (3, 7): 210, (3, 8): 158, (3, 9): 226,
    (3, 10): 163,
    (3, 11): 161, (3, 12): 241, (3, 13): 187, (3, 14): 211, (3, 15): 175, (3, 16): 214, (3, 17): 156, (3, 18): 204,
    (3, 19): 213, (3, 20): 159,
    (3, 21): 230, (3, 22): 184, (3, 23): 182, (3, 24): 238, (3, 25): 170, (3, 26): 172, (3, 27): 209, (3, 28): 223,
    (3, 29): 157, (3, 30): 219,

    (4, 1): 224, (4, 2): 231, (4, 3): 188, (4, 4): 212, (4, 5): 177, (4, 6): 233, (4, 7): 191, (4, 8): 163, (4, 9): 205,
    (4, 10): 181,
    (4, 11): 224, (4, 12): 182, (4, 13): 230, (4, 14): 220, (4, 15): 156, (4, 16): 212, (4, 17): 207, (4, 18): 218,
    (4, 19): 195, (4, 20): 161,
    (4, 21): 241, (4, 22): 165, (4, 23): 174, (4, 24): 199, (4, 25): 245, (4, 26): 247, (4, 27): 235, (4, 28): 224,
    (4, 29): 171, (4, 30): 211,

    (5, 1): 245, (5, 2): 183, (5, 3): 205, (5, 4): 195, (5, 5): 191, (5, 6): 160, (5, 7): 227, (5, 8): 222, (5, 9): 192,
    (5, 10): 196,
    (5, 11): 231, (5, 12): 190, (5, 13): 187, (5, 14): 170, (5, 15): 232, (5, 16): 236, (5, 17): 178, (5, 18): 240,
    (5, 19): 247, (5, 20): 199,
    (5, 21): 217, (5, 22): 165, (5, 23): 238, (5, 24): 227, (5, 25): 218, (5, 26): 182, (5, 27): 160, (5, 28): 222,
    (5, 29): 220, (5, 30): 247,

    (6, 1): 170, (6, 2): 186, (6, 3): 239, (6, 4): 209, (6, 5): 212, (6, 6): 223, (6, 7): 245, (6, 8): 156, (6, 9): 231,
    (6, 10): 197,
    (6, 11): 182, (6, 12): 185, (6, 13): 170, (6, 14): 178, (6, 15): 227, (6, 16): 207, (6, 17): 231, (6, 18): 214,
    (6, 19): 203, (6, 20): 177,
    (6, 21): 226, (6, 22): 203, (6, 23): 244, (6, 24): 227, (6, 25): 237, (6, 26): 240, (6, 27): 231, (6, 28): 160,
    (6, 29): 190, (6, 30): 158
}

c = {
    (1,1): 201, (1,2): 169, (1,3): 212, (1,4): 215, (1,5): 170, (1,6): 210, (1,7): 196, (1,8): 242, (1,9): 179, (1,10): 227,
    (1,11): 227, (1,12): 203, (1,13): 175, (1,14): 232, (1,15): 243, (1,16): 210, (1,17): 178, (1,18): 235, (1,19): 213, (1,20): 207,
    (1,21): 205, (1,22): 239, (1,23): 205, (1,24): 228, (1,25): 238, (1,26): 186, (1,27): 241, (1,28): 240, (1,29): 198, (1,30): 228,

    (2,1): 171, (2,2): 159, (2,3): 227, (2,4): 234, (2,5): 211, (2,6): 243, (2,7): 201, (2,8): 214, (2,9): 211, (2,10): 248,
    (2,11): 232, (2,12): 202, (2,13): 158, (2,14): 169, (2,15): 228, (2,16): 159, (2,17): 194, (2,18): 209, (2,19): 239, (2,20): 213,
    (2,21): 229, (2,22): 159, (2,23): 186, (2,24): 217, (2,25): 211, (2,26): 158, (2,27): 231, (2,28): 159, (2,29): 222, (2,30): 221,

    (3,1): 231, (3,2): 178, (3,3): 214, (3,4): 219, (3,5): 198, (3,6): 231, (3,7): 213, (3,8): 156, (3,9): 160, (3,10): 232,
    (3,11): 177, (3,12): 243, (3,13): 235, (3,14): 195, (3,15): 203, (3,16): 202, (3,17): 233, (3,18): 161, (3,19): 169, (3,20): 190,
    (3,21): 232, (3,22): 230, (3,23): 184, (3,24): 156, (3,25): 204, (3,26): 169, (3,27): 190, (3,28): 185, (3,29): 236, (3,30): 198,

    (4,1): 235, (4,2): 217, (4,3): 198, (4,4): 197, (4,5): 240, (4,6): 239, (4,7): 222, (4,8): 198, (4,9): 205, (4,10): 234,
    (4,11): 210, (4,12): 240, (4,13): 246, (4,14): 175, (4,15): 247, (4,16): 210, (4,17): 237, (4,18): 168, (4,19): 183, (4,20): 212,
    (4,21): 188, (4,22): 170, (4,23): 213, (4,24): 161, (4,25): 185, (4,26): 179, (4,27): 227, (4,28): 239, (4,29): 159, (4,30): 245,

    (5,1): 194, (5,2): 248, (5,3): 239, (5,4): 195, (5,5): 160, (5,6): 186, (5,7): 213, (5,8): 230, (5,9): 160, (5,10): 168,
    (5,11): 241, (5,12): 233, (5,13): 190, (5,14): 179, (5,15): 238, (5,16): 245, (5,17): 182, (5,18): 191, (5,19): 221, (5,20): 177,
    (5,21): 205, (5,22): 235, (5,23): 240, (5,24): 246, (5,25): 243, (5,26): 190, (5,27): 240, (5,28): 161, (5,29): 234, (5,30): 182,

    (6,1): 164, (6,2): 211, (6,3): 163, (6,4): 174, (6,5): 186, (6,6): 201, (6,7): 189, (6,8): 160, (6,9): 209, (6,10): 235,
    (6,11): 175, (6,12): 199, (6,13): 235, (6,14): 182, (6,15): 196, (6,16): 235, (6,17): 205, (6,18): 205, (6,19): 175, (6,20): 208,
    (6,21): 192, (6,22): 240, (6,23): 236, (6,24): 159, (6,25): 221, (6,26): 243, (6,27): 235, (6,28): 206, (6,29): 238, (6,30): 156
}

# 成本
C0 = 625  # 固定运营成本（$/h）
C1 = 0.5  # 库存成本（$/TEU·h）
C_late = {1:500, 2:255, 3:260,4:300,5:320,6:290}  # 在港口 p 的迟到惩罚成本（$/h）
F_q = 1000  # 加油的固定成本（$）

# 燃油参数
# 低硫燃油价格（$/吨）
P_M = {
    (1, 1): 697, (1, 2): 702, (1, 3): 705, (1, 4): 710, (1, 5): 715, (1, 6): 710, (1, 7): 705,
    (2, 1): 708, (2, 2): 713, (2, 3): 716, (2, 4): 721, (2, 5): 726, (2, 6): 721, (2, 7): 721,
    (3, 1): 701, (3, 2): 706, (3, 3): 709, (3, 4): 714, (3, 5): 719, (3, 6): 717, (3, 7): 715,
    (4, 1): 722, (4, 2): 727, (4, 3): 730, (4, 4): 735, (4, 5): 740, (4, 6): 735, (4, 7): 736,
    (5, 1): 658, (5,2): 655, (5,3): 655, (5,4): 661, (5,5): 683, (5,6): 682, (5,7): 667,
    (6,1): 690, (6,2): 688, (6,3): 679, (6,4): 680, (6,5): 695, (6,6): 705, (6,7): 705
}

# 高硫燃油价格（$/吨）
P_H = {
    (1, 1): 537, (1, 2): 532, (1, 3): 525, (1, 4): 500, (1, 5): 501, (1, 6): 504, (1, 7): 504,
    (2, 1): 540, (2, 2): 535, (2, 3): 528, (2, 4): 503, (2, 5): 504, (2, 6):506, (2, 7):507,
    (3, 1): 545, (3, 2): 540, (3, 3): 533, (3, 4): 508, (3, 5): 509, (3, 6): 512, (3, 7): 508,
    (4, 1): 540, (4,2): 535, (4,3): 528, (4,4): 503, (4,5): 504, (4,6): 507, (4,7):505,
    (5,1): 481, (5,2): 480, (5,3): 472, (5,4): 475, (5,5): 489, (5,6): 491, (5,7): 491,
    (6,1): 483, (6,2): 480, (6,3): 443, (6,4): 475, (6,5): 490, (6,6): 496, (6,7): 496
}

alpha_M = 3.081
alpha_H = 3.0
F = 1.71  # 燃油消耗系数
v_max = 25  # 最大航速（节）
v_min = 10  # 最小航速（节）

# 距离参数
l_p = {1:193.89, 2:448.95, 3:186.94,4:221.12,5:28.75}  # 从港口 p 到 p+1 的不绕行距离（海里）
L_p = {1:180, 2:428.92, 3: 140.83, 4:200.45, 5:20.32} # 港口p到p+1的陆地直线距离（海里）
d_p = {1:12, 2:12, 3:12,4:12,5:12,6:12}  # 到达ECA边界的垂直距离（海里）

# 货物和库存
W_p1 = {1: 1027, 2: 750, 3: 30, 4: 3, 5: 670, 6: 350}
W_p2 = {1: 800, 2: 900, 3: 45, 4: 5, 5: 740, 6: 320}
W_sp = {1: 3000, 2: 2773, 3: 2923, 4: 2898, 5: 2896, 6: 2926}

# 燃油容量
L_M = 1000  # 低硫燃油最大容量（吨）
L_H = 1000  # 高硫燃油最大容量（吨）
S_M = 300  # 低硫燃油最小容量（吨）
S_H = 300  # 高硫燃油最小容量（吨）
B_M1 = 400  # 到达港口 1 时的初始低硫燃油量（吨）
B_H1 = 450  # 到达港口 1 时的初始高硫燃油量（吨）

# 大 M 值
M_big = 1e7

# 情景概率
p_omega = {omega: 1 / len(Omega) for omega in Omega}  # 假设等概率

# PHA 参数
rho_t = 10.0  # 对 t_eta 的罚参数
rho_x = 10.0  # 对 x 的罚参数
max_iter = 100  # 最大迭代次数
epsilon_t = 1e-3  # 收敛阈值
epsilon_x = 1e-10  # 收敛阈值

# 初始化协调变量和拉格朗日乘子
t_eta_bar = {p: 0.0 for p in P}
x_bar = {(p, k): 1.0 / len(K) for p in P for k in K}  # 均匀分配初值

# 新增协调变量
Q_M_bar = {p: 0.0 for p in P}
Q_H_bar = {p: 0.0 for p in P}
v_eca_bar = {p: (v_min + v_max) / 2 for p in P_}
v_neca_bar = {p: (v_min + v_max) / 2 for p in P_}
y_bar = {p: 0.0 for p in P_}  # 初始化为 0
x_b_bar = {p: 0.0 for p in P}  # 初始化为 0

# 初始化拉格朗日乘子
mu_t = {omega: {p: 0.0 for p in P} for omega in Omega}
mu_x = {omega: {(p, k): 0.0 for p in P for k in K} for omega in Omega}

# 新增拉格朗日乘子
mu_Q_M = {omega: {p: 0.0 for p in P} for omega in Omega}
mu_Q_H = {omega: {p: 0.0 for p in P} for omega in Omega}
mu_v_eca = {omega: {p: 0.0 for p in P_} for omega in Omega}
mu_v_neca = {omega: {p: 0.0 for p in P_} for omega in Omega}
mu_y = {omega: {p: 0.0 for p in P_} for omega in Omega}
mu_x_b = {omega: {p: 0.0 for p in P} for omega in Omega}

# 初始化情景变量的存储
scenario_solutions = {}

# 初始出发时间（港口 1 的 ETA），可以由用户设定
initial_departure_time = 8.0  # 例如，设置为 8:00

# 定义一个非常小的正数 epsilon
epsilon = 1e-6

for n in range(max_iter):
    print(f"\n--- PHA 迭代 {n+1} ---")
    # 对于每个情景，建立并求解子问题
    scenario_objs = []
    for omega in Omega:
        # 创建 Gurobi 模型 参数如下：
        model = Model(f"TwoStageShipRouting_Scenario_{omega}")
        model.Params.OutputFlag = 1
        model.Params.NonConvex = 2  # 允许非凸二次约束
        model.Params.MIPGap = 0.1
        model.Params.Threads = 6

        # 一阶段变量（针对每个情景）
        t_eta = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_eta")
        # 将 x[p,k] 定义为连续变量
        x = model.addVars(P, K, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
        # 新增二进制变量 z[p,k]
        z = model.addVars(P, K, vtype=GRB.BINARY, name="z")

        # 新增：t_eta_mod 和 n_days
        t_eta_mod = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, ub=372.6, name="t_eta_mod")
        n_days = model.addVars(P, vtype=GRB.INTEGER, lb=0, name="n_days")

        # 二阶段变量（针对每个情景）
        t_arr = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_arr")
        t_dep = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_dep")
        t_wait = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_wait")
        t_late = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_late")
        t_sail = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="t_sail")
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


        # 一阶段目标函数（加入罚项和拉格朗日乘子）
        FirstStageCost = quicksum(c[p, k] * x[p, k] * (W_p1[p] + W_p2[p]) for p in P for k in K)

        penalty_t = quicksum(
            rho_t / 2 * (t_eta[p] - t_eta_bar[p])**2 + mu_t[omega][p] * (t_eta[p] - t_eta_bar[p]) for p in P
        )

        penalty_x = quicksum(
            rho_x / 2 * (x[p, k] - x_bar[p, k])**2 + mu_x[omega][p, k] * (x[p, k] - x_bar[p, k]) for p in P for k in K
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
            FirstStageCost + SecondStageCost + penalty_t + penalty_x ,
            GRB.MINIMIZE
        )

        # 一阶段约束
        # 固定初始出发时间
        model.addConstr(t_eta[1] == initial_departure_time, name="InitialETA")
        # t_eta_mod 和 n_days 的关系
        model.addConstrs((t_eta[p] == 372.6 * n_days[p] + t_eta_mod[p] for p in P), name="ETA_Modulo")
        model.addConstrs((t_eta_mod[p] >= 0 for p in P), name="ETA_Modulo_NonNegative")
        model.addConstrs((t_eta_mod[p] <= 372.6 for p in P), name="ETA_Modulo_Max")

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
                model.addConstr(t_arr[p] == 372.6 * n_days_arrival[p] + t_arr_mod[p], name=f"ArrivalTimeModulo_{p}")
                model.addConstr(t_arr_mod[p] >= 0, name=f"ArrivalModNonNegative_{p}")
                model.addConstr(t_arr_mod[p] <= 372.6, name=f"ArrivalModMax_{p}")

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


        # 计算我们的最优他theta以及对应的sin,cos
        for p in P_:
            d_p_total = d_p[p] + d_p.get(p + 1, 0)
            L_p_value = L_p[p]  # 获取 L_p
            # 定义函数
            def objective(theta, d_p_total, L_p_value):
                return d_p_total * np.sin(theta) + L_p_value * np.cos(theta)
            initial_theta=0
            # 最小化
            result = minimize(objective, initial_theta, args=(d_p_total, L_p_value))
            theta_opt = result.x[0]
            optimal_theta[p] = theta_opt  # 存储最优theta
            sin_theta_values[p] = math.sin(theta_opt)
            cos_theta_values[p] = math.cos(theta_opt)
            tan_theta_values[p] = math.tan(theta_opt)
            cot_theta_values[p] = 1 / tan_theta_values[p]  # 计算cot(theta)
            print(f"港口 {p} 最优theta值: {theta_opt:.4f} radians")
            print(f"sin(theta) = {sin_theta_values[p]:.4f}, cos(theta) = {cos_theta_values[p]:.4f}, tan(theta) = {tan_theta_values[p]:.4f}, cot(theta) = {cot_theta_values[p]:.4f}")

        # 航行时间计算（对于 p in P_）
        for p in P_:
            # 定义辅助变量
            t_sail1 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_sail1_{p}")
            t_sail2 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_sail2_{p}")
            t_sail3 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_sail3_{p}")
            # 角度限制
            model.addConstr(theta[p] >= 0.1, name=f"ThetaMin_{p}")
            model.addConstr(theta[p] <= (np.pi / 2) - 0.1, name=f"ThetaMax_{p}")
            #最优的theta加入到咱们的约束里面
            model.addConstr(theta[p] == optimal_theta[p], name=f"OptimalTheta_{p}")

            # 航线距离计算
            d_p_total = d_p[p] + d_p.get(p + 1, 0)
            # l_e[p] * sin_theta[p] == d_p_total * y[p]
            model.addConstr(l_e[p] * sin_theta_values[p] == d_p_total * y[p],
                            name=f"DistanceECA_{p}")
            # l_ne[p] == (l_p[p] - d_p_total * cot_theta[p]) * y[p]
            model.addConstr(l_ne[p] == (l_p[p] - d_p_total * cot_theta_values[p]) * y[p],
                            name=f"DistanceNECA_{p}")

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

        model.Params.IntFeasTol = 1e-8  # 将整数容差调整为更小的值


        # def my_callback(model, where):
        #     if where == GRB.Callback.MIPSOL:
        #         # 输出当前的 z[p,k] 变量值
        #         print("\nCurrent solution:")
        #         for p in P:
        #             for k in K:
        #                 z_val = model.cbGetSolution(z[p, k])
        #                 print(f"z[{p},{k}] = {z_val}")

        # 求解模型
        model.optimize()


        if model.status == GRB.OPTIMAL:
            # 存储情景解
            scenario_solutions[omega] = {
                't_eta': {p: t_eta[p].X for p in P},
                't_eta_mod': {p: t_eta_mod[p].X for p in P},
                'n_days': {p: n_days[p].X for p in P},
                'x': {(p, k): x[p, k].X for p in P for k in K},
                'z': {(p, k): z[p, k].X for p in P for k in K},
                'mu_t': mu_t[omega].copy(),
                'mu_x': mu_x[omega].copy(),
                'obj': model.ObjVal,
                # 存储二阶段变量以便输出
                't_arr': {p: t_arr[p].X for p in P},
                't_arr_mod': {p: t_arr_mod[p].X if p !=1 else t_eta_mod[p].X for p in P},
                'n_days_arrival': {p: n_days_arrival[p].X if p !=1 else n_days[p].X for p in P},
                't_port_entry': {p: t_port_entry[p].X for p in P},
                't_wait': {p: t_wait[p].X for p in P},
                't_late': {p: t_late[p].X for p in P},
                't_dep': {p: t_dep[p].X for p in P},
                't_sail': {p: t_sail[p].X if p in P_ else 0.0 for p in P},
                't_stay': {p: t_stay[p].X for p in P},
                'v_eca': {p: v_eca[p].X if p in P_ else None for p in P},
                'v_neca': {p: v_neca[p].X if p in P_ else None for p in P},
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
                # 新增拉格朗日乘子
                'mu_Q_M': mu_Q_M[omega].copy(),
                'mu_Q_H': mu_Q_H[omega].copy(),
                'mu_v_eca': mu_v_eca[omega].copy(),
                'mu_v_neca': mu_v_neca[omega].copy(),
                'mu_y': mu_y[omega].copy(),
                'mu_x_b': mu_x_b[omega].copy(),
            }
            scenario_objs.append(model.ObjVal)

            # 输出每次迭代的结果
            print(f"\n情景 {omega} 的结果：")
            for p in P:
                print(f"港口 {p}:")
                # 加油策略
                refuel_strategy = '加油' if scenario_solutions[omega]['x_b'][p] > 0.5 else '不加油'
                print(f"  加油策略: {refuel_strategy}")
                # 绕行策略（仅对 p in P_）
                if p in P_:
                    detour_strategy = '绕行' if scenario_solutions[omega]['y'][p] > 0.5 else '直接航线'
                    print(f"  从港口 {p} 到港口 {p+1} 的绕行策略: {detour_strategy}")
                    # 最优速度
                    print(f"  从港口 {p} 到港口 {p+1} 的 ECA 航速: {scenario_solutions[omega]['v_eca'][p]:.2f} 节")
                    print(f"  从港口 {p} 到港口 {p+1} 的非 ECA 航速: {scenario_solutions[omega]['v_neca'][p]:.2f} 节")
                else:
                    print(f"  没有后续航段。")
                # 加油量
                if scenario_solutions[omega]['x_b'][p] > 0.5:
                    print(f"  加油量（低硫燃油）: {scenario_solutions[omega]['Q_M'][p]:.2f} 吨")
                    print(f"  加油量（高硫燃油）: {scenario_solutions[omega]['Q_H'][p]:.2f} 吨")
                else:
                    print(f"  加油量（低硫燃油）: 0.00 吨")
                    print(f"  加油量（高硫燃油）: 0.00 吨")
                # ETA
                print(f"  ETA（预计到达时间）: {scenario_solutions[omega]['t_eta'][p]:.2f} 小时")
                # 实际到达时间
                print(f"  实际到达时间: {scenario_solutions[omega]['t_arr'][p]:.2f} 小时")
                # 选择的时间窗
                print(f"  {scenario_solutions[omega]['z'][p, 1]}, {scenario_solutions[omega]['z'][p, 2]}, {scenario_solutions[omega]['z'][p, 3]}, {scenario_solutions[omega]['z'][p, 4]}")
                # 进港时间
                print(f"  进港时间: {scenario_solutions[omega]['t_port_entry'][p]:.2f} 小时")
                # 等待时间
                print(f"  等待时间: {scenario_solutions[omega]['t_wait'][p]:.2f} 小时")

                # 时间窗选择
                selected_window = max(scenario_solutions[omega]['z'][p, k] for k in K)
                for k in K:
                    if scenario_solutions[omega]['z'][p, k] == selected_window:
                        print(f"  选择的时间窗口: {k}")
                        break
                # 到达时间是当日具体时间
                if p ==1:
                    print(f"  当日到达时间（15d）: {scenario_solutions[omega]['t_eta_mod'][p]:.2f} 小时")
                else:
                    print(f"  当日到达时间（15d）: {scenario_solutions[omega]['t_arr_mod'][p]:.2f} 小时")
        else:
            print(f"情景 {omega} 求解未找到最优解。")
            break

    else:
        # 在更新协调变量之前计算收敛值
        convergence_t = sum(
            (scenario_solutions[omega]['t_eta'][p] - t_eta_bar[p])**2 for omega in Omega for p in P
        )
        convergence_x = sum(
            (scenario_solutions[omega]['x'][p, k] - x_bar[p, k])**2 for omega in Omega for p in P for k in K
        )

        print(f"\nt的收敛值：{convergence_t}")
        print(f"x的收敛值：{convergence_x}")

        # 检查收敛条件
        if convergence_t < epsilon_t and convergence_x < epsilon_x:
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
        for p in P_:
            v_eca_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['v_eca'][p] for omega in Omega)
            v_neca_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['v_neca'][p] for omega in Omega)
            y_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['y'][p] for omega in Omega)
        for p in P:
            x_b_bar[p] = sum(p_omega[omega] * scenario_solutions[omega]['x_b'][p] for omega in Omega)

        # 更新拉格朗日乘子
        for omega in Omega:
            for p in P:
                mu_t[omega][p] += rho_t * (scenario_solutions[omega]['t_eta'][p] - t_eta_bar[p])
            for p in P:
                for k in K:
                    mu_x[omega][p, k] += rho_x * (scenario_solutions[omega]['x'][p, k] - x_bar[p, k])


        # 继续下一次迭代

# 输出最终结果
print("\n最终结果：")
# 输出最终协调后的一阶段决策变量
print("协调后的一阶段决策变量：")
for p in P:
    print(f"  港口 {p}:")
    print(f"    ETA（预计到达时间）: {t_eta_bar[p]:.2f}")
    for k in K:
        print(f"    时间窗口 {k}: 权重 = {x_bar[p, k]:.2f}")

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

# 计算平均目标函数值
average_obj = sum(p_omega[omega] * scenario_solutions[omega]['obj'] for omega in Omega)
print(f"\n平均目标函数值：{average_obj}")

