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
P = range(1, N + 1)  # 港口集合 [1,2,3,4,5,6]
P_plus = range(1, N + 2)  # 包含最后一个虚拟港口 N+1
Omega = [1, 2, 3, 4, 5, 6, 7]  # 情景集合
K = [1, 2, 3, 4]  # 每个港口的时间窗口集合

# 定义 P_，用于那些从港口 p 到 p+1 的距离已知的情况
P_ = range(1, N)  # [1,2,3,4,5]

# 参数

# 时间窗口和处理效率

a = {
    (1,1): 2.50, (1,2): 14.92, (1,3): 27.34, (1,4): 39.76, (1,5): 52.18, (1,6): 64.60, (1,7): 77.02, (1,8): 89.44, (1,9): 101.86, (1,10): 114.28,
    (1,11): 126.70, (1,12): 139.12, (1,13): 151.54, (1,14): 163.96, (1,15): 176.38, (1,16): 188.80, (1,17): 201.22, (1,18): 213.64, (1,19): 226.06, (1,20): 238.48,
    (1,21): 250.90, (1,22): 263.32, (1,23): 275.74, (1,24): 288.16, (1,25): 300.58, (1,26): 313.00, (1,27): 325.42, (1,28): 337.84, (1,29): 350.26, (1,30): 362.68,

    (2,1): 6.00, (2,2): 18.42, (2,3): 30.84, (2,4): 43.26, (2,5): 55.68, (2,6): 68.10, (2,7): 80.52, (2,8): 92.94, (2,9): 105.36, (2,10): 117.78,
    (2,11): 130.20, (2,12): 142.62, (2,13): 155.04, (2,14): 167.46, (2,15): 179.88, (2,16): 192.30, (2,17): 204.72, (2,18): 217.14, (2,19): 229.56, (2,20): 241.98,
    (2,21): 254.40, (2,22): 266.82, (2,23): 279.24, (2,24): 291.66, (2,25): 304.08, (2,26): 316.50, (2,27): 328.92, (2,28): 341.34, (2,29): 353.76, (2,30): 366.18,

    (3,1): 0.00, (3,2): 12.42, (3,3): 24.84, (3,4): 37.26, (3,5): 49.68, (3,6): 62.10, (3,7): 74.52, (3,8): 86.94, (3,9): 99.36, (3,10): 111.78,
    (3,11): 124.20, (3,12): 136.62, (3,13): 149.04, (3,14): 161.46, (3,15): 173.88, (3,16): 186.30, (3,17): 198.72, (3,18): 211.14, (3,19): 223.56, (3,20): 235.98,
    (3,21): 248.40, (3,22): 260.82, (3,23): 273.24, (3,24): 285.66, (3,25): 298.08, (3,26): 310.50, (3,27): 322.92, (3,28): 335.34, (3,29): 347.76, (3,30): 360.18,

    (4,1): 6.20, (4,2): 18.62, (4,3): 31.04, (4,4): 43.46, (4,5): 55.88, (4,6): 68.30, (4,7): 80.72, (4,8): 93.14, (4,9): 105.56, (4,10): 117.98,
    (4,11): 130.40, (4,12): 142.82, (4,13): 155.24, (4,14): 167.66, (4,15): 180.08, (4,16): 192.50, (4,17): 204.92, (4,18): 217.34, (4,19): 229.76, (4,20): 242.18,
    (4,21): 254.60, (4,22): 267.02, (4,23): 279.44, (4,24): 291.86, (4,25): 304.28, (4,26): 316.70, (4,27): 329.12, (4,28): 341.54, (4,29): 353.96, (4,30): 366.38,

    (5,1): 4.19, (5,2): 16.61, (5,3): 29.03, (5,4): 41.45, (5,5): 53.87, (5,6): 66.29, (5,7): 78.71, (5,8): 91.13, (5,9): 103.55, (5,10): 115.97,
    (5,11): 128.39, (5,12): 140.81, (5,13): 153.23, (5,14): 165.65, (5,15): 178.07, (5,16): 190.49, (5,17): 202.91, (5,18): 215.33, (5,19): 227.75, (5,20): 240.17,
    (5,21): 252.59, (5,22): 265.01, (5,23): 277.43, (5,24): 289.85, (5,25): 302.27, (5,26): 314.69, (5,27): 327.11, (5,28): 339.53, (5,29): 351.95, (5,30): 364.37,

    (6,1): 6.00, (6,2): 18.42, (6,3): 30.84, (6,4): 43.26, (6,5): 55.68, (6,6): 68.10, (6,7): 80.52, (6,8): 92.94, (6,9): 105.36, (6,10): 117.78,
    (6,11): 130.20, (6,12): 142.62, (6,13): 155.04, (6,14): 167.46, (6,15): 179.88, (6,16): 192.30, (6,17): 204.72, (6,18): 217.14, (6,19): 229.56, (6,20): 241.98,
    (6,21): 254.40, (6,22): 266.82, (6,23): 279.24, (6,24): 291.66, (6,25): 304.08, (6,26): 316.50, (6,27): 328.92, (6,28): 341.34, (6,29): 353.76, (6,30): 366.18
}

b = {
    (1,1): 8.71, (1,2): 21.13, (1,3): 33.55, (1,4): 45.97, (1,5): 58.39, (1,6): 70.81, (1,7): 83.23, (1,8): 95.65, (1,9): 108.07, (1,10): 120.49,
    (1,11): 132.91, (1,12): 145.33, (1,13): 157.75, (1,14): 170.17, (1,15): 182.59, (1,16): 195.01, (1,17): 207.43, (1,18): 219.85, (1,19): 232.27, (1,20): 244.69,
    (1,21): 257.11, (1,22): 269.53, (1,23): 281.95, (1,24): 294.37, (1,25): 306.79, (1,26): 319.21, (1,27): 331.63, (1,28): 344.05, (1,29): 356.47, (1,30): 368.89,

    (2,1): 12.21, (2,2): 24.63, (2,3): 37.05, (2,4): 49.47, (2,5): 61.89, (2,6): 74.31, (2,7): 86.73, (2,8): 99.15, (2,9): 111.57, (2,10): 123.99,
    (2,11): 136.41, (2,12): 148.83, (2,13): 161.25, (2,14): 173.67, (2,15): 186.09, (2,16): 198.51, (2,17): 210.93, (2,18): 223.35, (2,19): 235.77, (2,20): 248.19,
    (2,21): 260.61, (2,22): 273.03, (2,23): 285.45, (2,24): 297.87, (2,25): 310.29, (2,26): 322.71, (2,27): 335.13, (2,28): 347.55, (2,29): 359.97, (2,30): 372.39,

    (3,1): 6.21, (3,2): 18.63, (3,3): 31.05, (3,4): 43.47, (3,5): 55.89, (3,6): 68.31, (3,7): 80.73, (3,8): 93.15, (3,9): 105.57, (3,10): 117.99,
    (3,11): 130.41, (3,12): 142.83, (3,13): 155.25, (3,14): 167.67, (3,15): 180.09, (3,16): 192.51, (3,17): 204.93, (3,18): 217.35, (3,19): 229.77, (3,20): 242.19,
    (3,21): 254.61, (3,22): 267.03, (3,23): 279.45, (3,24): 291.87, (3,25): 304.29, (3,26): 316.71, (3,27): 329.13, (3,28): 341.55, (3,29): 353.97, (3,30): 366.39,

    (4,1): 12.41, (4,2): 24.83, (4,3): 37.25, (4,4): 49.67, (4,5): 62.09, (4,6): 74.51, (4,7): 86.93, (4,8): 99.35, (4,9): 111.77, (4,10): 124.19,
    (4,11): 136.61, (4,12): 149.03, (4,13): 161.45, (4,14): 173.87, (4,15): 186.29, (4,16): 198.71, (4,17): 211.13, (4,18): 223.55, (4,19): 235.97, (4,20): 248.39,
    (4,21): 260.81, (4,22): 273.23, (4,23): 285.65, (4,24): 298.07, (4,25): 310.49, (4,26): 322.91, (4,27): 335.33, (4,28): 347.75, (4,29): 360.17, (4,30): 372.59,

    (5,1): 10.40, (5,2): 22.82, (5,3): 35.24, (5,4): 47.66, (5,5): 60.08, (5,6): 72.50, (5,7): 84.92, (5,8): 97.34, (5,9): 109.76, (5,10): 122.18,
    (5,11): 134.60, (5,12): 147.02, (5,13): 159.44, (5,14): 171.86, (5,15): 184.28, (5,16): 196.70, (5,17): 209.12, (5,18): 221.54, (5,19): 233.96, (5,20): 246.38,
    (5,21): 258.80, (5,22): 271.22, (5,23): 283.64, (5,24): 296.06, (5,25): 308.48, (5,26): 320.90, (5,27): 333.32, (5,28): 345.74, (5,29): 358.16, (5,30): 370.58,

    (6,1): 12.21, (6,2): 24.63, (6,3): 37.05, (6,4): 49.47, (6,5): 61.89, (6,6): 74.31, (6,7): 86.73, (6,8): 99.15, (6,9): 111.57, (6,10): 123.99,
    (6,11): 136.41, (6,12): 148.83, (6,13): 161.25, (6,14): 173.67, (6,15): 186.09, (6,16): 198.51, (6,17): 210.93, (6,18): 223.35, (6,19): 235.77, (6,20): 248.19,
    (6,21): 260.61, (6,22): 273.03, (6,23): 285.45, (6,24): 297.87, (6,25): 310.29, (6,26): 322.71, (6,27): 335.13, (6,28): 347.55, (6,29): 359.97, (6,30): 372.39
}


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
v_min = 15  # 最小航速（节）

# 距离参数
l_p = {1:193.89, 2:448.95, 3:186.94,4:221.12,5:28.75}  # 从港口 p 到 p+1 的直线距离（海里）
d_p = {1:12, 2:15, 3:14,4:13,5:12,6:15}  # 到达ECA边界的垂直距离（海里）

# 货物和库存
W_p1 = {1:200, 2:250, 3:240,4:330,5:320,6:290}  # 在港口 p 卸载的货物量（TEU）
W_p2 = {1:200, 2:250, 3:240,4:320,5:310,6:290}  # 在港口 p 装载的货物量（TEU）
W_sp = {1:1000, 2:1050, 3:1100,4:1200,5:1300,6:1100}  # 在港口 p 和 p+1 之间的库存量（TEU）

# 燃油容量
L_M = 5000  # 低硫燃油最大容量（吨）
L_H = 5000  # 高硫燃油最大容量（吨）
S_M = 500  # 低硫燃油最小容量（吨）
S_H = 500  # 高硫燃油最小容量（吨）
B_M1 = 600  # 到达港口 1 时的初始低硫燃油量（吨）
B_H1 = 500  # 到达港口 1 时的初始高硫燃油量（吨）

# 大 M 值
M_big = 1e7

# 情景概率
p_omega = {omega: 1 / len(Omega) for omega in Omega}  # 假设等概率

# PHA 参数
rho_t = 100.0  # 对 t_eta 的罚参数
rho_x = 100.0  # 对 x 的罚参数
rho_Q_M = 100.0  # 对 Q_M 的罚参数
rho_Q_H = 100.0  # 对 Q_H 的罚参数
rho_v_eca = 100.0  # 对 v_eca 的罚参数
rho_v_neca = 100.0  # 对 v_neca 的罚参数
rho_y = 100.0  # 对 y 的罚参数
rho_x_b = 100.0  # 对 x_b 的罚参数
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
            # l_ne[p] == (l_p[p] - d_p_total * cot_theta[p]) * y[p]
            model.addConstr(l_ne[p] == (l_p[p] - d_p_total * cot_theta[p]) * y[p],
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

            # 新增变量的拉格朗日乘子更新
            for p in P:
                mu_Q_M[omega][p] += rho_Q_M * (scenario_solutions[omega]['Q_M'][p] - Q_M_bar[p])
                mu_Q_H[omega][p] += rho_Q_H * (scenario_solutions[omega]['Q_H'][p] - Q_H_bar[p])
            for p in P_:
                mu_v_eca[omega][p] += rho_v_eca * (scenario_solutions[omega]['v_eca'][p] - v_eca_bar[p])
                mu_v_neca[omega][p] += rho_v_neca * (scenario_solutions[omega]['v_neca'][p] - v_neca_bar[p])
                mu_y[omega][p] += rho_y * (scenario_solutions[omega]['y'][p] - y_bar[p])
            for p in P:
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

