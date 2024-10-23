import math


def solve_theta_range(d, l, L):
    # 计算 R 和 φ
    R = math.sqrt(d ** 2 + l ** 2)
    if R == 0:
        raise ValueError("d and l cannot both be zero.")

    phi = math.atan2(l, d)  # atan2 返回的值在 [-π, π] 范围内

    # 检查是否有解
    if L / R > 1:
        print("No solution exists because L/R > 1.")
        return None

    # 计算 θ 的上限
    theta_max = math.asin(L / R) - phi

    # θ 的范围必须在 (0, 2/π) 之间
    lower_bound = 0
    upper_bound = min(theta_max, 2 / math.pi)

    if upper_bound <= lower_bound:
        print("No valid solution for theta in the given range.")
        return None

    return (lower_bound, upper_bound)


# 示例用法
d = 3
l = 4
L = 5
theta_range = solve_theta_range(d, l, L)
if theta_range:
    print(f"Theta range: {theta_range[0]:.4f} to {theta_range[1]:.4f} radians")
