# 1 海里 = 1.852 公里
# 定义各港口间的直线距离（公里）
distances_km = {
    "Shanghai-Ningbo": 163,
    "Ningbo-Quanzhou": 675,
    "Quanzhou-Shantou": 265,
    "Shantou-Shekou": 301,
    "Shekou-Nansha": 29
}

# 换算成海里
distances_nm = {key: round(value / 1.852, 2) for key, value in distances_km.items()}
print(distances_nm)
