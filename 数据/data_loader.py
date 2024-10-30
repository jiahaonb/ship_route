import json
import pandas as pd
from datetime import datetime

# 读取JSON文件
with open('fij_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# c存储xlsx
output_data = []

# 解析燃油种类-本地
print(f"这是AE FJR的燃油")
for fuel_type, day_data in data["api"]["AE FJR"]["data"]['day_list'].items():
    print(f"燃油种类: {fuel_type}")

    # 将日期编号与时间戳进行匹配
    day_to_timestamp = day_data

    # 获取价格数据
    price_data = data["api"]["AE FJR"]["data"]['prices'][fuel_type]['dayprice']

    # 逐一处理价格数据
    for entry in price_data:
        day_number = entry[0]
        price = entry[1]

        # 获取时间戳并转换为日期
        if str(day_number) in day_to_timestamp:
            timestamp = day_to_timestamp[str(day_number)]
            date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')

            # 添加到结果列表
            output_data.append([fuel_type, date, price])

            # 打印输出
            print(f"{fuel_type} - 日期: {date} - 价格: {price}")


# 将数据转换为 DataFrame
df = pd.DataFrame(output_data, columns=["燃油种类", "日期", "价格"])

# 保存到 Excel 文件
df.to_excel("AE FJR燃油价格数据.xlsx", index=False)

print(f"这是全球的燃油")
out_global = []
for fuel_type, day_data in data["api"]["AV G20"]["data"]['day_list'].items():
    print(f"燃油种类: {fuel_type}")

    # 将日期编号与时间戳进行匹配
    day_to_timestamp = day_data

    # 获取价格数据
    price_data = data["api"]["AV G20"]["data"]['prices'][fuel_type]['dayprice']

    # 逐一处理价格数据
    for entry in price_data:
        day_number = entry[0]
        price = entry[1]

        # 获取时间戳并转换为日期
        if str(day_number) in day_to_timestamp:
            timestamp = day_to_timestamp[str(day_number)]
            date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')

            # 添加到结果列表
            out_global.append([fuel_type, date, price])

            # 打印输出
            print(f"{fuel_type} - 日期: {date} - 价格: {price}")


# 将数据转换为 DataFrame
df = pd.DataFrame(output_data, columns=["燃油种类", "日期", "价格"])

