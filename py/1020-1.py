import datetime
from datetime import timedelta

# 定义涨潮周期，假设每个港口的潮汐周期为12.42小时
TIDE_CYCLE_HOURS = 12.42

# 初始化每个港口的当前涨潮时间（假设）
# 你可以根据实际潮汐表输入当前的涨潮时间
ports_initial_high_tide = {
    "上海": datetime.datetime(2024, 10, 20, 3, 0),  # 假设上海的初始涨潮时间为10月20日凌晨3点
    "宁波": datetime.datetime(2024, 10, 20, 4, 0),  # 假设宁波的初始涨潮时间为10月20日凌晨4点
    "泉州": datetime.datetime(2024, 10, 20, 5, 0),  # 假设泉州的初始涨潮时间为10月20日凌晨5点
    "汕头": datetime.datetime(2024, 10, 20, 6, 0),  # 假设汕头的初始涨潮时间为10月20日凌晨6点
    "蛇口": datetime.datetime(2024, 10, 20, 7, 0),  # 假设蛇口的初始涨潮时间为10月20日凌晨7点
    "南沙": datetime.datetime(2024, 10, 20, 8, 0),  # 假设南沙的初始涨潮时间为10月20日凌晨8点
}

# 定义要计算的天数
days_to_forecast = 3  # 预测未来3天的涨潮时间

def calculate_high_tides(initial_time, cycle_hours, days):
    high_tides = []
    current_time = initial_time
    for _ in range(days * 2):  # 每天两个涨潮
        high_tides.append(current_time)
        current_time += timedelta(hours=cycle_hours)
    return high_tides

# 输出每个港口未来几天的涨潮时间
for port, initial_time in ports_initial_high_tide.items():
    print(f"\n{port}港未来{days_to_forecast}天的涨潮时间窗口：")
    high_tides = calculate_high_tides(initial_time, TIDE_CYCLE_HOURS, days_to_forecast)
    for tide_time in high_tides:
        print(tide_time.strftime('%Y-%m-%d %H:%M'))

