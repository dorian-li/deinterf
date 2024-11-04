import sys
import os

# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录的绝对路径（假设项目根目录与 deinterf 同级）
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import matplotlib.pyplot as plt
from sgl2020 import Sgl2020

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate, noise_level
from deinterf.utils.data_ioc import DataIoC
from deinterf.utils.expanded_tl import Current, Wire, LoadVibration
import pandas as pd


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 12})

surv_d = (
        Sgl2020()
        .line(["1002.02"])
        .source(
            [       "mag_1_c",
                    "mag_5_uc",
                    "flux_d_x",
                    "flux_d_y",
                    "flux_d_z",
                    "ins_pitch",
                    "ins_roll",
                    "ins_yaw",
                    "cur_com_1",
                    "cur_ac_hi",
                    "cur_ac_lo",
                    "cur_tank",
                    "cur_flap",
                    "cur_strb",
                    "cur_srvo_o",
                    "cur_srvo_m",
                    "cur_srvo_i",
                    "cur_heat",
                    "cur_acpwr",
                    "cur_outpwr",
                    "cur_bat_1",
                    "cur_bat_2",
            ]
        )
        .take()
    )
def prepare_for_tl(flt_d):
    container = DataIoC().with_data(
        MagVector(bx=flt_d["flux_d_x"], by=flt_d["flux_d_y"], bz=flt_d["flux_d_z"]),
        Current[0](current=flt_d["cur_com_1"]),
        Current[1](current=flt_d["cur_ac_hi"]),
        Current[2](current=flt_d["cur_ac_lo"]),
        Current[3](current=flt_d["cur_tank"]),
        Current[4](current=flt_d["cur_flap"]),
        Current[5](current=flt_d["cur_strb"]),
        Current[6](current=flt_d["cur_srvo_o"]),
        Current[7](current=flt_d["cur_srvo_m"]),
        Current[8](current=flt_d["cur_srvo_i"]),
        Current[9](current=flt_d["cur_heat"]),
        Current[10](current=flt_d["cur_acpwr"]),
        Current[11](current=flt_d["cur_outpwr"]),
        Current[12](current=flt_d["cur_bat_1"]),
        Current[13](current=flt_d["cur_bat_2"]),
    )
    expanded_terms = (
        Terms.Permanent
        | Terms.Induced_5
        | Terms.Eddy_8
        | LoadVibration()
        | Wire()[0]
        | Wire()[1]
        | Wire()[2]
        | Wire()[3]
        | Wire()[4]
        | Wire()[5]
        | Wire()[6]
        | Wire()[7]
        | Wire()[8]
        | Wire()[9]
        | Wire()[10]
        | Wire()[11]
        | Wire()[12]
        | Wire()[13]
    )
    
    # 添加调试信息
    valid_terms = []
    for term in expanded_terms:
        term_data = container[term]
        if term_data is None:
            print(f"Term: {term} has no data.")
        else:
            print(f"Term: {term}, Shape: {term_data.shape}")
            valid_terms.append(term)
    
    return container, expanded_terms


flt_d = surv_d["1002.02"]

container, expanded_terms = prepare_for_tl(flt_d)

tmi_with_interf100202 = Tmi(tmi=flt_d["mag_5_uc"])
ground_truth100202 = Tmi(tmi=flt_d["mag_1_c"])



# 创建补偿器
# compensator = TollesLawson(terms=Terms.Terms_16)
compensator = TollesLawson(terms=expanded_terms)
compensator.fit(container, tmi_with_interf100202)
# 默认使用交叉验证的岭回归，可替换为其他回归器
# from sklearn.linear_model import BayesianRidge
# compensator = TollesLawson(terms=Terms.Terms_16, estimator=BayesianRidge())
# compensator.fit(fom_data, tmi_with_interf100202)

# 补偿给定信号
# tmi_clean = compensator.transform(fom_data, tmi_with_interf100202)

# # 或者一步到位，拟合与补偿自身
# tmi_clean = compensator.fit_transform(fom_data, tmi_with_interf100202)


tmi_clean100202 = compensator.transform(container, tmi_with_interf100202)

# # 仅预测磁干扰
# interf = compensator.predict(fom_data)


# # 评估磁补偿性能
# comped_noise_level = noise_level(tmi_clean)
# print(f"{comped_noise_level=}")

# ir = improve_rate(tmi_with_interf100202, tmi_clean)
ir100202 = improve_rate(tmi_with_interf100202, tmi_clean100202)
# print(f"{ir=}")
print(f"{ir100202=}")

# # 简要绘图
# plt.figure(figsize=(9, 4))
# plt.plot(tmi_with_interf100202, label="未补偿数据")
# plt.plot(tmi_clean, label="补偿后数据")
# plt.legend()
# plt.title(f"改善比={ir : 2f}, 补偿后噪声水平={comped_noise_level: 4f} nT")
# plt.show()

plt.figure(figsize=(9, 4))
plt.plot(tmi_with_interf100202, label="未补偿数据")
plt.plot(tmi_clean100202, label="补偿后数据")
plt.legend()
plt.show()