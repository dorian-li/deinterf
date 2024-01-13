from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.data import DataGroup, MagVector
from deinterf.metrics.fom import improve_rate, noise_level

if __name__ == "__main__":
    flt_d = pd.read_csv(Path(__file__).resolve().parent / "test_data.csv")

    # 数据准备
    tmi_with_interf = flt_d["mag_3_uc"]
    mag_vec = MagVector().fit(flt_d, bx="flux_b_x", by="flux_b_y", bz="flux_b_z")
    fom_data = DataGroup(
        (
            mag_vec.to_tmi(),
            mag_vec.to_dir_cosine(),
        )
    )

    # 创建补偿器
    compensator = TollesLawson(terms=Terms.Terms_16)
    compensator.fit(fom_data, tmi_with_interf)

    # 补偿给定信号
    tmi_clean = compensator.transform(fom_data, tmi_with_interf)

    # 或者一步到位，拟合与补偿自身
    # tmi_pure = compensation.fit_transform(fom_data, tmi_with_interf)

    # 仅预测磁干扰
    interf = compensator.predict(fom_data)

    # 评估磁补偿性能
    comped_noise_level = noise_level(tmi_clean)
    print(f"{comped_noise_level=}")

    ir = improve_rate(tmi_with_interf, tmi_clean)
    print(f"{ir=}")

    # 简要绘图
    plt.plot(tmi_with_interf, label="tmi_with_interf")
    plt.plot(tmi_clean, label="tmi_clean")
    plt.legend()
    plt.show()
