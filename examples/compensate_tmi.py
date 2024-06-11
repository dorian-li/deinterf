import matplotlib.pyplot as plt
from sgl2020 import Sgl2020

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate, noise_level
from deinterf.utils.data_ioc import DataIoC

if __name__ == "__main__":
    surv_d = (
        Sgl2020()
        .line(["1002.02"])
        .source(
            [
                "flux_b_x",
                "flux_b_y",
                "flux_b_z",
                "mag_3_uc",
            ]
        )
        .take()
    )
    flt_d = surv_d["1002.02"]

    # 数据准备
    tmi_with_interf = Tmi(tmi=flt_d["mag_3_uc"])
    fom_data = DataIoC().add(
        MagVector(bx=flt_d["flux_b_x"], by=flt_d["flux_b_y"], bz=flt_d["flux_b_z"])
    )

    # 创建补偿器
    compensator = TollesLawson(terms=Terms.Terms_16)
    # 默认使用交叉验证的岭回归，可替换为其他回归器
    # from sklearn.linear_model import BayesianRidge
    # compensator = TollesLawson(terms=Terms.Terms_16, estimator=BayesianRidge())
    compensator.fit(fom_data, tmi_with_interf)

    # 补偿给定信号
    tmi_clean = compensator.transform(fom_data, tmi_with_interf)

    # 或者一步到位，拟合与补偿自身
    tmi_clean = compensator.fit_transform(fom_data, tmi_with_interf)

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
