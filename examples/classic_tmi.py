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

    # Data preparation
    tmi_with_interf = Tmi(tmi=flt_d["mag_3_uc"])
    fom_data = DataIoC().add(
        MagVector(bx=flt_d["flux_b_x"], by=flt_d["flux_b_y"], bz=flt_d["flux_b_z"])
    )

    # Create compensator
    compensator = TollesLawson(terms=Terms.Terms_16)
    # Default uses cross-validated ridge regression, can be replaced with other regressors
    # from sklearn.linear_model import BayesianRidge
    # compensator = TollesLawson(terms=Terms.Terms_16, estimator=BayesianRidge())
    compensator.fit(fom_data, tmi_with_interf)

    # Compensate given signal
    tmi_clean = compensator.transform(fom_data, tmi_with_interf)

    # Or fit and transform in one step
    tmi_clean = compensator.fit_transform(fom_data, tmi_with_interf)

    # Only predict magnetic interference
    interf = compensator.predict(fom_data)

    # Evaluate magnetic compensation performance
    comped_noise_level = noise_level(tmi_clean)
    print(f"{comped_noise_level=}")

    ir = improve_rate(tmi_with_interf, tmi_clean)
    print(f"{ir=}")

    # Simple plot
    plt.plot(tmi_with_interf, label="tmi_with_interf")
    plt.plot(tmi_clean, label="tmi_clean")
    plt.legend()
    plt.show()
