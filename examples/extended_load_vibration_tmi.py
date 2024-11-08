import numpy as np
import matplotlib.pyplot as plt
from sgl2020 import Sgl2020
from deinterf.foundation import ComposableTerm
from deinterf.foundation.sensors import MagIntensity, DirectionalCosine
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate, noise_level
from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.utils.data_ioc import DataIoC

class LoadVibration(ComposableTerm):
    def __build__(self, container: DataIoC) -> np.ndarray:
        intensity = container[MagIntensity]
        direction_cos = container[DirectionalCosine]
        dcx, dcy, dcz = direction_cos.T
        
        dcx_derivative = np.gradient(dcx) 
        dcy_derivative = np.gradient(dcy)
        dcz_derivative = np.gradient(dcz)
        
        # Third-order directional cosine terms (10 features)
        triple_direction_terms = intensity[:, None] * np.column_stack(
            (
                dcx * dcx * dcx,
                dcx * dcx * dcy,
                dcx * dcx * dcz,
                dcx * dcy * dcy,
                dcx * dcy * dcz,
                dcx * dcz * dcz,
                dcy * dcy * dcy,
                dcy * dcy * dcz, 
                dcy * dcz * dcz,
                dcz * dcz * dcz,
            )
        )
        
        # Second-order directional cosine and derivative terms (18 features)
        double_direction_derivative_terms = intensity[:, None] * np.column_stack(
            (
                dcx * dcx * dcx_derivative,
                dcx * dcx * dcy_derivative,
                dcx * dcx * dcz_derivative,
                dcx * dcy * dcx_derivative,
                dcx * dcy * dcy_derivative,
                dcx * dcy * dcz_derivative,
                dcx * dcz * dcx_derivative,
                dcx * dcz * dcy_derivative,
                dcx * dcz * dcz_derivative,
                dcy * dcy * dcx_derivative,
                dcy * dcy * dcy_derivative,
                dcy * dcy * dcz_derivative,
                dcy * dcz * dcx_derivative,
                dcy * dcz * dcy_derivative,
                dcy * dcz * dcz_derivative,
                dcz * dcz * dcx_derivative,
                dcz * dcz * dcy_derivative,
                dcz * dcz * dcz_derivative,
            )
        )
        
        # First-order derivative terms (3 features)
        derivative_terms = intensity[:, None] * np.column_stack(
            (
                dcx_derivative,
                dcy_derivative, 
                dcz_derivative,
            )
        )
        
        return np.column_stack((triple_direction_terms, 
                              double_direction_derivative_terms, 
                              derivative_terms))

if __name__ == "__main__":
    surv_d = (
        Sgl2020()
        .line(["1002.02"])
        .source(
            [
                "flux_d_x",
                "flux_d_y",
                "flux_d_z",
                "mag_5_uc",
            ]
        )
        .take()
    )
    flt_d = surv_d["1002.02"]

    tmi_with_interf = Tmi(tmi=flt_d["mag_5_uc"])
    fom_data = DataIoC().add(
        MagVector(bx=flt_d["flux_d_x"], by=flt_d["flux_d_y"], bz=flt_d["flux_d_z"])
    )

    compensator = TollesLawson(terms=Terms.Terms_16 | LoadVibration())

    tmi_clean = compensator.fit_transform(fom_data, tmi_with_interf)

    ir = improve_rate(tmi_with_interf, tmi_clean, verbose=True)
    print(f"{ir=}")

    plt.plot(tmi_with_interf, label="tmi_with_interf")
    plt.plot(tmi_clean, label="tmi_clean")
    plt.legend()
    plt.show()