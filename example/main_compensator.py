from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from deinterf import TollesLawsonCompensator

test_file = Path("C:/Users/Dorian/Desktop/flux_and_op.csv")
flt_d = pd.read_csv(test_file)

compor = TollesLawsonCompensator()
compor.fit(
    flt_d["flux_b_x"],
    flt_d["flux_b_y"],
    flt_d["flux_b_z"],
    flt_d["mag_3_uc"],
)
# compor.sampling_rate(10)
# compor.bpf(False)
# compor.using_permanent()
# compor.using_induced()
# compor.using_eddy()

comped, interf = compor.apply(
    flt_d["flux_b_x"],
    flt_d["flux_b_y"],
    flt_d["flux_b_z"],
    flt_d["mag_3_uc"],
)
compor.evaluate_src()
compor.evaluate(flt_d["mag_3_uc"], comped)

plt.plot(comped)
plt.plot(flt_d["mag_3_uc"])
plt.show()
