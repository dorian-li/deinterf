from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from deinterf import TollesLawsonCompensator

test_file = Path(__file__).parent / Path("test_data.csv")
flt_d = pd.read_csv(test_file)

compensator = TollesLawsonCompensator()
comped = compensator.fit_transform(
    flt_d[["flux_b_x", "flux_b_y", "flux_b_z"]],
    flt_d["mag_3_uc"],
)
# compensator.adjust_sampling_rate(10)
# compensator.enable_bpf(True)
# compensator.use_permanent(True)
# compensator.use_induced(True)
# compensator.use_eddy(True)

compensator.evaluate_src()
compensator.evaluate(flt_d["mag_3_uc"], comped)

plt.plot(comped, label="comped")
plt.plot(flt_d["mag_3_uc"], label="uncomped")
plt.xlabel("sample[point]")
plt.ylabel("magnetic[nT]")
plt.legend()
plt.grid()
plt.show()
