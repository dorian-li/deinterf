from deinterf.compensator.tmi.linear.kernel import (
    Eddy,
    Eddy8,
    Eddy9,
    Induced,
    Induced5,
    Induced6,
    Permanent,
)


class Terms:
    """磁总场补偿线性模型项"""

    Permanent = Permanent()

    Induced_5 = Induced5()
    Induced_6 = Induced6()
    Induced = Induced()

    Eddy_8 = Eddy8()
    Eddy_9 = Eddy9()
    Eddy = Eddy()

    Terms_16 = Permanent | Induced_5 | Eddy_8
    Terms_18 = Permanent | Induced_6 | Eddy_9
