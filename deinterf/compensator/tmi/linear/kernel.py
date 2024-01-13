import numpy as np
from numpy import ndarray
from typing_extensions import Union

from deinterf.data import DataGroup, DirectionalCosine, Tmi


class Feature:
    @staticmethod
    def make(data: DataGroup) -> ndarray:
        raise NotImplementedError

    def __or__(self, other: "Feature") -> "TermsCollection":
        if isinstance(other, Feature):
            return TermsCollection(self, other)
        raise TypeError(
            f"unsupported operand type(s) for |: {type(self)} and {type(other)}"
        )


class Permanent(Feature):
    @staticmethod
    def make(data: DataGroup) -> ndarray:
        return data[DirectionalCosine]


class Induced6(Feature):
    @staticmethod
    def make(data: DataGroup) -> ndarray:
        tmi = data[Tmi]
        cos_x, cos_y, cos_z = data[DirectionalCosine].T
        # (12301, 1) * (12301, 6) -> (12301, 6)
        return np.einsum(
            "ij,ij->ij",
            tmi,
            np.column_stack(
                (
                    cos_x * cos_x,
                    cos_x * cos_y,
                    cos_x * cos_z,
                    cos_y * cos_y,  # removed in Terms.Induced_5 version
                    cos_y * cos_z,
                    cos_z * cos_z,
                )
            ),
        )


class Induced5(Feature):
    @staticmethod
    def make(data: DataGroup) -> ndarray:
        feats = Induced6.make(data)
        return np.delete(feats, feats.shape[1] // 2, 1)


class Induced(Induced5):
    ...


class Eddy9(Feature):
    @staticmethod
    def make(data: DataGroup) -> ndarray:
        tmi = data[Tmi]
        cos_x, cos_y, cos_z = data[DirectionalCosine].T
        cos_x_dot = np.gradient(cos_x)
        cos_y_dot = np.gradient(cos_y)
        cos_z_dot = np.gradient(cos_z)
        return np.einsum(
            "ij,ij->ij",
            tmi,
            np.column_stack(
                (
                    cos_x * cos_x_dot,
                    cos_x * cos_y_dot,
                    cos_x * cos_z_dot,
                    cos_y * cos_x_dot,
                    cos_y * cos_y_dot,  # removed in Terms.Eddy_8 version
                    cos_y * cos_z_dot,
                    cos_z * cos_x_dot,
                    cos_z * cos_y_dot,
                    cos_z * cos_z_dot,
                )
            ),
        )


class Eddy8(Feature):
    @staticmethod
    def make(data: DataGroup):
        feats = Eddy9.make(data)
        return np.delete(feats, feats.shape[1] // 2, 1)


class Eddy(Eddy8):
    ...


class TermsCollection:
    def __init__(self, *terms: Feature) -> None:
        self.terms = terms

    def make(self, data: DataGroup) -> ndarray:
        return np.column_stack([term.make(data) for term in self.terms])

    def __or__(self, other: Union[Feature, "TermsCollection"]) -> "TermsCollection":
        if isinstance(other, Feature):
            return TermsCollection(*self.terms, other)

        if isinstance(other, TermsCollection):
            return TermsCollection(*self.terms, *other.terms)

        raise TypeError(
            f"unsupported operand type(s) for |: {type(self)} and {type(other)}"
        )
