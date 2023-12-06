from typing import Union

from numpy.typing import NDArray


class DataShapes:
    def __init__(self, rois):
        self.rois = rois
        self.patient_shape = (128, 128, 128)

    @property
    def full_roi_list(self):
        return sum(map(list, self.rois.values()), [])

    @property
    def num_rois(self):
        return len(self.full_roi_list)

    @property
    def dose(self) -> tuple[int, int, int, int]:
        """Dose deposited within the patient tensor"""
        return self.patient_shape + (1,)

    @property
    def sample_dose(self) -> tuple[int, int, int, int]:
        """Sample dose deposited within the patient tensor"""
        return self.patient_shape + (1,)

    @property
    def predicted_dose(self) -> tuple[int, int, int, int]:
        """Predicted dose that should be deposited within the patient tensor"""
        return self.dose

    @property
    def ct(self) -> tuple[int, int, int, int]:
        """CT image grey scale within the patient tensor"""
        return self.patient_shape + (1,)

    @property
    def structure_masks(self) -> tuple[int, int, int, int]:
        """Mask of all structures in patient"""
        return self.patient_shape + (self.num_rois,)

    @property
    def possible_dose_mask(self) -> tuple[int, int, int, int]:
        """Mask where dose can be deposited"""
        return self.patient_shape + (1,)

    @property
    def voxel_dimensions(self) -> tuple[float]:
        """Physical dimensions of patient voxels (in mm)"""
        return tuple((3,))

    def from_data_names(self, data_names: list[str]) -> dict[str, Union[NDArray, tuple[float]]]:
        data_shapes = {}
        for name in data_names:
            data_shapes[name] = getattr(self, name)
        return data_shapes
