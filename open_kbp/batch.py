from __future__ import annotations

from typing import Iterable, Iterator, Optional

import numpy as np
from numpy.typing import NDArray


class DataBatch:
    def __init__(
        self,
        dose: Optional[NDArray] = None,
        sample_dose: Optional[NDArray] = None,
        predicted_dose: Optional[NDArray] = None,
        ct: Optional[NDArray] = None,
        structure_masks: Optional[NDArray] = None,
        structure_mask_names: Optional[list[str]] = None,
        possible_dose_mask: Optional[NDArray] = None,
        voxel_dimensions: Optional[NDArray] = None,
        patient_list: Optional[list[str]] = None,
    ):
        self.patient_list = patient_list
        self.dose = dose
        self.sample_dose = sample_dose
        self.predicted_dose = predicted_dose
        self.ct = ct
        self.structure_masks = structure_masks
        self.structure_mask_names = structure_mask_names
        self.possible_dose_mask = possible_dose_mask
        self.voxel_dimensions = voxel_dimensions

    @property
    def null_values(self) -> NDArray[np.nan]:
        return np.full(shape=(self.size, 1), fill_value=np.nan)

    @property
    def num_rois(self) -> int:
        return len(self.structure_mask_names)

    @property
    def size(self) -> int:
        return len(self.patient_list)

    @classmethod
    def initialize_from_required_data(cls, data_dimensions: dict[str, NDArray], batch_size: int) -> DataBatch:
        attribute_values = {}
        for data_name, dimensions in data_dimensions.items():
            batch_data_dimensions = (batch_size, *dimensions)
            attribute_values[data_name] = np.zeros(batch_data_dimensions)
        return cls(**attribute_values)

    def set_values(self, data_name: str, batch_index: int, values: NDArray) -> None:
        getattr(self, data_name)[batch_index] = values

    def get_index_structure_from_structure(self, structure_name: str):
        return self.structure_mask_names.index(structure_name)

    def yield_roi_labels_and_masks(self, get_full_ptvs: bool = False) -> Iterable[str, NDArray]:
        for index, roi_name in enumerate(self.structure_mask_names):
            if get_full_ptvs and "PTV" in roi_name and self.structure_masks[:, :, :, :, [index]].any():
                full_ptv_indices = [
                    index for index, name in enumerate(self.structure_mask_names) if "PTV" in name and int(name[-1:]) <= int(roi_name[-1:])
                ]
                roi_masks = np.expand_dims(np.sum(self.structure_masks[:, :, :, :, full_ptv_indices], axis=-1), axis=-1)
            else:
                roi_masks = self.structure_masks[:, :, :, :, [index]]
            yield roi_name, roi_masks

    def yield_roi_dose(self, dose_samples: Optional[NDArray[float]] = None) -> Iterator[tuple[str, list[NDArray[float]]]]:
        """
        Get the dose delivered to each ROI by a dose.
        """
        dose_samples = self.dose if dose_samples is None else dose_samples
        for roi_name, batch_roi_mask in self.yield_roi_labels_and_masks(get_full_ptvs=True):
            mean_roi_doses = [
                dose[np.where(roi_mask)] if np.any(roi_mask) else np.array(np.nan) for roi_mask, dose in zip(batch_roi_mask, dose_samples)
            ]
            yield roi_name, mean_roi_doses
