import os
from ctypes import c_void_p, c_wchar_p
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imsave
from numpy.typing import NDArray

from results_manager.base import BaseStructure

DOSE_COLOR_MAP = plt.get_cmap("jet")


class Images(BaseStructure):
    def __init__(self, experiment_id: str, model_name: c_wchar_p, lambda_: c_void_p, iteration: c_void_p):
        super().__init__(experiment_id, "images", model_name, lambda_, iteration)

    @property
    def state_path(self) -> Path:
        return self.parent_dir / self.lambda_dir / self.iteration_dir

    def get_image_path(self, patient_id: str, epoch: int) -> Path:
        return self.state_path / patient_id / f"epoch_{epoch}.png"

    def write(self, final_image: NDArray, patient_id: str, epoch: int):
        save_path = self.get_image_path(patient_id, epoch)
        os.makedirs(save_path.parent, exist_ok=True)
        imsave(save_path, final_image)

    def make_dose_with_ct(self, dose_sample, batch, epoch):
        """Make image predicted and true dose beside each other"""
        for index, patient_id in enumerate(batch.patient_list):
            # Get predictions for each patient in patient_ids
            true_dose = batch.dose[index]
            sample_dose = dose_sample[index]
            ct = batch.ct[index]
            slice_index = 64

            # Prepare ct image
            ct_image = ct[:, :, slice_index]
            ct_image += 1
            ct_image *= 4095 / 2
            ct_image /= ct_image.max()
            ct_image = np.rot90(ct_image, 3)
            ct_image = np.repeat(ct_image, 4, axis=-1)

            # Prepare real image
            true_image = DOSE_COLOR_MAP((true_dose[:, :, 64] + 1) / 2)
            true_image = np.squeeze(true_image)
            true_image = np.rot90(true_image, 3)

            # Prepare fake image
            fake_image = DOSE_COLOR_MAP((sample_dose[:, :, 64] + 1) / 2)
            fake_image = np.squeeze(fake_image)
            fake_image = np.rot90(fake_image, 3)

            # Concatenate images together, and rotate to correct orientation
            final_image = np.concatenate((ct_image, true_image, fake_image), axis=1)
            self.write(final_image, patient_id, epoch)
