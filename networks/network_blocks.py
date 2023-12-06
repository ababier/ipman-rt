from typing import Optional, Union

from keras.layers import Conv3D, Conv3DTranspose, LeakyReLU, MaxPool3D, SpatialDropout3D, concatenate
from keras.src.engine.keras_tensor import KerasTensor
from keras.src.layers import BatchNormalization


def make_block(
    x: KerasTensor,
    num_filters: int = 0,
    kernel_size: tuple[int, int, int] = (1, 1, 1),
    strides: tuple[int, int, int] = (1, 1, 1),
    batch_norm: bool = True,
    dropout_rate: float = 0,
    relu_alpha: Optional[float] = None,
    x_skip: Optional[KerasTensor] = None,
    layer: Union[type[Conv3D], type[Conv3DTranspose], type[MaxPool3D]] = Conv3D,
) -> KerasTensor:
    """Standard convolution block structure that we use in networks."""
    x = x if x_skip is None else concatenate([x, x_skip])

    if layer in {Conv3D, Conv3DTranspose}:
        use_bias = not batch_norm
        x = layer(num_filters, kernel_size, strides, padding="same", use_bias=use_bias)(x)
    elif layer == MaxPool3D:
        x = layer(strides)(x)
    else:
        raise ValueError(f"{layer} is not supported in `make_base_block`.")

    if batch_norm:
        x = BatchNormalization()(x)
    if dropout_rate:
        x = SpatialDropout3D(dropout_rate)(x)
    if relu_alpha is not None:
        x = LeakyReLU(relu_alpha)(x)
    return x
