import math
from typing import Dict, Sequence
from typing import Union, Optional

def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def bound_fn(
        min_val: Union[float, int], max_val: Union[float, int], value: Union[float, int]
) -> Union[float, int]:
    return max(min_val, min(max_val, value))


def get_configuration(**kwargs) -> Dict:
    width_multiplier = kwargs['width_multi']

    ffn_multiplier = (
        2  # bound_fn(min_val=2.0, max_val=4.0, value=2.0 * width_multiplier)
    )
    mv2_exp_mult = 2  # max(1.0, min(2.0, 2.0 * width_multiplier))

    layer_0_dim = bound_fn(min_val=16, max_val=64, value=32 * width_multiplier)
    layer_0_dim = int(make_divisible(layer_0_dim, divisor=8, min_value=16))
    config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": layer_0_dim,
        },
        "layer1": {
            "out_channels": int(make_divisible(64 * width_multiplier, divisor=16)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2",
        },
        "layer2": {
            "out_channels": int(make_divisible(128 * width_multiplier, divisor=8)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 2,
            "stride": 2,
            "block_type": "mv2",
        },
        "layer3": {  # 28x28
            "out_channels": int(make_divisible(256 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(128 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "layer4": {  # 14x14
            "out_channels": int(make_divisible(384 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(192 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "layer5": {  # 7x7
            "out_channels": int(make_divisible(512 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(256 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
    }

    return config
