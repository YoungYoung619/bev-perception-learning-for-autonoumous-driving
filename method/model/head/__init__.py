import copy

from .gfl_head import GFLHead
from .nanodet_head import NanoDetHead
from .bev_segm_head import BEVSegmHead
from .bevcos3d_head import BevCos3DHead
from .bevcenter3d_head import BEVCenter3DHead

def build_head(cfg, head_cfg):
    head_cfg = copy.deepcopy(head_cfg)
    name = head_cfg.pop("name")
    if name == "GFLHead":
        return GFLHead(**head_cfg)
    elif name == "NanoDetHead":
        return NanoDetHead(**head_cfg)
    elif name == "BEVSegmHead":
        return BEVSegmHead(
            xbound=cfg.model.arch.bev_generator.xbound,
            ybound=cfg.model.arch.bev_generator.ybound,
            zbound=cfg.model.arch.bev_generator.zbound,
            dbound=cfg.model.arch.bev_generator.dbound,
            **head_cfg)
    elif name == "BevCos3DHead":
        return BevCos3DHead(
            xbound=cfg.model.arch.bev_generator.xbound,
            ybound=cfg.model.arch.bev_generator.ybound,
            zbound=cfg.model.arch.bev_generator.zbound,
            dbound=cfg.model.arch.bev_generator.dbound,
            **head_cfg)
    elif name == "BEVCenter3DHead":
        return BEVCenter3DHead(
            xbound=cfg.model.arch.bev_generator.xbound,
            ybound=cfg.model.arch.bev_generator.ybound,
            zbound=cfg.model.arch.bev_generator.zbound,
            dbound=cfg.model.arch.bev_generator.dbound,
            **head_cfg,
        )
    else:
        raise NotImplementedError
