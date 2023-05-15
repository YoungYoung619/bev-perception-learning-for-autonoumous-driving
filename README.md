# bev-perception-learning-for-autonoumous-driving
使用nuscenes数据集，复现了一些bev算法，仅供学习

## Lift Splat Shoot
### Train
以下脚本用于开启训练，请在对应配置文件中指定好您的nusences数据集位置和您的训练配置（gpu数量等）
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cfg_file=./config/lift_splat_shoot/lss_segm.yml
python tools/train.py --config ${cfg_file}
```

### Demo
以下脚本用于可视化模型检测效果，请在对应配置文件中指定好您的数据集位置，然后您将会看到如图1.所示的检测效果，其中，左侧六张图像为camera输入，右侧上方为真值，右侧下方为网络预测值，预训练的网络可以在model zoo中下载 (LSS & Segm)
```bash
python demo/demo_segm.py --cfg_file=${cfg_file} --model_file=${model_ckpt} --device=mps
```
<div align=center><img src="docs/figs/lss_segm_nuscenes_mini.gif"></div>
<div align="center">图.1 Lift splat shoot 分割检测头效果</div>

## Bevdepth
### Train
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cfg_file=./config/bevdepth/centerhead_3ddet.yml
python tools/train.py --config ${cfg_file}
```

### Demo
python demo/demo_3ddet.py --cfg_file=${cfg_file} --model_file=${model_ckpt} --device=mps
<div align=center><img src="docs/figs/bevdepth_3d.gif"></div>
<div align="center">图.2 Bevdepth 3d目标检测头效果</div>

## Model Zoo
以下为模型预训练文件及训练log，验证集指标可以在log中查看

| Method | Type | Model | Log |
|--------|---------------|------| ----|
|  LSS | Segm  | [Google](https://drive.google.com/file/d/1eOdib9VxuRg33AaQBIIN4PE65fN7gmZS/view?usp=share_link) | [Google](https://drive.google.com/file/d/17_PxiWg43fv_dQl8_zt4TkmYdEsQ4WPq/view?usp=share_link)|
| bevdepth     | 3D检测 | [Google](https://drive.google.com/file/d/1Me7rh4a-BMhNIM3urq3PKIL2m_w0kolv/view?usp=share_link) | [Google](https://drive.google.com/file/d/1StPE5SAeP-62qTu2oV-XGRiiorxr1NWi/view?usp=share_link) |