# bev-perception-learning-for-autonoumous-driving
使用nuscenes数据集，复现了一些bev算法，仅供学习

### Lift Splat Shoot
#### Train
以下脚本用于开启训练，请在对应配置文件中指定好您的nusences数据集位置和您的训练配置（gpu数量等）
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cfg_file=./config/lift_splat_shoot/lss_segm.yml
python tools/train.py --config ${cfg_file}
```

#### Demo
以下脚本用于可视化模型检测效果，请在对应配置文件中指定好您的数据集位置，然后您将会看到如图1.所示的检测效果
```bash
python demo/demo_segm.py --cfg_file=${cfg_file} --model_file=${model_ckpt}
```

<div align=center><img src="docs/figs/imgs/lss_segm_nuscenes_mini.gif"></div>
<div align="center">图.1 Lift splat shoot 分割检测头效果</div>