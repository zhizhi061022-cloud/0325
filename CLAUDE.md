<!--
 * @Author: zhizhi061022 zhizhi061022@gmail.com
 * @Date: 2026-03-25 10:25:39
 * @LastEditors: zhizhi061022 zhizhi061022@gmail.com
 * @LastEditTime: 2026-03-25 10:30:34
 * @FilePath: \PLN_baseline_1\CLAUDE.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
## 任务背景
- 任务：复现 Point Linking Network for Object Detection（论文：PLN.pdf）
- 数据集：Pascal VOC 2007 + 2012
- Backbone：ResNet-18（代替原论文 VGG-16）
- 目标：尽可能高的 mAP@0.5

## 当前最佳结果
指令（640 分辨率 + Cosine LR）：
```
python train.py \
      --voc07 /root/data/VOCdevkit/VOC2007 \
      --voc12 /root/data/VOCdevkit/VOC2012 \
      --img_size 640 \
      --epochs 135 \
      --batch 64 \
      --lr 1e-3 \
      --warmup 3 \
      --warmup_lr 2e-4 \
      --cosine \
      --save_dir runs/pln_640_cosine \
      --resume runs/pln_640_cosine/last.pth
```
Epoch 130：**mAP@0.5 = 0.3994**
| 类别 | AP | 类别 | AP |
|---|---|---|---|
| aeroplane | 0.4706 | bicycle | 0.5309 |
| bird | 0.2586 | boat | 0.2070 |
| bottle | 0.1041 | bus | 0.6057 |
| car | 0.5455 | cat | 0.5640 |
| chair | 0.1268 | cow | 0.2929 |
| diningtable | 0.3607 | dog | 0.4598 |
| horse | 0.5877 | motorbike | 0.6058 |
| person | 0.4884 | pottedplant | 0.1447 |
| sheep | 0.3257 | sofa | 0.3814 |
| train | 0.6062 | tvmonitor | 0.3217 |

## 代码结构
- `model/pln.py`：PLN 模型，ResNet-18 backbone + SharedConv + 4 个 BranchHead（lt/rt/lb/rb）
- `model/loss.py`：hybrid 损失（BCE for P，CE for Q/Lx/Ly，MSE for xy）
- `model/decoder.py`：PLNDecoder，4个branch合并后做per-class NMS
- `model/target.py`：GT编码，每个box生成5个关键点，按branch分配center/corner slot
- `data/voc_dataset.py`：VOCDataset，augment=SSD crop 或 YOLO jitter + hflip + resize_to_square
- `train.py`：训练主循环，RMSprop优化器

## 待优化方向（已分析）
### 高优先级（改动小、收益大）
1. **颜色增广缺失**：当前无 HSV / brightness / contrast / saturation jitter，这是 SSD/YOLO 类检测器的标配，对 bottle(0.10)、chair(0.13) 等小类帮助很大
2. **训练时用 resize_to_square 会拉伸宽高比**：换成 letterbox 更合理
3. **训练轮数不够**：cosine schedule 在135轮还未完全收敛，建议训到200~250轮

### 中优先级（改动中等）
4. **FPN 多尺度特征**：目前只用 ResNet-18 layer4（stride 32），小目标检测差；引入 layer2/3/4 的 FPN 可显著提升小类 AP
5. **Dilated backbone**：layer4 改成 dilation，stride 16，grid 从 20×20 变 40×40，提升定位精度
6. **B=3**：每格允许3个目标，减少密集场景漏检

### 低优先级
7. 测试时 TTA（水平翻转+多尺度）
8. Label smoothing
