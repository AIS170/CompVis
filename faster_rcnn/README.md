# Faster R-CNN：Beach Bird 独立实验

本次 A–E 实验已完成，按冻结记录，最终方案为 **D**。可直接使用 [精选结果和字段说明](results/README.md) 作组内比较，无需重新训练或运行 test。原始大型运行产物留在本机 `outputs/`，不提交 Git。两张图的 smoke 仅作流程检查，不能作为正式结果。旧 CompVis 指标不属于本实验。

## 环境与审计

记录结果的环境为 Python3.12.4、PyTorch2.6.0+cu124、TorchVision0.21.0+cu124、GPU RTX4070 Laptop 8GB、CPU i9-13980HX，计时4 threads。这是原实验环境记录，不代表队友的硬件或全新安装环境已经验证。新克隆的独立安装见下文；无需兄弟目录或个人环境。已读取课程PDF；按本次用户明确提出的协议实施。

通过 `kagglehub.dataset_download('liugalhigh/beach-bird/versions/3')` 定位并检查本地缓存结构。实际路径见 outputs/audit.json。images/{train,val,test} 对应 labels/{train,val,test}；YOLO class0 normalized xc/yc/w/h 转原图绝对像素 xyxy，bird=1、background=0，不重新划分。

| split | 图片 | 框 | 负样本 | 类别数 |
|---|---:|---:|---:|---:|
| train | 1369 | 8663 | 154 | 1 |
| val | 291 | 1843 | 33 | 1 |
| test | 240 | 1518 | 27 | 1 |

逐图解码并检查缺失、非有限/无效/越界框、空标注、重复stem及孤立label；无异常。负样本保留为空(0,4) boxes。异常直接报错，不静默丢弃；仅对归一化坐标1e-6浮点容差裁剪。RGB转换仅在内存，原数据不写入。训练保存train/val图像和标注SHA256 manifest。初始test仅作结构与文件身份审计，无预测或调参。

results/teammate_alignment.json 已核对队友test protocol全部240张原图和标注SHA256；validation五条件共1455张图片逐像素一致，1455份标注字节一致。

## 模型与配置

固定TorchVision COCO_V1预训练ResNet50-FPN Faster R-CNN，两类新分类头，backbone后三阶段可训练；使用已有完整权重的FPN V1，不混用V2。每次独立实验重设seed42，从相同COCO权重及相同随机分类头初始化，不跨增强实验续训。

原分辨率RGB float32[0,1]，内部COCO mean/std normalization；短边800、长边上限1333、保持比例，batch内stride32 padding。输出自动恢复到原图xyxy，再直接送evaluator，不再letterbox。预测导出记录实际 input_hw/resized_hw/padded_batch_shape。已实测300x600 -> 666x1333，padding672x1344。

五层FPN的anchor sizes分别16/32/64/128/256；每层height/width ratios=.5/1/2。原候选8/16/32/64/128的宽泛sqrt(area)4至181范围包含86.38% train框；train缩放框sqrt(area)中位67.80、P5=20.16、P95=280.52、最大771.61，因此正式A-D前仅依据train统计上移一档以照顾大框。上述统计使用800/1333的解析缩放近似；整数取整后的实际尺寸随预测导出。极端小/大/细长框仍可能覆盖不足，anchor不是可检测尺寸硬边界。未使用test或validation预测作此选择。

统一SGD lr=.005、momentum=.9、weight_decay=.0005，15 epochs；MultiStepLR完成epoch10及13后乘.1：epochs1-10=.005，11-13=.0005，14-15=.00005。batch2、AMP关闭、梯度累积1、workers0。GPU smoke通过；正式异常停止记录，不私自更改单个实验参数。

Python/NumPy/Torch/CUDA seed42，DataLoader独立generator seed42，workers0无需worker seeds；cuDNN benchmark=False、deterministic=True，跨硬件/版本不保证逐位一致。每epoch验证，按clean validation mAP50-95最大值选checkpoint，平分保留最早epoch。保存best/last、optimizer/scheduler、分项损失、lr和训练耗时。A-D除增强外配置相同。

## A-D增强

原分辨率顺序：flip -> photometric -> blur -> tensor -> 模型内部预处理。

- A无随机增强。
- B p=.5水平翻转，框同步 xmin'=W-xmax，xmax'=W-xmin。
- C在B上增加p=.5共享触发，独立uniform alpha[.8,1.2]与beta[-25.5,25.5]；像素 round(clip(alpha*RGB_uint8+beta,0,255)).astype(uint8)。
- D在C上增加p=.3 PIL GaussianBlur，radius uniform[.5,2]。

Val/test没有随机训练增强。

## Evaluator及组内对齐

直接从官方Ultralytics8.4.208 wheel提取并复用box_iou/compute_ap/ap_per_class/match_predictions等原函数，匹配队友记录版本，不把COCO AP冒充YOLO AP。不执行wheel包初始化，不安装第二套Ultralytics。wheel不提交仓库，通过 `python -m faster_rcnn.fetch_evaluator` 从固定 PyPI 文件地址获取，每次加载前校验 SHA256。来源、版本和hash见 `evaluator.lock.json`，第三方声明见 `THIRD_PARTY.md`；已记录函数hash见 `results/verification.json`。获取流程不改变原函数/AP/matching逻辑。

单类YOLO0对应FR-CNN1。IoU=.50:.05:.95、原版101点插值积分、use_scipy=False匹配去重。ROI内部保留score>.001，直接导出返回框，满足cutoff .001；每图最多300。P/R先过滤confidence>=.25再调用同一matcher、IoU>=.50，跨图micro TP/FP/FN，不用最大F1自动阈值。

ROI NMS=.7、RPN NMS=.7，与AP匹配IoU独立；RPN pre-NMS train/test=2000/1000、post-NMS=2000/1000，均大于300，但score/NMS仍可能使输出少于300，不补框。

已完成本地同库函数、数据身份及validation腐蚀一致性检查；尚未与队友本人共同复核。最终比较前确认共同重评、P/R重匹配、图片身份和计时边界，不能声称已严格完成跨模型对齐。

## 鲁棒性、失败分析与改进E

固定Clean、dark_mild像素乘.7、dark_strong乘.4、PIL GaussianBlur radius1/2。原分辨率腐蚀后进入模型，annotations不变；暗光clip(float32 RGB*factor,0,255).astype(uint8)截断，不四舍五入、不重压JPEG。全部validation腐蚀已与队友逐像素核对。

A-D验证完成后按FN、FP、鸟群数量、小框数量选择代表图。GT绿色命中/橙色漏检；预测青色命中/红色误检，显示分数。绘图使用score-greedy诊断匹配，报告指标仍用upstream matcher。遮挡/NMS、低对比度、模糊边缘只是待看图核实的原因假设。

E设计规则在test前固定：以五条件validation mean AP50-95从A-D选基础方案；比较strong dark/blur AP，针对较弱条件仅改一项。本次记录选择 D 为基础，弱项为 blur，E 唯一变化是 blur_probability .3→.6。E重新COCO初始化，其余参数和预算不变。`results/improvement_plan.json`记录依据；保留E与基础方案验证结果，不默认E有效。最终冻结方案仍为 D。

A-E验证完成后，以五条件validation mean AP50-95确定最终方案，平分按字母顺序；freeze.json固定方案和所有checkpoint SHA256。然后每模型/条件只评估一次test，输出目录要求全新，拒绝重复覆盖；中断保留痕迹停止。不能根据test反选或重训；最终方案来自freeze，不来自test表。

## CPU计时及限制

复用timing_protocol.json中的队友30张固定图片和三次图片顺序，按stem映射原始jpg；队友png是clean派生副本。计时前加载解码RGB。CPU batch1、4 threads，每repeat先5次warmup，再30图，重复3次。

计时含PIL-to-tensor、normalization、resize、模型、NMS和恢复原坐标；不含磁盘解码、模型加载、绘图、evaluator。输出每次总秒数、每图延迟和FPS；汇总FPS=90/sum(seconds)，不平均单次FPS。

重要：队友CPU为Family6 Model170，本机i9-13980HX为Model183；队友Torch2.15CPU，本环境2.6CUDA（计时用CPU）。相同流程不能消除软硬件差异；严格速度比较需要双方模型在同一机器环境重测，不能把目前FPS差异直接归因于架构。

## 新克隆安装与运行

以下命令从仓库根目录执行。使用 Python 3.12 新建本仓库自己的环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 原实验使用的 CUDA 12.4 构建；需自行具备兼容的 NVIDIA 驱动/GPU
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r faster_rcnn/requirements.txt
python -m faster_rcnn.fetch_evaluator
python -m faster_rcnn.fetch_evaluator --check
```

若只使用CPU，将上面的torch安装命令改为 `python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu`。Linux/macOS激活方式为 `source .venv/bin/activate`，硬件对应的torch构建需自行选择。上述为安装说明；本次没有从零安装所有平台或重现CPU训练，不能据此声称跨平台已验证。依赖版本来自现有实验环境快照。

只查看结果无需数据、权重或训练：打开 `faster_rcnn/results/results.csv`，或安装依赖后用 `FasterRCNN.ipynb` 查看精选表。

默认数据入口固定Kaggle `liugalhigh/beach-bird/versions/3`；本地数据可通过 `--data-root`、环境变量 `FASTER_RCNN_DATA_ROOT`，或 `--data-config faster_rcnn/data_config.example.json` 设置。模板路径相对于配置文件目录，结构为 `images/{train,val,test}` 和 `labels/{train,val,test}`。本地目录必须由提供者确认是v3，路径声明不等于已验证数据身份；可与已有manifest比对。不要提交填入个人路径的配置副本。

```powershell
python -m faster_rcnn.run --help
python -m faster_rcnn.pipeline --help
# 可选：审计和独立功能检查；verify可能下载COCO权重，不训练模型
python -m faster_rcnn.run audit --data-root "data/Beach Bird Detection Dataset v1.0"
python -m faster_rcnn.verify
# 以下是新实验的复现命令，不是查看已有结果的必要步骤
python -m faster_rcnn.run train --experiment A --data-config faster_rcnn/data_config.example.json
python -m faster_rcnn.run robust --experiment A --data-config faster_rcnn/data_config.example.json
python -u -m faster_rcnn.pipeline --data-config faster_rcnn/data_config.example.json
```

train支持A-D；E由pipeline按validation证据生成。完整pipeline会训练、冻结并最终评估test，不要为比较现有结果重复执行；已有输出目录会拒绝覆盖。新克隆不包含checkpoint，`robust/test/timing`必须先具备对应的本地权重，不能仅凭精选结果文件运行。原始冻结记录位于 `results/`，不自动复制到新实验的 `outputs/`。

协作入口的检查命令为 `python -m unittest faster_rcnn.tests.test_submission -v`。本次整理已在临时源码导出目录验证固定wheel获取/校验、CLI入口、五项单元检查和隐藏CUDA后的CPU功能检查；使用的仍是原有实验依赖环境，不是全新安装或所有平台验证，也没有运行训练或数据集test。原始 `outputs/verification.json` 未覆盖；`results/verification.json` 保留原实验记录。

## 产物、时间及提交

- audit.json：split统计、异常、train缩放框分布及环境。
- verification.json、teammate_alignment.json：功能/evaluator/数据一致性证据。
- A-E的config.json、data_manifest.json、history.json、summary.json、best/last.pth。
- best_val_predictions.json，val/test五条件原图坐标预测与CSV/JSON指标。
- failures/：validation失败图、索引、学习曲线。
- improvement_plan.json、freeze.json、cpu_timing.json。
- results.csv/json：区分split，mAP50、mAP50-95、固定点P/R、训练秒数、CPU延迟/FPS、所选epoch和配置。
- status.json与pipeline stdout/stderr：进度和异常traceback。

训练耗时含训练loader和计算，排除验证/保存，不等于完整pipeline墙钟耗时。提交源码ZIP排除权重、输入图片和结果图片，遵守课程25MB限制。

来源：旧CompVis数据/训练代码经独立复制修改；重新设定优化器、缩放、anchors、checkpoint标准和evaluator。模型依赖TorchVision；评估复用固定官方Ultralytics wheel。方法参考Ren et al., Faster R-CNN (NeurIPS2015)、Lin et al., FPN (CVPR2017)。数据：Kai Liu, Beach Bird Detection Dataset v1.0, Kaggle version3。
