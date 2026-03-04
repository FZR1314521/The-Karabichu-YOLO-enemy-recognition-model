# 卡拉比丘手机端人物识别模型

## 项目简介

本项目是一个基于YOLO模型的人物识别打标工具，专为卡拉比丘手机端游戏设计。它能够自动检测游戏中的人物并生成标注文件，用于模型训练和数据标注。

## 主要功能

1. **自动打标**：使用YOLO模型自动检测游戏中的人物并生成标注文件
2. **批量处理**：支持批量处理多张图片的打标工作
3. **数据集管理**：提供数据集配置和管理功能

## 技术实现

- 使用OpenCV进行图像处理
- 使用YOLO模型进行人物检测
- 使用Ultralytics YOLO库进行模型推理

## 项目结构

```
卡拉比丘手机端人物识别模型/
├── README.md              # 项目说明文档
├── auto_label.py          # 单张图片打标脚本
├── auto_label_batch.py    # 批量图片打标脚本
├── data.yaml              # 数据集配置文件
└── runs/
    └── train/
        └── weights/
            └── best.pt    # 训练好的YOLO模型
```

## 运行要求

1. **Python 3.7+**
2. **依赖包**：
   - opencv-python
   - numpy
   - ultralytics
   - Pillow

3. **文件要求**：
   - `runs/train/weights/best.pt` - YOLO模型文件（已包含）
   - `data.yaml` - 数据集配置文件（已包含）

## 运行方法

### 单张图片打标

```
python auto_label.py --image path/to/image.jpg
```

### 批量图片打标

```
python auto_label_batch.py --input_dir path/to/images --output_dir path/to/labels
```

## 数据集配置

`data.yaml` 文件包含了数据集的配置信息，包括类别名称、训练集和验证集的路径等。你可以根据实际情况修改此文件。

## 模型训练

如果需要训练自己的模型，请参考以下步骤：

1. 准备数据集，包含带有标注的图片
2. 修改 `data.yaml` 文件，配置数据集路径和类别
3. 使用YOLOv8训练模型
4. 将训练好的模型文件命名为`best.pt`，并放置在`runs/train/weights/`目录下

## 输出格式

打标脚本会生成YOLO格式的标注文件，每个图片对应一个`.txt`文件，格式为：

```
<class_id> <x_center> <y_center> <width> <height>
```

其中：
- `class_id`：类别ID（从0开始）
- `x_center`, `y_center`：目标中心坐标（归一化到0-1）
- `width`, `height`：目标宽度和高度（归一化到0-1）

## 故障排除

- 如果模型检测失败，请确保`best.pt`文件存在且正确
- 如果打标结果不准确，可以尝试调整模型的置信度阈值
- 如果批量处理失败，请检查输入目录和输出目录是否存在

## 许可证

本项目仅供学习和研究使用，请勿用于任何商业用途或违反游戏规则的行为。
