import os
import cv2
from ultralytics import YOLO
import json

# 配置参数
MODEL_PATH = "best.pt"  # 训练好的模型路径
INPUT_DIR = "d:\\语音生成器\\打标素材\\待打标素材"  # 输入图片目录
OUTPUT_DIR = "d:\\语音生成器\\打标素材\\模型打标素材"  # 输出结果目录
CONF_THRESHOLD = 0.25  # 置信度阈值

# 类别映射（与训练时一致）
CLASS_MAPPING = {
    0: "敌人",
    1: "用户",
    2: "未开镜",
    # 添加更多类别
}

print(f"开始批量自动打标")
print(f"模型路径: {MODEL_PATH}")
print(f"输入目录: {INPUT_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"置信度阈值: {CONF_THRESHOLD}")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 检查模型文件是否存在
if not os.path.exists(MODEL_PATH):
    print(f"错误: 模型文件 {MODEL_PATH} 不存在")
    print("请先运行 train.py 训练模型")
    exit(1)

# 加载模型
model = YOLO(MODEL_PATH)
print(f"模型加载完成: {MODEL_PATH}")

# 获取所有jpg图片文件
jpg_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]
print(f"找到 {len(jpg_files)} 个jpg图片文件")

# 统计变量
processed_count = 0
labeled_count = 0
skipped_count = 0

# 遍历所有jpg文件
for jpg_file in jpg_files:
    try:
        jpg_path = os.path.join(INPUT_DIR, jpg_file)
        
        # 读取图片
        img = cv2.imread(jpg_path)
        if img is None:
            skipped_count += 1
            print(f"跳过: {jpg_file} (无法读取图片)")
            continue
        
        # 获取图片尺寸
        img_height, img_width = img.shape[:2]
        
        # 使用模型进行预测
        results = model.predict(
            source=jpg_path,
            conf=CONF_THRESHOLD,
            save=False,
            show=False
        )
        
        # 生成打标结果
        label_data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": jpg_file,
            "imageData": None,
            "imageHeight": img_height,
            "imageWidth": img_width
        }
        
        # 处理预测结果
        labeled = False
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # 获取置信度
                conf = box.conf[0].item()
                # 获取类别ID
                class_id = int(box.cls[0].item())
                
                # 获取类别名称
                class_name = CLASS_MAPPING.get(class_id, f"class_{class_id}")
                
                # 添加到标注数据
                shape = {
                    "label": class_name,
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                label_data["shapes"].append(shape)
                labeled = True
        
        # 保存打标结果
        if labeled:
            json_file = os.path.splitext(jpg_file)[0] + ".json"
            json_path = os.path.join(OUTPUT_DIR, json_file)
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(label_data, f, ensure_ascii=False, indent=2)
            
            # 复制图片到输出目录
            output_jpg_path = os.path.join(OUTPUT_DIR, jpg_file)
            import shutil
            shutil.copy2(jpg_path, output_jpg_path)
            
            labeled_count += 1
            print(f"打标完成: {jpg_file} (检测到 {len(label_data['shapes'])} 个目标)")
        else:
            skipped_count += 1
            print(f"跳过: {jpg_file} (未检测到目标)")
        
        processed_count += 1
        
    except Exception as e:
        skipped_count += 1
        print(f"处理失败: {jpg_file} - {str(e)}")

# 打印统计信息
print("\n批量自动打标完成!")
print(f"总计处理: {processed_count} 个文件")
print(f"成功打标: {labeled_count} 个文件")
print(f"跳过: {skipped_count} 个文件")

if labeled_count > 0:
    print(f"\n打标结果保存至: {OUTPUT_DIR}")
else:
    print("\n未生成任何打标结果")

print("批量自动打标完成！")
