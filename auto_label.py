import os
import json
from ultralytics import YOLO

# 模型路径
MODEL_PATH = "runs/train/weights/best.pt"
INPUT_DIR = r"d:\语音生成器\打标素材\待打标素材"
INPUT_DIR = "d:\\语音生成器\\打标素材\\待打标素材"
# 模型打标素材目录（保存结果）
OUTPUT_DIR = "d:\\语音生成器\\打标素材\\模型打标素材"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 类别名称映射
CLASS_NAMES = ['敌人', '用户', '未开镜', '已开镜']

# 加载模型
print(f"加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("模型加载完成！")

# 获取待打标文件列表
image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jpg')]
print(f"发现 {len(image_files)} 个待打标文件")

# 统计变量
total_files = len(image_files)
processed_files = 0
success_count = 0
error_count = 0

print("\n开始自动打标...")

# 处理每个文件
for i, jpg_file in enumerate(image_files):
    jpg_path = os.path.join(INPUT_DIR, jpg_file)
    json_file = jpg_file.replace('.jpg', '.json')
    json_path = os.path.join(OUTPUT_DIR, json_file)
    
    try:
        # 显示进度
        processed_files += 1
        progress = (processed_files / total_files) * 100
        print(f"[{processed_files}/{total_files} ({progress:.1f}%)] 处理: {jpg_file}")
        
        # 使用模型预测
        results = model.predict(source=jpg_path, conf=0.25, imgsz=640)
        
        # 生成标签数据
        label_data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": jpg_file,
            "imageData": None
        }
        
        # 获取图片尺寸
        if results and len(results) > 0:
            result = results[0]
            label_data["imageWidth"] = int(result.orig_shape[1])
            label_data["imageHeight"] = int(result.orig_shape[0])
            
            # 处理预测结果
            if result.boxes:
                for box in result.boxes:
                    # 获取类别和置信度
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # 添加到标签数据
                    shape = {
                        "label": CLASS_NAMES[class_id],
                        "points": [[x1, y1], [x2, y2]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }
                    label_data["shapes"].append(shape)
        
        # 保存JSON标签文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, ensure_ascii=False, indent=2)
        
        # 复制图片到输出目录
        output_jpg_path = os.path.join(OUTPUT_DIR, jpg_file)
        import shutil
        shutil.copy2(jpg_path, output_jpg_path)
        
        success_count += 1
        print(f"  ✓ 成功: 检测到 {len(label_data['shapes'])} 个目标")
        
    except Exception as e:
        error_count += 1
        print(f"  ✗ 错误: {str(e)}")

print("\n自动打标完成！")
print(f"总计: {total_files} 个文件")
print(f"成功: {success_count} 个文件")
print(f"错误: {error_count} 个文件")
print(f"打标结果保存到: {OUTPUT_DIR}")