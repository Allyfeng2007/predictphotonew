# app.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
import asyncio
import uuid
import edge_tts

app = Flask(__name__)

# 加载 YOLOv8 模型（确保 yolov8n.pt 与 app.py 同目录）
# 仅 CPU 推理即可满足你的场景
model = YOLO('./yolov8n.pt')


def safe_crop(img, x1, y1, x2, y2):
    """保证裁剪框在图像范围内"""
    h, w = img.shape[:2]
    xx1 = max(0, min(w - 1, x1))
    yy1 = max(0, min(h - 1, y1))
    xx2 = max(0, min(w - 1, x2))
    yy2 = max(0, min(h - 1, y2))
    if xx2 <= xx1 or yy2 <= yy1:
        return None
    return img[yy1:yy2, xx1:xx2]


def generate_alert_text(detections, img_w):
    """根据检测结果生成提示文本（中文）"""
    alert_texts = []
    for det in detections:
        cls_name = det["class"]
        x1, y1, x2, y2 = det["box"]
        if cls_name == 'traffic light':
            # 交通灯颜色识别逻辑在 /predict 内裁剪 ROI 后完成，这里只使用传入的颜色字段
            color = det.get("color")
            if color in ('红灯', '绿灯', '黄灯'):
                alert_texts.append(f"前方{color}")
            else:
                alert_texts.append("前方交通灯")
        else:
            center_x = (x1 + x2) / 2
            pos = "左侧" if center_x < img_w * 0.3 else ("右侧" if center_x > img_w * 0.7 else "前方")
            cname_zh = {
                'person': '行人', 'bicycle': '自行车', 'car': '汽车',
                'motorbike': '摩托车', 'bus': '公交车', 'truck': '卡车'
            }.get(cls_name, "障碍物")
            alert_texts.append(f"{pos}有{cname_zh}")

    # 去重并组合
    alert_texts = list(dict.fromkeys(alert_texts))
    return "，".join(alert_texts) + "。" if alert_texts else "未检测到目标"


@app.route('/predict', methods=['POST'])
def predict():
    """接收小程序上传的图片并返回识别结果（文本 + 检测框）"""
    try:
        body = request.get_json(silent=True) or {}
        image_data = body.get('image')
        if not image_data:
            return jsonify({'error': '未提供图片数据'}), 400

        # Base64解码 -> OpenCV格式
        # 兼容 dataURL 和纯 base64
        if ',' in image_data:
            _, data = image_data.split(',', 1)
        else:
            data = image_data
        img_bytes = base64.b64decode(data)
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': '图片解析失败'}), 400

        h, w = img.shape[:2]

        # YOLOv8 目标检测
        results = model(img, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            det = {"class": cls_name, "box": [x1, y1, x2, y2]}
            # 若为交通灯，做简单颜色判断（红/绿/黄）
            if cls_name == 'traffic light':
                roi = safe_crop(img, x1, y1, x2, y2)
                if roi is not None and roi.size > 0:
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    # 红色可能跨 0° 和 180°：两段阈值
                    mask_red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | \
                               cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
                    mask_green = cv2.inRange(hsv, (50, 100, 100), (90, 255, 255))
                    red_count = cv2.countNonZero(mask_red)
                    green_count = cv2.countNonZero(mask_green)
                    # 简单阈值
                    if red_count > 50 and red_count > green_count:
                        det["color"] = "红灯"
                    elif green_count > 50 and green_count > red_count:
                        det["color"] = "绿灯"
                    else:
                        det["color"] = "黄灯"
            detections.append(det)

        # 生成语音提示文本
        alert_text = generate_alert_text(detections, img_w=w)
        return jsonify({
            'result': alert_text,
            'detections': detections
        })

    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


# ========= TTS：服务端用 edge-tts 合成 MP3，返回 base64 =========
async def synth_to_file(text, voice="zh-CN-XiaoxiaoNeural", rate="+0%"):
    fn = f"/tmp/tts_{uuid.uuid4().hex}.mp3"
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(fn)
    return fn

@app.route('/tts', methods=['POST'])
def tts():
    try:
        body = request.get_json(silent=True) or {}
        text = (body.get('text') or '').strip()
        voice = body.get('voice') or "zh-CN-XiaoxiaoNeural"
        if not text:
            return jsonify({'error': '缺少 text'}), 400

        fn = asyncio.run(synth_to_file(text, voice=voice))
        with open(fn, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        try:
            os.remove(fn)
        except Exception:
            pass

        return jsonify({'audio_base64': 'data:audio/mpeg;base64,' + b64})
    except Exception as e:
        return jsonify({'error': f'TTS失败: {e}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'yolo-detection'})


if __name__ == '__main__':
    # 本地调试时可用；云托管用 gunicorn 启动
    app.run(host='0.0.0.0', port=80, debug=False)
