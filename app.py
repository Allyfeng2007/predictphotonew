# app.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
import json

# ==== 腾讯云 TTS ====
from tencentcloud.common import credential
from tencentcloud.tts.v20190823 import tts_client, models

app = Flask(__name__)

# 加载 YOLOv8 模型（确保 yolov8n.pt 与 app.py 同目录）
model = YOLO('./yolov8n.pt')


def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    xx1 = max(0, min(w - 1, x1))
    yy1 = max(0, min(h - 1, y1))
    xx2 = max(0, min(w - 1, x2))
    yy2 = max(0, min(h - 1, y2))
    if xx2 <= xx1 or yy2 <= yy1:
        return None
    return img[yy1:yy2, xx1:xx2]


def generate_alert_text(dets, img_w):
    alert = []
    for det in dets:
        cls_name = det["class"]
        x1, y1, x2, y2 = det["box"]
        if cls_name == "traffic light":
            color = det.get("color")
            if color in ("红灯", "绿灯", "黄灯"):
                alert.append(f"前方{color}")
            else:
                alert.append("前方交通灯")
        else:
            cx = (x1 + x2) / 2
            pos = "左侧" if cx < img_w * 0.3 else ("右侧" if cx > img_w * 0.7 else "前方")
            cname_zh = {
                'person': '行人', 'bicycle': '自行车', 'car': '汽车',
                'motorbike': '摩托车', 'bus': '公交车', 'truck': '卡车'
            }.get(cls_name, "障碍物")
            alert.append(f"{pos}有{cname_zh}")
    alert = list(dict.fromkeys(alert))
    return "，".join(alert) + "。" if alert else "未检测到目标"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(silent=True) or {}
        image_data = body.get('image')
        if not image_data:
            return jsonify({'error': '未提供图片数据'}), 400

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
        results = model(img, verbose=False)[0]
        dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det = {"class": cls_name, "box": [x1, y1, x2, y2]}
            if cls_name == 'traffic light':
                roi = safe_crop(img, x1, y1, x2, y2)
                if roi is not None and roi.size > 0:
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask_red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | \
                               cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
                    mask_green = cv2.inRange(hsv, (50, 100, 100), (90, 255, 255))
                    red_count = int(cv2.countNonZero(mask_red))
                    green_count = int(cv2.countNonZero(mask_green))
                    if red_count > 50 and red_count > green_count:
                        det["color"] = "红灯"
                    elif green_count > 50 and green_count > red_count:
                        det["color"] = "绿灯"
                    else:
                        det["color"] = "黄灯"
            dets.append(det)

        text = generate_alert_text(dets, img_w=w)
        return jsonify({'result': text, 'detections': dets})
    except Exception as e:
        return jsonify({'error': f'处理失败: {e}'}), 500


@app.route('/tts', methods=['POST'])
def tts():
    """
    使用腾讯云语音合成，将 text 合成为 MP3（base64）并返回。
    需要在“云托管控制台 → 环境变量”配置：
    - TENCENTCLOUD_SECRET_ID
    - TENCENTCLOUD_SECRET_KEY
    """
    try:
        body = request.get_json(silent=True) or {}
        text = (body.get('text') or '').strip()
        if not text:
            return jsonify({'error': '缺少 text'}), 400

        sid = os.environ.get("TENCENTCLOUD_SECRET_ID")
        sk  = os.environ.get("TENCENTCLOUD_SECRET_KEY")
        if not sid or not sk:
            return jsonify({'error': '未配置腾讯云密钥(TENCENTCLOUD_SECRET_ID/KEY)'}), 500

        cred = credential.Credential(sid, sk)
        client = tts_client.TtsClient(cred, "ap-guangzhou")  # 可换其他地域

        # 音色/参数按需调整；文本过长可分段或截断
        req = models.TextToVoiceRequest()
        params = {
            "Text": text[:500],
            "SessionId": "s1",
            "ModelType": 1,
            "VoiceType": 101001,  # 女声示例，见官方音色表
            "Codec": "mp3",
            "SampleRate": 16000,
            "Speed": 0,
            "Volume": 0
        }
        req.from_json_string(json.dumps(params))
        resp = client.TextToVoice(req)  # resp.Audio 是 base64

        return jsonify({'audio_base64': 'data:audio/mpeg;base64,' + resp.Audio})
    except Exception as e:
        return jsonify({'error': f'TTS失败: {e}'}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'yolo-detection'})


# 调试：列出已注册路由，便于排查 404（上线可删除）
@app.route('/routes', methods=['GET'])
def routes():
    return jsonify(sorted([f"{sorted(r.methods)} {r.rule}" for r in app.url_map.iter_rules()]))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)


