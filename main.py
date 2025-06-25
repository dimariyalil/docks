import os
import subprocess
import uuid
from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import openai

app = Flask(__name__)

CLIPS_DIR = "./videos/clips"
TRANSCRIPTS_DIR = "./videos/transcripts"
OUTPUT_DIR = "./videos/output"
TEMP_DIR = "./videos/temp"
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/")
def index():
    return "Video API is running!"

def run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout

@app.route('/cut', methods=['POST'])
def cut():
    data = request.json
    input_path = data["input_path"]
    output_dir = data["output_dir"]
    start = data["start"]
    end = data["end"]
    segment_name = data["segment_name"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, segment_name)
    duration = round(end - start, 2)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(round(start, 2)),
        "-i", input_path,
        "-t", str(duration),
        "-c:v", "h264_nvenc",
        "-c:a", "aac",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return jsonify({"error": result.stderr}), 500
    return jsonify({"result": "ok", "output_path": output_path})

@app.route('/analyze', methods=['POST'])
def analyze_transcript():
    text = request.json['text']
    system_prompt = "Ты видеоредактор. Получи транскрипт видео и найди 3 самых интересных момента. Для каждого момента верни time_start и time_end (секунды) и краткое описание."
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    # Лучше вернуть moments как список, если OpenAI вернул JSON-строку
    import json
    try:
        moments = json.loads(response.choices[0].message.content)
    except Exception:
        moments = response.choices[0].message.content
    return jsonify({"moments": moments})

@app.route('/cut-moments', methods=['POST'])
def cut_moments():
    video_path = request.json['video_path']
    moments = request.json['moments']  # [{'time_start':, 'time_end': }]
    clips = []
    for i, m in enumerate(moments):
        out_clip = os.path.join(CLIPS_DIR, f"clip_{i}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(m['time_start']),
            "-to", str(m['time_end']),
            "-c", "copy",
            out_clip
        ]
        run_cmd(cmd)
        clips.append(out_clip)
    return jsonify({"clips": clips})

@app.route('/center-face', methods=['POST'])
def center_face():
    input_clip = request.json['clip_path']
    out_clip = os.path.join(TEMP_DIR, f"centered_{os.path.basename(input_clip)}")
    cap = cv2.VideoCapture(input_clip)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_width, target_height = 720, 1280  # для Shorts/TikTok

    out = cv2.VideoWriter(out_clip, fourcc, fps, (target_width, target_height))
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                cx = int((bbox.xmin + bbox.width / 2) * width)
                cy = int((bbox.ymin + bbox.height / 2) * height)
                x1 = max(0, cx - target_width // 2)
                y1 = max(0, cy - target_height // 2)
                x2 = min(width, x1 + target_width)
                y2 = min(height, y1 + target_height)
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (target_width, target_height))
                out.write(crop)
        else:
            x1 = max(0, width // 2 - target_width // 2)
            y1 = max(0, height // 2 - target_height // 2)
            crop = frame[y1:y1+target_height, x1:x1+target_width]
            crop = cv2.resize(crop, (target_width, target_height))
            out.write(crop)
    cap.release()
    out.release()
    return jsonify({"centered_clip": out_clip})

@app.route('/concat-clips', methods=['POST'])
def concat_clips():
    clips = request.json['clips']            # обязательно
    list_path = os.path.join(TEMP_DIR, "concat_list.txt")
    data        = request.json
    clips       = data['clips']              # обязательно
    output_dir  = data.get('output_dir', OUTPUT_DIR)      # дефолт = /videos/output
    output_name = data.get('output_name', 'final_video.mp4')

    os.makedirs(output_dir, exist_ok=True)                # создаём, как в /cut :contentReference[oaicite:0]{index=0}

    list_path = os.path.join(TEMP_DIR, "concat_list.txt")
    with open(list_path, "w") as f:
        for c in clips:
            f.write(f"file '{os.path.abspath(c)}'\n")
    out_path = os.path.join(OUTPUT_DIR, "final_video.mp4")
    out_path = os.path.join(output_dir, output_name)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        out_path
    ]
    run_cmd(cmd)
    return jsonify({"final_video": out_path})
   

@app.route('/add-subs', methods=['POST'])
def add_subs():
    video_path = request.json['video_path']
    srt_path = request.json['srt_path']
    out_path = os.path.join(OUTPUT_DIR, "subtitled_video.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles={srt_path}",
        out_path
    ]
    run_cmd(cmd)
    return jsonify({"subtitled_video": out_path})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)