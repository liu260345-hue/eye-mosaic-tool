"""
视频眼部马赛克工具 - Web 版
Flask 服务端：处理视频上传、调度处理任务、提供进度查询和文件下载。
"""

import os
import sys
import uuid
import time
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template

# 将桌面版引擎加入路径
TOOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, TOOL_DIR)
from eye_mosaic import process_video

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 允许的视频格式
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".ts", ".mts"}

# 任务状态存储
tasks: dict[str, dict] = {}


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _process_task(task_id: str, input_path: str, output_path: str,
                  strength: int, padding: float):
    """后台线程执行视频处理。"""
    def progress_cb(pct: float, msg: str):
        if pct >= 0:
            tasks[task_id]["progress"] = int(pct)
            tasks[task_id]["message"] = msg
        else:
            tasks[task_id]["message"] = msg

    tasks[task_id]["status"] = "processing"
    tasks[task_id]["progress"] = 0

    success = process_video(
        input_path=input_path,
        output_path=output_path,
        mosaic_strength=strength,
        eye_padding=padding,
        progress_callback=progress_cb,
    )

    if success:
        tasks[task_id]["status"] = "done"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "处理完成"
    else:
        tasks[task_id]["status"] = "error"
        if "message" not in tasks[task_id] or not tasks[task_id]["message"]:
            tasks[task_id]["message"] = "处理失败"

    # 处理完成后删除上传的原始文件
    try:
        os.remove(input_path)
    except OSError:
        pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "未选择文件"}), 400

    file = request.files["file"]
    if not file.filename or not _allowed_file(file.filename):
        return jsonify({"error": "不支持的文件格式"}), 400

    strength = int(request.form.get("strength", 8))
    padding = float(request.form.get("padding", 0.5))
    strength = max(2, min(20, strength))
    padding = max(0.0, min(2.0, padding))

    task_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix
    safe_name = task_id + ext
    input_path = os.path.join(UPLOAD_DIR, safe_name)
    output_path = os.path.join(OUTPUT_DIR, task_id + "_mosaic" + ext)

    file.save(input_path)

    original_name = file.filename
    tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "message": "排队中...",
        "filename": original_name,
        "output_path": output_path,
        "output_name": Path(original_name).stem + "_眼部马赛克" + ext,
    }

    thread = threading.Thread(
        target=_process_task,
        args=(task_id, input_path, output_path, strength, padding),
        daemon=True,
    )
    thread.start()

    return jsonify({"task_id": task_id, "filename": original_name})


@app.route("/progress/<task_id>")
def progress(task_id: str):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify({
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
        "filename": task["filename"],
    })


@app.route("/download/<task_id>")
def download(task_id: str):
    task = tasks.get(task_id)
    if not task or task["status"] != "done":
        return jsonify({"error": "文件不可用"}), 404

    return send_file(
        task["output_path"],
        as_attachment=True,
        download_name=task["output_name"],
    )


if __name__ == "__main__":
    import webbrowser
    port = 5000
    print("视频眼部马赛克工具 - Web 版")
    print(f"打开浏览器访问: http://localhost:{port}")
    # 延迟 1.5 秒后自动打开浏览器
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=False)
