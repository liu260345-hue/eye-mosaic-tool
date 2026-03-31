"""
视频眼部马赛克工具 - Web 版独立启动入口
PyInstaller 打包时以此文件为入口。
"""
import os
import sys
import threading
import webbrowser

# 修正 PyInstaller 打包后的路径
if getattr(sys, "frozen", False):
    # exe 所在目录
    APP_DIR = os.path.dirname(sys.executable)
    # PyInstaller 解压的临时目录
    BASE_DIR = sys._MEIPASS
    # 将模板和静态文件指向打包内的 web 目录
    template_folder = os.path.join(BASE_DIR, "web", "templates")
    static_folder = os.path.join(BASE_DIR, "web", "static")
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = APP_DIR
    template_folder = os.path.join(APP_DIR, "web", "templates")
    static_folder = os.path.join(APP_DIR, "web", "static")

# 将项目根目录加入路径以便导入 eye_mosaic
sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify, send_file, render_template
from pathlib import Path
import uuid

from eye_mosaic import process_video

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

# uploads / outputs 放在 exe 所在目录（用户可见的目录），不放临时目录
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
OUTPUT_DIR = os.path.join(APP_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".ts", ".mts"}

tasks: dict[str, dict] = {}


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _process_task(task_id, input_path, output_path, strength, padding):
    def progress_cb(pct, msg):
        if pct >= 0:
            tasks[task_id]["progress"] = int(pct)
            tasks[task_id]["message"] = msg
        else:
            tasks[task_id]["message"] = msg

    tasks[task_id]["status"] = "processing"
    tasks[task_id]["progress"] = 0

    try:
        success = process_video(
            input_path=input_path,
            output_path=output_path,
            mosaic_strength=strength,
            eye_padding=padding,
            progress_callback=progress_cb,
        )
    except Exception as e:
        print(f"[错误] 处理异常: {e}", flush=True)
        import traceback
        traceback.print_exc()
        success = False
        tasks[task_id]["message"] = f"处理出错: {e}"

    if success:
        tasks[task_id]["status"] = "done"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "处理完成"
    else:
        tasks[task_id]["status"] = "error"
        if not tasks[task_id].get("message"):
            tasks[task_id]["message"] = "处理失败"
        print(f"[错误] 任务 {task_id} 失败: {tasks[task_id]['message']}", flush=True)

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
def progress(task_id):
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
def download(task_id):
    task = tasks.get(task_id)
    if not task or task["status"] != "done":
        return jsonify({"error": "文件不可用"}), 404
    return send_file(
        task["output_path"],
        as_attachment=True,
        download_name=task["output_name"],
    )


if __name__ == "__main__":
    import socket

    # 自动选择可用端口（macOS 12+ 的 AirPlay 默认占用 5000）
    def _find_free_port(preferred=8080):
        for p in [preferred, 8081, 8082, 8888, 0]:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", p))
                    return s.getsockname()[1]
            except OSError:
                continue
        return 8080

    port = _find_free_port()

    print("=" * 50)
    print("  视频眼部马赛克工具 - Web 版")
    print("=" * 50)

    # 启动诊断：检查 FFmpeg 是否可用
    try:
        from eye_mosaic import get_ffmpeg, get_ffprobe
        ffmpeg_path = get_ffmpeg()
        ffprobe_path = get_ffprobe()
        print(f"  FFmpeg:  {ffmpeg_path}")
        print(f"  FFprobe: {ffprobe_path}")
    except Exception as e:
        print(f"  [警告] FFmpeg 未找到: {e}")
        print(f"  处理视频时将尝试自动下载...")

    print(f"\n浏览器将自动打开: http://localhost:{port}")
    print("关闭此窗口即可停止服务\n")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=False)
