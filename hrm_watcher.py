#!/usr/bin/env python3
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH_DIRS = [
    str(Path.home() / "code"),
    str(Path.home() / "projects"),
    str(Path.home() / "scripts"),
    str(Path.home() / "hrm_flashrenderer"),
    str(Path.home() / "Documents"),
]

INDEX_DIR = "./model_coding"          # your live index
CORPUS_FILE = "./live_coding_corpus.txt"

class HRMWatchHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory or event.src_path.endswith(".git") or event.src_path.endswith(".pyc"):
            return
        print(f"🔄 Change detected: {event.src_path} → rebuilding HRM index...")
        self.rebuild_index()

    def rebuild_index(self):
        try:
            # Rebuild corpus from all watched files (fast)
            with open(CORPUS_FILE, "w", encoding="utf-8") as f:
                for d in WATCH_DIRS:
                    for file in Path(d).rglob("*.py"):
                        try:
                            f.write(f"# FILE: {file}\n")
                            f.write(file.read_text(encoding="utf-8", errors="ignore") + "\n\n")
                        except:
                            pass
                    for file in Path(d).rglob("*.sh"):
                        try:
                            f.write(f"# FILE: {file}\n")
                            f.write(file.read_text(encoding="utf-8", errors="ignore") + "\n\n")
                        except:
                            pass

            # Rebuild HRM index
            subprocess.run(["./hrm_core/build/hrm", "prep", "--input", CORPUS_FILE, "--out", "coding_payloads.jsonl", "--cluster-size", "300"], check=True)
            subprocess.run(["./hrm_core/build/hrm", "build", "--payloads", "coding_payloads.jsonl", "--outdir", INDEX_DIR], check=True)
            print("✅ HRM index updated in realtime")
        except Exception as e:
            print(f"⚠️ Rebuild failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting realtime HRM filewatcher...")
    print("Watching:", WATCH_DIRS)
    print("Index:", INDEX_DIR)
    print("Press Ctrl+C to stop")

    event_handler = HRMWatchHandler()
    observer = Observer()
    for directory in WATCH_DIRS:
        if Path(directory).exists():
            observer.schedule(event_handler, directory, recursive=True)

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
