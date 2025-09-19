# launcher.py
import os
import sys
import time
import socket
import subprocess
from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile
from PySide6.QtGui import QIcon

APP_TITLE = "Memory Graph"
APP_ICON_FILENAME = "app.ico"

# ---------------- Robust single-instance lock ----------------
_LOCK_HANDLE = None
_LOCK_FILE = None

def resource_path(rel_path: str) -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return str(Path(sys._MEIPASS) / rel_path)  # type: ignore[attr-defined]
    return str(Path(__file__).resolve().parent / rel_path)

def acquire_single_instance_lock() -> bool:
    # 1) Windows named mutex if pywin32 is available
    if os.name == "nt":
        try:
            import win32event
            import win32api
            import win32con
            name = "Global\\MemoryGraph_SingleInstance_Lock"
            handle = win32event.CreateMutex(None, False, name)
            last_err = win32api.GetLastError()
            if last_err == win32con.ERROR_ALREADY_EXISTS:
                return False
            globals()["_LOCK_HANDLE"] = handle
            return True
        except Exception:
            pass  # fall back to file lock

    # 2) Cross-platform file lock
    lock_path = Path(resource_path("MemoryGraph.lock"))
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(lock_path, "a+")
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                f.close()
                return False
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                return False
        globals()["_LOCK_FILE"] = f
        return True
    except Exception:
        return True  # don’t block startup on rare errors

def release_single_instance_lock():
    try:
        if _LOCK_FILE:
            if os.name == "nt":
                import msvcrt
                try:
                    _LOCK_FILE.seek(0)
                    msvcrt.locking(_LOCK_FILE.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(_LOCK_FILE.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            try:
                path = Path(resource_path("MemoryGraph.lock"))
                _LOCK_FILE.close()
                path.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        pass

# ---------------- Net helpers ----------------
def find_free_port(start: int = 8501, span: int = 200) -> int:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    for p in range(start, start + span):
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            continue
    raise RuntimeError("No free port available")

def wait_for_server(url: str, timeout: float = 30.0, interval: float = 0.25) -> bool:
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            time.sleep(interval)
    return False

# ---------------- Start Streamlit headless ----------------
def run_streamlit(port: int) -> subprocess.Popen:
    app_dir = Path(__file__).resolve().parent
    python = sys.executable
    cmd = [
        python, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.fileWatcherType", "none",      # avoid auto-reload spawning
        "--browser.serverAddress", "localhost",
        "--browser.gatherUsageStats", "false",
    ]
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(
        cmd,
        cwd=str(app_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=creation,
    )

# ---------------- Qt window hosting the web app ----------------
class MainWindow(QMainWindow):
    def __init__(self, url: str, proc: subprocess.Popen, app_icon: QIcon | None = None):
        super().__init__()
        self._proc = proc
        self.setWindowTitle(APP_TITLE)
        self.resize(1280, 860)

        if app_icon is not None and not app_icon.isNull():
            self.setWindowIcon(app_icon)

        view = QWebEngineView(self)

        # Accept downloads into user's Downloads folder
        profile = QWebEngineProfile.defaultProfile()
        profile.downloadRequested.connect(self._on_download_requested)

        settings = view.settings()
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)

        view.setContextMenuPolicy(Qt.NoContextMenu)
        view.setUrl(QUrl(url))

        self.setCentralWidget(view)

    def _on_download_requested(self, download):
        try:
            downloads_dir = Path.home() / "Downloads"
            downloads_dir.mkdir(parents=True, exist_ok=True)
            # API differences across PySide6 versions
            if hasattr(download, "downloadFileName"):
                fname = Path(download.downloadFileName()).name or "download.bin"
            else:
                fname = "download.bin"
            target = downloads_dir / fname
            if hasattr(download, "setDownloadDirectory") and hasattr(download, "setDownloadFileName"):
                download.setDownloadDirectory(str(downloads_dir))
                download.setDownloadFileName(fname)
            elif hasattr(download, "setPath"):
                download.setPath(str(target))
            download.accept()
        except Exception:
            try:
                download.accept()
            except Exception:
                pass

    def closeEvent(self, event):
        try:
            if self._proc and (self._proc.poll() is None):
                self._proc.terminate()
                for _ in range(15):
                    if self._proc.poll() is not None:
                        break
                    time.sleep(0.2)
                if self._proc.poll() is None:
                    self._proc.kill()
        except Exception:
            pass
        return super().closeEvent(event)

# ---------------- Entry ----------------
def main():
    # Acquire single-instance lock before anything else
    if not acquire_single_instance_lock():
        return

    # Clean conflicting Qt env
    for key in ["QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QTWEBENGINEPROCESS_PATH"]:
        os.environ.pop(key, None)

    # Enable WebGL (with fallbacks) when frozen
    if getattr(sys, "frozen", False):
        flags = ["--ignore-gpu-blocklist", "--enable-webgl"]
        if os.name == "nt":
            flags.append("--use-angle=d3d11")  # try "warp" if d3d11 fails
        else:
            flags.append("--use-gl=swiftshader")  # software fallback for Linux/macOS without GL
        os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", " ".join(flags))
        os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

    port = find_free_port()
    proc = run_streamlit(port)

    url = f"http://172.20.10.2:{port}"
    if not wait_for_server(url, timeout=35.0, interval=0.25):
        try:
            if proc and proc.stdout:
                print(proc.stdout.read().decode("utf-8", errors="ignore"))
        except Exception:
            pass
        release_single_instance_lock()
        raise SystemExit("Failed to start Streamlit server")

    app = QApplication(sys.argv)

    icon_path = resource_path(APP_ICON_FILENAME)
    app_icon = QIcon(icon_path) if Path(icon_path).exists() else QIcon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    win = MainWindow(url, proc, app_icon=app_icon)
    win.show()

    try:
        sys.exit(app.exec())
    finally:
        release_single_instance_lock()

if __name__ == "__main__":
    main()
