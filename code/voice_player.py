#!/usr/bin/env python3
"""调度语音播报：assets/voice/ 下的 wav 按词元拼接播放，独立线程串行。

车牌动态 → 逐字文件（省份字 p_/字母 c_/数字 n_）运行时拼接；
短语与车位名单独文件。token 之间插入静音（wave 帧级拼接，无需音频库）。
板上播放链路（健身项目实测同款）: paplay 优先（走 PipeWire 系统路由，
跟随用户选择的输出设备，蓝牙耳机可用）→ aplay 直出 es8389 兜底。
"""
import os
import queue
import subprocess
import threading
import time
import wave
from pathlib import Path

VOICE_DIR = Path(__file__).resolve().parent.parent / "assets" / "voice"
GAP_S = 0.20          # token 间静音秒数（太短听感急促，实测 0.2s 舒适）
PLATE_KEYS = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼") | \
    set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | set("0123456789")


def plate_tokens(plate: str):
    """车牌字符串 → 语音词元列表（省份字 p_ / 字母 c_ / 数字 n_）"""
    out = []
    for ch in plate.upper():
        if ch in PLATE_KEYS:
            if ch.isdigit():
                out.append(f"n_{ch}")
            elif ch.isascii():
                out.append(f"c_{ch}")
            else:
                out.append(f"p_{ch}")
    return out


def _read_frames(path: Path, framerate: int):
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != framerate or w.getnchannels() != 1 \
                or w.getsampwidth() != 2:
            return None          # 采样率不一致时不拼（正常不会发生）
        return w.readframes(w.getnframes())


def build_announce(tokens, out_path: Path):
    """词元列表 → 临时 wav（拼接 + 静音）。返回 True 成功。"""
    framerate = 22050
    frames = []
    for t in tokens:
        f = _read_frames(VOICE_DIR / f"{t}.wav", framerate)
        if f is None:
            # 重读一次取实际采样率（以首个存在文件为准）
            p = VOICE_DIR / f"{t}.wav"
            if not p.exists():
                return False
            with wave.open(str(p), "rb") as w:
                framerate = w.getframerate()
            f = _read_frames(p, framerate)
            if f is None:
                return False
        frames.append(f)
        frames.append(b"\x00\x00" * int(framerate * GAP_S))
    try:
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(framerate)
            w.writeframes(b"".join(frames))
        return True
    except OSError:
        return False


class VoicePlayer:
    """串行播放队列；put() 非阻塞，避免卡 UI 线程"""

    def __init__(self, data_dir=None):
        self.enabled = os.environ.get('PLATE_VOICE', '1') != '0'
        self.q: "queue.Queue[list]" = queue.Queue()
        self._stop = threading.Event()
        self._proc = None
        self._tmp = Path(data_dir or VOICE_DIR.parent.parent / 'data') / '_voice'
        self._tmp.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        # 录制支持：keep=True 保留拼接整句 wav（不删）；on_play 开播回调
        # fn(t_wall, tokens, wav_path)，在 voice 线程触发，调用方自行保证线程安全
        self.keep = False
        self.on_play = None
        if self.enabled:
            threading.Thread(target=self._run, daemon=True).start()

    def say(self, *tokens, on_start=None):
        """入队一条播报：say('p_皖','c_A',...,'please_go','A01')
        on_start: 开播回调（开播前触发；拼接失败也触发，避免 UI 横幅队列卡死）"""
        if not self.enabled or not tokens:
            return
        self.q.put((list(tokens), on_start))

    def announce_plate(self, plate: str, phrase: str, spot: str = None,
                       on_start=None):
        """便捷入口：先车牌（是谁）→ 再指令（做什么）→ 车位（去哪）
        播报顺序 = 真实调度广播惯例: 「苏A53D9P，请前往 A01」"""
        tokens = plate_tokens(plate) + [phrase]
        if spot:
            tokens.append(spot)
        self.say(*tokens, on_start=on_start)

    def announce_arrived(self, plate: str, spot: str = None, on_start=None):
        """到位播报：车位必须嵌在句中，不能落到句尾。
        「苏AQ2L95，已到位，A02，开始接客」
        —— 若用 announce_plate 固定顺序，车位会跟在「开始接客」后面，
        听成「…已到位开始接客 A02」，语义错乱。"""
        tokens = plate_tokens(plate) + ['arrived_verb']
        if spot:
            tokens.append(spot)
        tokens.append('serve_start')
        self.say(*tokens, on_start=on_start)

    def shutdown(self):
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def _try(self, cmd) -> bool:
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._proc.wait(timeout=15)
            return self._proc.returncode == 0
        except Exception:
            return False

    def _su_paplay_cmd(self, wav: str):
        """root 时以会话用户身份 paplay（Pulse 协议拒 root 直连非 root 会话，
        且 q 的 PipeWire 独占声卡令 aplay 也打不开——su 是 root 会话唯一活路）"""
        import glob as _glob
        import pwd
        for udir in sorted(_glob.glob('/run/user/*')):
            if not os.path.exists(os.path.join(udir, 'pulse', 'native')):
                continue
            try:
                uid = int(udir.rsplit('/', 1)[1])
                name = pwd.getpwuid(uid).pw_name
            except (ValueError, KeyError):
                continue
            return ['su', '-', name, '-c',
                    f'XDG_RUNTIME_DIR={udir} paplay {wav}']
        return None

    def _play(self, wav: Path) -> bool:
        # 1) 本用户直连系统路由（跟随输出设备，含蓝牙耳机）
        if self._try(["paplay", str(wav)]):
            return True
        # 2) root 会话：su 会话用户走其 PipeWire
        if os.geteuid() == 0:
            cmd = self._su_paplay_cmd(str(wav))
            if cmd and self._try(cmd):
                return True
        # 3) 最后兜底：ALSA 硬件直出
        return self._try(["aplay", "-q", "-D", "plughw:0,0", str(wav)])

    def _run(self):
        while not self._stop.is_set():
            try:
                tokens, on_start = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            self._seq += 1
            wav = self._tmp / f"ann_{self._seq}.wav"
            if not build_announce(tokens, wav):
                print(f"[voice] WARN: 拼接失败 {tokens}", flush=True)
                if on_start:
                    on_start()          # 失败也要推进 UI 队列
                continue
            if on_start:
                on_start()              # 开播即回调：横幅与语音同步切换
            if self.on_play:
                try:
                    self.on_play(time.time(), tokens, str(wav))
                except Exception:
                    pass
            played = self._play(wav)
            print(f"[voice] {'played' if played else 'no-audio'}: {tokens}",
                  flush=True)
            if not self.keep:
                try:
                    wav.unlink()
                except OSError:
                    pass
