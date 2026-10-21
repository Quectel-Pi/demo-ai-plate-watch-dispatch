#!/usr/bin/env python3
"""网约车调度终端语音库生成：edge-tts 晓晓音色 → assets/voice/*.wav。

板上完全离线播放（只播 wav），TTS 仅在主机生成时联网。
复用健身项目验证过的方案（tools/gen_tts.py 同款链路与参数）。

语音由运行时按 token 拼接，因此车牌逐字生成（省份字 + 字母 + 数字），
短语与车位名单独成文件。字符间静音由 voice_player 播放时插入。

依赖: PYTHONPATH=/home/q/work/temp/tmp/fitness-coach/edge-tts-py
用法: PYTHONPATH=/home/q/work/temp/tmp/fitness-coach/edge-tts-py python3 tools/gen_voice.py
输出: assets/voice/<name>.wav（16bit PCM，paplay 自动重采样）
"""
import subprocess
import sys
import wave
from pathlib import Path

TMP = Path("/home/q/work/temp/tmp/fitness-coach")
VOICE_DIR = Path(__file__).resolve().parent.parent / "assets" / "voice"

VOICE = "zh-CN-XiaoxiaoNeural"
RATE = "+10%"
PITCH = "+6Hz"

# 车牌逐字库
PROVINCES = list("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼")
LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
DIGITS = [str(i) for i in range(10)]

# 短语与车位（文件名 = 键名）
PHRASES = {
    "please_go": "请前往",
    "wait_queue": "车位已满，请在候车队列等待",
    # 到位播报车位需嵌在句中（车牌+已到位+A02+开始接客），故拆成两个词元
    "arrived_verb": "已到位",
    "serve_start": "开始接客",
    "finished": "接客完成，车位已释放",
    "cancelled": "订单已取消",
    "timeout": "已超时未到位，请尽快前往",
    "unregistered": "发现未登记车辆",
    "welcome": "智能车牌识别调度终端已就绪",
}
SPOTS = {"A01": "A零一", "A02": "A零二", "A03": "A零三", "A04": "A零四"}


def trim_silence(samples, threshold=600, pad=200):
    """裁掉头尾静音（edge-tts 单字自带 ~0.7s 停顿，不裁拼接会拖很长）"""
    import array
    a = array.array('h', samples)
    n = len(a)
    i0, i1 = 0, n - 1
    while i0 < n and abs(a[i0]) <= threshold:
        i0 += 1
    while i1 > i0 and abs(a[i1]) <= threshold:
        i1 -= 1
    if i0 >= i1:                      # 全静音（异常保护）
        return samples
    i0 = max(0, i0 - pad)
    i1 = min(n - 1, i1 + pad)
    return a[i0:i1 + 1].tobytes()


def synth(name: str, text: str):
    """edge-tts 出 mp3 -> miniaudio 解码降混单声道 -> 裁静音 -> wav"""
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    mp3 = TMP / f"_voice_{name}.mp3"
    wav = VOICE_DIR / f"{name}.wav"
    subprocess.run(
        [sys.executable, "-m", "edge_tts",
         "--voice", VOICE, "--rate", RATE, "--pitch", PITCH,
         "--text", text, "--write-media", str(mp3)],
        check=True, capture_output=True,
        env={"PYTHONPATH": str(TMP / "edge-tts-py"), "PATH": "/usr/bin:/bin"},
    )
    import miniaudio
    dec = miniaudio.decode_file(str(mp3))
    samples = dec.samples
    if dec.nchannels == 2:
        samples = samples[0::2]
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(dec.sample_rate)
        w.writeframes(trim_silence(samples))
    mp3.unlink()
    print(f"[ok] {name}: {text}")


def main():
    tasks = {}
    for ch in PROVINCES:
        tasks[f"p_{ch}"] = ch
    for ch in LETTERS:
        tasks[f"c_{ch}"] = ch
    for d in DIGITS:
        tasks[f"n_{d}"] = d
    tasks.update(PHRASES)
    tasks.update(SPOTS)
    for name, text in tasks.items():
        if (VOICE_DIR / f"{name}.wav").exists():
            continue
        synth(name, text)
    print(f"完成: {len(tasks)} 条 -> {VOICE_DIR}")


if __name__ == "__main__":
    main()
