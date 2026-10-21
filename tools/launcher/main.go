// 智能车牌识别调度系统 双击启动器（产物名 plate-recognition，窗口标题仍是中文「智能车牌识别调度系统」）
// 交叉编译: GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o plate-recognition .
// 产物为静态 ELF —— GNOME 文件管理器双击二进制直接运行（无「运行/显示」弹框），
// 通过 /proc/self/exe 定位自身路径，上两级即应用根目录。
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
)

func main() {
	self, err := os.Executable()
	if err != nil {
		fmt.Fprintln(os.Stderr, "[launcher] 找不到自身路径:", err)
		os.Exit(1)
	}
	root := filepath.Dir(filepath.Dir(self)) // <root>/code/智能车牌识别调度系统 -> <root>

	// 单实例: NPU 不能被两个进程同时占用，双击=重启
	// run.sh 以 "python3 -u app.py" 运行（cmdline 无绝对路径），按文件名匹配
	_ = exec.Command("pkill", "-f", "[a]pp.py").Run()

	// 日志落盘（双击启动无终端）
	logDir := filepath.Join(os.Getenv("HOME"), ".local", "state", "plate-rec")
	_ = os.MkdirAll(logDir, 0o755)
	logFile, err := os.OpenFile(filepath.Join(logDir, "app.log"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		fmt.Fprintln(os.Stderr, "[launcher] 打不开日志:", err)
		os.Exit(1)
	}
	defer logFile.Close()

	cmd := exec.Command(filepath.Join(root, "code", "run.sh"), os.Args[1:]...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.Stdin = nil
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true} // 脱离进程组
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(logFile, "[launcher] 启动失败:", err)
		os.Exit(1)
	}
	fmt.Fprintln(logFile, "[launcher] started pid", cmd.Process.Pid)
}
