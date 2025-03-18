# **tshark 使用文档 (Ubuntu 命令行操作)**

`tshark` 是 Wireshark 的命令行版本，可用于捕获和分析网络流量。本文档介绍如何在 Ubuntu 上使用 `tshark` 捕获 `eth0` 适配器的流量，并将结果保存到 `pcap` 文件。

---

## **1. 安装 Tshark**

如果系统未安装 `tshark`，可以使用以下命令安装：

```sh
sudo apt update
sudo apt install tshark -y
```

安装后，验证 `tshark` 版本：

```sh
tshark -v
```

---

## **2. 允许普通用户使用 Tshark**

默认情况下，`tshark` 需要 `root` 权限。如果希望普通用户可以使用，可以运行：

```sh
sudo dpkg-reconfigure wireshark-common
```

选择 **"Yes"** 允许非 root 用户捕获流量。

然后，将当前用户添加到 `wireshark` 组：

```sh
sudo usermod -aG wireshark $(whoami)
newgrp wireshark
```

这样可以避免每次运行 `tshark` 需要 `sudo`。

---

## **3. 查看可用网络接口**

在开始捕获之前，可以查看当前可用的网络接口：

```sh
tshark -D
```

示例输出：

```
1. eth0
2. lo
3. wlan0
```

`eth0` 是我们要监听的网卡。

---

## **4. 捕获 eth0 适配器的流量**

### **4.1 仅在终端显示流量**

如果只是想实时查看 `eth0` 的流量：

```sh
tshark -i eth0
```

### **4.2 将流量保存到 pcap 文件**

如果要将捕获的流量保存到 `capture.pcap` 文件：

```sh
tshark -i eth0 -w capture.pcap
```

---

## **5. 设置捕获时长或数据包数量**

### **5.1 指定捕获数据包数量**

```sh
tshark -i eth0 -c 100 -w capture.pcap
```

### **5.2 限制捕获时间**

```sh
tshark -i eth0 -a duration:60 -w capture.pcap
```

---

## **6. 过滤特定流量**

### **6.1 仅捕获 TCP 流量**

```sh
tshark -i eth0 -w capture.pcap tcp
```

### **6.2 仅捕获来自 192.168.1.1 的流量**

```sh
tshark -i eth0 -w capture.pcap host 192.168.1.1
```

### **6.3 仅捕获 HTTP 流量**

```sh
tshark -i eth0 -w capture.pcap port 80
```

---

## **7. 读取和分析已保存的 pcap 文件**

### **7.1 读取 pcap 文件**

```sh
tshark -r capture.pcap
```

### **7.2 仅显示 HTTP 流量**

```sh
tshark -r capture.pcap -Y "http"
```

### **7.3 提取数据包详细信息**

```sh
tshark -r capture.pcap -V
```

---

## **8. 其他实用选项**

### **8.1 仅显示感兴趣的字段**

```sh
tshark -i eth0 -T fields -e ip.src -e ip.dst -e tcp.port
```

### **8.2 将流量保存为 JSON 格式**

```sh
tshark -i eth0 -T json > capture.json
```

---

## **9. 停止 Tshark**

在终端运行 `tshark` 时，可以使用 `Ctrl + C` 停止捕获。

如果 `tshark` 作为后台进程运行：

```sh
pkill tshark
```

或者：

```sh
killall tshark
```

---

## **10. 总结**

| 任务             | 命令                                              |
| -------------- | ----------------------------------------------- |
| 安装 `tshark`    | `sudo apt install tshark -y`                    |
| 查看网卡接口         | `tshark -D`                                     |
| 监听 `eth0`      | `tshark -i eth0`                                |
| 捕获数据并保存为 pcap  | `tshark -i eth0 -w capture.pcap`                |
| 仅捕获 100 个数据包   | `tshark -i eth0 -c 100 -w capture.pcap`         |
| 仅捕获 60 秒       | `tshark -i eth0 -a duration:60 -w capture.pcap` |
| 仅捕获 TCP 数据包    | `tshark -i eth0 -w capture.pcap tcp`            |
| 读取 pcap 文件     | `tshark -r capture.pcap`                        |
| 提取 HTTP 流量     | `tshark -r capture.pcap -Y "http"`              |
| 仅显示源 IP 和目标 IP | `tshark -i eth0 -T fields -e ip.src -e ip.dst`  |
| 终止 `tshark`    | `Ctrl + C` 或 `pkill tshark`                     |

---

### **结束语**

`tshark` 是强大的命令行网络分析工具，适用于远程服务器、嵌入式设备或无 GUI 环境下的抓包需求。本指南涵盖了最常用的 `tshark` 操作，如有更复杂需求，可以参考：

```sh
man tshark
```

希望本指南能帮到你！🚀