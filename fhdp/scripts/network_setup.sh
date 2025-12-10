#!/bin/bash
# FHDP测试平台网络自动化配置脚本
# 支持WireGuard VPN、ZeroTier SD-WAN和网络优化

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置参数
CONFIG_FILE="${HOME}/.fhdp_network_config"
NETWORK_RANGE="10.0.0.0/24"
VPN_PORT="51820"
ZEROTIER_NETWORK=""

# 读取配置
load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE"
        log_info "已加载网络配置文件"
    fi
}

# 保存配置
save_config() {
    cat > "$CONFIG_FILE" <<EOF
# FHDP网络配置文件
NETWORK_RANGE="$NETWORK_RANGE"
VPN_PORT="$VPN_PORT"
ZEROTIER_NETWORK="$ZEROTIER_NETWORK"
WIREGUARD_PRIVATE_KEY="$WIREGUARD_PRIVATE_KEY"
WIREGUARD_PUBLIC_KEY="$WIREGUARD_PUBLIC_KEY"
SERVER_IP="$SERVER_IP"
EOF
    log_success "网络配置已保存到 $CONFIG_FILE"
}

# 检查系统要求
check_system() {
    log_info "检查系统要求..."
    
    # 检查操作系统
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_error "此脚本仅支持Linux系统"
        exit 1
    fi
    
    # 检查权限
    if [[ $EUID -ne 0 ]]; then
        log_error "此脚本需要root权限运行"
        exit 1
    fi
    
    # 检查网络接口
    if ! ip link show | grep -q "eth0\|ens"; then
        log_warning "未检测到标准网络接口"
    fi
    
    log_success "系统检查通过"
}

# 安装网络工具
install_tools() {
    log_info "安装网络工具..."
    
    # 更新包管理器
    apt update
    
    # 安装基础网络工具
    apt install -y \
        wireguard \
        wireguard-tools \
        zerotier-one \
        iptables \
        iproute2 \
        net-tools \
        dnsutils \
        curl \
        wget \
        htop \
        nmap \
        tcpdump \
        mtr
    
    log_success "网络工具安装完成"
}

# 配置WireGuard VPN
setup_wireguard() {
    local role=${1:-"server"}  # server or client
    local server_ip=${2:-""}
    local peer_public_key=${3:-""}
    
    log_info "配置WireGuard VPN ($role)..."
    
    # 生成密钥对
    if [[ ! -f /etc/wireguard/private.key ]]; then
        wg genkey | tee /etc/wireguard/private.key | wg pubkey > /etc/wireguard/public.key
        chmod 600 /etc/wireguard/private.key
    fi
    
    WIREGUARD_PRIVATE_KEY=$(cat /etc/wireguard/private.key)
    WIREGUARD_PUBLIC_KEY=$(cat /etc/wireguard/public.key)
    
    # 获取服务器IP
    if [[ "$role" == "server" ]]; then
        SERVER_IP=$(ip route get 8.8.8.8 | awk '{print $7; exit}')
        log_info "服务器IP: $SERVER_IP"
    fi
    
    # 创建WireGuard配置
    if [[ "$role" == "server" ]]; then
        # 服务器配置
        cat > /etc/wireguard/wg0.conf <<EOF
[Interface]
PrivateKey = $WIREGUARD_PRIVATE_KEY
Address = 10.0.0.1/24
ListenPort = $VPN_PORT
SaveConfig = true

# PostUp和PostDown用于配置NAT
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -t nat -A POSTROUTING -o $(ip route | grep default | awk '{print $5}') -j MASQUERADE
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -t nat -D POSTROUTING -o $(ip route | grep default | awk '{print $5}') -j MASQUERADE
EOF
        
        # 启用IP转发
        sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/' /etc/sysctl.conf
        sysctl -p
        
        log_info "WireGuard服务器配置完成"
        log_info "公钥: $WIREGUARD_PUBLIC_KEY"
        log_info "客户端连接配置:"
        echo "[Peer]"
        echo "PublicKey = $WIREGUARD_PUBLIC_KEY"
        echo "Endpoint = $SERVER_IP:$VPN_PORT"
        echo "AllowedIPs = 10.0.0.0/24"
        
    else
        # 客户端配置
        if [[ -z "$server_ip" || -z "$peer_public_key" ]]; then
            log_error "客户端模式需要服务器IP和公钥参数"
            exit 1
        fi
        
        cat > /etc/wireguard/wg0.conf <<EOF
[Interface]
PrivateKey = $WIREGUARD_PRIVATE_KEY
Address = 10.0.0.2/24
DNS = 8.8.8.8, 8.8.4.4

[Peer]
PublicKey = $peer_public_key
Endpoint = $server_ip:$VPN_PORT
AllowedIPs = 10.0.0.0/24
PersistentKeepalive = 25
EOF
        
        log_info "WireGuard客户端配置完成"
    fi
    
    # 启动WireGuard
    systemctl enable wg-quick@wg0
    systemctl start wg-quick@wg0
    
    # 验证连接
    sleep 5
    wg show
    
    log_success "WireGuard VPN配置完成"
}

# 配置ZeroTier SD-WAN
setup_zerotier() {
    local network_id=${1:-""}
    
    log_info "配置ZeroTier SD-WAN..."
    
    # 启动ZeroTier服务
    systemctl enable zerotier-one
    systemctl start zerotier-one
    
    # 加入网络
    if [[ -n "$network_id" ]]; then
        zerotier-cli join "$network_id"
        log_info "已加入ZeroTier网络: $network_id"
        
        # 显示状态
        sleep 5
        zerotier-cli status
        zerotier-cli info
        
        log_info "请在ZeroTier控制台授权此设备"
    else
        log_warning "未提供网络ID，请手动加入网络"
        log_info "使用命令: zerotier-cli join <network_id>"
    fi
    
    log_success "ZeroTier配置完成"
}

# 配置防火墙规则
setup_firewall() {
    log_info "配置防火墙规则..."
    
    # 重置防火墙
    ufw --force reset
    
    # 默认规则
    ufw default deny incoming
    ufw default allow outgoing
    
    # 允许SSH
    ufw allow 22/tcp comment "SSH"
    
    # 允许WireGuard
    ufw allow $VPN_PORT/udp comment "WireGuard VPN"
    
    # 允许k3s集群通信
    ufw allow 6443/tcp comment "Kubernetes API"
    ufw allow 8472/udp comment "Flannel VXLAN"
    ufw allow 10250/tcp comment "Kubelet API"
    ufw allow 30000:32767/tcp comment "NodePort services"
    
    # 允许监控端口
    ufw allow 9090/tcp comment "Prometheus"
    ufw allow 3000/tcp comment "Grafana"
    ufw allow 3100/tcp comment "Loki"
    ufw allow 9093/tcp comment "Alertmanager"
    
    # 允许V2X仿真端口
    ufw allow 8080/tcp comment "FHDP API"
    ufw allow 8081/tcp comment "V2X Simulation"
    ufw allow 8082/tcp comment "Pipeline Communication"
    
    # 允许WireGuard转发
    ufw route allow in on wg0 out on $(ip route | grep default | awk '{print $5}')
    
    # 启用防火墙
    ufw --force enable
    
    log_success "防火墙配置完成"
    ufw status verbose
}

# 优化网络参数
optimize_network() {
    log_info "优化网络参数..."
    
    # 创建网络优化配置
    cat > /etc/sysctl.d/99-fhdp-network.conf <<'EOF'
# FHDP网络优化参数

# TCP优化
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_congestion_control = bbr

# UDP优化 (V2X通信)
net.core.rmem_default = 262144
net.core.wmem_default = 262144
net.ipv4.udp_rmem_min = 8192
net.ipv4.udp_wmem_min = 8192

# 连接跟踪优化
net.netfilter.nf_conntrack_max = 1048576
net.netfilter.nf_conntrack_udp_timeout = 30
net.netfilter.nf_conntrack_udp_timeout_stream = 180

# 网络缓冲区优化
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_no_metrics_save = 1
net.ipv4.tcp_ecn = 1

# 内存压力优化
vm.swappiness = 10
vm.vfs_cache_pressure = 50
EOF
    
    # 应用配置
    sysctl -p /etc/sysctl.d/99-fhdp-network.conf
    
    log_success "网络参数优化完成"
}

# 配置网络性能监控
setup_network_monitoring() {
    log_info "配置网络性能监控..."
    
    # 创建网络监控脚本
    cat > /usr/local/bin/network-monitor.sh <<'EOF'
#!/bin/bash
# 网络性能监控脚本

METRICS_FILE="/tmp/network_metrics.txt"
INTERVAL=10

# 清空指标文件
> "$METRICS_FILE"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 基础网络指标
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    MEMORY_USAGE=$(free -m | awk 'NR==2{printf "%.1f", $3*100/$2}')
    
    # 网络指标
    RX_BYTES=$(cat /proc/net/dev | grep eth0 | awk '{print $2}')
    TX_BYTES=$(cat /proc/net/dev | grep eth0 | awk '{print $10}')
    RX_PACKETS=$(cat /proc/net/dev | grep eth0 | awk '{print $3}')
    TX_PACKETS=$(cat /proc/net/dev | grep eth0 | awk '{print $11}')
    
    # VPN指标 (如果存在)
    if ip link show wg0 &> /dev/null; then
        VPN_STATUS="UP"
        VPN_RX=$(cat /proc/net/dev | grep wg0 | awk '{print $2}')
        VPN_TX=$(cat /proc/net/dev | grep wg0 | awk '{print $10}')
    else
        VPN_STATUS="DOWN"
        VPN_RX=0
        VPN_TX=0
    fi
    
    # 写入指标
    echo "$TIMESTAMP,CPU:$CPU_USAGE,MEM:$MEMORY_USAGE,RX_BYTES:$RX_BYTES,TX_BYTES:$TX_BYTES,RX_PACKETS:$RX_PACKETS,TX_PACKETS:$TX_PACKETS,VPN:$VPN_STATUS,VPN_RX:$VPN_RX,VPN_TX:$VPN_TX" >> "$METRICS_FILE"
    
    # 保持最近1000条记录
    tail -n 1000 "$METRICS_FILE" > "${METRICS_FILE}.tmp" && mv "${METRICS_FILE}.tmp" "$METRICS_FILE"
    
    sleep $INTERVAL
done
EOF
    
    chmod +x /usr/local/bin/network-monitor.sh
    
    # 创建systemd服务
    cat > /etc/systemd/system/network-monitor.service <<'EOF'
[Unit]
Description=Network Performance Monitor
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/network-monitor.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl enable network-monitor
    systemctl start network-monitor
    
    log_success "网络监控服务已启动"
    log_info "监控数据位置: /tmp/network_metrics.txt"
}

# 配置DHCP服务 (可选)
setup_dhcp_server() {
    log_info "配置DHCP服务器..."
    
    # 安装DHCP服务器
    apt install -y isc-dhcp-server
    
    # 配置网络接口
    cat > /etc/default/isc-dhcp-server <<'EOF'
INTERFACESv4="eth1"
EOF
    
    # 配置DHCP池
    cat > /etc/dhcp/dhcpd.conf <<EOF
# FHDP DHCP服务器配置

default-lease-time 600;
max-lease-time 7200;
option domain-name-servers 8.8.8.8, 8.8.4.4;

subnet 192.168.100.0 netmask 255.255.255.0 {
    range 192.168.100.100 192.168.100.200;
    option routers 192.168.100.1;
    option broadcast-address 192.168.100.255;
    option subnet-mask 255.255.255.0;
}

# Jetson设备预留地址
host jetson-01 {
    hardware ethernet 00:04:4b:01:02:03;
    fixed-address 192.168.100.11;
}

host jetson-02 {
    hardware ethernet 00:04:4b:01:02:04;
    fixed-address 192.168.100.12;
}

host jetson-03 {
    hardware ethernet 00:04:4b:01:02:05;
    fixed-address 192.168.100.13;
}

host jetson-04 {
    hardware ethernet 00:04:4b:01:02:06;
    fixed-address 192.168.100.14;
}
EOF
    
    # 启用IP转发
    sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/' /etc/sysctl.conf
    sysctl -p
    
    # 启动DHCP服务
    systemctl enable isc-dhcp-server
    systemctl restart isc-dhcp-server
    
    log_success "DHCP服务器配置完成"
}

# 生成网络报告
generate_network_report() {
    log_info "生成网络配置报告..."
    
    REPORT_FILE="$HOME/fhdp_network_report.txt"
    
    cat > "$REPORT_FILE" <<EOF
FHDP测试平台网络配置报告
生成时间: $(date)

=== 系统信息 ===
主机名: $(hostname)
操作系统: $(uname -a)
内核版本: $(uname -r)

=== 网络接口 ===
$(ip addr show)

=== 路由表 ===
$(ip route show)

=== 防火墙状态 ===
$(ufw status verbose)

=== WireGuard状态 ===
$(wg show 2>/dev/null || echo "WireGuard未配置")

=== ZeroTier状态 ===
$(zerotier-cli status 2>/dev/null || echo "ZeroTier未配置")

=== 网络连接测试 ===
- 外网连接: $(ping -c 1 8.8.8.8 >/dev/null 2>&1 && echo "正常" || echo "失败")
- DNS解析: $(nslookup google.com >/dev/null 2>&1 && echo "正常" || echo "失败")

=== 监控服务状态 ===
- Network Monitor: $(systemctl is-active network-monitor)
- WireGuard: $(systemctl is-active wg-quick@wg0 2>/dev/null || echo "未启用")
- ZeroTier: $(systemctl is-active zerotier-one 2>/dev/null || echo "未启用")

=== 性能参数 ===
$(sysctl net.core.rmem_max net.core.wmem_max net.ipv4.tcp_rmem net.ipv4.tcp_wmem)

=== 建议配置 ===
1. 确保所有节点在相同网络段
2. 检查防火墙规则是否正确
3. 验证VPN连接稳定性
4. 定期检查网络性能指标

=== 故障排查命令 ===
- 检查网络接口: ip addr show
- 检查路由: ip route show
- 测试连接: ping <target>
- 检查端口: netstat -tlnp
- 监控流量: iftop -i <interface>
- 分析包: tcpdump -i <interface>

EOF
    
    log_success "网络报告已生成: $REPORT_FILE"
    cat "$REPORT_FILE"
}

# 主函数
main() {
    echo "FHDP测试平台网络自动化配置"
    echo "============================="
    echo
    
    # 解析命令行参数
    COMMAND=${1:-"setup"}
    VPN_TYPE=${2:-"wireguard"}
    ROLE=${3:-"server"}
    SERVER_IP=${4:-""}
    PEER_PUBLIC_KEY=${5:-""}
    ZEROTIER_NETWORK_ID=${6:-""}
    
    case "$COMMAND" in
        "setup")
            check_system
            install_tools
            optimize_network
            setup_firewall
            setup_network_monitoring
            
            case "$VPN_TYPE" in
                "wireguard")
                    setup_wireguard "$ROLE" "$SERVER_IP" "$PEER_PUBLIC_KEY"
                    ;;
                "zerotier")
                    setup_zerotier "$ZEROTIER_NETWORK_ID"
                    ;;
                "both")
                    setup_wireguard "$ROLE" "$SERVER_IP" "$PEER_PUBLIC_KEY"
                    setup_zerotier "$ZEROTIER_NETWORK_ID"
                    ;;
                *)
                    log_error "不支持的VPN类型: $VPN_TYPE"
                    exit 1
                    ;;
            esac
            
            save_config
            generate_network_report
            ;;
            
        "dhcp")
            setup_dhcp_server
            ;;
            
        "monitor")
            setup_network_monitoring
            ;;
            
        "report")
            generate_network_report
            ;;
            
        "help")
            echo "用法: $0 [command] [options]"
            echo
            echo "命令:"
            echo "  setup <vpn_type> <role> [server_ip] [peer_key] [zerotier_id]  # 完整网络配置"
            echo "  dhcp                                                        # 配置DHCP服务器"
            echo "  monitor                                                     # 仅配置监控"
            echo "  report                                                      # 生成网络报告"
            echo "  help                                                        # 显示帮助"
            echo
            echo "VPN类型: wireguard, zerotier, both"
            echo "角色: server, client"
            echo
            echo "示例:"
            echo "  $0 setup wireguard server                    # 配置WireGuard服务器"
            echo "  $0 setup wireguard client 192.168.1.100 key  # 配置WireGuard客户端"
            echo "  $0 setup zerotier server 1234567890        # 配置ZeroTier"
            echo "  $0 setup both server                        # 配置两种VPN"
            ;;
            
        *)
            log_error "未知命令: $COMMAND"
            echo "使用 '$0 help' 查看帮助"
            exit 1
            ;;
    esac
    
    log_success "网络配置完成！"
    echo
    log_info "下一步操作:"
    echo "1. 检查网络连接: ping 10.0.0.1"
    echo "2. 查看配置报告: cat $HOME/fhdp_network_report.txt"
    echo "3. 测试VPN连接: wg show 或 zerotier-cli status"
    echo "4. 监控网络性能: tail -f /tmp/network_metrics.txt"
}

# 运行主函数
main "$@"