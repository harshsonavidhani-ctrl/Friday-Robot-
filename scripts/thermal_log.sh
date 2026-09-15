#!/bin/bash
# Friday's power/thermal black box. Appends every 2s and fsyncs, so the last
# lines survive a hard power cut. The rsts line says WHY the last boot happened:
# 0x1000 = power-on reset (power was lost), 0x20 = watchdog/reboot.
L=/home/harsh/friday/logs/thermal.log
echo "=== boot $(date '+%F %T')  rsts=$(od -An -tx4 --endian=big /proc/device-tree/chosen/bootloader/rsts 2>/dev/null | tr -d ' ')" >> "$L"
sync "$L"
while :; do
    echo "$(date +%T) $(vcgencmd measure_temp) $(vcgencmd get_throttled) load=$(cut -d' ' -f1 /proc/loadavg) avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB" >> "$L"
    sync "$L"
    sleep 2
done
