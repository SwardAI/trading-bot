#!/bin/bash
# Health check script — run via: ssh root@134.122.73.180 -i ~/.ssh/digitalocean 'bash ~/trading/scripts/health_check.sh'
# Or locally on the droplet: bash ~/trading/scripts/health_check.sh

echo "=========================================="
echo "  TRADING BOT HEALTH CHECK"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="

cd ~/trading

# Container status
echo ""
echo "--- CONTAINER STATUS ---"
docker-compose ps 2>/dev/null

# System resources
echo ""
echo "--- SYSTEM RESOURCES ---"
echo "CPU load: $(cat /proc/loadavg | awk '{print $1, $2, $3}')"
echo "Memory:   $(free -h | awk '/^Mem:/ {print $3 "/" $2 " used"}')"
echo "Disk:     $(df -h / | awk 'NR==2 {print $3 "/" $2 " used (" $5 ")"}')"

# Recent logs (last 30 lines)
echo ""
echo "--- RECENT LOGS (last 30 lines) ---"
docker-compose logs --tail=30 2>/dev/null

# Error count in last 100 lines
echo ""
echo "--- ERROR SUMMARY (last 200 lines) ---"
ERROR_COUNT=$(docker-compose logs --tail=200 2>/dev/null | grep -c "ERROR\|CRITICAL")
WARNING_COUNT=$(docker-compose logs --tail=200 2>/dev/null | grep -c "WARNING")
echo "Errors/Critical: $ERROR_COUNT"
echo "Warnings: $WARNING_COUNT"

# Show actual errors if any
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo ""
    echo "--- ERRORS ---"
    docker-compose logs --tail=200 2>/dev/null | grep "ERROR\|CRITICAL"
fi

# Generate report
echo ""
echo "--- PERFORMANCE REPORT ---"
docker exec crypto-bot python scripts/generate_report.py 2>/dev/null

# Uptime
echo ""
echo "--- CONTAINER UPTIME ---"
docker inspect crypto-bot --format='Started: {{.State.StartedAt}}' 2>/dev/null
docker inspect crypto-bot --format='Status: {{.State.Status}}' 2>/dev/null

echo ""
echo "=========================================="
echo "  END HEALTH CHECK"
echo "=========================================="
