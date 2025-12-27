#!/bin/bash
# Setup monitoring cron jobs for AI-Stocks
# Usage: bash setup_monitoring_cron.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHON_SERVICE_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PYTHON_SERVICE_ROOT/config"
CRON_FILE="$CONFIG_DIR/ai-stocks-monitoring.cron"

echo "==============================================="
echo "AI-Stocks Monitoring Cron Setup"
echo "==============================================="
echo ""

# Check if cron file exists
if [ ! -f "$CRON_FILE" ]; then
    echo "ERROR: Cron configuration file not found: $CRON_FILE"
    exit 1
fi

echo "Cron file location: $CRON_FILE"
echo ""

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    echo "This script requires sudo privileges to install cron jobs."
    echo ""
    echo "Run with: sudo bash setup_monitoring_cron.sh"
    exit 1
fi

echo "Installing cron jobs..."
echo ""

# Copy cron file to /etc/cron.d/
TARGET_CRON="/etc/cron.d/ai-stocks-monitoring"
cp "$CRON_FILE" "$TARGET_CRON"
chmod 644 "$TARGET_CRON"

echo "✓ Installed cron job to: $TARGET_CRON"
echo ""

# Create monitoring_logs directory if it doesn't exist
mkdir -p "$PYTHON_SERVICE_ROOT/monitoring_logs"
chmod 755 "$PYTHON_SERVICE_ROOT/monitoring_logs"
echo "✓ Created/verified monitoring_logs directory: $PYTHON_SERVICE_ROOT/monitoring_logs"
echo ""

# Verify installation
echo "Verifying installation..."
echo ""

if crontab -l | grep -q "ai-stocks"; then
    echo "✓ Cron jobs found in crontab"
else
    echo "⚠ WARNING: Cron jobs not found in crontab yet (may take a moment to register)"
fi

echo ""
echo "Cron schedule:"
echo "  - Health check:  Every weekday at 6:00 AM"
echo "  - Monitoring:    Every weekday at 6:00 PM"
echo "  - Log retention: Monthly cleanup of logs older than 90 days"
echo ""
echo "Log files: $PYTHON_SERVICE_ROOT/monitoring_logs/"
echo ""

echo "==============================================="
echo "Setup Complete!"
echo "==============================================="
echo ""
echo "Next steps:"
echo "1. Verify the cron job is installed:"
echo "   sudo crontab -l | grep ai-stocks"
echo ""
echo "2. Test health check manually:"
echo "   cd $PYTHON_SERVICE_ROOT"
echo "   python scripts/health_check.py"
echo ""
echo "3. Test monitoring manually:"
echo "   cd $PYTHON_SERVICE_ROOT"
echo "   python scripts/monitor_models.py"
echo ""
echo "4. Monitor cron execution:"
echo "   tail -f $PYTHON_SERVICE_ROOT/monitoring_logs/cron.log"
echo ""
