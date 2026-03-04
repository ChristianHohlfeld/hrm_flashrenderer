#!/usr/bin/env bash

# Unified script to start the telemetry relay and the C++ integer engine.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[*] Cleaning up old relay processes..."
fuser -k -9 9998/tcp 9999/udp >/dev/null 2>&1 || true

echo "[*] Starting Hardware Telemetry Relay..."
/home/chris/myenv2/bin/python3 "$DIR/telemetry_relay/relay.py" &
RELAY_PID=$!

# Give the relay 2 seconds to bind its ports and become ready
sleep 2

echo "================================================================="
echo "[*] Relay is running."
echo "[*] Access the UI at: http://localhost:9998"
echo "================================================================="
echo ""
echo "[*] Starting Engine (run3.sh) in chat mode..."
echo "[*] Note: Passing any provided arguments to the engine."
echo ""

cd "$DIR"
# Run the engine and pass all user arguments (like --hf deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
./run3.sh --chat "$@"

# Cleanup when the engine exits
echo ""
echo "[*] Engine exited. Shutting down relay..."
kill $RELAY_PID 2>/dev/null || true
echo "[*] Done."
