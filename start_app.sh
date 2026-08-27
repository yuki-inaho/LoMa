#!/bin/bash
cd "$(dirname "$0")"
PORT="${1:-7861}"
HOST="${2:-127.0.0.1}"
pkill -f "python app.py --host $HOST --port $PORT" 2>/dev/null
sleep 1
setsid uv run python app.py --host "$HOST" --port "$PORT" \
  </dev/null >app_data/app.log 2>&1 &
disown
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/" || true)
  if [ "$code" = "200" ]; then echo "READY http://$HOST:$PORT/"; exit 0; fi
  sleep 2
done
echo "FAILED to start"; tail -30 app_data/app.log; exit 1
