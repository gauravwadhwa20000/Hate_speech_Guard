"""
Keep-alive for Render free tier.
Pings the app every 5-14 minutes to prevent it from sleeping.

Usage:
  - As background thread inside Flask:  from keep_alive import start_keep_alive; start_keep_alive()
  - Standalone:                         python keep_alive.py
"""

import os
import time
import random
import threading
import urllib.request
import logging

log = logging.getLogger(__name__)

URL = os.environ.get("RENDER_EXTERNAL_URL", "https://hate-speech-guard.onrender.com/")
MIN_INTERVAL = 5 * 60   # 5 minutes
MAX_INTERVAL = 14 * 60  # 14 minutes


def _ping():
    try:
        req = urllib.request.Request(URL, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            log.info("Keep-alive ping OK – status %s", resp.status)
    except Exception as exc:
        log.warning("Keep-alive ping failed: %s", exc)


def _loop():
    log.info("Keep-alive thread started – pinging %s every %d-%d s", URL, MIN_INTERVAL, MAX_INTERVAL)
    while True:
        _ping()
        time.sleep(random.randint(MIN_INTERVAL, MAX_INTERVAL))


def start_keep_alive():
    """Start the keep-alive pinger as a daemon thread (safe to call once at app startup)."""
    t = threading.Thread(target=_loop, daemon=True)
    t.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _loop()
