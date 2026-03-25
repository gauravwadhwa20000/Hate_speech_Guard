"""
Keep-alive script for Render free tier.
Pings the app every 14 minutes to prevent it from sleeping.
Run with: python keep_alive.py
"""

import time
import random
import urllib.request
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

URL = "https://hate-speech-guard.onrender.com/"
MIN_INTERVAL = 5 * 60   # 5 minutes in seconds
MAX_INTERVAL = 14 * 60  # 14 minutes in seconds


def ping():
    try:
        req = urllib.request.Request(URL, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            logging.info("Ping OK – status %s", resp.status)
    except Exception as exc:
        logging.warning("Ping failed: %s", exc)


if __name__ == "__main__":
    logging.info("Keep-alive started – pinging %s every %d-%d seconds", URL, MIN_INTERVAL, MAX_INTERVAL)
    while True:
        ping()
        delay = random.randint(MIN_INTERVAL, MAX_INTERVAL)
        logging.info("Next ping in %d seconds (%.1f min)", delay, delay / 60)
        time.sleep(delay)
