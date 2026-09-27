# to see and monitor debug lines; easier for debugging

import os

LOG_PATH = os.environ.get("CHAT_DEBUG_LOG", "/content/chat_debug.log")


def log(message: str) -> None:
    print(message)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(message + "\n")
    except Exception:
        pass  # never let logging break the request
