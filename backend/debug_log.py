# to see and monitor debug lines; easier for debugging
# always prints to stdout (visible in Vercel's runtime logs). A log FILE is only written when
# CHAT_DEBUG_LOG is set explicitly, or in Colab (/content) -- never on Vercel, whose filesystem is read-only.

import os

_ON_VERCEL = bool(os.environ.get("VERCEL"))
LOG_PATH = os.environ.get("CHAT_DEBUG_LOG") or (
    "/content/chat_debug.log" if os.path.isdir("/content") and not _ON_VERCEL else None
)


def log(message: str) -> None:
    print(message, flush=True)
    if not LOG_PATH or _ON_VERCEL:
        return
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass  # never let logging break the request
