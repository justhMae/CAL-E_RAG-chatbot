# Vercel Python serverless entrypoint.
# Vercel detects the WSGI `app` object exported here; vercel.json rewrites every route to this function
# (except /api/forms/*, which is served straight from data/forms/ by Vercel's CDN).
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import app  # noqa: E402

__all__ = ["app"]
