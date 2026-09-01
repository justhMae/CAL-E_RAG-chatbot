from flask import Flask, request, jsonify, render_template, send_from_directory, abort, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import secrets

from backend.rag_pipeline import answer_question
from backend.config import FORMS_DIR

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)
app.secret_key = secrets.token_hex(16)  # needed for session cookies to work
CORS(app, supports_credentials=True)     # supports_credentials so cookies are sent cross-origin (ngrok)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()

    if not question:
        return jsonify({"error": "message is required"}), 400

    history = session.get("history", [])

    try:
        result = answer_question(question, history)
        session["history"] = result["history"]
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {e}"}), 500


@app.route("/api/chat/reset", methods=["POST"])
def reset_chat():
    session.pop("history", None)
    return jsonify({"status": "reset"})


@app.route("/api/forms/<path:filename>")
def download_form(filename):
    safe_name = secure_filename(filename)
    if safe_name != filename or not (FORMS_DIR / safe_name).is_file():
        abort(404)
    return send_from_directory(FORMS_DIR, safe_name, as_attachment=True)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)