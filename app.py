"""Flask dashboard and inference API."""

from dataclasses import asdict
import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from detector import Detector, ROOT


MAX_TEXT = 4096
MAX_BODY = 64 * 1024
ARTIFACTS = ROOT / "artifacts"
MODELS = {"rules": "Word and phrase rules", "naive_bayes": "Naive Bayes"}
NO_WORDS = "Enter a comment with at least one analyzable word or number."


def create_app(config=None):
    app = Flask(__name__)
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=MAX_BODY,
        ARTIFACT_PATH=ARTIFACTS,
    )
    app.config.update(config or {})
    artifacts = Path(app.config["ARTIFACT_PATH"])
    detector = Detector.load(artifacts / "detector.json")
    metrics = detector.evaluation
    if metrics is None:
        metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    if (not isinstance(metrics, dict) or metrics.get("selected_model") != detector.selected
            or metrics.get("nb_threshold") != detector.threshold):
        raise ValueError("The evaluation and detector artifacts report different model settings")

    @app.after_request
    def disable_cache(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def dashboard():
        return render_template("index.html", metrics=metrics, models=MODELS, max_text=MAX_TEXT)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": detector.selected}

    @app.errorhandler(413)
    def payload_too_large(_error):
        return jsonify(error="The request is too large."), 413

    @app.post("/api/check")
    def check():
        data = request.get_json(silent=True)
        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
            return jsonify(error=f"Enter between 1 and {MAX_TEXT} characters."), 400
        counts = detector.preprocessor.counts(text)
        if not counts:
            return jsonify(error=NO_WORDS), 400
        return jsonify(**asdict(detector.predict_counts(counts)), model=detector.selected, threshold=detector.threshold)

    return app


if __name__ == "__main__":
    from waitress import create_server

    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    server = create_server(create_app(), host=host, port=port, max_request_body_size=MAX_BODY)
    print(f"Toxic Language Detector is ready at http://{host}:{server.effective_port}", flush=True)
    server.run()
