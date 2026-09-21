import os
"""Flask backend wiring the web UI to detector.py's existing functions.
The UI (static/index.html) uploads a lost-item photo plus candidate photos in
one request; this server saves them to a run folder, calls
detector.compare_images() / detector.describe_image() / detector.make_gradcam(),
and returns the ranked results as JSON. detector.py and chatbot.py are NOT
modified. Run:  python server.py   ->   open http://127.0.0.1:5000
"""
import inspect
import os
import re
import shutil
import threading
import time
import uuid

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import detector  # importing loads ResNet18 once; we only use its functions

app = Flask(__name__)
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(RUNS_DIR, exist_ok=True)
match_lock = threading.Lock()  # detector's model/hooks aren't thread-safe


def unique_name(directory, filename, fallback="upload.jpg"):
    """Sanitize a client filename and make it unique inside directory."""
    name = secure_filename(filename) or fallback
    base, ext = os.path.splitext(name)
    if not ext:
        ext = ".jpg"
    candidate, i = base + ext, 0
    while os.path.exists(os.path.join(directory, candidate)):
        i += 1
        candidate = f"{base}_{i}{ext}"
    return candidate


def cleanup_old_runs(keep=20):
    """Keep only the newest `keep` run folders so runs/ doesn't grow forever."""
    try:
        entries = sorted((os.path.join(RUNS_DIR, d) for d in os.listdir(RUNS_DIR)),
                         key=os.path.getmtime)
        for old in entries[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
    except OSError:
        pass  # cleanup is best-effort; never fail a request because of it


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/model-info")
def model_info():
    """Introspect detector.py's live configuration — no hardcoded duplicates.
    Values are derived from the loaded model object and detector.py's source."""
    m = detector.model
    with match_lock:  # shares the model with /api/match; serialize access
        # architecture from block layout: ResNet18 = 4 stages of 2 BasicBlocks
        blocks = [len(m.layer1), len(m.layer2), len(m.layer3), len(m.layer4)]
        basic = type(m.layer4[0]).__name__ == "BasicBlock"
        arch = "ResNet18" if basic and blocks == [2, 2, 2, 2] else type(m).__name__
        # embedding dim measured from a real forward pass (fc replaced by Identity)
        with torch.no_grad():
            dim = int(m(torch.zeros(1, 3, 224, 224)).shape[-1])
    # Grad-CAM layer parsed from make_gradcam's actual source
    hit = re.search(r"model\.(\w+)\.register_forward_hook",
                    inspect.getsource(detector.make_gradcam))
    gradcam_layer = hit.group(1) if hit else "unknown"
    # similarity metric read from compare_images' source
    metric = ("Cosine" if "cos" in inspect.getsource(detector.compare_images).lower()
              else "Unknown")
    # pretrained weights + training state from detector.py's configuration
    src = inspect.getsource(detector)
    pretrained = "ImageNet" if "IMAGENET1K_V1" in src else "random-init"
    frozen = not any("optim" in k.lower() for k in vars(detector))
    return jsonify({
        "model": f"{arch} ({'frozen' if frozen else 'trainable'}, "
                 f"{pretrained} pretrained)",
        "embedding_dim": dim,
        "similarity_metric": metric,
        "gradcam_layer": gradcam_layer,
        "device": next(m.parameters()).device.type.upper(),
    })


@app.route("/api/match", methods=["POST"])
def api_match():
    target_file = request.files.get("target")
    candidate_files = [f for f in request.files.getlist("candidates")
                       if f and f.filename]
    if target_file is None or not target_file.filename:
        return jsonify({"error": "No lost-item (target) image uploaded."}), 400
    if not candidate_files:
        return jsonify({"error": "No candidate images uploaded."}), 400

    with match_lock:  # serialize runs: single shared model + hook pipeline
        run_id = uuid.uuid4().hex[:10]
        run_dir = os.path.join(RUNS_DIR, run_id)
        os.makedirs(run_dir)

        # save uploads with safe, collision-free names
        target_name = unique_name(run_dir, target_file.filename, "target.jpg")
        target_path = os.path.join(run_dir, target_name)
        target_file.save(target_path)
        candidate_paths = []
        for f in candidate_files:
            cpath = os.path.join(run_dir, unique_name(run_dir, f.filename))
            f.save(cpath)
            candidate_paths.append(cpath)

        try:
            t0 = time.time()
            # reuse detector.py's ranking logic, unchanged
            ranked = detector.compare_images(target_path, candidate_paths)
            best_path, best_pct = ranked[0]

            # --- diagnostics: raw cosine scores + embedding freshness check ---
            # Recompute cosines independently from detector.get_embedding().
            # Because compare_images() embeds every file again on each call, these
            # values must match its percentages (up to rounding); a stale-embedding
            # bug would show up here as a mismatch between the two computations.
            tgt_vec = detector.get_embedding(target_path)
            norm_t = np.linalg.norm(tgt_vec)
            print(f"[match {run_id}] target={target_name} embedding_norm={norm_t:.4f}")
            raw = {}
            for p in candidate_paths:
                v = detector.get_embedding(p)
                cos = float(np.dot(tgt_vec, v) / (norm_t * np.linalg.norm(v)))
                raw[os.path.basename(p)] = cos
            print(f"[match {run_id}] raw cosine scores: "
                  + ", ".join(f"{k}={v:.6f}" for k, v in
                              sorted(raw.items(), key=lambda kv: kv[1], reverse=True)))
            # cross-check: percentages implied by raw cosines vs compare_images
            raw_total = sum(raw.values())
            for p, pct in ranked:
                implied = raw[os.path.basename(p)] / raw_total * 100
                if abs(implied - pct) > 0.05:
                    print(f"[match {run_id}] WARNING: {os.path.basename(p)} "
                          f"compare_images={pct:.2f}% vs recomputed={implied:.2f}%")
            # Grad-CAM needs a live embedding too; verify target vector is nonzero
            # (an all-zero vector would mean a blank/black upload, not a bug)
            if norm_t < 1e-6:
                print(f"[match {run_id}] WARNING: target embedding norm ~0; "
                      "target image may be blank/uniform")

            # same natural-language verdict logic as detector.main()
            lc, ls = detector.describe_image(target_path)
            bc, bs = detector.describe_image(best_path)
            if bc == lc and bs == ls:
                reason = f"{bc} color and {bs} form"
            elif bc == lc:
                reason = f"its {bc} color"
            elif bs == ls:
                reason = f"its {bs} form"
            else:
                reason = "overall visual pattern"
            best_name = os.path.basename(best_path)
            sentence = (f"Your lost item is most likely {best_name}, with "
                        f"{best_pct:.1f}% confidence, based on matching {reason}.")

            # Grad-CAM heatmap for the best match only (written to this run dir)
            gradcam_name = "gradcam.jpg"
            detector.make_gradcam(target_path, best_path,
                                  out_path=os.path.join(run_dir, gradcam_name))
            elapsed_ms = round((time.time() - t0) * 1000, 1)
        except Exception as e:  # unreadable/corrupt image files etc.
            shutil.rmtree(run_dir, ignore_errors=True)
            return jsonify({"error": f"Inference failed: {e}"}), 500

        rankings = [{"name": os.path.basename(p), "percent": round(float(pct), 1),
                     "thumb": f"/runs/{run_id}/{os.path.basename(p)}"}
                    for p, pct in ranked]
        cleanup_old_runs()

    return jsonify({
        "rankings": rankings,
        "sentence": sentence,
        "gradcam": f"/runs/{run_id}/{gradcam_name}",
        "elapsed_ms": elapsed_ms,
        "best_name": best_name,
        "best_percent": round(float(best_pct), 1),
    })


@app.route("/runs/<run_id>/<path:fname>")
def run_file(run_id, fname):
    # serve uploaded thumbnails and the Grad-CAM heatmap for this run
    return send_from_directory(os.path.join(RUNS_DIR, run_id), fname)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

