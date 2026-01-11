import os
import sys
import time
import threading
import asyncio
from collections import deque
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template
import pyshark
from flow_features import compute_flow_features_78

MODEL_FOLDER = os.path.join(os.getcwd(), "trained_models")
FLOW_SIZE = 20
MAX_PACKETS_DISPLAY = 200
PRED_THRESHOLD = 0.5
DEBUG = False

print("[+] Loading ML models...")
try:
    rf_model = joblib.load(os.path.join(MODEL_FOLDER, "Random_Forest.pkl"))
    xgb_model = joblib.load(os.path.join(MODEL_FOLDER, "XGBoost.pkl"))
    lgbm_model = joblib.load(os.path.join(MODEL_FOLDER, "LightGBM.pkl"))
    stack_model = joblib.load(os.path.join(MODEL_FOLDER, "Stacking.pkl"))
    scaler = joblib.load(os.path.join(MODEL_FOLDER, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_FOLDER, "feature_columns.pkl"))
except Exception as e:
    print(" Error loading models or artifacts:", e)
    sys.exit(1)

try:
    expected_n_features = getattr(scaler, "n_features_in_", None) or getattr(scaler, "mean_", None).shape[0]
except Exception:
    expected_n_features = None

if expected_n_features is not None and len(feature_columns) != expected_n_features:
    print(f"[!] Warning: feature_columns length ({len(feature_columns)}) != scaler expected features ({expected_n_features}).")
    print("[!] This mismatch will likely produce wrong predictions. Ensure feature_columns.pkl and scaler.pkl were created together.")

print(" Models and feature columns loaded successfully!")

flows = {}
recent_packets = deque(maxlen=MAX_PACKETS_DISPLAY)

def safe_get_layer_port(pkt, layer_name, attr):
    try:
        if hasattr(pkt, layer_name):
            layer = getattr(pkt, layer_name)
            val = getattr(layer, attr, 0)
            return int(val)
    except Exception:
        try:
            return int(getattr(pkt, attr, 0))
        except Exception:
            return 0
    return 0

def get_model_prediction(model, X_scaled):
    try:
        proba = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_scaled)
            if p.ndim == 2 and p.shape[1] >= 2:
                proba = float(p[:, 1][0])
            else:
                proba = float(p.max(axis=1)[0])
        elif hasattr(model, "predict"):
            return int(model.predict(X_scaled)[0])
        if proba is None:
            return int(model.predict(X_scaled)[0])
        return 1 if proba >= PRED_THRESHOLD else 0
    except Exception:
        try:
            return int(model.predict(X_scaled)[0])
        except Exception:
            return 0

def process_packet(pkt, debug=False):
    if not hasattr(pkt, "ip"):
        return

    proto = getattr(pkt, "transport_layer", None)
    if proto not in ("TCP", "UDP"):
        return

    ts = float(getattr(pkt, "sniff_timestamp", time.time()))
    src = pkt.ip.src
    dst = pkt.ip.dst
    length = int(getattr(pkt, "length", 0))
    sport = safe_get_layer_port(pkt, proto, "srcport")
    dport = safe_get_layer_port(pkt, proto, "dstport")
    flags = pkt.tcp.flags if hasattr(pkt, "tcp") else ""
    win_bytes = 0
    try:
        if hasattr(pkt, "tcp") and hasattr(pkt.tcp, "window_size_value"):
            win_bytes = int(pkt.tcp.window_size_value)
        elif hasattr(pkt, "tcp") and hasattr(pkt.tcp, "window_size"):
            win_bytes = int(pkt.tcp.window_size)
    except Exception:
        win_bytes = 0

    flow_key = (src, sport, dst, dport, proto)
    reverse_key = (dst, dport, src, sport, proto)

    pkt_dict = {
        "ts": ts,
        "src": src,
        "dst": dst,
        "length": length,
        "flags": flags,
        "sport": sport,
        "dst_port": dport,
        "win_bytes": win_bytes,
        "dir": "fwd"
    }

    if flow_key in flows:
        flows[flow_key].append(pkt_dict)
    elif reverse_key in flows:
        pkt_dict["dir"] = "bwd"
        flows[reverse_key].append(pkt_dict)
    else:
        flows[flow_key] = [pkt_dict]

    active_key = flow_key if flow_key in flows else reverse_key
    if len(flows[active_key]) >= FLOW_SIZE:
        features_dict = compute_flow_features_78(flows[active_key])
        if not features_dict:
            flows[active_key] = []
            return

        df = pd.DataFrame([features_dict])
        df = df.reindex(columns=feature_columns, fill_value=0)

        if debug or DEBUG:
            print("[DEBUG] Reindexed df columns count:", df.shape[1])
            print(df.iloc[0].head(20).to_dict())

        if expected_n_features is not None and df.shape[1] != expected_n_features:
            if df.shape[1] < expected_n_features:
                for i in range(expected_n_features - df.shape[1]):
                    df[f"_pad_{i}"] = 0
            else:
                df = df.iloc[:, :expected_n_features]

        try:
            df_scaled = scaler.transform(df)
            if df_scaled.ndim == 1:
                df_scaled = df_scaled.reshape(1, -1)
        except Exception:
            preds = {
                "Random Forest": 0,
                "XGBoost": 0,
                "LightGBM": 0,
                "Stacking": 0
            }
        else:
            preds = {
                "Random Forest": get_model_prediction(rf_model, df_scaled),
                "XGBoost": get_model_prediction(xgb_model, df_scaled),
                "LightGBM": get_model_prediction(lgbm_model, df_scaled),
                "Stacking": get_model_prediction(stack_model, df_scaled)
            }

        status = "Attack" if preds.get("Stacking", 0) == 1 else "Normal"
        recent_packets.appendleft({
            "time": time.strftime("%H:%M:%S", time.localtime(ts)),
            "src": src,
            "dst": dst,
            "length": length,
            "predictions": preds,
            "status": status
        })

        flows[active_key] = []

def start_sniffer(interface_name="Wi-Fi", debug=False):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print(f"[+] Starting live capture on interface: {interface_name}")
    try:
        capture = pyshark.LiveCapture(interface=interface_name, display_filter="tcp or udp")
        for pkt in capture.sniff_continuously():
            try:
                process_packet(pkt, debug=debug)
            except Exception as e:
                if debug or DEBUG:
                    print("[DEBUG] packet processing error:", e)
    except Exception as e:
        print(f"[!] Capture error: {e}")

app = Flask(__name__)

@app.route("/")
def home():
    return "<h3> IDS Running — Visit <a href='/dashboard'>/dashboard</a></h3>"

@app.route("/flows")
def get_flows():
    return jsonify(list(recent_packets))

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    threading.Thread(
        target=start_sniffer,
        kwargs={"interface_name": "Wi-Fi", "debug": DEBUG},
        daemon=True
    ).start()
    app.run(host="0.0.0.0", port=5000)
