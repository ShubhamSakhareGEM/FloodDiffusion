from flask import Flask, request, jsonify
from flask_sock import Sock
from transformers import AutoModel
import torch
import time
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
sock = Sock(app) # Initialize WebSocket support

print("[SYSTEM] Booting up Network Server...")
print("[SYSTEM] Loading FloodDiffusionTiny model from Hugging Face...")

# 1. Load the model
model = AutoModel.from_pretrained(
    "ShandaAI/FloodDiffusionTiny",
    trust_remote_code=True
)

# 2. M1 Architecture Override
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)
print(f"[SYSTEM] Model loaded successfully onto device: {device}")


# --- THE NEW WEBSOCKET PIPELINE ---
@sock.route('/api/generate_stream')
def stream_motion(ws):
    print("\n[NETWORK] 🟢 WebSocket Connection Opened! Client connected.")
    
    # Keep the connection open forever
    while True:
        try:
            # 1. Wait for the live prompt from the client's text box
            raw_data = ws.receive() 
            if raw_data is None:
                continue
                
            data = json.loads(raw_data)
            text_prompt = data.get('prompt', '')
            print(f"[NETWORK] Live Prompt Received: '{text_prompt}'")
            
            start_time = time.time()
            
            # 2. Server Processing (Inference)
            motion_joints = model(text_prompt, length=15, output_joints=True)
            processing_time = (time.time() - start_time) * 1000 
            
            # 3. Format Network Payload
            payload = {
                "status": "success",
                "latency_ms": round(processing_time, 2),
                "tensor_shape": list(motion_joints.shape),
                "data": motion_joints.tolist() 
            }
            
            # 4. Push data back through the pipe instantly!
            ws.send(json.dumps(payload))
            print(f"[NETWORK] ⚡ Streamed 30 frames to client in {processing_time:.2f}ms")
            
        except Exception as e:
            print(f"[NETWORK] 🔴 WebSocket Error or Disconnect: {e}")
            break

if __name__ == '__main__':
    # Flask-Sock runs perfectly on the standard app.run
    app.run(host='0.0.0.0', port=8080, debug=False)