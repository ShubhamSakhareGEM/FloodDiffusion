from flask import Flask, request, jsonify
from transformers import AutoModel
import torch
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

print("[SYSTEM] Booting up Network Server...")
print("[SYSTEM] Loading FloodDiffusionTiny model from Hugging Face...")

# 1. Load the Tiny model (Downloads automatically the first time)
model = AutoModel.from_pretrained(
    "ShandaAI/FloodDiffusionTiny",
    trust_remote_code=True
)

# 2. The M1 Architecture Override
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)
print(f"[SYSTEM] Model loaded successfully onto device: {device}")

@app.route('/api/generate_stream', methods=['POST'])
def stream_motion():
    start_time = time.time()
    
    # 1. Receive Network Request
    data = request.json
    text_prompt = data.get('prompt', 'a person walking forward')
    print(f"\n[NETWORK] Received incoming prompt: '{text_prompt}'")
    
    # 2. Server Processing (Inference)
    # We use length=30 (approx 1 second of motion) to ensure ultra-low latency testing
    # output_joints=True directly bypasses video rendering!
    motion_joints = model(text_prompt, length=30, output_joints=True)
    
    processing_time = (time.time() - start_time) * 1000 # Convert to milliseconds
    
    # 3. Format Network Payload
    payload = {
        "status": "success",
        "latency_ms": round(processing_time, 2),
        "payload_type": "22x3_joint_coordinates",
        "tensor_shape": list(motion_joints.shape), # Will show [30, 22, 3]
        "data": motion_joints.tolist() 
    }
    
    print(f"[NETWORK] Processing complete. Latency: {processing_time:.2f}ms.")
    print("[NETWORK] Transmitting JSON payload to client.")
    
    return jsonify(payload)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)