from flask import Flask, request, jsonify
from flask_cors import CORS
from first import NervousnessDetector
import base64
import cv2
import numpy as np
import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import time

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

app = Flask(__name__)
CORS(app)

detector = NervousnessDetector(
    nervousness_threshold=0.005,
    save_dir="nervous_captures",
    buffer_size=40
)

# Track recent peaks per session to avoid duplicate uploads
session_peaks = {}
session_last_upload = {}  # Track last upload time per session
UPLOAD_COOLDOWN = 5  # Minimum seconds between uploads for same question

def upload_frame_to_cloudinary(frame, session_id, score, question_index):
    """Upload nervous frame to Cloudinary and return secure URL"""
    try:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        result = cloudinary.uploader.upload(
            buffer.tobytes(),
            folder=f"interview_highlights/{session_id}",
            public_id=f"q{question_index}_score{int(score*100)}_{int(np.random.rand()*1000)}",
            resource_type="image",
            overwrite=False,
            transformation=[
                {'width': 640, 'height': 480, 'crop': 'limit'},
                {'quality': 'auto:good'}
            ]
        )
        
        print(f"✅ Uploaded to Cloudinary: {result['secure_url']}")
        return result["secure_url"]
        
    except Exception as e:
        print(f"❌ Cloudinary upload failed: {str(e)}")
        return None


@app.post("/analyze-frame")
def analyze_frame():
    """Analyze frame for nervousness and upload to Cloudinary if threshold exceeded"""
    data = request.json
    frame_b64 = data.get("frame")
    session_id = data.get("sessionId")
    question_index = data.get("questionIndex", 0)

    if not frame_b64 or not session_id:
        return jsonify({"error": "Missing frame or sessionId"}), 400

    try:
        # Decode base64 frame
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",")[1]
        
        try:
            img_bytes = base64.b64decode(frame_b64)
        except Exception:
            return jsonify({"imageUrl": None}), 200

        if not img_bytes or len(img_bytes) < 100:   # 🔴 CRITICAL CHECK
            return jsonify({"imageUrl": None}), 200

        np_arr = np.frombuffer(img_bytes, np.uint8)

        if np_arr.size == 0:                         # 🔴 CRITICAL CHECK
            return jsonify({"imageUrl": None}), 200

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"imageUrl": None}), 200


        if frame is None or frame.size == 0:
            return jsonify({"error": "Failed to decode frame"}), 400

        # Analyze nervousness
        try:
            result = detector.analyze_frame_for_api(frame)
            print(
                f"[ANALYZE] nervous={result['nervous']} | "
                f"score={result['score']:.3f} | "
                f"conf={result['confidence']:.3f} | "
                f"q={question_index}"
            )

        except Exception as e:
            print("❌ Detector crash:", e)
            return jsonify({
                "nervous": False,
                "score": 0.0,
                "confidence": 0.0,
                "imageUrl": None
            }), 200


        nervous = result["nervous"]
        score = result["score"]
        confidence = result["confidence"]
        image_base64 = result["imageBase64"]

        
        # nervous = score > detector.threshold
        image_url = None
        
        # Upload to Cloudinary if nervous and confident
        if image_base64 and nervous and confidence > 0.1:
            # Check if this is a new peak for this session
            session_key = f"{session_id}_{question_index}"
            current_time = time.time()
            
            if session_key not in session_peaks:
                session_peaks[session_key] = {"score": 0, "uploaded": False}
                session_last_upload[session_key] = 0
            
            # Check cooldown - prevent uploads within 5 seconds for same question
            time_since_last_upload = current_time - session_last_upload[session_key]
            
            # Only upload if this is a higher score AND cooldown passed
            if score > session_peaks[session_key]["score"] and time_since_last_upload >= UPLOAD_COOLDOWN:
                image_url = upload_frame_to_cloudinary(
                    frame, 
                    session_id, 
                    score, 
                    question_index
                )
                
                if image_url:
                    session_peaks[session_key] = {
                        "score": score,
                        "uploaded": True,
                        "url": image_url
                    }
                    session_last_upload[session_key] = current_time
                    print(f"🎯 New peak for Q{question_index}: {score:.2f} (uploaded after {time_since_last_upload:.1f}s)")
            else:
                if time_since_last_upload < UPLOAD_COOLDOWN:
                    print(f"⏳ Cooldown active for Q{question_index} ({UPLOAD_COOLDOWN - time_since_last_upload:.1f}s remaining)")
                else:
                    print(f"📊 Score {score:.2f} not higher than peak {session_peaks[session_key]['score']:.2f}")

        return jsonify({
            "nervous": bool(nervous),
            "score": float(score),
            "confidence": float(confidence),
            "imageUrl": image_url,  # Will be None if not uploaded
            
        })

    except Exception as e:
        print(f"❌ Error analyzing frame: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.post("/cleanup-session")
def cleanup_session():
    """Clean up session data after interview ends"""
    data = request.json
    session_id = data.get("sessionId")
    
    if session_id:
        # Remove session peaks from memory
        keys_to_remove = [k for k in session_peaks.keys() if k.startswith(session_id)]
        for key in keys_to_remove:
            del session_peaks[key]
            if key in session_last_upload:
                del session_last_upload[key]
        
        print(f"🧹 Cleaned up session: {session_id}")
        return jsonify({"success": True, "message": "Session cleaned up"})
    
    return jsonify({"error": "No sessionId provided"}), 400


@app.get("/health")
def health():
    return jsonify({
        "status": "OK", 
        "detector": "ready",
        "cloudinary": "configured" if cloudinary.config().cloud_name else "not configured"
    })


@app.post("/test-cloudinary")
def test_cloudinary():
    """Test endpoint to verify Cloudinary upload works"""
    try:
        # Create a simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:, :] = [0, 0, 255]  # Red in BGR
        
        # Upload to Cloudinary
        _, buffer = cv2.imencode('.jpg', test_img)
        
        result = cloudinary.uploader.upload(
            buffer.tobytes(),
            folder="test_uploads",
            resource_type="image"
        )
        
        return jsonify({
            "status": "OK",
            "message": "Cloudinary upload successful",
            "url": result["secure_url"]
        })
        
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    print("🚀 Nervousness Detection Server Starting...")
    print(f"📁 Save directory: {os.path.abspath(detector.save_dir)}")
    print(f"☁️  Cloudinary configured: {cloudinary.config().cloud_name or 'NOT CONFIGURED'}")
    app.run(host="127.0.0.1", port=5050, debug=True)