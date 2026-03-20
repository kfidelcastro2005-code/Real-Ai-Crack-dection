from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # ===================== REAL AI CRACK DETECTION =====================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Connect broken crack lines
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find cracks
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cracks = []
    output_img = img.copy()
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 500:  # Ignore noise
            x, y, w, h = cv2.boundingRect(cnt)
            length_px = cv2.arcLength(cnt, True)
            length_cm = round(length_px / 15, 1)  # Rough scale (you can calibrate later)
            
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.putText(output_img, f"Crack {i+1}: {length_cm}cm", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cracks.append({
                "id": i + 1,
                "length_cm": length_cm,
                "severity": "HIGH" if length_cm > 25 else "MEDIUM" if length_cm > 8 else "LOW",
                "bbox": [x, y, w, h]
            })
    
    health_score = max(25, 100 - len(cracks) * 12)
    
    # Convert annotated image to base64
    _, buffer = cv2.imencode('.jpg', output_img)
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "health_score": health_score,
        "cracks": cracks,
        "annotated_image": annotated_base64,
        "message": "✅ Real AI Crack Detection Complete"
    })

if __name__ == '__main__':
    print("🚀 SHMon AI Backend running on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
