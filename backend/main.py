from flask import Flask, request, jsonify
import cv2
import os
from autogluon.multimodal import MultiModalPredictor
from IPython.display import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def capture_frames(video_path, output_folder='frames'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidObj = cv2.VideoCapture(video_path)
    count = 0
    frame_interval = 1
    fps = int(vidObj.get(cv2.CAP_PROP_FPS))
    frame_time = 1 / fps

    frame_urls = []

    while True:
        success, image = vidObj.read()

        if not success:
            break

        current_time = vidObj.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if current_time >= count * frame_interval:
            frame_name = os.path.join(output_folder, "frame%d.jpg" % count)
            cv2.imwrite(frame_name, image)
            frame_urls.append(frame_name)
            count += 1

    vidObj.release()

    return frame_urls

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        frame_urls = capture_frames(filename)

        predictor = MultiModalPredictor(problem_type="zero_shot_image_classification")

        class_labels = ['Violence', 'Normal']
        result = {'frames': []}

        for frame_url in frame_urls:
            prob = predictor.predict_proba({"image": [frame_url]}, {"text": class_labels})
            violence_prob = prob[0][0]
            result['frames'].append({'url': frame_url, 'probability': violence_prob})
        
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
