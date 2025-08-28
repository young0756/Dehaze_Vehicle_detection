from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
from enhance import net_enhance
from recognize import yolov5_recognize

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' in request.files:
        # 原图
        file = request.files['image']
        image = Image.open(file.stream)
        result_1 = yolov5_recognize(image)
        buf = io.BytesIO()
        result_1.save(buf, format='JPEG')
        img_str_1 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({'image': img_str_1})
    elif request.is_json and 'image_base64' in request.json:
        # 增强后
        img_data = base64.b64decode(request.json['image_base64'])
        image = Image.open(io.BytesIO(img_data))
        result_0 = yolov5_recognize(image)
        buf = io.BytesIO()
        result_0.save(buf, format='JPEG')
        img_str_0 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({'image': img_str_0})
    else:
        return jsonify({'result': 'No valid images were received'}), 400

@app.route('/enhance', methods=['POST'])
def enhance():
    file = request.files['image']
    image = Image.open(file.stream)
    enhanced_image = net_enhance(image)
    buf = io.BytesIO()
    enhanced_image.save(buf, format='JPEG')
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return jsonify({'image': img_str})

if __name__ == '__main__':
    app.run(debug=True)