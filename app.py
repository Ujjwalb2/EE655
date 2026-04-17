from flask import Flask, render_template, request, send_file, jsonify
import os
from model import stylize_image, progress

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        content = request.files["content"]
        style = request.files["style"]

        content_path = os.path.join(UPLOAD_FOLDER, content.filename)
        style_path = os.path.join(UPLOAD_FOLDER, style.filename)

        content.save(content_path)
        style.save(style_path)

        output_path = os.path.join(OUTPUT_FOLDER, "result.jpg")

        stylize_image(content_path, style_path, output_path)

        return send_file(output_path, mimetype='image/jpeg')

    return render_template("index.html")


@app.route("/progress")
def get_progress():
    return jsonify(progress)


if __name__ == "__main__":
    app.run(debug=True)