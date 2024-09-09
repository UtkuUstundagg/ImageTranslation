from flask import Blueprint, render_template, request, jsonify, send_file

from main import getSingleReceiptForFinding, getSingleReceiptForSegmentation, \
    getSingleReceiptForTranslate

views = Blueprint(__name__, "views")


@views.route("/")
def home():
    return render_template("index.html")


@views.route("/translate")
def translate():
    return render_template("translate.html")


@views.route("/ocr")
def ocr():
    return render_template("ocr.html")


@views.route("/receipt")
def receipt():
    return render_template("receipt.html")


@views.route("/segment")
def segment():
    return render_template("segment.html")


@views.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_url = data['image_url']
    translate_lang = data['translate_lang']

    translated_results = getSingleReceiptForTranslate(image_url, translate_lang, "text")

    return jsonify(translated_results)


@views.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.json
    image_url = data.get('image_url')
    translation = data.get('translation')
    fileType = data.get('fileType')
    result_file_path = getSingleReceiptForTranslate(image_url, translation, fileType)

    return send_file(result_file_path, as_attachment=True)


@views.route('/find_receipt', methods=['POST'])
def find_receipt():
    data = request.json
    image_url = data.get('image_url')

    result_file_path = getSingleReceiptForFinding(image_url)

    return send_file(result_file_path, as_attachment=True)


@views.route('/segment_receipt', methods=['POST'])
def segment_receipt():
    data = request.json
    image_url = data.get('image_url')
    method_type = data.get('method_type')

    result_file_path = getSingleReceiptForSegmentation(image_url, method_type)

    return send_file(result_file_path, as_attachment=True)
