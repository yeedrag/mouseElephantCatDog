from flask import Flask, render_template, url_for, request, send_file, send_from_directory
from ModelBuilder.buildModelClass import modelBuilder
import os
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1


@app.route('/')
def load_page():
    return render_template('index.html')

app.config['model_file'] = os.getcwd()
print('dir',app.config['model_file'])

@app.route('/index', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        jsonData = request.get_json()
        modelBuilder(jsonData["backendOutput"])
    return "this is success"

@app.route('/download', methods=["GET", "POST"])
def download():
    name = 'test.onnx'
    result = send_from_directory(app.config['model_file'], name, as_attachment = True)
    print('download_result',result)
    return result

if __name__ == "__main__":
  app.run(debug=True)