from flask import Flask, render_template, url_for, request
from ModelBuilder.buildModelClass import modelBuilder
app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
@app.route('/index', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        jsonData = request.get_json()
        modelBuilder(jsonData["backendOutput"])

        return {
            'response' : 'I am the response'
        }
    return render_template('index.html')

if __name__ == "__main__":
  app.run(debug=True)