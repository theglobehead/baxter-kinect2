import json

from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def main():
    with open("angle_data.txt", 'r') as file:
        content = file.read()
        json_data = json.loads(content)

    return json_data


if __name__ == "__main__":
    app.run(host="192.168.1.152", port=5000, debug=True)
