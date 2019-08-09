from flask import Flask, request, jsonify, make_response
import json

app = Flask(__name__)

@app.route('/post_data', methods=['GET', 'POST'])
def check():
    if request.method == 'POST':
        print(request.json)
        f = open("pose_action.json", "w")
        json.dump(request.json, f, ensure_ascii=False, indent=2, sort_keys=True, separators=(',', ': '))
        response = { 'status': 'OK' }
        return make_response(jsonify(response))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)
