from flask import Flask, request, Response

app = Flask(__name__)


@app.route("/webhook", methods=["POST"])
def respond():
    print(request.json)  # Handle webhook request here
    return Response(status=200)
