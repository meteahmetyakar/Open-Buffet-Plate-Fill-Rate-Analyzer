from flask import Flask, Response, request
import time
import queue
import threading

app = Flask(__name__)

# SSE için bir kuyruğu global olarak tanımla
message_queue = queue.Queue()

# SSE veri akışı
def generate_sse():
    """
    Kuyruktaki mesajları istemcilere gönderir.
    """
    while True:
        try:
            # Kuyruktan bir mesaj alın (mesaj varsa)
            message = message_queue.get(timeout=1)  # 1 saniye bekler, yoksa devam eder
            yield f"data: {message}\n\n"
        except queue.Empty:
            # Kuyruk boşsa tekrar bekler
            continue

@app.route('/stream', methods=['GET'])
def stream():
    """
    SSE istemcilerine veri akışı.
    """
    print("istek istek")
    return Response(generate_sse(), content_type='text/event-stream')

@app.route('/send', methods=['POST'])
def send_message():
    """
    Dış kaynaklardan veri alır ve kuyruk aracılığıyla SSE istemcilerine gönderir.
    """
    data = request.get_json()  # Gelen POST isteğindeki JSON veriyi al
    if not data:
        return {"error": "No data provided"}, 400

    print(data)
    # JSON verisini kuyruğa ekle
    message_queue.put(data)

    return {"status": "Message received and forwarded to SSE clients"}, 200


if __name__ == "__main__":
    # Flask sunucusunu başlat
    app.run(debug=True, threaded=True, host="127.0.0.1", port=5000)
