from flask import Flask, Response, request
import time
import threading

app = Flask(__name__)

# Bir SSE bağlantısı için kullanılacak veri kuyruğu
clients = []

def event_stream():
    """Her istemci için bir jeneratör işlevi."""
    try:
        while True:
            if clients:
                data = clients.pop(0)
                yield f"data: {data}\n\n"
            else:
                time.sleep(0.1)  # Veri yoksa kısa süre bekle
    except GeneratorExit:
        print("Connection closed by client")

@app.route('/stream')
def stream():
    """SSE endpoint."""
    return Response(event_stream(), content_type='text/event-stream')

@app.route('/send', methods=['POST'])
def send_message():
    """Mesajları kuyruğa ekleyen bir endpoint."""
    message = request.form.get('message')
    if message:
        clients.append(message)
        return "Message sent!", 200
    return "No message provided", 400

@app.route('/')
def index():
    """Basit bir istemci arayüzü."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>SSE Demo</title>
    </head>
    <body>
        <h1>Server-Sent Events (SSE) Demo</h1>
        <div id="messages"></div>
        <script>
            const eventSource = new EventSource('/stream');
            eventSource.onmessage = function(event) {
                const messagesDiv = document.getElementById('messages');
                const message = document.createElement('p');
                message.textContent = event.data;
                messagesDiv.appendChild(message);
            };
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
