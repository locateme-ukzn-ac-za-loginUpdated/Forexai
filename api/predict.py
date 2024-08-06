import json
import joblib
import numpy as np
from http.server import BaseHTTPRequestHandler

model = joblib.load('./model.pkl')

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        
        features = np.array([data['features']])
        prediction = model.predict(features).tolist()
        
        response = {'prediction': prediction}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
