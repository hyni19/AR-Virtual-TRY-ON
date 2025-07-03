from flask import Flask, render_template, request
import subprocess, os

app = Flask(__name__)

SHIRT_DIR = 'static/images/shirts'
PANT_DIR = 'static/images/pants'
SAREE_DIR = 'static/images/saree'

@app.route('/tryon-room')
def tryon_room():
    shirts = sorted([f for f in os.listdir(SHIRT_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    pants  = sorted([f for f in os.listdir(PANT_DIR)  if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    sarees = sorted([f for f in os.listdir(SAREE_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    return render_template('tryon_room.html', shirts=shirts, pants=pants, sarees=sarees)

@app.route('/')
def index():
    shirts = sorted([f for f in os.listdir(SHIRT_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    pants = sorted([f for f in os.listdir(PANT_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    sarees = sorted([f for f in os.listdir(SAREE_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    return render_template('index.html', shirts=shirts, pants=pants, sarees=sarees)

@app.route('/start-tryon', methods=['POST'])
def start_tryon():
    shirt = request.form.get('shirt', '')
    pant = request.form.get('pant', '')
    saree = request.form.get('saree', '')
    shirt_color = request.form.get('shirt_color', '#ffffff')
    pant_color = request.form.get('pant_color', '#ffffff')
    saree_color = request.form.get('saree_color', '#ffffff')
    args = [
    "python", "test-2.py",
    os.path.join(SHIRT_DIR, shirt) if shirt else "",
    os.path.join(PANT_DIR, pant) if pant else "",
    os.path.join(SAREE_DIR, saree) if saree else "",
    shirt_color, pant_color, saree_color
    ]
    subprocess.Popen(args)
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)