from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO
import cv2, os, json, time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from shapely.geometry import Point, Polygon
from werkzeug.utils import secure_filename
import threading
import uuid

# ---------------- Flask Setup ---------------- #
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app, cors_allowed_origins="*")

ZONES_FILE = 'zones.json'
if not os.path.exists(ZONES_FILE):
    with open(ZONES_FILE, 'w') as f:
        json.dump([], f)

current_source = None
cap = None
frame_lock = threading.Lock()
current_frame = None

# ---------------- Models ---------------- #
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- Auth Routes ---------------- #
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        if User.query.filter((User.username==username)|(User.email==email)).first():
            flash('Username or Email already exists!', 'danger')
            return redirect(url_for('register'))
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login unsuccessful. Check email and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ---------------- Dashboard ---------------- #
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

# ---------------- Video & Analytics ---------------- #
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

def load_zones():
    with open(ZONES_FILE, 'r') as f:
        try:
            return json.load(f)
        except:
            return []

def save_zones(zones):
    with open(ZONES_FILE, 'w') as f:
        json.dump(zones, f, indent=4)

def inside_zone(point, zone):
    try:
        if "points" in zone:
            poly = Polygon(zone["points"])
            return poly.contains(Point(point))
        else:
            poly = Polygon([zone["topleft"], [zone["bottomright"][0], zone["topleft"][1]], zone["bottomright"], [zone["topleft"][0], zone["bottomright"][1]]])
            return poly.contains(Point(point))
    except Exception as e:
        print(f"Error checking zone: {e}")
        return False

def _resolve_source(src):
    if src == "camera" or src == 0:
        return 0
    path = os.path.join(app.config['UPLOAD_FOLDER'], src)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    return path

def generate_frames():
    global current_frame, cap, current_source
    if not current_source:
        yield b''
        return

    cap_src = _resolve_source(current_source)
    if cap_src is None:
        yield b''
        return

    cap = cv2.VideoCapture(cap_src)
    is_file = isinstance(cap_src, str) and os.path.exists(cap_src)
    if not cap.isOpened():
        print(f"Error: Could not open video source {cap_src}")
        yield b''
        return

    # --- Frame Limit Logic ---
    FRAME_LIMIT = 30  # <-- Stops after 30 frames
    frame_counter = 0
    # -------------------------

    while True:
        success, frame = cap.read()
        
        # --- Check for End-of-Stream ---
        if not success:
            if is_file:
                # If it's a file, we stop instead of looping back
                break  
            else:
                # If it's a camera that failed, we break
                break
        
        # --- Frame Counter Stop ---
        if frame_counter >= FRAME_LIMIT:
            print(f"Frame limit of {FRAME_LIMIT} reached. Stopping stream.")
            break 
        # ----------------------------

        results = model(frame)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)
        zones = load_zones()
        zone_counts = {z.get("name", f"zone_{uuid.uuid4().hex}"): 0 for z in zones}
        heatmap_points = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
            heatmap_points.append((cx, cy))
            
            for z in zones:
                if "points" in z and inside_zone((cx, cy), z):
                    zone_counts[z["name"]] = zone_counts.get(z["name"], 0) + 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {track.track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Emit real-time data to the frontend
        socketio.emit("update", {"counts": zone_counts, "heatmap": heatmap_points, "timestamp": time.time()})

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        # --- INCREMENT COUNTER ---
        frame_counter += 1
        # -------------------------

    # --- Clean up after loop breaks ---
    cap.release()
    print("Video stream finished and resources released.")
    # ----------------------------------

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_source', methods=['POST'])
@login_required
def set_source():
    global current_source
    data = request.get_json()
    src = data.get("source")
    if src:
        current_source = src
        return jsonify({"status":"source updated","source":current_source})
    return jsonify({"status":"error", "message": "No source provided"}), 400

@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    if 'video' not in request.files:
        return jsonify({"status": "no file"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "no filename"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    global current_source
    current_source = filename
    return jsonify({"status":"saved","filename":filename})

# ---------------- Zone APIs ---------------- #
@app.route('/save_zone', methods=['POST'])
@login_required
def save_zone():
    data = request.get_json()
    if not data or "name" not in data or "points" not in data:
        return jsonify({"status": "invalid data"}), 400
    
    zones = load_zones()
    if any(z.get("name") == data.get("name") for z in zones):
        return jsonify({"status":"zone exists"}), 400

    zones.append(data)
    save_zones(zones)
    return jsonify({"status":"success"})

@app.route('/preview_zones', methods=['GET'])
@login_required
def preview_zones():
    return jsonify(load_zones())

@app.route('/delete_zone', methods=['POST'])
@login_required
def delete_zone():
    data = request.get_json()
    zones = load_zones()
    zones = [z for z in zones if z.get("name") != data.get('name')]
    save_zones(zones)
    return jsonify({"status":"deleted"})

@app.route('/update_threshold', methods=['POST'])
@login_required
def update_threshold():
    data = request.get_json()
    zone_name = data.get('name')
    new_threshold = data.get('threshold')
    zones = load_zones()
    for z in zones:
        if z.get("name") == zone_name:
            z['threshold'] = int(new_threshold)
            save_zones(zones)
            return jsonify({"status":"threshold updated", "zone": zone_name, "new_threshold": int(new_threshold)})
    return jsonify({"status": "zone not found"}), 404

# ---------------- Main ---------------- #
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)