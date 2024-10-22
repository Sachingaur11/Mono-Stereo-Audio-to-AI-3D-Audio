from flask import Flask, request, render_template, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import subprocess
from pymongo import MongoClient
from urllib.parse import quote_plus
from gridfs import GridFS
import sys

app = Flask(__name__)

# Configure MongoDB
username = quote_plus("sachingaur")
password = quote_plus("Sachin@1234")
client = MongoClient(f"mongodb+srv://{username}:{password}@cluster0.9xtyu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['audio_database']

# Allowed extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the files
        if 'vo_file' not in request.files or 'sfx_file' not in request.files or 'bgm_file' not in request.files:
            return "No file part", 400

        vo_file = request.files['vo_file']
        sfx_file = request.files['sfx_file']
        bgm_file = request.files['bgm_file']

        # Save files to MongoDB with specific collections
        vo_id = GridFS(db, collection='input_VO_files').put(vo_file, filename=secure_filename(vo_file.filename))
        sfx_id = GridFS(db, collection='input_SFX_files').put(sfx_file, filename=secure_filename(sfx_file.filename))
        bgm_id = GridFS(db, collection='input_BGM_files').put(bgm_file, filename=secure_filename(bgm_file.filename))

        # Wait for all files to upload to MongoDB by checking their presence
        if db['input_VO_files.files'].find_one({'_id': vo_id}) and \
           db['input_SFX_files.files'].find_one({'_id': sfx_id}) and \
           db['input_BGM_files.files'].find_one({'_id': bgm_id}):
            print("All files have been uploaded successfully.")
        else:
            print("Error: One or more files were not uploaded successfully.")
            sys.exit(1)

        # Run the processing script with the MongoDB file IDs
        subprocess.run(['python3', 'run.py', str(sfx_id), str(vo_id), str(bgm_id)])

        # Find the latest output file
        final_file = db['Final_output_files.files'].find_one(sort=[('_id', -1)])
        if final_file:
            output_path = os.path.join('Final', final_file['filename'])
            with open(output_path, "wb") as f:
                f.write(final_file['data'])

            # Return the file name to the client
            return jsonify({'filename': final_file['filename']})

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
