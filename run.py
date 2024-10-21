import os
import sys
import time
from pydub import AudioSegment
from pymongo import MongoClient
from urllib.parse import quote_plus
from gridfs import GridFS
import subprocess
from gridfs.errors import NoFile  # Import the correct exception
from bson import ObjectId

# Configure MongoDB
username = quote_plus("sachingaur")
password = quote_plus("Sachin@1234")
client = MongoClient(f"mongodb+srv://{username}:{password}@cluster0.9xtyu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['audio_database']

def get_file_from_mongo(file_id, output_path, collection):
    try:
        file_data = GridFS(db, collection=collection).get(file_id).read()
        with open(output_path, "wb") as f:
            f.write(file_data)
        return output_path
    except NoFile:
        print(f"[ERROR] No file found in collection {collection} with _id {file_id}")
        sys.exit(1)

def merge_audio_files(sfx_path, vo_path, bgm_path, output_file):
    # Load the audio files
    sfx = AudioSegment.from_file(sfx_path)
    vo = AudioSegment.from_file(vo_path)
    bgm = AudioSegment.from_file(bgm_path)

    # Ensure all tracks are the same length by matching the shortest track
    min_duration = min(len(sfx), len(vo), len(bgm))

    # Trim the files to the same length
    sfx = sfx[:min_duration]
    vo = vo[:min_duration]
    bgm = bgm[:min_duration]

    # Adjust volumes if needed (optional)
    sfx = sfx - 10       # Lower volume by 10 dB (optional)
    vo = vo - 5          # Lower volume by 5 dB (optional)
    bgm = bgm - 15       # Lower volume by 15 dB (optional)

    # Overlay the tracks (sfx, vo, bgm)
    combined = bgm.overlay(sfx).overlay(vo)

    # Export the combined audio to a file
    combined.export(output_file, format="mp3")

    print(f"[INFO] Merged file saved as {output_file}")

    # Upload merged file to MongoDB
    with open(output_file, "rb") as f:
        output_data = f.read()
        db['Final_output_files'].insert_one({"filename": os.path.basename(output_file), "data": output_data})

def cleanup_files(*files):
    for file in files:
        try:
            os.remove(file)
            print(f"[INFO] Deleted temporary file: {file}")
        except OSError as e:
            print(f"[WARNING] Could not delete file {file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("[ERROR] Please provide the SFX, VO, and BGM file IDs as arguments.")
        sys.exit(1)

    sfx_id = sys.argv[1]
    vo_id = sys.argv[2]
    bgm_id = sys.argv[3]

    print(f"[INFO] SFX ID: {sfx_id}")
    print(f"[INFO] VO ID: {vo_id}")
    print(f"[INFO] BGM ID: {bgm_id}")

    # Fetch files from MongoDB
    # fileId1 = ObjectId(sfx_id)
    # fileId2 = ObjectId(vo_id)
    fileId3 = ObjectId(bgm_id)
    # sfx_file = get_file_from_mongo(fileId1, "sfx_input.mp3", 'input_SFX_files')
    # vo_file = get_file_from_mongo(fileId2, "vo_input.mp3", 'input_VO_files')
    bgm_file = get_file_from_mongo(fileId3, "bgm_input.mp3", 'input_BGM_files')

    # Run processing script for SFX and wait for it to complete
    sfx_process = subprocess.run(['python3', 'GPT_SFX.py', sfx_id], check=True)
    if sfx_process.returncode == 0:
        # Fetch processed SFX file from MongoDB
        latest_sfx_file = db['output_SFX_files.files'].find_one(sort=[('_id', -1)])
        if latest_sfx_file:
            processed_sfx_file = get_file_from_mongo(latest_sfx_file['_id'], "processed_sfx.mp3", 'output_SFX_files')
        else: 
            print("[ERROR] No processed SFX file found in the database.")
            sys.exit(1)
    else:
        print("[ERROR] SFX processing script failed.")
        sys.exit(1)

    # Run processing script for VO and wait for it to complete
    vo_process = subprocess.run(['python3', 'GPT_VO.py', vo_id], check=True)
    if vo_process.returncode == 0:
        # Fetch processed VO file from MongoDB
        latest_vo_file = db['output_VO_files.files'].find_one(sort=[('_id', -1)])
        if latest_vo_file:
            processed_vo_file = get_file_from_mongo(latest_vo_file['_id'], "processed_vo.mp3", 'output_VO_files')
        else:
            print("[ERROR] No processed VO file found in the database.")
            sys.exit(1)
    else:
        print("[ERROR] VO processing script failed.")
        sys.exit(1)

    # Merge the audio files
    current_time = int(time.time())
    output_file = f"Final_3D_audio_{current_time}.mp3"
    merge_audio_files(processed_sfx_file, processed_vo_file, bgm_file, output_file=output_file)

    print(f"[INFO] Process completed successfully! Final output file: {output_file}")

    # Clean up temporary files
    cleanup_files( bgm_file, processed_sfx_file, processed_vo_file, output_file)
