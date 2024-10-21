from pydub import AudioSegment
import soundfile as sf
import numpy as np
import requests
import json
import os
import sys
from pymongo import MongoClient
from urllib.parse import quote_plus
from gridfs import GridFS
from gridfs.errors import NoFile
import time
from scipy.io import wavfile
from scipy.signal import spectrogram
from bson import ObjectId

AUDIO_FORMATS = {'.mp3', '.mp4', '.wav'}
SPEED_OF_SOUND = 343  
MAX_DELAY_SEC = 0.001  
ATTENUATION_MIN = 0.15
ATTENUATION_MAX = 0.7
ATTENUATION_DISTANCE_FACTOR = 4

# Your OpenAI API key
api_key = "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"

# Encode the username and password
username = quote_plus("sachingaur")
password = quote_plus("Sachin@1234")

# Use the encoded username and password in the connection string
client = MongoClient(f"mongodb+srv://{username}:{password}@cluster0.9xtyu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&ssl=true")
db = client['audio_database']
fs = GridFS(db)
input_collection = db['input_SFX_files.files']
output_collection = db['output_SFX_files']

# Convert audio channels to stereo if necessary
def convert_audio_channels(input_file, output_file, target_channels=2):
    print("[INFO] Checking audio channels...")
    audio = AudioSegment.from_file(input_file)
    
    if audio.channels != target_channels:
        print(f"[INFO] Converting audio to {target_channels}-channel...")
        channels_audio = audio.set_channels(target_channels)
        channels_audio.export(output_file, format="wav")
        print(f"[INFO] {target_channels}-channel audio saved as: {output_file}")
        return output_file
    else:
        print(f"[INFO] Audio is already {target_channels}-channel.")
        return input_file

# Convert audio file format
def convert_audio_format(input_file, target_format=".wav"):
    print(f"[INFO] Converting {input_file} to {target_format} format...")
    file_name, file_extension = os.path.splitext(input_file)
    
    if file_extension.lower() not in AUDIO_FORMATS:
        raise ValueError(f"[ERROR] Unsupported file format: {file_extension}.")
    
    audio = AudioSegment.from_file(input_file)
    output_file = f"{file_name}{target_format}"
    audio.export(output_file, format=target_format.strip('.'))
    print(f"[INFO] File successfully converted to: {output_file}")
    return output_file


def find_audio_peaks_using_spectrogram(audio_file, min_interval_sec=6, threshold=20):
    print("[INFO] Analyzing audio to find peaks using the spectrogram...")

    # Read the audio file
    sample_rate, data = wavfile.read(audio_file)
    
    # If stereo, take only one channel
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Generate the spectrogram
    frequencies, times, Sxx = spectrogram(data, sample_rate)
    
    # Normalize the spectrogram and apply thresholding
    Sxx_log = 10 * np.log10(Sxx)
    Sxx_log = np.where(Sxx_log < threshold, 0, Sxx_log)
    
    # Identify time frames where audio exceeds the threshold
    is_significant = np.any(Sxx_log > 0, axis=0)
    
    # Find continuous segments of significant sound
    significant_times = times[np.where(is_significant)] * 1000  # Convert to milliseconds
    
    # Determine start and end times for each peak interval
    peaks = []
    last_end_time = None

    for start_time in significant_times:
        if last_end_time is None or (start_time - last_end_time) >= (min_interval_sec * 1000):
            if last_end_time is not None:
                peaks.append((last_end_time, start_time))
            last_end_time = start_time

    # Add the last interval to the end of the audio duration if it exists
    if last_end_time is not None:
        audio_duration = len(data) / sample_rate * 1000  # Convert to milliseconds
        peaks.append((last_end_time, audio_duration))

    print(f"[INFO] Found {len(peaks)} peaks in the audio with a minimum interval of {min_interval_sec} seconds.")
    
    return peaks



def generate_gpt_3d_positions(peaks, system_prompt):
    print("[INFO] Sending onset data to GPT for 3D position generation...")
    
    gpt_positions = []
    batch_size = 30  # Split requests into batches of 30 onsets

    for i in range(0, len(peaks), batch_size):
        batch_peaks = peaks[i:i+batch_size]
        onset_data = [{"start_time": start, "end_time": end} for start, end in batch_peaks]
        
        script = f"Given the following onset times in milliseconds: {onset_data}, please generate 3D positions with azimuth, elevation, and distance for each onset."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": script}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()  # Raise error for bad status codes
            result = response.json()

            # Debug: Check the GPT response
            print(f"[DEBUG] GPT Response Content: {result}")

            if 'choices' in result and len(result['choices']) > 0:
                gpt_response = result['choices'][0]['message']['content']
                
                try:
                    # Parse the GPT response
                    batch_positions = json.loads(gpt_response)  # Expecting GPT to return JSON-like positions
                    gpt_positions.extend(batch_positions)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to decode GPT response as JSON: {e}")
                    print(f"[DEBUG] GPT response was: {gpt_response}")
                    return None
            else:
                raise ValueError("Unexpected response format from GPT.")
        
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] API request failed: {e}")
            return None

    return gpt_positions



# Use GPT-generated positions to process 3D audio
def generate_location_data_gpt(peaks, system_prompt):
    # Get positions from GPT
    gpt_positions = generate_gpt_3d_positions(peaks, system_prompt)
    
    if gpt_positions is None:
        print("[ERROR] Failed to generate 3D positions with GPT.")
        return []

    # Check if the number of positions matches the number of peaks
    if len(gpt_positions) < len(peaks):
        print(f"[WARNING] GPT generated fewer positions ({len(gpt_positions)}) than peaks ({len(peaks)}). Filling the rest with default values.")
    elif len(gpt_positions) > len(peaks):
        print(f"[WARNING] GPT generated more positions ({len(gpt_positions)}) than peaks ({len(peaks)}). Trimming the excess positions.")

    # Convert GPT positions into a format usable by the audio processing system
    location_data = []
    for i in range(min(len(peaks), len(gpt_positions))):
        location_data.append({
            "start_time": peaks[i][0],
            "end_time": peaks[i][1],
            "azimuth": gpt_positions[i]['azimuth'],
            "elevation": gpt_positions[i]['elevation'],
            "distance": gpt_positions[i]['distance']
        })

    # If GPT returned fewer positions than peaks, fill the rest with default values
    for i in range(len(gpt_positions), len(peaks)):
        location_data.append({
            "start_time": peaks[i][0],
            "end_time": peaks[i][1],
            "azimuth": 0,  # Default azimuth
            "elevation": 0,  # Default elevation
            "distance": 3  # Default distance (e.g., middle distance)
        })

    print(f"[INFO] Successfully generated 3D positions for {len(location_data)} segments.")
    return location_data


# Apply 3D effects to the audio
def apply_3d_effects(input_wav_file, location_data):
    print(f"[INFO] Applying 3D effects to {input_wav_file}...")
    data, sample_rate = sf.read(input_wav_file)
    
    if data.ndim == 1:
        print("[INFO] Converting mono audio to stereo...")
        input_wav_file = convert_audio_channels(input_wav_file, "stereo_audio.wav")
        data, sample_rate = sf.read(input_wav_file)

    processed_data = process_audio_chunks(data, sample_rate, location_data)
    current_time = int(time.time())
    output_file = f"SFX_{current_time}_output.wav"
    sf.write(output_file, processed_data, sample_rate)
    print(f"[INFO] 3D audio effects applied and saved to: {output_file}")

    return output_file

# Process audio chunks using location data
def process_audio_chunks(data, sample_rate, location_data):
    processed_data = []
    for i, interval in enumerate(location_data):
        start_sample, end_sample = get_sample_interval(interval, sample_rate)
        if not is_valid_chunk(start_sample, end_sample, data):
            continue

        print(f"[DEBUG] Processing chunk {i + 1}/{len(location_data)}: Start sample {start_sample}, End sample {end_sample}")
        
        chunk = data[start_sample:end_sample]
        next_azimuth = get_next_azimuth(i, location_data)
        processed_chunk = apply_3d_effects_to_chunk(
            chunk, sample_rate, interval, next_azimuth
        )
        processed_data.append(processed_chunk)

    # Ensure smooth transition between chunks by panning
    processed_data = pan_transitions(processed_data, sample_rate)

    return normalize_audio_volume(np.concatenate(processed_data, axis=0))

# New transition function that just pans to the next location
def pan_transitions(chunks, sample_rate):
    if not chunks:
        return np.array([])

    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]

        # Calculate the azimuth transition for panning from current chunk to the position of the next chunk
        azimuth_transition_to_next = np.linspace(0, 1, len(current_chunk))
        
        # Apply panning to current chunk towards the position of the next chunk
        current_chunk[:, 0] *= (1 - azimuth_transition_to_next)  # Pan left channel
        current_chunk[:, 1] *= azimuth_transition_to_next  # Pan right channel

    return chunks


# Helper functions for 3D effect processing
def get_sample_interval(interval, sample_rate):
    start_sample = int(interval["start_time"] * sample_rate / 1000)
    end_sample = int(interval["end_time"] * sample_rate / 1000)
    return start_sample, end_sample

def is_valid_chunk(start_sample, end_sample, data):
    if start_sample >= end_sample or start_sample >= len(data):
        print(f"[WARNING] Skipping invalid chunk: start={start_sample}, end={end_sample}")
        return False
    return True

def get_next_azimuth(index, location_data):
    return location_data[index + 1]["azimuth"] if index + 1 < len(location_data) else location_data[index]["azimuth"]

def apply_3d_effects_to_chunk(chunk, sample_rate, interval, next_azimuth):
    left_channel, right_channel = chunk[:, 0], chunk[:, 1]
    
    azimuth_transition = np.linspace(interval["azimuth"], next_azimuth, len(left_channel))

    delay_samples = np.array([calculate_delay_samples(sample_rate, az) for az in azimuth_transition])
    
    left_channel_delayed, right_channel_delayed = apply_delay_to_channels(
        left_channel, right_channel, azimuth_transition, delay_samples
    )

    left_channel_attenuated = np.zeros_like(left_channel_delayed)
    right_channel_attenuated = np.zeros_like(right_channel_delayed)

    for i, azimuth in enumerate(azimuth_transition):
        if 45 <= azimuth <= 135:
            left_channel_attenuated[i] = left_channel_delayed[i] * min(0.2, 1 - calculate_attenuation_factor(azimuth, interval["distance"]))
            right_channel_attenuated[i] = right_channel_delayed[i] * calculate_attenuation_factor(azimuth, interval["distance"])
        elif -135 <= azimuth <= -45:
            left_channel_attenuated[i] = left_channel_delayed[i] * calculate_attenuation_factor(azimuth, interval["distance"])
            right_channel_attenuated[i] = right_channel_delayed[i] * min(0.2, 1 - calculate_attenuation_factor(azimuth, interval["distance"]))
        else:
            left_channel_attenuated[i] = left_channel_delayed[i] * 0.5
            right_channel_attenuated[i] = right_channel_delayed[i] * 0.5

    return normalize_stereo_channels(left_channel_attenuated, right_channel_attenuated)

def calculate_attenuation_factor(azimuth, distance):
    base_attenuation = 0.5 * (1 + np.cos(np.radians(azimuth)))
    distance_factor = 1 / (distance ** 2) if distance > 0 else 1
    attenuation_factor = base_attenuation * distance_factor
    return np.clip(attenuation_factor, ATTENUATION_MIN, ATTENUATION_MAX)

def apply_delay_to_channels(left_channel, right_channel, azimuths, delay_samples):
    left_channel_delayed = np.zeros_like(left_channel)
    right_channel_delayed = np.zeros_like(right_channel)

    for i in range(len(delay_samples)):
        if azimuths[i] > 0:
            left_channel_delayed[i] = apply_fractional_delay_single_sample(left_channel, i, delay_samples[i])
            right_channel_delayed[i] = right_channel[i]
        elif azimuths[i] < 0:
            right_channel_delayed[i] = apply_fractional_delay_single_sample(right_channel, i, delay_samples[i])
            left_channel_delayed[i] = left_channel[i]
        else:
            left_channel_delayed[i] = left_channel[i]
            right_channel_delayed[i] = right_channel[i]

    return left_channel_delayed, right_channel_delayed

def apply_fractional_delay_single_sample(channel, index, delay_sample):
    int_delay = int(delay_sample)
    frac_delay = delay_sample - int_delay

    if index - int_delay < 0:
        return channel[index]

    delayed_value = channel[index - int_delay]
    if frac_delay > 0 and (index - int_delay - 1) >= 0:
        delayed_value += frac_delay * (channel[index - int_delay] - channel[index - int_delay - 1])

    return delayed_value

def calculate_delay_samples(sample_rate, azimuth):
    return min(sample_rate * np.abs(np.sin(np.radians(azimuth))) / SPEED_OF_SOUND, sample_rate * MAX_DELAY_SEC)

def normalize_audio_volume(data):
    max_amplitude = np.max(np.abs(data))
    if max_amplitude > 0:
        normalization_factor = 1.0 / max_amplitude
        data *= normalization_factor
    return data

def normalize_stereo_channels(left_channel, right_channel):
    max_amplitude = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
    if max_amplitude > 0:
        normalization_factor = 1.0 / max_amplitude
        left_channel *= normalization_factor
        right_channel *= normalization_factor
    return np.column_stack((left_channel, right_channel))

def get_file_from_mongo(file_id, output_path, collection):
    try:
        file_data = GridFS(db, collection=collection).get(file_id).read()
        with open(output_path, "wb") as f:
            f.write(file_data)
        return output_path
    except NoFile:
        print(f"[ERROR] No file found in collection {collection} with _id {file_id}")
        sys.exit(1)

# Main function to orchestrate the process
def main(input_file_id):
    print("[INFO] Starting the process...")
    current_time = int(time.time())
    fileId = ObjectId(input_file_id)
    input_file = get_file_from_mongo(fileId, "sfx_input.mp3", 'input_SFX_files')
    output_file_name = f"SFX_{current_time}_output.wav"
    
    wav_file = convert_audio_format(input_file)
    system_prompt = """
You are an expert in generating realistic 3D audio positions for sound effects. Given the onset times in milliseconds, generate 3D positions (azimuth, elevation, and distance) for each onset to create a natural and balanced 3D audio effect. The values should follow these guidelines:

Think of a pattern in which you want the sound to rotate in 3d and then generate following data: 

1. Azimuth: The horizontal position of the sound, ranging from -180 to 180 degrees except (the angle between-45 to 45 degrees), where -180 represents the far left, 0 is center, and 180 is the far right. Ensure the azimuth values alternate or change smoothly between onsets to avoid abrupt changes in direction.
   
2. Elevation: The vertical position of the sound, ranging from -45 to 45 degrees. These values should gradually vary, ensuring no abrupt jumps in height. Balance the elevation between positive and negative values to create a sense of depth.

3. Distance: The distance from the listener, 3 units fixed. 


Please provide the output strictly in the following JSON format, no extra text or comments:

[
    {
        "azimuth": <azimuth_value>,
        "elevation": <elevation_value>,
        "distance": <distance_value>
    },
    {
        "azimuth": <azimuth_value>,
        "elevation": <elevation_value>,
        "distance": <distance_value>
    },
    ...
]

"""
    
    audio = AudioSegment.from_file(wav_file)
    total_duration = len(audio)
    
    peaks = find_audio_peaks_using_spectrogram(wav_file)
    location_data = generate_location_data_gpt(peaks, system_prompt)
    output_wav_file = apply_3d_effects(wav_file, location_data)
    
    output_mp3_file = output_wav_file.replace(".wav", ".mp3")
    convert_audio_format(output_wav_file, target_format=".mp3")
    
    processed_duration = len(AudioSegment.from_file(output_mp3_file)) / 1000
    print(f"[INFO] Final processed duration: {processed_duration} seconds.")
    print("[INFO] Process completed successfully!")

    # Upload output files to MongoDB using GridFS
    with open(output_mp3_file, "rb") as f:
        output_data = f.read()
        GridFS(db, collection='output_SFX_files').put(output_data, filename=os.path.basename(output_mp3_file))
        print(f"[INFO] Output file uploaded to MongoDB with filename: {os.path.basename(output_mp3_file)}")

    # Clean up temporary files
    os.remove(output_wav_file)
    os.remove(wav_file)  # Delete the intermediate WAV file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("[ERROR] Please provide the input file ID as an argument.")
        sys.exit(1)
    
    input_file_id = sys.argv[1]
    print(f"[INFO] Processing file with ID: {input_file_id}")
    main(input_file_id)
