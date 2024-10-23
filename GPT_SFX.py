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
import librosa

AUDIO_FORMATS = {'.mp3', '.mp4', '.wav'}
SPEED_OF_SOUND = 343  
MAX_DELAY_SEC = 0.001  
ATTENUATION_MIN = 0.15
ATTENUATION_MAX = 0.7
ATTENUATION_DISTANCE_FACTOR = 4

api_key = "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"

username = quote_plus("sachingaur")
password = quote_plus("Sachin@1234")

client = MongoClient(f"mongodb+srv://{username}:{password}@cluster0.9xtyu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&ssl=true")
db = client['audio_database']
fs = GridFS(db)
input_collection = db['input_SFX_files.files']
output_collection = db['output_SFX_files']

def convert_audio_channels(input_file, output_file, target_channels=2):
    audio = AudioSegment.from_file(input_file)
    
    if audio.channels != target_channels:
        channels_audio = audio.set_channels(target_channels)
        channels_audio.export(output_file, format="wav")
        return output_file
    else:
        return input_file

def convert_audio_format(input_file, target_format=".wav"):
    file_name, file_extension = os.path.splitext(input_file)
    
    if file_extension.lower() not in AUDIO_FORMATS:
        raise ValueError(f"Unsupported file format: {file_extension}.")
    
    audio = AudioSegment.from_file(input_file)
    output_file = f"{file_name}{target_format}"
    audio.export(output_file, format=target_format.strip('.'))
    return output_file

def find_audio_peaks_using_intervals(audio_file, interval_sec=6, min_interval_sec=2):
    y, sr = librosa.load(audio_file)
    amplitude = np.abs(y)
    time_frames = []
    amplitudes = []
    interval_samples = int(interval_sec * sr)
    min_interval_samples = int(min_interval_sec * sr)
    current_idx = 0

    while current_idx < len(amplitude):
        start_idx = current_idx
        end_idx = min(current_idx + min_interval_samples, len(amplitude))
        min_amplitude = np.min(amplitude[start_idx:end_idx])
        min_idx = np.argmin(amplitude[start_idx:end_idx]) + start_idx
        time_frames.append(librosa.samples_to_time(min_idx, sr=sr))
        amplitudes.append(min_amplitude)
        current_idx = min_idx + interval_samples

    if time_frames:
        time_frames[0] = 0.0
        time_frames[-1] = librosa.samples_to_time(len(y), sr=sr)

    peaks = []
    for i in range(len(time_frames) - 1):
        start_time = time_frames[i] * 1000
        end_time = time_frames[i + 1] * 1000
        peaks.append((start_time, end_time))

    if len(time_frames) > 1:
        peaks.append((time_frames[-2] * 1000, time_frames[-1] * 1000))

    if len(peaks) > 1 and peaks[-1] == peaks[-2]:
        peaks.pop()

    return peaks

def generate_gpt_3d_positions(peaks, system_prompt):
    gpt_positions = []
    batch_size = 30

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
            response.raise_for_status()
            result = response.json()

            if 'choices' in result and len(result['choices']) > 0:
                gpt_response = result['choices'][0]['message']['content']
                
                try:
                    batch_positions = json.loads(gpt_response)
                    gpt_positions.extend(batch_positions)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode GPT response as JSON: {e}")
                    return None
            else:
                raise ValueError("Unexpected response format from GPT.")
        
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None

    return gpt_positions

def generate_location_data_gpt(peaks, system_prompt):
    gpt_positions = generate_gpt_3d_positions(peaks, system_prompt)
    
    if gpt_positions is None:
        print("Failed to generate 3D positions with GPT.")
        return []

    if len(gpt_positions) < len(peaks):
        print(f"GPT generated fewer positions ({len(gpt_positions)}) than peaks ({len(peaks)}). Filling the rest with default values.")
    elif len(gpt_positions) > len(peaks):
        print(f"GPT generated more positions ({len(gpt_positions)}) than peaks ({len(peaks)}). Trimming the excess positions.")

    location_data = []
    for i in range(min(len(peaks), len(gpt_positions))):
        location_data.append({
            "start_time": peaks[i][0],
            "end_time": peaks[i][1],
            "azimuth": gpt_positions[i]['azimuth'],
            "elevation": gpt_positions[i]['elevation'],
            "distance": gpt_positions[i]['distance']
        })

    for i in range(len(gpt_positions), len(peaks)):
        location_data.append({
            "start_time": peaks[i][0],
            "end_time": peaks[i][1],
            "azimuth": 0,
            "elevation": 0,
            "distance": 3
        })

    return location_data

def apply_3d_effects(input_wav_file, location_data):
    data, sample_rate = sf.read(input_wav_file)
    
    if data.ndim == 1:
        input_wav_file = convert_audio_channels(input_wav_file, "stereo_audio.wav")
        data, sample_rate = sf.read(input_wav_file)

    processed_data = process_audio_chunks(data, sample_rate, location_data)
    current_time = int(time.time())
    output_file = f"SFX_{current_time}_output.wav"
    sf.write(output_file, processed_data, sample_rate)

    return output_file

def process_audio_chunks(data, sample_rate, location_data):
    processed_data = []
    for i, interval in enumerate(location_data):
        start_sample, end_sample = get_sample_interval(interval, sample_rate)
        if not is_valid_chunk(start_sample, end_sample, data):
            continue

        chunk = data[start_sample:end_sample]
        next_azimuth = get_next_azimuth(i, location_data)
        processed_chunk = apply_3d_effects_to_chunk(
            chunk, sample_rate, interval, next_azimuth
        )
        processed_data.append(processed_chunk)

    processed_data = pan_transitions(processed_data)

    return normalize_audio_volume(np.concatenate(processed_data, axis=0))

def pan_transitions(chunks):
    if not chunks:
        return np.array([])

    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]

        if i % 2 == 0:
            azimuth_transition_to_next = np.linspace(0, 1, len(current_chunk))
        else:
            azimuth_transition_to_next = np.linspace(1, 0, len(current_chunk))
        
        current_chunk[:, 0] *= (1 - azimuth_transition_to_next)
        current_chunk[:, 1] *= azimuth_transition_to_next

    return chunks

def get_sample_interval(interval, sample_rate):
    start_sample = int(interval["start_time"] * sample_rate / 1000)
    end_sample = int(interval["end_time"] * sample_rate / 1000)
    return start_sample, end_sample

def is_valid_chunk(start_sample, end_sample, data):
    if start_sample >= end_sample or start_sample >= len(data):
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
        print(f"No file found in collection {collection} with _id {file_id}")
        sys.exit(1)

def main(input_file_id):
    current_time = int(time.time())
    fileId = ObjectId(input_file_id)
    input_file = get_file_from_mongo(fileId, "sfx_input.mp3", 'input_SFX_files')
    output_file_name = f"SFX_{current_time}_output.wav"
    
    wav_file = convert_audio_format(input_file)
    
    audio = AudioSegment.from_file(wav_file)
    
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
    
    peaks = find_audio_peaks_using_intervals(wav_file)
    location_data = generate_location_data_gpt(peaks, system_prompt)
    output_wav_file = apply_3d_effects(wav_file, location_data)
    
    processed_audio = AudioSegment.from_file(output_wav_file)
    
    output_mp3_file = output_wav_file.replace(".wav", ".mp3")
    convert_audio_format(output_wav_file, target_format=".mp3")
    
    final_audio = AudioSegment.from_file(output_mp3_file)
    processed_duration = len(final_audio) / 1000
    
    with open(output_mp3_file, "rb") as f:
        output_data = f.read()
        GridFS(db, collection='output_SFX_files').put(output_data, filename=os.path.basename(output_mp3_file))

    os.remove(output_wav_file)
    os.remove(wav_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide the input file ID as an argument.")
        sys.exit(1)
    
    input_file_id = sys.argv[1]
    main(input_file_id)