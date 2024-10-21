import sys
import time
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import requests
import json
import os
import glob
from pymongo import MongoClient
from urllib.parse import quote_plus
from gridfs import GridFS
from gridfs.errors import NoFile
from bson import ObjectId

AUDIO_FORMATS = {'.mp3', '.mp4', '.wav'}
SPEED_OF_SOUND = 343  # Speed of sound in air in m/s
MAX_DELAY_SEC = 0.001  # Maximum delay in seconds
ATTENUATION_MIN = 0.15
ATTENUATION_MAX = 0.7
ATTENUATION_DISTANCE_FACTOR = 3

# Your OpenAI API key
api_key = "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"

# Encode the username and password
username = quote_plus("sachingaur")
password = quote_plus("Sachin@1234")

# Use the encoded username and password in the connection string
client = MongoClient(f"mongodb+srv://{username}:{password}@cluster0.9xtyu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['audio_database']
fs = GridFS(db)
input_collection = db['input_VO_files']
output_collection = db['output_VO_files']

# Convert audio channels to stereo if necessary
def convert_audio_channels(input_file, output_file, target_channels=2):
    print("[INFO] Checking audio channels...")
    try:
        audio = AudioSegment.from_file(input_file)
    except Exception as e:
        print(f"[ERROR] Failed to load audio file: {e}")
        return None
    
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
    
    try:
        audio = AudioSegment.from_file(input_file)
    except Exception as e:
        print(f"[ERROR] Failed to load audio file: {e}")
        return None
    
    output_file = f"{file_name}{target_format}"
    audio.export(output_file, format=target_format.strip('.'))
    print(f"[INFO] File successfully converted to: {output_file}")
    return output_file

# Function to create a job and return the jobId
def create_transcription_job(audio_file):
    url = "https://api.spectropic.ai/v1/transcribe"
    files = {
        'file': (audio_file, open(audio_file, 'rb'))
    }
    data = {
        "model": "standard",
        "language": "hi",
        "vocabulary": "Spectropic, AI, LLama, Mistral, Whisper.",
        "group_segments": True,
        "transcript_output_format": "segments_only",
        "webhook": "https://example.com/webhook"
    }
    headers = {
        "Authorization": "Bearer sk-81b0a5f4363242cfaf5017e7efc70bc0"
    }

    try:
        with open(audio_file, 'rb') as f:
            response = requests.post(url, files={'file': (audio_file, f)}, data=data, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        job_id = response_data.get("jobId")
        print(f"Job created successfully. Job ID: {job_id}")
        return job_id
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return None
    except json.JSONDecodeError:
        print("[ERROR] Failed to decode response as JSON.")
        return None

# Function to check the status of the job and print the transcription result

def check_job_status(job_id, system_prompt):
    url = f"https://api.spectropic.ai/v1/jobs/{job_id}"
    headers = {"Authorization": "Bearer sk-81b0a5f4363242cfaf5017e7efc70bc0"}

    while True:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            print(f"Job Status: {response_data.get('status')}")
            
            if response_data.get("status") == "succeeded":
                print("Job succeeded.")
                output = response_data.get('output', {})
                segments = output.get('segments', [])
                
                # Extracting and printing speaker, segment details (start, end), and text
                formatted_output = []
                last_end = 0
                for idx, segment in enumerate(segments):
                    speaker = segment.get('speaker')
                    start = segment.get('start')
                    end = segment.get('end')
                    text = segment.get('text')
                    
                    # Adjust start and end times with 0.2-second buffer, handle edge cases
                    if start > 0:  # Do not modify if it's the first segment starting at 0
                        start = max(0, start - 0.2)
                    end = end + 0.2

                    # Handle missing parts and ensure gaps are limited to 0.2 seconds
                    if idx > 0:
                        # Calculate the gap between last_end and the current start
                        gap = start - last_end
                        if gap > 0.2:
                            # Distribute the extra time equally to the end of the last segment and start of the next
                            extra_time = (gap - 0.2) / 2
                            last_segment = formatted_output[-1]
                            last_segment['end'] += extra_time  # Extend the end of the last segment
                            start -= extra_time  # Adjust the start of the current segment

                        # Ensure the adjusted start is not overlapping the last end time
                        start = max(last_end, start)

                    # Final adjustment to ensure the segment has valid start and end times
                    formatted_output.append({
                        'speaker': speaker,
                        'start': start,
                        'end': end,
                        'text': text
                    })

                    # Update the last_end for the next iteration
                    last_end = end
                
                print("Formatted Output:")
                print(json.dumps(formatted_output, indent=4, ensure_ascii=False))  # Ensuring correct display of non-ASCII characters
                
                # Generate 3D positions using GPT
                location_data = generate_gpt_3d_positions(formatted_output, system_prompt)
                if not location_data:
                    print("[ERROR] Failed to generate 3D positions.")
                    return None
                
                return location_data

            elif response_data.get("status") == "failed":
                print("Job failed. Please try again.")
                break
            else:
                print("Job is not completed yet. Waiting for 30 seconds before checking again...")
                time.sleep(30)  # Wait for 30 seconds before checking the status again

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] API request failed: {e}")
            break
        except json.JSONDecodeError:
            print("[ERROR] Failed to decode response as JSON.")
            break

# GPT function modified to take input in the required format
def generate_gpt_3d_positions(transcription_result, system_prompt):
    print("[INFO] Sending transcription data to GPT for 3D position generation...")
    
    gpt_positions = []
    batch_size = 30  # Split requests into batches of 30 segments

    for i in range(0, len(transcription_result), batch_size):
        batch_transcription = transcription_result[i:i + batch_size]

        onset_data = [
            {
                "speaker": segment["speaker"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            for segment in batch_transcription
        ]
        
        script = (
            f"Given the following transcript segments: {onset_data}, "
            f"please generate 3D positions with azimuth, elevation, and distance for each speaker."
        )

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
            "max_tokens": 2000
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()  # Raise error for bad status codes
            result = response.json()

            if 'choices' in result and len(result['choices']) > 0:
                gpt_response = result['choices'][0]['message']['content']
                
                try:
                    # Parse the GPT response
                    batch_positions = json.loads(gpt_response)  # Expecting GPT to return JSON-like positions
                    gpt_positions.extend(batch_positions)
                    print(gpt_positions)
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

# Process audio using location data
def process_audio(data, sample_rate, location_data):
    processed_data = []
    last_end_sample = 0

    # Iterate over the location data and process the audio in chunks
    for i, interval in enumerate(location_data):
        start_sample, end_sample = get_sample_interval(interval, sample_rate)

        # If there is a gap between the current start and the previous end, keep the audio unchanged
        if start_sample > last_end_sample:
            print(f"[DEBUG] Keeping unchanged interval: Start sample {last_end_sample}, End sample {start_sample}")
            processed_data.append(data[last_end_sample:start_sample])

        # Process the current chunk if valid
        if is_valid_chunk(start_sample, end_sample, data):
            print(f"[DEBUG] Processing interval {i + 1}/{len(location_data)}: Start sample {start_sample}, End sample {end_sample}")
            
            chunk = data[start_sample:end_sample]
            next_azimuth = get_next_azimuth(i, location_data)
            processed_chunk = apply_3d_effects_to_chunk(
                chunk, sample_rate, interval, next_azimuth
            )
            processed_data.append(processed_chunk)
        
        last_end_sample = end_sample

    # If there's audio left after the last interval, keep it unchanged
    if last_end_sample < len(data):
        print(f"[DEBUG] Keeping unchanged interval: Start sample {last_end_sample}, End sample {len(data)}")
        processed_data.append(data[last_end_sample:])

    # Normalize the audio
    return normalize_audio_volume(np.concatenate(processed_data, axis=0))

# Helper functions for 3D effect processing
def get_sample_interval(interval, sample_rate):
    start_sample = int(interval["start"] * sample_rate )
    end_sample = int(interval["end"] * sample_rate )
    return start_sample, end_sample

def is_valid_chunk(start_sample, end_sample, data):
    if start_sample >= end_sample or start_sample >= len(data):
        print(f"[WARNING] Skipping invalid chunk: start={start_sample}, end={end_sample}")
        return False
    return True

def get_next_azimuth(index, location_data):
    # Check if the next index exists
    if index + 1 < len(location_data):
        next_azimuth = location_data[index + 1].get("azimuth", 0.0)
        print(f"[DEBUG] Next azimuth at index {index + 1}: {next_azimuth}")
    else:
        next_azimuth = location_data[index].get("azimuth", 0.0)
        print(f"[DEBUG] Current/Last azimuth at index {index}: {next_azimuth}")
    
    return next_azimuth


def apply_3d_effects_to_chunk(chunk, sample_rate, interval, next_azimuth):
    left_channel, right_channel = chunk[:, 0], chunk[:, 1]
    
    azimuth = interval["azimuth"]
    print(f"[DEBUG] Azimuth: {azimuth}")

    delay_samples = calculate_delay_samples(sample_rate, azimuth)

    left_channel_delayed, right_channel_delayed = apply_delay_to_channels(
        left_channel, right_channel, [azimuth] * len(left_channel), [delay_samples] * len(left_channel)
    )

    # Debugging attenuation factors
    left_channel_attenuated = np.zeros_like(left_channel_delayed)
    right_channel_attenuated = np.zeros_like(right_channel_delayed)

    for i in range(len(left_channel)):
        if -90 <= azimuth < 0:
            left_channel_attenuated[i] = left_channel_delayed[i] * calculate_attenuation_factor(azimuth, interval["distance"])
            right_channel_attenuated[i] = right_channel_delayed[i] * 0.15
        elif 0 < azimuth <= 90:
            left_channel_attenuated[i] = left_channel_delayed[i] * 0.15
            right_channel_attenuated[i] = right_channel_delayed[i] * calculate_attenuation_factor(azimuth, interval["distance"])
        else:
            left_channel_attenuated[i] = left_channel_delayed[i] * 0.5
            right_channel_attenuated[i] = right_channel_delayed[i] * 0.5

    return normalize_stereo_channels(left_channel_attenuated, right_channel_attenuated)


def calculate_attenuation_factor(azimuth, distance):
    base_attenuation = 0.5 * (1 + np.cos(np.radians(azimuth)))
    distance_factor = 1 / (distance ** 2) if distance > 0 else 1
    attenuation_factor = base_attenuation * distance_factor
    # print(f"[DEBUG] Azimuth: {azimuth}, Distance: {distance}, Attenuation factor: {attenuation_factor}")
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

        # Debug info for delays applied to channels
        # print(f"[DEBUG] Sample {i}, Azimuth: {azimuths[i]}, Left delayed: {left_channel_delayed[i]}, Right delayed: {right_channel_delayed[i]}")

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
    # Calculate delay based on azimuth
    delay = min(sample_rate * np.abs(np.sin(np.radians(azimuth))) / SPEED_OF_SOUND, sample_rate * MAX_DELAY_SEC)
    # print(f"[DEBUG] Azimuth: {azimuth}, Delay samples: {delay}")
    return delay

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

# Apply 3D effects to the audio
def apply_3d_effects(input_wav_file, location_data):
    print(f"[INFO] Applying 3D effects to {input_wav_file}...")
    try:
        data, sample_rate = sf.read(input_wav_file)
    except Exception as e:
        print(f"[ERROR] Failed to read audio file: {e}")
        return None
    
    if data.ndim == 1:
        print("[INFO] Converting mono audio to stereo...")
        input_wav_file = convert_audio_channels(input_wav_file, "stereo_audio.wav")
        if input_wav_file is None:
            return None
        data, sample_rate = sf.read(input_wav_file)

    processed_data = process_audio(data, sample_rate, location_data)

    current_time = int(time.time())
    output_file = f"VO_{current_time}_output.wav"
    sf.write(output_file, processed_data, sample_rate)
    print(f"[INFO] 3D audio effects applied and saved to: {output_file}")
    return output_file

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
    
    # Use the provided method to get the input file
    fileId = ObjectId(input_file_id)
    input_file = get_file_from_mongo(fileId, "vo_input.mp3", 'input_VO_files')
    
    output_file_name = f"VO_{current_time}_output.wav"
    
    # Convert input audio to WAV format
    wav_file = convert_audio_format(input_file)
    if wav_file is None:
        print("[ERROR] Failed to convert audio format.")
        return

    system_prompt = """
    You are an expert in generating realistic 3D audio positions for vocals. Given the transcript segments with times and speaker roles, generate 3D positions (azimuth, elevation, and distance) for each segment.
    Identify the total speaker roles and assign them fix values to the respective characters.
    1. Narrator: Fixed position with azimuth and elevation of 0 degrees, diatance = 3.
    2. Characters: Assign azimuth between -90 to -45 and 45 to 90 degrees, such as SPEAKER_XX to left, then SPEAKER_XY to right and then same pattern and keep them fixed.
    3. Elevation: Remains between -15 to 15 degrees to create a natural effect.
    4. Distance: 2
    5. Speaker: Assign a unique speaker ID to each character.

    Provide output strictly in the following JSON format, no extra text or comments or explanation:

    [
        {
            "start": <start_time>,
            "end": <end_time>,
            "azimuth": <azimuth_value>,
            "elevation": <elevation_value>,
            "distance": <distance_value>,
            "speaker": <speaker_value>
        },
        ...
    ]

    no extra text or comments or explanation.
    """

    # Create transcription job
    job_id = create_transcription_job(input_file)
    if not job_id:
        print("[ERROR] Failed to create transcription job.")
        return
    
    # Check transcription job status and retrieve 3D position data
    location_data = check_job_status(job_id, system_prompt)
    if not location_data:
        print("[ERROR] Failed to retrieve transcription data.")
        return

    # Apply 3D effects to the audio based on location data
    output_wav_file = apply_3d_effects(wav_file, location_data)
    if output_wav_file is None:
        print("[ERROR] Failed to apply 3D effects.")
        return

    # Convert the processed WAV file back to MP3
    output_mp3_file = output_wav_file.replace(".wav", ".mp3")
    output_mp3_file = convert_audio_format(output_wav_file, target_format=".mp3")
    if not output_mp3_file:
        print("[ERROR] Failed to convert processed audio to MP3 format.")
        return

    # Calculate and display the duration of the processed audio file
    try:
        processed_duration = len(AudioSegment.from_file(output_mp3_file)) / 1000
        print(f"[INFO] Final processed duration: {processed_duration} seconds.")
    except Exception as e:
        print(f"[ERROR] Failed to calculate processed duration: {e}")

    print("[INFO] Process completed successfully!")

    # Upload output files to MongoDB using GridFS
    with open(output_mp3_file, "rb") as f:
        output_data = f.read()
        GridFS(db, collection='output_VO_files').put(output_data, filename=os.path.basename(output_mp3_file))

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
