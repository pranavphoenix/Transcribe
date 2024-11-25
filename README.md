# Transcribe
## Transcribe audio in any language to English using Whisper

### Python Code Example
```python
pip install transformers torchaudio
```
Upload audio file in colab

```python
from google.colab import files

# Upload the file
uploaded = files.upload()

# Display the uploaded file(s)
for filename in uploaded.keys():
    print(f"Uploaded file: {filename}")
    file_path = filename
```
Transcribe using whisper to any language needed
```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Load Whisper processor and large-v2 model
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

# Ensure the model is on the appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to preprocess audio file
def preprocess_audio(file_path, sampling_rate=16000):
    # Load the audio file
    audio, sr = torchaudio.load(file_path)
    # Resample to 16kHz if necessary
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio.squeeze(0)  # Remove channel dimension if stereo

# Function to split long audio into chunks
def split_audio(audio, chunk_duration, sampling_rate):
    num_samples_per_chunk = chunk_duration * sampling_rate
    return [audio[i:i + num_samples_per_chunk] for i in range(0, len(audio), num_samples_per_chunk)]

 # Replace with the actual file path

# Preprocess the audio
audio_tensor = preprocess_audio(file_path)

# Split the audio into smaller chunks (30 seconds each)
chunk_duration = 30  # Duration of each chunk in seconds
chunks = split_audio(audio_tensor, chunk_duration, sampling_rate=16000)

# Transcribe each chunk and concatenate results
transcriptions = []
for i, chunk in enumerate(chunks):
    input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ta", task="translate") #change language here
    generated_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
        max_new_tokens=444,  # Allow enough tokens for longer transcription
    )
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    transcriptions.append(transcription)
    print(f"Chunk {i + 1}/{len(chunks)} transcribed.")

# Combine all chunks into the final transcription
full_transcription = " ".join(transcriptions)
print("\nFull Transcription:\n", full_transcription)


