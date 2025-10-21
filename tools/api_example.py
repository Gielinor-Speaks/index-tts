"""
Simple example script demonstrating how to use the IndexTTS2 API.

Usage:
    # First, start the API server:
    uv run api.py --fp16

    # Then run this example:
    python tools/api_example.py <speaker_audio.wav> <output.wav> "Your text here"
"""

import base64
import requests
import sys
from pathlib import Path


def encode_audio_file(file_path: str) -> str:
    """Encode an audio file to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def save_audio_response(audio_b64: str, output_path: str):
    """Save base64-encoded audio to file."""
    audio_bytes = base64.b64decode(audio_b64)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)


def main():
    if len(sys.argv) < 4:
        print("Usage: python api_example.py <speaker_audio> <output> <text>")
        print("\nExample:")
        print('  python api_example.py examples/voice_01.wav output.wav "Hello world"')
        sys.exit(1)

    speaker_audio_path = sys.argv[1]
    output_path = sys.argv[2]
    text = sys.argv[3]

    # Validate input file exists
    if not Path(speaker_audio_path).exists():
        print(f"Error: Speaker audio file not found: {speaker_audio_path}")
        sys.exit(1)

    print(f"Loading speaker audio: {speaker_audio_path}")
    speaker_audio_b64 = encode_audio_file(speaker_audio_path)

    print(f"Generating speech for: {text}")

    # Make API request
    api_url = "http://localhost:8000/synthesize"
    response = requests.post(
        api_url,
        json={
            "text": text,
            "speaker_audio": speaker_audio_b64,
            "format": "wav",
            # Optional: customize parameters
            "temperature": 0.8,
            "emotion_weight": 0.65,
        },
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()

        # Save output
        save_audio_response(result["audio"], output_path)

        print(f"\nSuccess!")
        print(f"  Output: {output_path}")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Segments: {result['num_segments']}")
        print(f"  Sample rate: {result['sample_rate']} Hz")
    else:
        print(f"\nError: {response.status_code}")
        print(response.text)
        sys.exit(1)


if __name__ == "__main__":
    main()
