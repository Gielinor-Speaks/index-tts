"""
Test script for the IndexTTS2 FastAPI server.

This script tests the API endpoints with various configurations.

Usage:
    # Start the API server first:
    uv run api.py --fp16

    # Then run this test script:
    PYTHONPATH="$PYTHONPATH:." uv run tools/test_api.py
"""

import base64
import time
import requests
import soundfile as sf
import os
from pathlib import Path


API_BASE_URL = "http://localhost:8000"


def encode_audio_file(file_path: str) -> str:
    """Encode an audio file to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def save_audio_response(response_data: dict, output_path: str):
    """Save audio from API response to file."""
    audio_bytes = base64.b64decode(response_data["audio"])
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    print(f"  Saved to: {output_path}")
    print(f"  Format: {response_data['format']} ({response_data['sample_rate']} Hz)")
    print(f"  Duration: {response_data['duration']:.2f}s")
    print(f"  Segments: {response_data['num_segments']}")


def test_health():
    """Test the health check endpoint."""
    print("\n" + "=" * 80)
    print("TEST 1: Health Check")
    print("=" * 80)

    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Service status: {data['status']}")
        print(f"Model loaded: {data['model_loaded']}")
        print(f"Device: {data['device']}")
        print(f"Version: {data['version']}")
        return True
    else:
        print(f"ERROR: {response.text}")
        return False


def test_basic_tts(speaker_audio_b64: str):
    """Test basic TTS generation."""
    print("\n" + "=" * 80)
    print("TEST 2: Basic TTS Generation")
    print("=" * 80)

    response = requests.post(
        f"{API_BASE_URL}/synthesize",
        json={
            "text": "Hello, this is a test of the IndexTTS2 API. The quality should be excellent.",
            "speaker_audio": speaker_audio_b64,
            "format": "wav"
        },
        timeout=60
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        save_audio_response(data, "outputs/test_basic.wav")
        return True
    else:
        print(f"ERROR: {response.text}")
        return False


def test_emotion_reference(speaker_audio_b64: str, emotion_audio_b64: str):
    """Test emotion control with reference audio."""
    print("\n" + "=" * 80)
    print("TEST 3: Emotion Control (Reference Audio)")
    print("=" * 80)

    response = requests.post(
        f"{API_BASE_URL}/synthesize",
        json={
            "text": "This sentence should reflect the emotion from the reference audio.",
            "speaker_audio": speaker_audio_b64,
            "emotion_mode": "reference_audio",
            "emotion_audio": emotion_audio_b64,
            "emotion_weight": 0.8,
            "format": "wav"
        },
        timeout=60
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        save_audio_response(data, "outputs/test_emotion_ref.wav")
        return True
    else:
        print(f"ERROR: {response.text}")
        return False


def test_emotion_vector(speaker_audio_b64: str):
    """Test emotion control with manual vector."""
    print("\n" + "=" * 80)
    print("TEST 4: Emotion Control (Manual Vector)")
    print("=" * 80)

    # Happy and surprised emotion
    emotion_vector = [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0]

    response = requests.post(
        f"{API_BASE_URL}/synthesize",
        json={
            "text": "I just won the lottery! This is amazing!",
            "speaker_audio": speaker_audio_b64,
            "emotion_mode": "emotion_vector",
            "emotion_vector": emotion_vector,
            "emotion_weight": 0.75,
            "format": "wav"
        },
        timeout=60
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        save_audio_response(data, "outputs/test_emotion_vector.wav")
        return True
    else:
        print(f"ERROR: {response.text}")
        return False


def test_custom_parameters(speaker_audio_b64: str):
    """Test with custom generation parameters."""
    print("\n" + "=" * 80)
    print("TEST 5: Custom Generation Parameters")
    print("=" * 80)

    response = requests.post(
        f"{API_BASE_URL}/synthesize",
        json={
            "text": "Testing custom parameters for fine-tuned control.",
            "speaker_audio": speaker_audio_b64,
            "temperature": 0.9,
            "top_p": 0.85,
            "top_k": 40,
            "num_beams": 5,
            "repetition_penalty": 12.0,
            "format": "wav"
        },
        timeout=60
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        save_audio_response(data, "outputs/test_custom_params.wav")
        return True
    else:
        print(f"ERROR: {response.text}")
        return False


def test_segmentation(speaker_audio_b64: str):
    """Test text segmentation with long text."""
    print("\n" + "=" * 80)
    print("TEST 6: Text Segmentation")
    print("=" * 80)

    long_text = (
        "This is a very long text that will be automatically split into multiple segments. "
        "Each segment will be generated separately and then concatenated together. "
        "The segmentation is based on the token limit per segment, which helps manage "
        "memory usage and generation quality for very long inputs. "
        "You can control the maximum tokens per segment and the silence between segments."
    )

    # First preview the segmentation
    print("\nPreviewing segmentation...")
    preview_response = requests.post(
        f"{API_BASE_URL}/synthesize/segment-preview",
        json={
            "text": long_text,
            "max_text_tokens_per_segment": 60
        }
    )

    if preview_response.status_code == 200:
        preview_data = preview_response.json()
        print(f"Text will be split into {preview_data['num_segments']} segments:")
        for i, (segment, tokens) in enumerate(zip(preview_data['segments'], preview_data['tokens_per_segment'])):
            print(f"  Segment {i+1} ({tokens} tokens): {segment[:60]}...")
    else:
        print(f"Preview ERROR: {preview_response.text}")

    # Now generate with segmentation
    print("\nGenerating with segmentation...")
    response = requests.post(
        f"{API_BASE_URL}/synthesize",
        json={
            "text": long_text,
            "speaker_audio": speaker_audio_b64,
            "max_text_tokens_per_segment": 60,
            "interval_silence": 300,
            "format": "wav"
        },
        timeout=120
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        save_audio_response(data, "outputs/test_segmentation.wav")
        return True
    else:
        print(f"ERROR: {response.text}")
        return False


def test_mp3_output(speaker_audio_b64: str):
    """Test MP3 output format."""
    print("\n" + "=" * 80)
    print("TEST 7: MP3 Output Format")
    print("=" * 80)

    response = requests.post(
        f"{API_BASE_URL}/synthesize",
        json={
            "text": "Testing MP3 output format.",
            "speaker_audio": speaker_audio_b64,
            "format": "mp3"
        },
        timeout=60
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        save_audio_response(data, "outputs/test_output.mp3")
        return True
    else:
        print(f"ERROR: {response.text}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("IndexTTS2 API Test Suite")
    print("=" * 80)
    print(f"API Base URL: {API_BASE_URL}")

    # Check if outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Find speaker audio files
    examples_dir = Path("examples")
    prompts_dir = Path("prompts")

    speaker_audio_path = None
    emotion_audio_path = None

    # Look for audio files
    for directory in [examples_dir, prompts_dir]:
        if directory.exists():
            audio_files = list(directory.glob("*.wav")) + list(directory.glob("*.mp3"))
            if audio_files:
                speaker_audio_path = str(audio_files[0])
                if len(audio_files) > 1:
                    emotion_audio_path = str(audio_files[1])
                else:
                    emotion_audio_path = speaker_audio_path
                break

    if speaker_audio_path is None:
        print("\nERROR: No audio files found in examples/ or prompts/ directories")
        print("Please provide at least one audio file for testing.")
        return

    print(f"\nUsing speaker audio: {speaker_audio_path}")
    if emotion_audio_path:
        print(f"Using emotion audio: {emotion_audio_path}")

    # Encode audio files
    print("\nEncoding audio files...")
    speaker_audio_b64 = encode_audio_file(speaker_audio_path)
    emotion_audio_b64 = encode_audio_file(emotion_audio_path) if emotion_audio_path else speaker_audio_b64

    # Run tests
    results = []

    try:
        results.append(("Health Check", test_health()))
        time.sleep(1)

        results.append(("Basic TTS", test_basic_tts(speaker_audio_b64)))
        time.sleep(1)

        results.append(("Emotion Reference", test_emotion_reference(speaker_audio_b64, emotion_audio_b64)))
        time.sleep(1)

        results.append(("Emotion Vector", test_emotion_vector(speaker_audio_b64)))
        time.sleep(1)

        results.append(("Custom Parameters", test_custom_parameters(speaker_audio_b64)))
        time.sleep(1)

        results.append(("Segmentation", test_segmentation(speaker_audio_b64)))
        time.sleep(1)

        results.append(("MP3 Output", test_mp3_output(speaker_audio_b64)))

    except requests.exceptions.ConnectionError:
        print("\n" + "=" * 80)
        print("ERROR: Could not connect to API server")
        print("=" * 80)
        print("Make sure the API server is running:")
        print("  uv run api.py --fp16")
        return

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! ✓")
    else:
        print(f"\n{total - passed} test(s) failed.")


if __name__ == "__main__":
    main()
