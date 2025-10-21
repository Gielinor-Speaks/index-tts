"""
FastAPI server for IndexTTS2.

Provides a comprehensive REST API exposing all features available in the Gradio webui,
compliant with the OpenAPI specification in tools/local-tts-api-spec.yaml.

Usage:
    uv run api.py [--host HOST] [--port PORT] [--model-dir DIR] [--config PATH] [--fp16] [--cuda-kernel]

Example:
    uv run api.py --port 8000 --fp16
"""

import argparse
import os
import tempfile
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from indextts.infer_v2 import IndexTTS2
from indextts.utils.api_utils import (
    TTSRequest,
    TTSResponse,
    SegmentPreviewRequest,
    SegmentPreviewResponse,
    HealthResponse,
    EmotionMode,
    decode_audio_base64,
    encode_audio_base64,
    convert_audio_file,
    normalize_emotion_vector,
)


# Global TTS model instance
tts_model: Optional[IndexTTS2] = None
config: dict = {
    "model_dir": "checkpoints",
    "cfg_path": "checkpoints/config.yaml",
    "use_fp16": False,
    "use_cuda_kernel": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup TTS model."""
    global tts_model, config

    print("Initializing IndexTTS2 model...")
    print(f"  Model directory: {config['model_dir']}")
    print(f"  Config path: {config['cfg_path']}")
    print(f"  FP16: {config['use_fp16']}")
    print(f"  CUDA kernel: {config['use_cuda_kernel']}")

    try:
        tts_model = IndexTTS2(
            cfg_path=config["cfg_path"],
            model_dir=config["model_dir"],
            use_fp16=config["use_fp16"],
            use_cuda_kernel=config["use_cuda_kernel"],
        )
        print(f"Model loaded successfully on device: {tts_model.device}")
    except Exception as e:
        print(f"ERROR: Failed to load TTS model: {e}")
        traceback.print_exc()
        raise

    yield

    # Cleanup
    print("Shutting down IndexTTS2 model...")
    if tts_model is not None:
        del tts_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="IndexTTS2 API",
    description="REST API for IndexTTS2 - Emotionally Expressive Zero-Shot Text-to-Speech",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the service status and model information.
    """
    return HealthResponse(
        status="healthy" if tts_model is not None else "unhealthy",
        model_loaded=tts_model is not None,
        device=str(tts_model.device) if tts_model else "unknown",
        version="2.0.0",
    )


@app.post("/synthesize", response_model=TTSResponse, tags=["synthesis"])
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech with reference voice (zero-shot voice cloning).

    This endpoint exposes all features available in the Gradio webui:
    - Zero-shot voice cloning from reference audio
    - Multiple emotion control modes (speaker audio, separate audio, vector, text)
    - Fine-grained generation parameters (temperature, sampling, beam search, etc.)
    - Text segmentation control for long texts
    - Multiple output formats

    Args:
        request: Synthesis request containing text, prompt audio, and generation parameters

    Returns:
        Base64-encoded audio with metadata

    Raises:
        HTTPException: If model not loaded, invalid parameters, or generation fails
    """
    if tts_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS model not loaded",
        )

    # Create temporary files for audio inputs
    temp_files = []
    try:
        # Decode speaker audio
        speaker_audio_path = decode_audio_base64(request.speaker_audio)
        temp_files.append(speaker_audio_path)

        # Prepare emotion parameters based on mode
        emo_audio_path = None
        emo_vector = None
        use_emo_text = False
        emo_text = None

        if request.emotion_mode == EmotionMode.SAME_AS_SPEAKER:
            # Mode 0: Use speaker audio for emotion (default)
            pass

        elif request.emotion_mode == EmotionMode.REFERENCE_AUDIO:
            # Mode 1: Separate emotion reference audio
            if request.emotion_audio is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="emotion_audio is required when emotion_mode=reference_audio",
                )
            emo_audio_path = decode_audio_base64(request.emotion_audio)
            temp_files.append(emo_audio_path)

        elif request.emotion_mode == EmotionMode.EMOTION_VECTOR:
            # Mode 2: Manual 8D emotion vector
            if request.emotion_vector is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="emotion_vector is required when emotion_mode=emotion_vector",
                )
            # Normalize the emotion vector
            emo_vector = normalize_emotion_vector(
                request.emotion_vector,
                apply_bias=True
            ).tolist()

        elif request.emotion_mode == EmotionMode.TEXT_DESCRIPTION:
            # Mode 3: Text-based emotion
            # if not request.emotion_text:
            #     raise HTTPException(
            #         status_code=status.HTTP_400_BAD_REQUEST,
            #         detail="emotion_text is required when emotion_mode=text_description",
            #     )
            use_emo_text = True
            emo_text = request.emotion_text # If emotion_text is None, it will be inferred from the prompt string.

        # Create output file
        fd, output_wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        temp_files.append(output_wav_path)

        # Call TTS inference
        # Tokenize to get segment count before inference
        text_tokens_list = tts_model.tokenizer.tokenize(request.text)
        segments_tokens = tts_model.tokenizer.split_segments(
            text_tokens_list,
            max_text_tokens_per_segment=request.max_text_tokens_per_segment,
        )
        num_segments = len(segments_tokens)

        # infer() returns the output path (as a string) when stream_return=False
        tts_model.infer(
            spk_audio_prompt=speaker_audio_path,
            text=request.text,
            output_path=output_wav_path,
            # Emotion parameters
            emo_audio_prompt=emo_audio_path,
            emo_vector=emo_vector,
            emo_alpha=request.emotion_weight,
            use_random=request.emotion_random,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            # Generation parameters
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            length_penalty=request.length_penalty,
            max_mel_tokens=request.max_mel_tokens,
            # Segmentation parameters
            max_text_tokens_per_segment=request.max_text_tokens_per_segment,
            interval_silence=request.interval_silence,
            # Other
            stream_return=False,
            verbose=True,
        )

        # Convert output if needed and encode to base64
        final_output_path = convert_audio_file(output_wav_path, request.output_audio_format)
        if final_output_path not in temp_files:
            temp_files.append(final_output_path)

        audio_b64, duration, sample_rate = encode_audio_base64(
            final_output_path,
            format=request.output_audio_format,
        )

        return TTSResponse(
            audio=audio_b64,
            format=request.output_audio_format,
            duration=duration,
            inference_time_seconds=0.0,  # TODO: track actual inference time
            sample_rate=sample_rate,
            num_segments=num_segments,
            metadata={
                "text_length": len(request.text),
                "audio_format": request.output_audio_format,
                "model_version": "IndexTTSv2",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS generation failed: {str(e)}",
        )
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {temp_file}: {e}")


@app.post("/synthesize/segment-preview", response_model=SegmentPreviewResponse, tags=["TTS"])
async def preview_segmentation(request: SegmentPreviewRequest):
    """
    Preview how text will be segmented before generation.

    This is useful for understanding how long texts will be split into
    multiple generation segments based on token limits.

    Args:
        request: Text and segmentation parameters

    Returns:
        List of text segments and token counts

    Raises:
        HTTPException: If model not loaded or segmentation fails
    """
    if tts_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS model not loaded",
        )

    try:
        # First tokenize the text
        text_tokens_list = tts_model.tokenizer.tokenize(request.text)

        # Then split into segments
        segments_tokens = tts_model.tokenizer.split_segments(
            text_tokens_list,
            max_text_tokens_per_segment=request.max_text_tokens_per_segment,
        )

        # Convert token lists back to text and get counts
        segments = [
            tts_model.tokenizer.convert_tokens_to_string(seg)
            for seg in segments_tokens
        ]
        tokens_per_segment = [len(seg) for seg in segments_tokens]

        return SegmentPreviewResponse(
            segments=segments,
            num_segments=len(segments),
            tokens_per_segment=tokens_per_segment,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation preview failed: {str(e)}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    traceback.print_exc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": f"Internal server error: {str(exc)}",
        },
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="IndexTTS2 FastAPI Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="checkpoints",
        help="Path to model directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="checkpoints/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 for faster inference with lower VRAM",
    )
    parser.add_argument(
        "--cuda-kernel",
        action="store_true",
        help="Use custom CUDA kernels for BigVGAN",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Update config globally BEFORE starting the server
    # This must happen before uvicorn.run() so the lifespan can access it
    global config
    config["model_dir"] = args.model_dir
    config["cfg_path"] = args.config
    config["use_fp16"] = args.fp16
    config["use_cuda_kernel"] = args.cuda_kernel

    print("=" * 80)
    print("IndexTTS2 FastAPI Server")
    print("=" * 80)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    print(f"FP16: {args.fp16}")
    print(f"CUDA Kernel: {args.cuda_kernel}")
    print("=" * 80)

    # Run the server
    import uvicorn

    uvicorn.run(
        app,  # Pass the app object directly, not the string
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
