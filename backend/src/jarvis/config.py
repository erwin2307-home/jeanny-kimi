"""Application settings, loaded from env with sensible defaults."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"


class Settings(BaseModel):
    ws_host: str = Field(default_factory=lambda: os.getenv("JARVIS_WS_HOST", "127.0.0.1"))
    ws_port: int = Field(default_factory=lambda: int(os.getenv("JARVIS_WS_PORT", "8765")))

    sample_rate: int = Field(default_factory=lambda: int(os.getenv("JARVIS_SAMPLE_RATE", "16000")))
    frame_ms: int = Field(default_factory=lambda: int(os.getenv("JARVIS_FRAME_MS", "30")))

    wake_word: str = Field(default_factory=lambda: os.getenv("JARVIS_WAKE_WORD", "jarvis"))
    wake_threshold: float = Field(default_factory=lambda: float(os.getenv("JARVIS_WAKE_THRESHOLD", "0.5")))
    wake_model_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("JARVIS_WAKE_MODEL", str(MODELS_DIR / "openwakeword" / "hey_jarvis_v0.1.onnx"))
        )
    )

    vad_threshold: float = Field(default_factory=lambda: float(os.getenv("JARVIS_VAD_THRESHOLD", "0.65")))
    silence_ms: int = Field(default_factory=lambda: int(os.getenv("JARVIS_SILENCE_MS", "800")))
    # Hard wall-clock cap on LISTENING, independent of VAD/silence tuning.
    # Background noise can keep resetting the silence counter so speech_end
    # never fires naturally; without a cap the recording just grows until
    # STT hallucinates nonsense from a huge buried-in-noise buffer.
    max_listen_s: float = Field(
        default_factory=lambda: float(os.getenv("JARVIS_MAX_LISTEN_S", "15.0"))
    )
    # Minimum recognised word count to treat a transcript as a real query.
    # Set to 0 to disable the filter. 2 catches one-word noise hallucinations
    # ("yeah.", "okay.") while still letting "what time" through.
    min_transcript_words: int = Field(
        default_factory=lambda: int(os.getenv("JARVIS_MIN_TRANSCRIPT_WORDS", "2"))
    )

    whisper_model: str = Field(default_factory=lambda: os.getenv("JARVIS_WHISPER_MODEL", "small"))
    whisper_compute: str = Field(default_factory=lambda: os.getenv("JARVIS_WHISPER_COMPUTE", "int8"))
    whisper_lang: str = Field(default_factory=lambda: os.getenv("JARVIS_WHISPER_LANG", "de"))
    # Biases Whisper's decoding toward this vocabulary so domain jargon
    # (brand names, loanwords) gets recognized/spelled correctly instead
    # of being mangled into phonetically-similar German nonsense — e.g.
    # "BeadChip" was observed coming out as "Watschup".
    whisper_initial_prompt: str = Field(
        default_factory=lambda: os.getenv(
            "JARVIS_WHISPER_INITIAL_PROMPT",
            "Illumina, BeadChip, Infinium, Proteinase K, Formamid, RA1, "
            "MA1, MA2, XC1, XC2, XC3, XC4, PB1, PB2, iScan, HiScan, "
            "Genotypisierung.",
        )
    )

    llm_host: str = Field(default_factory=lambda: os.getenv("JARVIS_LLM_HOST", "http://localhost:11434"))
    # Reverted to the original llama3.2:1b default (this fork briefly
    # tried qwen3:4b-instruct-2507-q4_K_M, but it was too slow/unresponsive
    # in practice on this hardware). llama3.2:1b stays the fast local/
    # fallback tier for simple queries; complex queries still route to
    # DeepSeek under "complexity" strategy (see route_strategy/use_cloud
    # below) — that's unaffected by this model choice.
    llm_model: str = Field(default_factory=lambda: os.getenv("JARVIS_LLM_MODEL", "llama3.2:1b"))
    llm_timeout_s: float = Field(
        default_factory=lambda: float(os.getenv("JARVIS_LLM_TIMEOUT_S", "300"))
    )
    llm_system_prompt: str = Field(
        default_factory=lambda: os.getenv(
            "JARVIS_SYSTEM_PROMPT",
            "Du bist Jeanny, eine knappe, trocken-humorvolle KI-Assistentin. "
            "Antworte auf Deutsch, natürlich und kurz. Maximal zwei kurze "
            "Sätze, außer der Nutzer bittet ausdrücklich um mehr Details. "
            "Wenn jemand dich mit 'Jeanny' anspricht oder fragt, wie du "
            "heißt, antworte, dass du Jeanny bist.",
        )
    )
    llm_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("JARVIS_LLM_MAX_TOKENS", "200"))
    )

    # Filler: play a short acknowledgement phrase while STT/LLM are running so
    # Jarvis seems responsive. Then the real answer is spoken with a prefix.
    use_filler: bool = Field(
        default_factory=lambda: os.getenv("JARVIS_USE_FILLER", "1") not in ("0", "false", "False", "")
    )
    response_prefix: str = Field(
        default_factory=lambda: os.getenv(
            "JARVIS_RESPONSE_PREFIX", "Hier ist, was ich gefunden habe. "
        )
    )

    # Hybrid routing. Set JARVIS_USE_CLOUD=0 to force local-only even if a
    # Groq/DeepSeek key is present (privacy / offline / cost reasons).
    use_cloud: bool = Field(
        default_factory=lambda: os.getenv("JARVIS_USE_CLOUD", "1") not in ("0", "false", "False", "")
    )
    deepseek_model: str = Field(
        default_factory=lambda: os.getenv("JARVIS_DEEPSEEK_MODEL", "deepseek-chat")
    )
    deepseek_base_url: str = Field(
        default_factory=lambda: os.getenv("JARVIS_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    )
    # Groq: OpenAI-compatible, much lower latency than DeepSeek (LPU hardware).
    # Preferred cloud backend when a key is available — see app.py wiring.
    groq_model: str = Field(
        default_factory=lambda: os.getenv("JARVIS_GROQ_MODEL", "llama-3.1-8b-instant")
    )
    groq_base_url: str = Field(
        default_factory=lambda: os.getenv("JARVIS_GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    complexity_threshold: float = Field(
        default_factory=lambda: float(os.getenv("JARVIS_COMPLEXITY_THRESHOLD", "1.0"))
    )
    # Hard token cap for the fast model's reaction-filler call (complex
    # queries only) — separate from llm_max_tokens, which is sized for full
    # answers. A small local model won't reliably follow "react briefly,
    # don't answer" on its own; this bounds both output length and, more
    # importantly, generation time, so the "fast" reaction stays fast.
    reaction_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("JARVIS_REACTION_MAX_TOKENS", "40"))
    )
    # "cloud_first": always cloud (Groq/DeepSeek), fallback to local on failure.
    # "complexity" (default): simple queries -> fast local model directly;
    # complex queries -> cloud, with the local model reacting to the
    # transcript as a filler while cloud computes (see orchestrator.py).
    route_strategy: str = Field(
        default_factory=lambda: os.getenv("JARVIS_ROUTE_STRATEGY", "complexity")
    )

    # German female voice. "low" quality tier — Piper's only German female
    # voices (eva_k, kerstin, ramona) are low/x-low; "medium"/"high" only
    # exist for the male Thorsten voice.
    tts_voice: str = Field(default_factory=lambda: os.getenv("JARVIS_TTS_VOICE", "de_DE-kerstin-low"))
    tts_model_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("JARVIS_TTS_MODEL", str(MODELS_DIR / "piper" / "de_DE-kerstin-low.onnx"))
        )
    )
    # >1.0 slows speech down (Piper/VITS: multiplies phoneme duration).
    # A low-quality-tier voice reads clearer when spoken more slowly.
    tts_length_scale: float = Field(
        default_factory=lambda: float(os.getenv("JARVIS_TTS_LENGTH_SCALE", "1.3"))
    )
    # "edge": Microsoft Edge's free neural TTS (edge-tts package, no API
    # key) — needs internet, but real neural quality. "piper": fully
    # offline, but the only German female Piper voices are low/x-low
    # tier (see tts_voice above). Chosen over Piper after Ramona/Kerstin
    # were both rejected as too low quality.
    tts_backend: str = Field(default_factory=lambda: os.getenv("JARVIS_TTS_BACKEND", "edge"))
    tts_edge_voice: str = Field(
        default_factory=lambda: os.getenv("JARVIS_TTS_EDGE_VOICE", "de-DE-KatjaNeural")
    )
    # Must match whichever backend is active: edge-tts/Katja outputs
    # 24000 Hz; Piper voices output 22050 Hz. RealSpeaker is constructed
    # with this value so playback pitch/speed is correct.
    tts_sample_rate: int = Field(
        default_factory=lambda: int(os.getenv("JARVIS_TTS_SAMPLE_RATE", "24000"))
    )

    # Remote text-chat API (remote_api.py) — a separate, minimal FastAPI
    # app meant to be tunneled to the public internet so a Streamlit
    # Cloud frontend can reach this machine's LLM. Deliberately exposes
    # ONLY plain chat (no qa_matcher side effects beyond fastpath_answer)
    # — this fork dropped the coding/games generation features entirely
    # for speed; see CLAUDE.md. Required (no default) so the server
    # refuses to start wide open; set via env, never committed.
    remote_api_key: str = Field(
        default_factory=lambda: os.getenv("JARVIS_REMOTE_API_KEY", "")
    )
    remote_api_port: int = Field(
        default_factory=lambda: int(os.getenv("JARVIS_REMOTE_API_PORT", "8800"))
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
