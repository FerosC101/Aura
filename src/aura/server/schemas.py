"""Request and response models for the control API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModeRequest(BaseModel):
    mode: str = Field(description="focus | commute | walk | social | custom")


class EnabledRequest(BaseModel):
    enabled: bool


class SuppressionRequest(BaseModel):
    strength: float = Field(ge=0.0, le=1.5)


class CategoryGainRequest(BaseModel):
    category: str
    #: ``None`` clears the override and returns the category to its mode default.
    gain_db: float | None = Field(default=None, ge=-40.0, le=12.0)
    learn: bool = True


class AudioPayload(BaseModel):
    """Raw mono audio for enrolment.

    Sent as plain float samples rather than an encoded file so the server never
    needs a decoder in the request path. The samples are used to compute an
    embedding and are then discarded -- they are never written to disk.
    """

    samples: list[float] = Field(description="Mono PCM in [-1, 1]")
    sample_rate: int = Field(default=48000, gt=0)


class EnrolSpeakerRequest(BaseModel):
    name: str
    clips: list[AudioPayload] = Field(default_factory=list)
    consent: bool = Field(
        default=False,
        description="Must be true. Speaker enrolment requires explicit consent.",
    )
    #: Build the profile from a synthesized voice instead of a recording.
    #: Used by the demo and the tests so Person Lock can be shown without a mic.
    demo_voice: str | None = None


class SelectSpeakerRequest(BaseModel):
    profile_id: str | None = None


class NameEnrolRequest(BaseModel):
    word: str
    clips: list[AudioPayload] = Field(default_factory=list)
    demo_word: str | None = None


class NameToggleRequest(BaseModel):
    enabled: bool


class ContextRequest(BaseModel):
    state: str | None = Field(
        default=None,
        description="stationary | walking | commuting | studying | social; null to auto-detect",
    )


class MotionRequest(BaseModel):
    speed_mps: float | None = None
    step_rate_hz: float | None = None
    place_category: str | None = None


class PrivacyRequest(BaseModel):
    store_speaker_embeddings: bool | None = None
    allow_motion_context: bool | None = None
    allow_cloud_inference: bool | None = None
    allow_name_awareness: bool | None = None
    auto_context_detection: bool | None = None
    retain_scene_history: bool | None = None


class OnboardingRequest(BaseModel):
    user_name: str = ""
    primary_uses: list[str] = Field(default_factory=list)
    safety_categories: list[str] = Field(default_factory=list)
    preferred_mode: str = "commute"
    completed: bool = True


class DemoStartRequest(BaseModel):
    scene: str = "psc_demo"
    #: raw | anc | aura
    comparison: str = "aura"


class StartEngineRequest(BaseModel):
    input_device: int | str | None = None
    output_device: int | str | None = None
    #: Start from a demo scene instead of the microphone.
    demo_scene: str | None = None
