"""Control API and the privacy guarantees it must enforce."""

import pytest
from fastapi.testclient import TestClient

from aura.config import EngineConfig
from aura.server.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(EngineConfig(
        sample_rate=16000, block_size=320,
        profile_dir=str(tmp_path / "profile"), publish_rate_hz=1000.0,
    ))
    with TestClient(app) as c:
        yield c


def test_health_and_status(client):
    assert client.get("/api/health").json()["ok"]
    body = client.get("/api/status").json()
    for key in ("running", "enabled", "mode", "backends", "metrics", "safety"):
        assert key in body


def test_reference_data(client):
    modes = client.get("/api/modes").json()["modes"]
    assert {m["mode"] for m in modes} == {"care", "focus", "commute", "walk", "social", "custom"}
    cats = client.get("/api/categories").json()["categories"]
    assert len(cats) >= 9
    assert any(c["safety_critical"] for c in cats)


def test_mode_switching(client):
    r = client.post("/api/mode", json={"mode": "walk"})
    assert r.status_code == 200
    assert r.json()["mode"] == "walk"
    assert client.post("/api/mode", json={"mode": "nonsense"}).status_code == 400


def test_master_toggle(client):
    assert client.post("/api/engine/enabled", json={"enabled": False}).json()["enabled"] is False
    assert client.post("/api/engine/enabled", json={"enabled": True}).json()["enabled"] is True


def test_mixer_override_and_reset(client):
    rows = client.post("/api/mixer", json={"category": "engine_noise", "gain_db": -20.0}).json()
    row = next(r for r in rows["categories"] if r["category"] == "engine_noise")
    assert row["override_db"] == -20.0

    rows = client.post("/api/mixer", json={"category": "engine_noise", "gain_db": None}).json()
    row = next(r for r in rows["categories"] if r["category"] == "engine_noise")
    assert row["override_db"] is None


def test_mixer_cannot_silence_a_safety_sound(client):
    """The API must enforce the same guarantee the engine does."""
    rows = client.post("/api/mixer", json={"category": "emergency_siren", "gain_db": -40.0}).json()
    row = next(r for r in rows["categories"] if r["category"] == "emergency_siren")
    assert row["gain_db"] >= -3.0
    assert row["always_audible"]


def test_speaker_enrolment_requires_consent(client):
    r = client.post("/api/speakers", json={"name": "Vince", "consent": False})
    assert r.status_code == 403
    assert "consent" in r.json()["detail"].lower()


def test_speaker_enrolment_and_deletion(client):
    r = client.post("/api/speakers", json={"name": "Vince", "consent": True, "demo_voice": "vince"})
    assert r.status_code == 200
    lock = r.json()["person_lock"]
    assert lock["profiles"][0]["name"] == "Vince"
    pid = lock["profiles"][0]["id"]

    # The API must never hand back the raw embedding vector.
    assert "centroid" not in r.text

    assert client.post("/api/speakers/select", json={"profile_id": pid}).status_code == 200
    assert client.delete(f"/api/speakers/{pid}").json()["profiles"] == []
    assert client.delete(f"/api/speakers/{pid}").status_code == 404


def test_name_awareness_requires_opt_in(client):
    r = client.post("/api/name/enrol", json={"word": "Vince", "demo_word": "vince"})
    assert r.status_code == 403

    client.patch("/api/privacy", json={"allow_name_awareness": True})
    assert client.post("/api/name/enrol", json={"word": "Vince", "demo_word": "vince"}).status_code == 200


def test_motion_requires_opt_in(client):
    assert client.post("/api/context/motion", json={"speed_mps": 1.4}).status_code == 403
    client.patch("/api/privacy", json={"allow_motion_context": True})
    assert client.post("/api/context/motion", json={"speed_mps": 1.4}).status_code == 200


def test_privacy_centre_describes_itself(client):
    body = client.get("/api/privacy").json()
    assert body["processes"] and body["stores"] and body["never_stores"]
    assert not body["settings"]["allow_cloud_inference"]
    joined = " ".join(body["never_stores"]).lower()
    assert "raw" in joined and "audio" in joined


def test_delete_all_data(client):
    client.post("/api/speakers", json={"name": "Vince", "consent": True, "demo_voice": "vince"})
    client.post("/api/mixer", json={"category": "music", "gain_db": -12.0})
    r = client.delete("/api/privacy/data").json()
    assert r["speaker_profiles_deleted"] >= 1
    assert client.get("/api/speakers").json()["profiles"] == []


def test_onboarding_sets_up_the_profile(client):
    r = client.post("/api/onboarding", json={
        "user_name": "Vince", "primary_uses": ["commuting"],
        "safety_categories": ["emergency_siren", "vehicle_horn"],
        "preferred_mode": "commute", "completed": True,
    })
    body = r.json()
    assert body["onboarded"] and body["user_name"] == "Vince"
    safety = client.get("/api/status").json()["safety"]["protected"]
    assert set(safety) == {"emergency_siren", "vehicle_horn"}


def test_demo_scenes_are_listed(client):
    body = client.get("/api/demo/scenes").json()
    assert {s["key"] for s in body["scenes"]} >= {"psc_demo", "jeepney", "classroom", "urban_walk"}
    assert {c["key"] for c in body["comparisons"]} == {"raw", "anc", "aura"}


def test_demo_comparison_modes_set_bypass(client):
    """Raw and ANC must bypass AURA, or the comparison would be meaningless."""
    client.post("/api/demo/start", json={"scene": "psc_demo", "comparison": "raw"})
    assert client.get("/api/status").json()["enabled"] is False
    client.post("/api/demo/start", json={"scene": "psc_demo", "comparison": "aura"})
    assert client.get("/api/status").json()["enabled"] is True
    client.post("/api/engine/stop")


def test_unknown_demo_scene_is_rejected(client):
    assert client.post("/api/demo/start", json={"scene": "nope"}).status_code == 404


def test_websocket_streams_scenes(client):
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["event"] == "status"
        ws.send_json({"action": "status"})
        assert ws.receive_json()["event"] == "status"


def test_learning_endpoint(client):
    client.post("/api/mode", json={"mode": "commute"})
    for _ in range(5):
        client.post("/api/mixer", json={"category": "announcement", "gain_db": 8.0})
    prefs = client.get("/api/learning").json()["preferences"]
    assert any(p["category"] == "announcement" for p in prefs)
    assert client.delete("/api/learning").json()["forgotten"] >= 1
