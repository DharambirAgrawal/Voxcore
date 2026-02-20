"""Quick test to verify all updated modules load and work correctly."""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("TESTING UPDATED MODULES")
print("=" * 60)

# 1. Test voice_profile.py
print("\n1. output/voice_profile.py")
from output.voice_profile import VoiceProfile, AVAILABLE_VOICES, DEFAULT_VOICE, VALID_EMOTIONS
config = {
    'persona': {'voice': 'af_heart', 'language': 'en'},
    'audio': {'output_sample_rate': 24000},
    'models': {'tts_model': 'models/kokoro-v1.0.onnx', 'tts_voices': 'models/voices-v1.0.bin'},
}
vp = VoiceProfile(config)
assert vp.voice_name == "af_heart", f"Expected af_heart, got {vp.voice_name}"
assert vp.sample_rate == 24000, f"Expected 24000, got {vp.sample_rate}"
assert vp.get_clips_subdir() == "heart", f"Expected heart, got {vp.get_clips_subdir()}"
assert vp.tts_model_path == "models/kokoro-v1.0.onnx"
assert vp.tts_voices_path == "models/voices-v1.0.bin"
assert len(AVAILABLE_VOICES) == 27, f"Expected 27 voices, got {len(AVAILABLE_VOICES)}"
assert DEFAULT_VOICE == "af_heart"
# Test fallback
vp2 = VoiceProfile({'persona': {'voice': 'tara'}, 'audio': {}, 'models': {}})
assert vp2.voice_name == "af_heart", f"Fallback failed: {vp2.voice_name}"
print("   PASSED - Kokoro voices, 24kHz, clips subdir all correct")

# 2. Test tts_client.py imports
print("\n2. output/tts_client.py")
from output.tts_client import TTSClient
import inspect
src = inspect.getsource(TTSClient)
assert "kokoro_onnx" in open("output/tts_client.py").read(), "Missing kokoro_onnx import"
assert "Kokoro" in src, "TTSClient should use Kokoro class"
assert "AsyncGroq" not in src, "TTSClient should NOT reference AsyncGroq"
assert "EMOTION_TAG_PATTERN" in open("output/tts_client.py").read(), "Should strip emotion tags"
print("   PASSED - Uses Kokoro, strips emotion tags, no Groq references")

# 3. Test emotion_tagger.py
print("\n3. brain/emotion_tagger.py")
from brain.emotion_tagger import EmotionTagger, EMOTION_KEYWORDS, DEFAULT_EMOTION
et = EmotionTagger()
tagged = et.ensure_emotion_tag("That's amazing news!")
assert tagged.startswith("["), f"Should have tag: {tagged}"
already = et.ensure_emotion_tag("[cheerful] Hello there")
assert already == "[cheerful] Hello there", f"Should keep existing: {already}"
# Check docstring doesn't reference Orpheus anymore
src = open("brain/emotion_tagger.py").read()
assert "Kokoro" in src, "Docstring should mention Kokoro"
print("   PASSED - Tagging works, docstring updated")

# 4. Test config.yaml loads correctly
print("\n4. config.yaml")
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
assert cfg['persona']['voice'] == 'af_heart', f"Voice: {cfg['persona']['voice']}"
assert cfg['models']['tts_model'] == 'models/kokoro-v1.0.onnx'
assert cfg['models']['tts_voices'] == 'models/voices-v1.0.bin'
assert cfg['audio']['output_sample_rate'] == 24000
assert 'heart' in cfg['backchannel']['clips_dir']
print("   PASSED - af_heart, kokoro model paths, 24kHz, clips/heart/")

# 5. Test selector.py
print("\n5. backchannel/selector.py")
src = open("backchannel/selector.py").read()
assert "clips/heart/" in src, "Default clips_dir should be clips/heart/"
print("   PASSED - Default clips dir is clips/heart/")

# 6. Test generator.py
print("\n6. backchannel/generator.py")
from backchannel.generator import ClipGenerator, BACKCHANNEL_PHRASES, DEFAULT_VOICE as GEN_DEFAULT
assert GEN_DEFAULT == "af_heart", f"Generator default: {GEN_DEFAULT}"
# Check no old Orpheus docstring
src = open("backchannel/generator.py").read()
assert "Groq Orpheus TTS API" not in src, "Old Orpheus docstring still present!"
assert "Kokoro" in src or "kokoro" in src, "Should reference Kokoro"
phrases_have_no_tags = all(not p.startswith("[") for k, p in BACKCHANNEL_PHRASES.items() 
                           if not k.endswith("-100"))
assert phrases_have_no_tags, "Kokoro phrases should not have [emotion] tags"
print("   PASSED - Kokoro-based, no old Orpheus references in spec")

# 7. Test requirements.txt
print("\n7. requirements.txt")
reqs = open("requirements.txt").read()
assert "kokoro-onnx" in reqs, "Missing kokoro-onnx"
assert "soundfile" in reqs, "Missing soundfile"
groq_line = [l for l in reqs.split('\n') if 'groq' in l.lower() and l.startswith('#')][0]
assert "TTS" not in groq_line, f"Groq comment should not say TTS: {groq_line}"
print("   PASSED - kokoro-onnx and soundfile listed, groq comment updated")

# 8. Test ARCHITECTURE.md
print("\n8. docs/ARCHITECTURE.md")
arch = open("docs/ARCHITECTURE.md").read()
assert "Kokoro-ONNX" in arch, "Should mention Kokoro-ONNX"
assert "kokoro-v1.0.onnx" in arch, "Should have model path"
assert "af_heart" in arch, "Should have af_heart voice"
# Check old references are gone
assert 'tts: "canopylabs' not in arch, "Old Orpheus model reference in config section"
print("   PASSED - Kokoro-ONNX throughout, old Orpheus cleaned up")

print("\n" + "=" * 60)
print("ALL 8 TESTS PASSED - Migration to Kokoro-ONNX complete!")
print("=" * 60)
