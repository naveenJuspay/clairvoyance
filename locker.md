# Speaker Verification System Setup Guide

## Overview
This guide covers the complete setup for the speaker verification system that provides real-time speaker authentication in voice conversations.

## Features
- **Auto-enrollment**: User gets enrolled after 2 queries automatically
- **Real-time verification**: Blocks unauthorized speakers from reaching STT
- **RTVI integration**: Sends events to frontend for user feedback
- **Fallback systems**: Multiple layers of audio processing (pyannote.audio → SpeechBrain → energy-based)
- **Audio debugging**: Saves debug files for troubleshooting

## Environment Variables

Add these to your `.env` file:

```bash
# Speaker Verification Configuration
ENABLE_SPEAKER_VERIFICATION=true
SPEAKER_VERIFICATION_TARGET_ID=user
SPEAKER_VERIFICATION_SIMILARITY_THRESHOLD=0.60

# HuggingFace Authentication (CRITICAL)
HF_TOKEN=
```

## Required Dependencies

Install the speaker verification dependencies:

```bash
pip install -r speaker_verification_requirements.txt
```

The requirements include:
- `speechbrain>=0.5.15` - Core speaker recognition
- `pyannote.audio>=3.1.0` - Speaker diarization (preferred)
- `huggingface-hub>=0.16.0` - Model authentication
- `transformers>=4.21.0` - Required by pyannote
- `torch` and `torchaudio` - Neural network backends

## HuggingFace Setup

### 1. Token Requirements
- Must be a **classic token** with full permissions (not fine-grained)
- Must have access to **gated repositories**
- Current working token: ``

### 2. Model Access Permissions
Visit these HuggingFace model pages and accept the user conditions:
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.1

### 3. Verification Test
```bash
python -c "
from huggingface_hub import hf_hub_download
config = hf_hub_download('pyannote/speaker-diarization-3.1', 'config.yaml', token='')
print('✅ HuggingFace access working')
"
```

## File Structure

The system creates these directories automatically:
```
clairvoyance/
├── enrollments/           # Speaker enrollment files
├── debug_audio/          # Debug audio files for troubleshooting
├── pretrained_models/    # Downloaded SpeechBrain models
└── app/agents/voice/automatic/
    └── processors/
        └── speaker_verification.py  # Main implementation
```

## Configuration Details

### Similarity Threshold
- **Default**: 0.60 (60% similarity required)
- **Lower values**: More permissive (easier to pass verification)
- **Higher values**: More strict (harder to pass verification)
- **Recommended range**: 0.55-0.70

### Speaker ID
- **Default**: "user" 
- Change `SPEAKER_VERIFICATION_TARGET_ID` to use different speaker names

## System Behavior

### Enrollment Phase (First 2 Queries)
1. User speaks → System saves enrollment audio
2. Audio is processed with diarization to extract clean speech
3. Speaker embeddings are extracted and stored
4. RTVI event `speaker-enrollment-progress` sent to frontend
5. After 2nd query → User gets locked, system switches to verification mode

### Verification Phase (After Enrollment)
1. User speaks → Audio is processed and compared with enrolled embeddings
2. If similarity ≥ threshold → Audio forwarded to STT
3. If similarity < threshold → Audio blocked, interruption sent
4. RTVI events sent: `speaker-verification-success` or `speaker-verification-failed`

## Audio Processing Pipeline

1. **Primary**: pyannote.audio diarization (requires HF token)
2. **Fallback 1**: SpeechBrain diarization (if available)
3. **Fallback 2**: Energy-based voice activity detection

## Troubleshooting

### Common Issues

#### HuggingFace Authentication Fails
```
Error: 403 Forbidden: Please enable access to public gated repositories
```
**Solution**: Use a classic token with gated repository access

#### Speaker Getting Rejected
```
Similarity: 0.422, Threshold: 0.60 → REJECTED
```
**Solutions**:
- Lower threshold: `SPEAKER_VERIFICATION_SIMILARITY_THRESHOLD=0.50`
- Check debug audio files for quality issues
- Re-enroll in quiet environment

#### No Audio Segments Found
```
No energy-based segments found, using full audio
```
**Solution**: Speak louder or check microphone levels

### Debug Files
Check these locations for troubleshooting:
- `debug_audio/` - Raw audio files with timestamps
- `enrollments/` - Enrollment audio files
- `server.log` - Detailed logging with `[SPEAKER_VERIFICATION]` tags

### Log Analysis
Look for these key log messages:
- `✅ Pyannote.audio pipeline loaded successfully` - Diarization working
- `Successfully logged in to HuggingFace Hub` - Auth working
- `🔒 USER LOCKED` - Enrollment completed
- `✅ VERIFIED` / `❌ REJECTED` - Verification results

## Integration Points

### RTVI Events Sent to Frontend
```javascript
// Enrollment progress
{ type: "speaker-enrollment-progress", data: { step: 1, total: 2 } }

// User locked after enrollment
{ type: "speaker-verification-locked", data: { speaker_id: "user" } }

// Verification results
{ type: "speaker-verification-success", data: { similarity: 0.75 } }
{ type: "speaker-verification-failed", data: { similarity: 0.45 } }
```

### Audio Flow
```
Microphone → VAD → Speaker Verification → STT → LLM
                 ↓ (if rejected)
                 Silence/Interruption
```

## Performance Notes

- **Enrollment**: Requires 2 voice samples (auto-collected)
- **Verification**: Real-time (~100ms processing)
- **Memory**: ~500MB for loaded models
- **Storage**: ~10MB per enrolled speaker

## Security Considerations

- Speaker embeddings are stored locally (not transmitted)
- Debug audio files contain voice data (clean up regularly)
- HF token has repository access (protect appropriately)

## Updates and Maintenance

### Model Updates
Models are cached locally. To update:
```bash
rm -rf ~/.cache/huggingface/transformers/models--pyannote*
```

### Re-enrollment
To re-enroll a speaker:
```bash
rm -rf enrollments/user_*
# Restart service - system will auto-enroll on next 2 queries
```

### Threshold Tuning
Monitor logs for rejection rates and adjust threshold accordingly:
```bash
grep "VERIFICATION RESULT" server.log | tail -20
```