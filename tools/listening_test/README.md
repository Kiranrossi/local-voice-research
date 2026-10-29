# Blind listening test (TTS)

32 clips: 8 sentences (4 everyday, 4 stress) x 4 TTS models, loudness-normalised (EBU R128, -20 LUFS)
and converted to 64 kbps MP3 so volume and file quality don't bias ratings. Each listener hears all 32 in a
random order, with no sentence played twice in a row and no model names shown.

Listeners rate naturalness on the standard 5-point ACR scale (1 Bad to 5 Excellent) and word accuracy
(yes / mostly / no), then send back a results code: `VLT1;<name>;<n>;<sentence>_<model>=<score><accuracy>,...`

Model codes: m1 = Kokoro-82M, m2 = Chatterbox, m3 = Qwen3-TTS 0.6B, m4 = VoxCPM2.
Audio files are not committed (see .gitignore); they are generated from `results/colab_t4/tts`.
