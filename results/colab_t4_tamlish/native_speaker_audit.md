# Native-speaker audit (Tamil)

Reviewer: the author (native Tamil speaker). Purpose: check that the automatic metrics agree with a human reading.

| Clip | Model | Output | Verdict |
|------|-------|--------|---------|
| ta_11 (என் account-ல balance check பண்ணணும்) | IndicConformer 600M | என் அக்கவுண்டில் பேலன்ஸ் செக் பண்ணணும் | Fully correct; English words in Tamil script |
| ta_11 | Whisper large-v3, turbo, Tamil medium | ஏன் … | Error: ஏன் ("why") instead of என் ("my") changes the meaning |

The phonetic metric scores ஏன்/என் as close (one vowel apart); a human reader marks it as a meaning-changing error.
This is a known limitation of the script-agnostic metric and is stated in the report.
