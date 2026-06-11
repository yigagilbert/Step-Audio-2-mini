# Luganda-to-English Speech Translation Evaluation Report

Date: 2026-06-11

## Executive Summary

This evaluation compares three systems on Luganda-to-English speech translation:

- **Base model**: `stepfun-ai/Step-Audio-2-mini` without the Luganda LoRA adapter.
- **Fine-tuned Step-Audio model**: Step-Audio-2-mini with the Luganda-to-English LoRA adapter.
- **Cascade pipeline**: Luganda ASR with `Sunbird/asr-whisper-large-v3-salt`,
  text translation with `Sunbird/translate-nllb-3.3b-salt`, and English TTS with
  `Sunbird/orpheus-3b-tts-multilingual`.

The base model is not usable for this task without adaptation. It scored near zero
on BLEU, produced very high WER, and emitted no valid speech tokens for audio
synthesis. Fine-tuning changed the model from unusable to competitive: BLEU rose
from `0.012` to `32.530`, chrF from `5.152` to `54.535`, and COMET from `0.386`
to `0.717`.

The cascade pipeline is the strongest system on text and semantic translation
metrics. It outperforms the fine-tuned Step-Audio model on BLEU, chrF, WER,
COMET, and BLASER. However, the fine-tuned Step-Audio model performs better on
SpeechBERTScore F1, suggesting stronger speech-level similarity to the English
reference audio under the WavLM-based proxy metric. MCD favors the cascade, but
MCD should be interpreted cautiously because the systems use different speech
generation mechanisms and voices.

Overall, the cascade is the current best quality baseline, while the fine-tuned
Step-Audio model is a successful end-to-end system that substantially closes the
gap from the base model and avoids the operational complexity of ASR + MT + TTS.

## Evaluation Setup

The evaluation was run on the validation split with a 200-sample limit for text
metrics and BLASER. Speech metrics were computed on the aligned subset of 197
samples because the fine-tuned Step-Audio audio synthesis produced valid WAV
files for 197 of the 200 examples.

Text metrics:

- **BLEU**: corpus-level n-gram overlap with reference English.
- **chrF**: character n-gram F-score, more tolerant of morphology and phrasing.
- **WER on text channel**: word error rate between generated English and reference
  English; lower is better.
- **COMET**: learned MT quality estimate using source, hypothesis, and reference.
- **BLASER 2.0 ref/QE**: semantic evaluation from SONAR embeddings. Higher is better.

Speech metrics:

- **SpeechBERTScore-style WavLM precision/recall/F1**: embedding similarity between
  generated English speech and reference English speech. Higher is better.
- **MCD**: MFCC + DTW mel-cepstral distortion. Lower is better, but the absolute
  values should be treated as diagnostic rather than directly comparable to
  conventional same-speaker MCD results.

## Main Text Results

All values below are computed on 200 validation examples.

| System | BLEU ↑ | chrF ↑ | WER ↓ | COMET ↑ | BLASER ref ↑ | BLASER QE ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Base Step-Audio-2-mini | 0.012 | 5.152 | 10.702 | 0.386 | 1.713 | 2.164 |
| Fine-tuned Step-Audio LoRA | 32.530 | 54.535 | 0.574 | 0.717 | 3.762 | 3.723 |
| Cascade ASR + MT | 36.778 | 57.971 | 0.521 | 0.737 | 3.839 | 3.776 |

### Text Result Interpretation

The base model performs extremely poorly. Its BLEU and chrF are close to zero,
WER is above `10`, and qualitative inspection shows frequent unrelated outputs
and repetitive hallucination. This is expected for a base model that has not been
adapted to this Luganda-to-English S2ST task.

The fine-tuned Step-Audio model shows a very large improvement over the base
model. Relative to the base model, it improves BLEU by `+32.518`, chrF by
`+49.383`, COMET by `+0.331`, and BLASER ref by `+2.050`. This confirms that
the LoRA adapter learned the task and substantially improved translation quality.

The cascade remains ahead of the fine-tuned model on text quality. It improves
over the fine-tuned model by `+4.248` BLEU, `+3.436` chrF, `-0.053` absolute WER,
`+0.019` COMET, `+0.077` BLASER ref, and `+0.052` BLASER QE. The margin is
consistent across lexical, semantic, and learned evaluation metrics.

## Speech Results

Speech metrics were computed on the 197 examples for which both the fine-tuned
Step-Audio model and the cascade had generated audio.

| System | Count | chrF ↑ | SpeechBERT P ↑ | SpeechBERT R ↑ | SpeechBERT F1 ↑ | MCD ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Fine-tuned Step-Audio LoRA | 197 | 54.582 | 0.644 | 0.648 | 0.645 | 629.718 |
| Cascade ASR + MT + TTS | 197 | 57.893 | 0.603 | 0.622 | 0.612 | 613.212 |

The base model is excluded from speech metrics because it generated no valid
speech-token sequences. Its synthesis stage wrote `0` WAV files, so SpeechBERTScore
and MCD cannot be computed for it.

### Speech Result Interpretation

The fine-tuned Step-Audio model is stronger on SpeechBERTScore F1 (`0.645` vs
`0.612`). This suggests that, among the systems with generated audio, the
fine-tuned end-to-end model produces speech that is more similar to the reference
English speech in WavLM embedding space.

The cascade is better on MCD (`613.212` vs `629.718`, lower is better). This may
indicate closer spectral-envelope similarity under the MFCC + DTW computation.
However, because the cascade uses Orpheus TTS while the fine-tuned Step-Audio
model uses Step-Audio token2wav, MCD is strongly affected by vocoder, speaker,
duration, and prosody differences. It should be treated as a supporting audio
diagnostic, not as a definitive translation-quality metric.

Taken together, the speech metrics are mixed: the cascade remains stronger on
text translation and MCD, while the fine-tuned model is stronger on the WavLM
speech-similarity proxy.

## Qualitative Observations

The examples below illustrate the pattern behind the aggregate metrics.

Example `lug_eng_0078394`

- Reference: Anybody who is arrested by the police must be officially notified.
- Base model: They are not in the country.
- Fine-tuned Step-Audio: Any person arrested by the police must be informed officially.
- Cascade: Anybody arrested by the police must be officially reported.

Example `lug_eng_0004112`

- Reference: Patients who recover quickly are discharged to save space in the wards.
- Base model: The world is very big.
- Fine-tuned Step-Audio: Patients who recover quickly are discharged in order to
  free hospital beds.
- Cascade: Patients who recover quickly are discharged to save space in the wards.

Example `lug_eng_0026138`

- Reference: This disagreement, in turn, can lead to the breakdown of your relationship.
- Base model: Repetitive condolence hallucination.
- Fine-tuned Step-Audio: This disagreement resulted in their friend's death.
- Cascade: The disagreement led to the breakdown of their friendship.

The base model often produces unrelated fluent English. In one inspected example,
it entered a long repetitive condolence loop. It also produced zero audio tokens
across the 200 evaluated examples, making it unsuitable as a zero-shot baseline
for this task.

The fine-tuned model usually produces relevant English and often good paraphrases,
but it still makes semantic errors. For example, it can preserve the topic while
changing the specific relation or event.

The cascade tends to produce the most accurate text, especially when the ASR step
captures the Luganda input correctly. Its errors are often traceable to ASR
ambiguity or literal translation choices.

## System-Level Discussion

### Base Model

The base Step-Audio-2-mini model should be treated as a negative control. Its poor
metrics show that the base model does not already solve Luganda-to-English speech
translation. This strengthens the evidence that the LoRA fine-tuning is responsible
for the improvement.

Key findings:

- Near-zero BLEU (`0.012`) and very low chrF (`5.152`).
- Very high WER (`10.702`), indicating severe mismatch with references.
- Low COMET (`0.386`) and BLASER ref (`1.713`).
- No valid generated audio tokens, so no audio metrics.

### Fine-Tuned Step-Audio Model

The fine-tuned model is a successful end-to-end Luganda-to-English S2ST system.
It is much stronger than the base model and produces both text and audio outputs
for nearly all examples.

Strengths:

- Large improvement over the base model across all text and semantic metrics.
- Generates valid speech audio for 197 of 200 examples.
- Best SpeechBERTScore F1 among systems with speech output.
- Single end-to-end model, which is operationally simpler than a cascade.

Weaknesses:

- Text metrics remain below the cascade.
- Some examples still show semantic drift.
- Three of 200 examples did not produce enough valid audio tokens for synthesis.

### Cascade Pipeline

The cascade is currently the strongest system for English text translation quality.
It benefits from specialized components for ASR and MT, with Orpheus providing a
separate English TTS stage.

Strengths:

- Best BLEU, chrF, WER, COMET, and BLASER scores.
- Exact or near-exact translations in many inspected examples.
- Lower MCD than the fine-tuned model.

Weaknesses:

- More operationally complex: three major model stages instead of one.
- Errors can compound across ASR, MT, and TTS.
- Lower SpeechBERTScore F1 than the fine-tuned model.
- Requires maintaining separate model dependencies and serving paths.

## Conclusions

The evaluation supports three main conclusions.

First, the base Step-Audio-2-mini model is not sufficient for Luganda-to-English
speech translation. It acts as a useful negative control but not as a viable
deployment candidate.

Second, the fine-tuned Step-Audio LoRA model is a meaningful success. It converts
an unusable base model into a functioning end-to-end S2ST system with strong text
metrics and generated speech. The improvement over the base model is large and
consistent across BLEU, chrF, WER, COMET, and BLASER.

Third, the cascade is still the best system for text quality. It should be treated
as the current upper baseline. The fine-tuned model does not yet surpass it on
translation metrics, but it is competitive enough to justify further work,
especially if the deployment objective favors a single end-to-end model.


## Source Artifacts

The report is based on the following evaluation artifacts:

- `base_metrics_200.txt`
- `base_validation_metrics.json`
- `base_validation_predictions.jsonl`
- `stepaudio_metrics_200.txt`
- `validation_metrics.json`
- `validation_predictions.jsonl`
- `cascade_metrics_200.txt`
- `cascade_validation_metrics.json`
- `cascade_validation_predictions.jsonl`
- `advanced_metrics_200_no_blaser.json`
- `blaser_metrics_200_with_base_aligned.json`
- `blaser_metrics_200_aligned.json`
