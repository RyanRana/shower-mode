# encore

Keeps AirPods Conversation Awareness (CA) on, except while you are singing along to the
song that is playing. Talking, someone else talking, or singing a different song all let
CA through. Any doubt, error, pause, or exit turns CA back on.

## How it works

Every 0.75 s the service (`encore.py`) looks at the last few seconds of mic audio and
decides "singing along" or not:

1. **Lyrics match**: faster-whisper transcribes the mic; the words are fuzzy-matched
   against the song's synced lyrics.
2. **Pitch match** (`melody.py`): the mic voice's pitch track is compared to the song's
   chroma, captured from system output by the `sysaudio` helper. A gradient-boosted model
   (`sing_model.joblib`) turns this into a probability. This covers songs without lyrics.
3. **Second voice** (`speaker.py`): a WeSpeaker embedding is compared to your enrolled
   voice. Another speaker always lets CA through (`speaker_threshold.json`).

### Outputs: CA toggle vs. volume ducking

The `airpods_ca` helper tries to toggle CA over Apple's AACP protocol. On current macOS
this fails: `bluetoothd` owns the AACP channel (L2CAP open returns `kIOReturnError`), and
the private AudioAccessoryServices API that System Settings uses requires the
`com.apple.bluetooth.system` entitlement, which only Apple-signed binaries get. So in
practice encore runs in **volume-ducking mode**: when you talk over music (or someone
else talks), system volume drops to 30% and comes back once you stop or start singing.
The helper is kept in case a future macOS or firmware opens the channel.

## Requirements

- macOS 13+ with Xcode command line tools (`xcode-select --install`)
- Python 3.10+
- Spotify or Apple Music for lyrics (pitch matching works with any audio)

## Install

```sh
./install.sh            # build helpers, venv, model, LaunchAgent; ends with --doctor
./uninstall.sh          # remove the LaunchAgent and turn CA back on
```

Runs at login as LaunchAgent `co.noxmetals.encore`, restarted on crash. Log: `encore.log`.

Grant **Screen & System Audio Recording** to Python (System Settings, Privacy & Security),
or `sysaudio` fails with `SCStreamErrorDomain -3801` and the pitch check is off. Capture
also stops when the screen locks or sleeps (`-3815`); encore restarts it with backoff.

## Check your setup

```sh
.venv/bin/python encore.py --doctor
```

```
encore doctor
  ok   microphone             MacBook Air Microphone
  off  now playing            nothing playing in Spotify/Music
  ok   system audio           capturing
  off  AirPods CA toggle      macOS blocks the AACP channel; volume ducking instead
  ok   pitch model            sing_model.joblib
  ok   speaker model          wespeaker_en_voxceleb_resnet34.onnx
  off  voice profile          0 samples (second-voice check needs 5)
```

Set `ENCORE_MIC` to part of a device name to use a different microphone. The AirPods mic
is never used: it forces the low-quality call codec.

## Files

| Path | What |
| --- | --- |
| `encore.py` | the service |
| `melody.py` | system audio capture and pitch features |
| `speaker.py` | second-voice check |
| `mert_sing.py` | experimental MERT + attention detector (not used by the service) |
| `airpods_ca.swift`, `sysaudio.swift` | macOS helpers |
| `install.sh`, `uninstall.sh` | LaunchAgent setup and removal |
| `tune/tune.py` | speaker threshold tuning, data loading, laptop-mic simulation |
| `tune/rawcache.py` | per-frame pitch/chroma/onset cache of the realistic set (fast feature rounds) |
| `tune/rounds.py` | feature rounds on the realistic set; `--live final` scores service-identical features, `--export` writes `sing_model.joblib` |
| `tune/climb.py` | hill climb of the pitch model; `--final` scores holdout, `--export` writes `sing_model.joblib` |
| `tune/hparam.py` | Optuna search over the GBM |
| `tune/e2e.py` | trains and scores the MERT detector |
| `tune/results.jsonl` | every experiment, one JSON line each |
| `data/` | MUSDB18, LibriSpeech, and feature caches (large, not needed to run) |

## Evaluation

Offline only, nothing touches audio devices. Songs come from MUSDB18; the isolated vocal
stem is run through a laptop-mic simulation to stand in for you singing. Negatives:

- `talk`: LibriSpeech speech over the song
- `other_song`: singing from a different track
- `bleed`: no voice, just the song leaking into the mic

Tracks are split 60/25/15 into train/dev/holdout. The threshold is set so at most 2% of
non-singing windows count as singing; the score is the share of singing windows caught.

## Results so far

Current model (`tune/rounds.py` `final`, realistic benchmark, trained on service-identical features):

| Model | Split | Caught | False singing |
| --- | --- | --- | --- |
| v1 GBM features, 3 s (reference) | holdout | 48.8% | 1.5% |
| **v2: wider lag, local-vs-key contrast, soft chroma, mic chroma, onset sync, lag stats, voicing; 4 s, smooth 4** | holdout | **62.0%** | 1.7% |
| same | dev | 75.4% | 2.4% |

What each round added on dev (cached frames): baseline 56.1, finer +-500 ms lag 59.8, contrast 60.7, soft chroma 61.4, mic chroma 61.4 (lower false), onset sync 68.4, lag stats + voicing 71.4, 4 s window 72.5, 4-step smoothing 75.2. The best-lag position feature was dropped (+4.6 on dev) because it likely learns the augmentation's synthetic lag. Previous model: `tune/sing_model_v1.joblib`.

Older results (clean benchmark):

| Model | Split | Caught | False singing | Worst case |
| --- | --- | --- | --- | --- |
| GBM, 3 s window, pYIN | holdout | 52% | 2.6% | bleed 7.7% |
| GBM, Optuna tuned, 4 s | holdout | 55% | 3.0% | bleed 9.1% |
| MERT + attention | dev | 18 to 28% | 1.4 to 2.2% | other_song 5.6% |

## Known limits

- Positives are the original studio vocal, which is perfectly in tune and in time. Real
  singing along is not, so offline numbers are probably optimistic.
- Bleed is the main source of false singing.
- The `airpods_ca` toggle is blocked by macOS (see above); the service runs in
  volume-ducking mode.
- The second-voice check only starts after 5 confident sing-along snippets have been
  enrolled into `voice_profile.npy`.
