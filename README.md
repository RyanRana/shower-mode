# shower-mode

stops airpods conversation awareness from turning your music down when you sing along.

conversation awareness treats any voice as the start of a conversation. shower-mode tells singing along apart from talking using three models on the mac's built-in mic: lyrics, pitch and speaker. talking, or someone else talking, still gets through. if anything is uncertain or fails, it falls back to normal airpods behavior.

## how it works

every 0.75 s the service (`shower_mode.py`) looks at the last few seconds of mic audio and decides "singing along" or not:

1. **lyrics match**: faster-whisper transcribes the mic and fuzzy-matches the words against the song's synced lyrics.
2. **pitch match** (`melody.py`): your voice's pitch track is compared to the song's chroma, captured from system output by the `sysaudio` helper. a gradient-boosted model (`sing_model.joblib`) turns that into a probability. covers songs without lyrics.
3. **second voice** (`speaker.py`): a wespeaker embedding is compared to your enrolled voice. another speaker always gets through (`speaker_threshold.json`).

### conversation awareness toggle vs. volume ducking

the `airpods_ca` helper tries to toggle conversation awareness over apple's aacp protocol. on current macos this fails: `bluetoothd` owns the aacp channel (l2cap open returns `kIOReturnError`), and the private audioaccessoryservices api that system settings uses needs the `com.apple.bluetooth.system` entitlement, which only apple-signed binaries get. so in practice shower-mode runs in **volume-ducking mode**: talk over music (or someone else talks) and system volume drops to 30%, then comes back once you stop or start singing. the helper stays in case a future macos or firmware opens the channel.

## requirements

- macos 13+ with xcode command line tools (`xcode-select --install`)
- python 3.10+
- spotify or apple music for lyrics (pitch matching works with any audio)

## install

```sh
./install.sh            # build helpers, venv, model, launchagent; ends with --doctor
./uninstall.sh          # remove the launchagent and turn conversation awareness back on
```

runs at login as launchagent `co.noxmetals.shower-mode`, restarted on crash. log: `shower-mode.log`.

give python **screen & system audio recording** (system settings, privacy & security), or `sysaudio` fails with `SCStreamErrorDomain -3801` and the pitch check is off. capture also stops when the screen locks or sleeps (`-3815`); shower-mode restarts it with backoff.

## check your setup

```sh
.venv/bin/python shower_mode.py --doctor
```

```
shower-mode doctor
  ok   microphone             MacBook Air Microphone
  off  now playing            nothing playing in Spotify/Music
  ok   system audio           capturing
  off  AirPods CA toggle      macOS blocks the AACP channel; volume ducking instead
  ok   pitch model            sing_model.joblib
  ok   speaker model          wespeaker_en_voxceleb_resnet34.onnx
  off  voice profile          0 samples (second-voice check needs 5)
```

set `SHOWER_MIC` to part of a device name to use a different mic. the airpods mic is never used: it forces the low-quality call codec.

## files

| path | what |
| --- | --- |
| `shower_mode.py` | the service |
| `melody.py` | system audio capture and pitch features |
| `speaker.py` | second-voice check |
| `mert_sing.py` | experimental mert + attention detector (not used by the service) |
| `airpods_ca.swift`, `sysaudio.swift` | macos helpers |
| `install.sh`, `uninstall.sh` | launchagent setup and removal |
| `tune/tune.py` | speaker threshold tuning, data loading, laptop-mic simulation |
| `tune/rawcache.py` | per-frame pitch/chroma/onset cache of the realistic set (fast feature rounds) |
| `tune/rounds.py` | feature rounds on the realistic set; `--live final` scores service-identical features, `--export` writes `sing_model.joblib` |
| `tune/climb.py` | hill climb of the pitch model; `--final` scores holdout, `--export` writes `sing_model.joblib` |
| `tune/hparam.py` | optuna search over the gbm |
| `tune/e2e.py` | trains and scores the mert detector |
| `tune/results.jsonl` | every experiment, one json line each |
| `data/` | musdb18, librispeech and feature caches (large, not needed to run) |

## evaluation

offline only, nothing touches audio devices. songs come from musdb18; the isolated vocal stem goes through a laptop-mic simulation to stand in for you singing. negatives:

- `talk`: librispeech speech over the song
- `other_song`: singing from a different track
- `bleed`: no voice, just the song leaking into the mic

tracks are split 60/25/15 into train/dev/holdout. the threshold is set so at most 2% of non-singing windows count as singing; the score is the share of singing windows caught.

## results so far

current model (`tune/rounds.py` `final`, realistic benchmark, trained on service-identical features):

| model | split | caught | false singing |
| --- | --- | --- | --- |
| v1 gbm features, 3 s (reference) | holdout | 48.8% | 1.5% |
| **v2: wider lag, local-vs-key contrast, soft chroma, mic chroma, onset sync, lag stats, voicing; 4 s, smooth 4** | holdout | **62.0%** | 1.7% |
| same | dev | 75.4% | 2.4% |

what each round added on dev (cached frames): baseline 56.1, finer +-500 ms lag 59.8, contrast 60.7, soft chroma 61.4, mic chroma 61.4 (lower false), onset sync 68.4, lag stats + voicing 71.4, 4 s window 72.5, 4-step smoothing 75.2. the best-lag position feature was dropped (+4.6 on dev) because it likely learns the augmentation's synthetic lag. previous model: `tune/sing_model_v1.joblib`.

older results (clean benchmark):

| model | split | caught | false singing | worst case |
| --- | --- | --- | --- | --- |
| gbm, 3 s window, pyin | holdout | 52% | 2.6% | bleed 7.7% |
| gbm, optuna tuned, 4 s | holdout | 55% | 3.0% | bleed 9.1% |
| mert + attention | dev | 18 to 28% | 1.4 to 2.2% | other_song 5.6% |

## known limits

- positives are the original studio vocal, perfectly in tune and in time. real singing along isn't, so offline numbers are probably optimistic.
- bleed is the main source of false singing.
- the `airpods_ca` toggle is blocked by macos (see above); the service runs in volume-ducking mode.
- the second-voice check only starts after 5 confident sing-along snippets are enrolled into `voice_profile.npy`.
