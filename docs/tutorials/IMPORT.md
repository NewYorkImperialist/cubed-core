# Add video

Route: `/import`

Add video stores a recording and its exact starting scramble in the local
workspace. It does not run Decode or control your camera.

## Record a new solve

This is the recommended path:

1. Open **Add video**.
2. Apply the displayed scramble to a solved cube. Choose **New scramble** before
   recording if you want a different one.
3. Record the complete solve with your normal camera app.
4. Transfer the original video file to the computer running the workbench.
5. Choose that recording and confirm that it starts from the displayed
   scramble.
6. Select **Add to workspace**.

Do not confirm a recording that starts after moves have already happened or
from a different cube state. Decode replays from the confirmed scramble.

## Use an existing recording

Open **Use an existing recording**, choose the original file, and enter the
exact scramble that was applied from solved before the recording began. Use
canonical face moves separated by spaces, such as `R U R' U'`.

If the scramble is unknown, do not guess it. You can still open the file
directly in Label, but it cannot be prepared honestly for Decode.

## Choose calibration

Decode also needs six color references. The recommended path creates them from
the imported video:

1. Open **Sample this video**.
2. Scrub to a clear view of a sticker in the target color.
3. Choose its target color and click the middle of the sticker.
4. Repeat for white, green, red, blue, orange, and yellow. Select a thumbnail
   again to replace a weak sample.
5. Review the six crops, then create and attach the calibration.

The browser sends only those small lossless crops. The local server performs
the OpenCV-Lab conversion, rejects dark or mixed-color samples, checks that the
six colors are distinct, and attaches the existing six-color calibration
format. There is no second import step. Choose the same recording in Decode and
its attached calibration is already available as **Sampled from this video**.

**Use another calibration source** remains available for the published shared
calibration, a calibration from another workspace recording, or a JSON upload.
These are secondary choices. The published calibration was measured on gtD1
under one lighting condition, so another cube, camera, lens, exposure, white
balance, or lighting setup can be less reliable.

Decode names those sources **Published shared calibration**, **From
_recording filename_**, or the safe uploaded filename. Internal schema IDs and
hashes stay out of the picker.

## Upload size and media rules

Every video uses the same byte limit, independent of frame rate. The service
reads it from `CUBED_CORE_MAX_UPLOAD_BYTES`, which defaults to
`1073741824` bytes, or 1 GiB. Change that environment value and restart the
service if your local workspace should accept a different maximum.

The tested profile is 120 fps with an encoded short edge of at least 1080
pixels. Readable nonstandard media can continue with warnings. Uploads must use
one of these extensions: `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, or `.webm`.

A native 220–242 fps original uses the same upload cap. Preserve that original,
then create its linked deterministic 120 fps derivative before Decode:

```console
uv run --no-sync cubed-core derive-240-to-120 <capture-id>
```

That derivative requirement is a Decode and media rule. It does not change the
upload cap.

## Published dataset videos

`make download-dataset` verifies the pinned corpus and registers every
published video in the workspace with a hard link. It does not duplicate video
bytes, copy teacher or IMU sidecars, attach calibration, lock a recording, or
run Decode.

Thirty-one videos carry their exact frame-zero starting scramble and can be
prepared normally. Select one in Decode, choose the published shared
calibration, lock its video and scramble, and run. The same lock supports later
attempts. You can replace calibration between attempts without re-importing the
video. `gtD2`, `gtD3`, `gtD4`, and `gtD5` start solved and scramble on-camera.
They are listed for inspection, but the current preflight has no explicit
starting-state contract for that sequence, so do not invent a scramble to make
them runnable.

See the
[pinned Hugging Face quickstart](DECODE.md#pinned-hugging-face-quickstart).

## What happens after upload

The service preserves the original bytes, records their SHA-256 and measured
media metadata, and creates a workspace recording. Uploading does not run the
tracker or decoder.

For Decode, attach calibration, lock the recording's video and scramble, and
choose local or remote CUDA compute. That lock supports any number of
sequential attempts. Replace calibration from the same preparation panel if a
later run shows that the color references were poor. For annotations, continue
to Label.

BLE, decoded teacher moves, and IMU files are optional evaluation or sensor
sidecars. The standard Decode request does not include them, and they are not
required here. This request boundary is not proof of an isolated upstream
environment.

Imported files stay under the configured local workspace. Do not attach private
videos, calibrations, teacher records, or sensor files to a public issue.

## Remove a workspace video

Open the collapsed **Manage workspace videos** section in Add video, then choose
**Move to Trash**. The local capture bundle and its workspace labels move to
the operating system's Trash. Existing Runs remain, but playback is unavailable
without the source video. Removing a published capture does not change the
published dataset files, which can be downloaded and registered again with
`make download-dataset`. The API returns `409` and leaves the capture in place
while it has a queued or running Decode job. Wait for that attempt to become
terminal before removing the video.

See:

- [Decode and Runs](DECODE.md)
- [Label](LABEL.md)
- [Dataset](../DATASET.md)
- [Privacy](../PRIVACY.md)

## Troubleshooting

### The recording is not listed in Decode

Refresh Decode after the upload finishes. For published data, run the supported
`make download-dataset` command so the verified videos are registered in this
workspace. Missing calibration or a scramble blocks preparation, but it does
not hide a recording from the list.

### I do not have a matching calibration

Use **Sample this video** and collect one clear sticker of each color from your
recording. If a color is unclear, scrub to another frame and replace its
thumbnail. You may instead select the published shared calibration after
reading its mismatch warning, with potentially lower reliability.

### The upload is too large

Check `CUBED_CORE_MAX_UPLOAD_BYTES` on the machine running the service. The
same limit applies to every frame rate. Keep enough free temporary disk space
for the multipart upload and workspace copy.
