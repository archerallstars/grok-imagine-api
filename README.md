# Grok Imagine video

A command-line script that generates a video with the [xAI Imagine video API](https://docs.x.ai/developers/model-capabilities/video/generation) and prints the finished URL.

You can use this project for any purpose. See [LICENSE](LICENSE).

## Setup

Python 3.11 or newer, and the official SDK:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create an API key in the [xAI console](https://console.x.ai/), then either export it or pass it for one run:

```bash
export XAI_API_KEY="YOUR_KEY"
```

`--api-key` overrides `XAI_API_KEY` when both are set. The key is never written into the script or into `generated-videos.json`.

## Examples

Flags can be given in any order. Run the script with no arguments, or with `-h`, to print help.

```bash
python3 ./grok-imagine-video.py -p "slow serene time-lapse" -d 10

python3 ./grok-imagine-video.py -d 8 -m grok-imagine-video-1.5 -i first:milkyway.png -p "slow serene time-lapse"

python3 ./grok-imagine-video.py -i ref:person.png ref:shirt.png -p "the person from <IMAGE_1> wears the shirt from <IMAGE_2>" -a eve

python3 ./grok-imagine-video.py -i first:open.png 3:middle.png last:close.png -d 8 -p "dolly through the room"

python3 ./grok-imagine-video.py -i loop:scene.png -d 6 -s -ar 9:16 -p "a gentle loop"

python3 ./grok-imagine-video.py -l
```

Stdout for a generation is only the video URL. Progress and errors go to stderr.

## Flags

| Flag | Meaning |
| --- | --- |
| `-p`, `--prompt` | Prompt text. Required unless a frame is pinned. |
| `-m`, `--model` | Model name. Default `grok-imagine-video-1.5`. |
| `-i`, `--images` | One or more local image files or `http(s)` URLs. |
| `-d`, `--duration` | Length in whole seconds, from 1 through 15. Omit it and the API uses 8 seconds. |
| `-ar`, `--aspect-ratio` | `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, or `2:3`. Omit it and the API uses `16:9`. |
| `-s`, `--silent` | Video with no audio track. Leave `-a` off when you use this. |
| `-a`, `--audio` | Up to 3 preset voice ids, such as `eve`, `ara`, `leo`, or `rex`. Example: `-a eve leo`. |
| `--api-key` | Key for this run. No short form. Overrides `XAI_API_KEY`. |
| `-l`, `--list-generated-videos` | Print saved URLs. Use this flag alone. |
| `-h`, `--help` | Help. Also shown when the script is started with no arguments. |

## Images

Each `--images` value is a local file (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`) or an `http` / `https` URL. A local file is sent as a data URL. A remote URL is sent unchanged.

| How you write it | What the API receives |
| --- | --- |
| One bare path or URL | Exact first frame |
| Several bare paths or URLs | Reference images, in the order given |
| `ref:` or `reference:` | Reference image |
| `first:` | Exact first frame |
| `last:` | Exact last frame |
| `loop:` | That same image as both the first frame and the last frame |
| `3:` or `3.5:` | Exact frame at that many seconds inside the clip |

Prefix every image, or leave every image bare. `loop:` already sets both ends, so leave out `first:` and `last:` when you use it. You can combine `ref:` with pinned frames.

Limits used by this script: 7 reference images, and 4 middle frames. A middle timestamp has to be greater than 0, less than the clip length, and at least 1/3 second away from the other middle frames. If you set a middle frame and omit `-d`, the script uses 8 seconds so those times can be checked.

In the prompt, tag reference images as `<IMAGE_1>`, `<IMAGE_2>`, and so on, in the order of the reference images only. Pinned frames do not use those tags. Tag voices as `<AUDIO_0>`, `<AUDIO_1>`, and `<AUDIO_2>`.

## Saved URLs

A successful run appends the returned URL to `generated-videos.json` beside the script. That file stays on your machine and is listed in `.gitignore`.

The xAI SDK documents `response.url` as valid for 24 hours. `-l` prints the saved entries, newest first, and deletes any record that is 24 hours old or older. It does not call the API.

## License

[MIT](LICENSE). Use, change, and redistribute this project for any purpose.
