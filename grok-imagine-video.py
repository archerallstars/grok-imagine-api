#!/usr/bin/env python3
"""
Generate one video with the xAI Imagine video API.

Run with no arguments, or with -h / --help, to see every flag.
Flags may be given in any order.
The API key is never stored in this file. Pass --api-key, or set XAI_API_KEY.
When both are set, --api-key is the one used for that run.
"""

# Upgraded for elementary-coder: descriptive names + required comments added;
# overall structure and ordering taken from supplied skeleton.

# Standard library imports
import argparse
import base64
import json
import os
import queue
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Third-party imports
import xai_sdk
from xai_sdk.types import UrlKeyframe, VideoAspectRatio, VoiceAudioRef

# =============================================================================
# CONFIGURATION - Customizable variables and switches (grouped by category)
# =============================================================================

# --- Model and clip length ---
default_video_generation_model_name: str = "grok-imagine-video-1.5"  # model used when -m is omitted
default_video_duration_seconds_when_omitted: int = 8  # API default, sent when a middle frame needs a known length
minimum_video_duration_seconds: int = 1  # shortest clip the API accepts
maximum_video_duration_seconds: int = 15  # longest clip the API accepts
minimum_keyframe_spacing_seconds: float = 1.0 / 3.0  # middle frames must stay at least this far apart

# --- Image and voice limits from the grok-imagine-video-1.5 reference-to-video API ---
maximum_reference_image_count: int = 7  # most reference images in one request
maximum_middle_keyframe_count: int = 4  # most pinned frames inside the clip, endpoints excluded
maximum_reference_voice_count: int = 3  # most preset voices in one request

# --- API key ---
environment_variable_name_for_api_key: str = "XAI_API_KEY"  # read when --api-key is omitted

# --- Aspect ratio values accepted by client.video.generate ---
allowed_video_aspect_ratio_by_text: dict[str, VideoAspectRatio] = {
    "1:1": "1:1",  # square
    "16:9": "16:9",  # wide, and the API default when -ar is omitted
    "9:16": "9:16",  # tall
    "4:3": "4:3",  # classic horizontal
    "3:4": "3:4",  # classic vertical
    "3:2": "3:2",  # photo horizontal
    "2:3": "2:3",  # photo vertical
}

# --- Saved video URLs. response.url is documented as valid for 24 hours. ---
generated_video_catalog_file_name: str = "generated-videos.json"  # stored beside this script
generated_video_url_lifetime_hours: int = 24  # SDK note on VideoResponse.url
generated_video_catalog_empty_message: str = "No videos have been saved yet."  # stdout when the list is empty

# --- Local image types that can be sent as data URLs ---
local_image_suffix_to_mime_type: dict[str, str] = {
    ".png": "image/png",  # png files
    ".jpg": "image/jpeg",  # jpg files
    ".jpeg": "image/jpeg",  # jpeg files
    ".webp": "image/webp",  # webp files
    ".gif": "image/gif",  # gif files
}

# --- Role words accepted before the colon on --images ---
image_role_prefix_reference: str = "ref"  # short prefix for a reference image
image_role_prefix_reference_long: str = "reference"  # long prefix for a reference image
image_role_prefix_first_frame: str = "first"  # exact opening frame
image_role_prefix_last_frame: str = "last"  # exact closing frame
image_role_prefix_loop: str = "loop"  # same image for the opening frame and the closing frame

# --- Help text (stdout). The example key is a placeholder, never a real key. ---
command_help_description: str = """
Generate a video with the xAI Imagine video API.

Flags can be given in any order.
Each --images value is a local file or an http(s) URL.

Image roles, written as a prefix on --images:
  one bare image          exact first frame
  several bare images     reference images, in the order given
  ref: or reference:      reference image
  first:                  exact first frame
  last:                   exact last frame
  loop:                   that image is both the first frame and the last frame
  3: or 3.5:              exact frame at that many seconds inside the clip

Reference images are tagged in the prompt as <IMAGE_1>, <IMAGE_2>, and so on,
following the order of the reference images only. Pinned frames do not use those tags.
Voices passed with -a are preset ids such as eve, ara, leo, and rex.
Tag them as <AUDIO_0>, <AUDIO_1>, and <AUDIO_2>. At most 3 voices.
An unknown voice id is rejected by the API, which returns the current voice list.
-s / --silent builds a video with no audio track. Leave -a off when you use it.
-ar / --aspect-ratio is one of 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3.
Leave it off and the API uses 16:9.

-l / --list-generated-videos prints URLs saved beside this script.
Use that flag alone. Each saved URL is kept for 24 hours.
Listing deletes any link that is 24 hours old or older.

--api-key supplies the key for this run and overrides XAI_API_KEY.
When --api-key is omitted, the key is read from XAI_API_KEY.
""".strip()

command_help_epilog: str = """
examples:
  python3 ./grok-imagine-video.py -p "slow serene time-lapse" -d 10
  python3 ./grok-imagine-video.py -d 8 -m grok-imagine-video-1.5 -i first:milkyway.png -p "slow serene time-lapse"
  python3 ./grok-imagine-video.py -i ref:person.png ref:shirt.png -p "the person from <IMAGE_1> wears the shirt from <IMAGE_2>" -a eve
  python3 ./grok-imagine-video.py -i first:open.png 3:middle.png last:close.png -d 8 -p "dolly through the room"
  python3 ./grok-imagine-video.py -i loop:scene.png -d 6 -s -p "a gentle loop"
  python3 ./grok-imagine-video.py -i "https://docs.x.ai/assets/api-examples/video/milkyway-still.png" -p "slow serene time-lapse"
  python3 ./grok-imagine-video.py --api-key YOUR_KEY -p "slow serene time-lapse" -d 10 -ar 9:16
  python3 ./grok-imagine-video.py -l
  python3 ./grok-imagine-video.py --list-generated-videos
""".strip()

# =============================================================================
# LOGGER AND ERROR HANDLER
# =============================================================================

# Lines for the user go to stderr through this queue so the main thread
# does not write them itself. The video URL stays alone on stdout.
standard_error_message_queue: queue.Queue = queue.Queue()


def standard_error_message_worker() -> None:
    """Print queued stderr lines.

    Input is one message string per queue item.
    Output is that line on stderr.
    The worker is a daemon thread started at import, so process exit is not blocked.
    """
    while True:
        message_text = standard_error_message_queue.get()
        try:
            print(message_text, file=sys.stderr, flush=True)
        finally:
            standard_error_message_queue.task_done()


def standard_error_message_write(message_text: str) -> None:
    """Queue one stderr line and wait until it has been printed.

    Input is the line to show. It must not contain an API key.
    Output is none. The call returns only after the line is on stderr,
    so a progress line appears before the video request starts.
    """
    standard_error_message_queue.put(message_text)
    standard_error_message_queue.join()
    return None  # the line is already on stderr


# Start the stderr worker once. Daemon so a normal exit does not wait on it.
standard_error_message_thread = threading.Thread(
    target=standard_error_message_worker,
    name="standard_error_message_worker",
    daemon=True,
)
standard_error_message_thread.start()

# =============================================================================
# LOCAL DATA SECTION
# =============================================================================


class VideoCommandError(Exception):
    """Raised when flags or local images cannot become a video request."""

    def __init__(self, message_text: str) -> None:
        """Store message_text for stderr. The text must never include an API key."""
        self.message_text = message_text
        super().__init__(message_text)


@dataclass(frozen=True)
class ImageArgumentTokenRole:
    """One --images token after the role prefix has been read."""

    role_name: str  # bare, reference, first, last, loop, or keyframe
    image_source_text: str  # local path or http(s) URL
    middle_keyframe_timestamp_seconds: float | None  # set only for a keyframe


@dataclass(frozen=True)
class VideoImageRoleAssignment:
    """Images split into the fields client.video.generate understands."""

    first_frame_source_text: str | None  # image_url source, before a file is read
    last_frame_source_text: str | None  # last_frame_url source
    loop_frame_source_text: str | None  # used for both ends when loop: is set
    reference_image_source_text_list: list[str]  # reference_image_urls sources, in order
    middle_keyframe_list: list[tuple[float, str]]  # (timestamp seconds, source) inside the clip


@dataclass(frozen=True)
class VideoCommandRequest:
    """Flags after checks, still with local paths rather than data URLs."""

    prompt_text: str  # may be empty when a frame is pinned
    video_generation_model_name: str
    duration_seconds: int | None  # None means leave duration unset for the API
    silent_video_requested: bool
    reference_voice_id_list: list[str]
    api_key_argument: str | None  # None means the SDK should read XAI_API_KEY
    aspect_ratio: VideoAspectRatio | None  # None lets the API use 16:9
    image_role_assignment: VideoImageRoleAssignment


@dataclass(frozen=True)
class GeneratedVideoRecord:
    """One finished video saved in generated-videos.json."""

    video_url: str  # response.url, documented as valid for 24 hours
    generated_at_utc: str  # UTC time the record was saved, YYYY-MM-DDTHH:MM:SSZ
    prompt_text: str
    video_generation_model_name: str
    aspect_ratio_text: str | None  # None when -ar was omitted
    duration_seconds: int | None  # None when -d was omitted


@dataclass(frozen=True)
class ResolvedVideoImageUrls:
    """Image fields ready to pass to client.video.generate."""

    first_frame_url: str | None
    last_frame_url: str | None
    reference_image_url_list: list[str]
    middle_keyframe_url_list: list[tuple[float, str]]  # (timestamp seconds, url or data URL)


def main() -> int:
    """Parse flags, generate one video, or list saved video URLs.

    Inputs are the process arguments in sys.argv.
    Returns 0 when help is shown, the video URL is printed, or the list is printed.
    Returns 1 when the flags are rejected or the generation fails.
    A generation writes only the video URL to stdout. -l writes the saved list.
    """
    if len(sys.argv) == 1:
        cli_argument_parser_build().print_help(sys.stdout)
        return 0  # no flags: show help and do not call the API
    else:
        argument_parser = cli_argument_parser_build()
        parsed_arguments = argument_parser.parse_args()
        try:
            if parsed_arguments.list_generated_videos_requested:
                cli_list_generated_videos_conflict_check(parsed_arguments)
                command_stdout_text = generated_video_catalog_text_for_list()
            else:
                video_command_request = cli_parsed_arguments_validate(parsed_arguments)
                command_stdout_text = video_generation_start(video_command_request)
        except VideoCommandError as video_command_error:
            standard_error_message_write(video_command_error.message_text)
            return 1  # the flags, the catalog, or the local files were rejected
        except Exception as generation_error:
            standard_error_message_write(str(generation_error))
            return 1  # the video API failed after the flags were accepted
        else:
            print(command_stdout_text)
            return 0  # a generation prints the video URL; -l prints the saved list


# =============================================================================
# IMAGE ROLE PARSING AND LOCAL FILE READS
# =============================================================================


def image_argument_bare_role_assignment_build(
    image_argument_token_role_list: list[ImageArgumentTokenRole],
) -> VideoImageRoleAssignment:
    """Turn a list of bare images into a first frame or into references.

    Input is every token, already classified as bare.
    One image becomes the first frame. More than one become reference images.
    Output is the role assignment. Raises VideoCommandError above the reference limit.
    """
    image_source_text_list = [
        image_argument_token_role.image_source_text
        for image_argument_token_role in image_argument_token_role_list
    ]
    if len(image_source_text_list) == 1:
        return VideoImageRoleAssignment(
            first_frame_source_text=image_source_text_list[0],
            last_frame_source_text=None,
            loop_frame_source_text=None,
            reference_image_source_text_list=[],
            middle_keyframe_list=[],
        )  # one bare image keeps the original image_url behavior
    else:
        if len(image_source_text_list) > maximum_reference_image_count:
            raise VideoCommandError(
                f"At most {maximum_reference_image_count} reference images are allowed."
            )
        else:
            return VideoImageRoleAssignment(
                first_frame_source_text=None,
                last_frame_source_text=None,
                loop_frame_source_text=None,
                reference_image_source_text_list=image_source_text_list,
                middle_keyframe_list=[],
            )  # several bare images are references, in the order given


def image_argument_classify_known_role_name(prefix_text_lower: str) -> str | None:
    """Map a role word to its role name.

    Input is the text before the first colon, already lowercased.
    Returns the role name, or None when the prefix is not a role word.
    A number is not handled here; the caller checks that separately.
    """
    if prefix_text_lower in (image_role_prefix_reference, image_role_prefix_reference_long):
        return "reference"  # ref: and reference: are the same role
    elif prefix_text_lower == image_role_prefix_first_frame:
        return "first"  # exact opening frame
    elif prefix_text_lower == image_role_prefix_last_frame:
        return "last"  # exact closing frame
    elif prefix_text_lower == image_role_prefix_loop:
        return "loop"  # one image used for both ends
    else:
        return None  # not a role word


def image_argument_classify_prefix_timestamp_check(prefix_text: str) -> bool:
    """Return whether prefix_text is a second count such as 3 or 3.5.

    Input is the text before the first colon, with its original spelling.
    Returns True when every character fits a whole or decimal second count.
    Returns False when the colon belongs to a URL or a path.
    """
    if prefix_text == "":
        return False  # an empty prefix is not a timestamp
    else:
        digit_seen = False
        decimal_point_seen = False
        prefix_is_timestamp = True
        for prefix_character in prefix_text:
            if prefix_character.isdigit():
                digit_seen = True
            elif prefix_character == "." and not decimal_point_seen:
                decimal_point_seen = True
            else:
                prefix_is_timestamp = False
        if prefix_is_timestamp and digit_seen and not prefix_text.endswith("."):
            return True  # the prefix is a timestamp in seconds
        else:
            return False  # the colon belongs to the path or URL


def image_argument_classify_token_role(image_argument_token: str) -> ImageArgumentTokenRole:
    """Classify one --images value.

    Input is one argument string.
    A role word or a timestamp before the first colon selects the role.
    https:// stays bare because "https" is not a role and not a timestamp.
    Returns the role, the path or URL, and the timestamp when the role is a middle frame.
    """
    stripped_image_argument_token = image_argument_token.strip()
    if stripped_image_argument_token == "":
        raise VideoCommandError("An image argument was blank.")
    else:
        if ":" not in stripped_image_argument_token:
            return ImageArgumentTokenRole(
                role_name="bare",
                image_source_text=stripped_image_argument_token,
                middle_keyframe_timestamp_seconds=None,
            )  # no role prefix, so this image is bare
        else:
            prefix_text, remainder_text = stripped_image_argument_token.split(":", 1)
            known_role_name = image_argument_classify_known_role_name(prefix_text.lower())
            stripped_remainder_text = remainder_text.strip()
            if known_role_name is not None:
                if stripped_remainder_text == "":
                    raise VideoCommandError(
                        f"The {prefix_text.lower()}: image is missing a file path or URL."
                    )
                else:
                    return ImageArgumentTokenRole(
                        role_name=known_role_name,
                        image_source_text=stripped_remainder_text,
                        middle_keyframe_timestamp_seconds=None,
                    )  # role word, then the path or URL
            else:
                if image_argument_classify_prefix_timestamp_check(prefix_text):
                    if stripped_remainder_text == "":
                        raise VideoCommandError(
                            f"The middle frame at {prefix_text} seconds is missing a file path or URL."
                        )
                    else:
                        return ImageArgumentTokenRole(
                            role_name="keyframe",
                            image_source_text=stripped_remainder_text,
                            middle_keyframe_timestamp_seconds=float(prefix_text),
                        )  # the number before the colon is the timestamp in seconds
                else:
                    return ImageArgumentTokenRole(
                        role_name="bare",
                        image_source_text=stripped_image_argument_token,
                        middle_keyframe_timestamp_seconds=None,
                    )  # the colon belongs to a URL or path, so the whole token stays bare


def image_argument_prefixed_role_assignment_build(
    image_argument_token_role_list: list[ImageArgumentTokenRole],
) -> VideoImageRoleAssignment:
    """Turn prefixed images into first, last, loop, reference, and middle frames.

    Input is every token, already classified, with no bare tokens mixed in.
    Output is the role assignment.
    Raises VideoCommandError when a role is repeated, loop: is combined with
    first: or last:, or a count is above the API limit.
    """
    first_frame_source_text: str | None = None
    last_frame_source_text: str | None = None
    loop_frame_source_text: str | None = None
    reference_image_source_text_list: list[str] = []
    middle_keyframe_list: list[tuple[float, str]] = []
    for image_argument_token_role in image_argument_token_role_list:
        role_name = image_argument_token_role.role_name
        image_source_text = image_argument_token_role.image_source_text
        if role_name == "reference":
            reference_image_source_text_list.append(image_source_text)
        elif role_name == "first":
            if first_frame_source_text is not None:
                raise VideoCommandError("Only one first: image is allowed.")
            else:
                first_frame_source_text = image_source_text
        elif role_name == "last":
            if last_frame_source_text is not None:
                raise VideoCommandError("Only one last: image is allowed.")
            else:
                last_frame_source_text = image_source_text
        elif role_name == "loop":
            if loop_frame_source_text is not None:
                raise VideoCommandError("Only one loop: image is allowed.")
            else:
                loop_frame_source_text = image_source_text
        elif role_name == "keyframe":
            middle_keyframe_timestamp_seconds = image_argument_token_role.middle_keyframe_timestamp_seconds
            if middle_keyframe_timestamp_seconds is None:
                raise VideoCommandError("A middle frame is missing its timestamp.")
            else:
                middle_keyframe_list.append((middle_keyframe_timestamp_seconds, image_source_text))
        else:
            raise VideoCommandError(f"Unknown image role {role_name}.")
    if loop_frame_source_text is not None and (
        first_frame_source_text is not None or last_frame_source_text is not None
    ):
        raise VideoCommandError(
            "loop: uses one image as both the opening frame and the closing frame. "
            "Leave out first: and last: when you use loop:."
        )
    else:
        pass
    if len(reference_image_source_text_list) > maximum_reference_image_count:
        raise VideoCommandError(
            f"At most {maximum_reference_image_count} reference images are allowed."
        )
    else:
        pass
    if len(middle_keyframe_list) > maximum_middle_keyframe_count:
        raise VideoCommandError(
            f"At most {maximum_middle_keyframe_count} middle frames are allowed."
        )
    else:
        pass
    return VideoImageRoleAssignment(
        first_frame_source_text=first_frame_source_text,
        last_frame_source_text=last_frame_source_text,
        loop_frame_source_text=loop_frame_source_text,
        reference_image_source_text_list=reference_image_source_text_list,
        middle_keyframe_list=middle_keyframe_list,
    )  # prefixed images mapped onto first, last, loop, references, and middle frames


def image_argument_resolved_role_assignment_build(
    image_argument_token_list: list[str] | None,
) -> VideoImageRoleAssignment:
    """Split --images into API roles.

    Input is the list of --images values, or None when the flag was omitted.
    Bare and prefixed tokens cannot be mixed.
    Returns an empty assignment when there are no images.
    """
    if image_argument_token_list is None or len(image_argument_token_list) == 0:
        return VideoImageRoleAssignment(
            first_frame_source_text=None,
            last_frame_source_text=None,
            loop_frame_source_text=None,
            reference_image_source_text_list=[],
            middle_keyframe_list=[],
        )  # no images: text-to-video
    else:
        image_argument_token_role_list = [
            image_argument_classify_token_role(image_argument_token)
            for image_argument_token in image_argument_token_list
        ]
        bare_role_count = 0
        prefixed_role_count = 0
        for image_argument_token_role in image_argument_token_role_list:
            if image_argument_token_role.role_name == "bare":
                bare_role_count += 1
            else:
                prefixed_role_count += 1
        if bare_role_count > 0 and prefixed_role_count > 0:
            raise VideoCommandError(
                "Prefix every image, or leave every image bare. "
                "A bare path cannot sit next to first:, last:, loop:, ref:, or a timestamp."
            )
        else:
            if prefixed_role_count == 0:
                return image_argument_bare_role_assignment_build(
                    image_argument_token_role_list,
                )  # every image is bare
            else:
                return image_argument_prefixed_role_assignment_build(
                    image_argument_token_role_list,
                )  # every image has a role prefix


def image_file_one_path_read_into_queue(
    local_image_file_path: str,
    local_image_read_result_queue: queue.Queue,
) -> None:
    """Read one local image and put the outcome on the result queue.

    Inputs are the file path and the queue shared by the parallel readers.
    The queue item is (path, file bytes or None, error text or None).
    This is the only place a local image file is opened.
    """
    try:
        with open(local_image_file_path, "rb") as local_image_file_handle:
            local_image_file_bytes = local_image_file_handle.read()
    except OSError as local_image_read_error:
        local_image_read_result_queue.put(
            (local_image_file_path, None, str(local_image_read_error)),
        )
    else:
        local_image_read_result_queue.put(
            (local_image_file_path, local_image_file_bytes, None),
        )
    return None  # the read result is on the queue


def image_file_paths_read_together(
    local_image_file_path_list: list[str],
) -> dict[str, bytes]:
    """Read every local image at the same time.

    Input is the local paths to open. Repeated paths are read once.
    Each file is opened on its own thread, and the bytes come back through a queue.
    Returns a map of path to file bytes. Raises VideoCommandError if any read fails.
    """
    unique_local_image_file_path_list = list(dict.fromkeys(local_image_file_path_list))
    if len(unique_local_image_file_path_list) == 0:
        return {}  # no local files to read
    else:
        local_image_read_result_queue: queue.Queue = queue.Queue()
        local_image_read_thread_list: list[threading.Thread] = []
        for local_image_file_path in unique_local_image_file_path_list:
            local_image_read_thread = threading.Thread(
                target=image_file_one_path_read_into_queue,
                args=(local_image_file_path, local_image_read_result_queue),
            )
            local_image_read_thread.start()
            local_image_read_thread_list.append(local_image_read_thread)
        for local_image_read_thread in local_image_read_thread_list:
            local_image_read_thread.join()
        local_image_bytes_by_path: dict[str, bytes] = {}
        local_image_read_failure_message_list: list[str] = []
        completed_read_count = 0
        while completed_read_count < len(unique_local_image_file_path_list):
            local_image_file_path, local_image_file_bytes, read_error_message_text = (
                local_image_read_result_queue.get()
            )
            completed_read_count += 1
            if read_error_message_text is None:
                if local_image_file_bytes is None:
                    local_image_read_failure_message_list.append(
                        f"{local_image_file_path}: no bytes were read",
                    )
                else:
                    local_image_bytes_by_path[local_image_file_path] = local_image_file_bytes
            else:
                local_image_read_failure_message_list.append(
                    f"{local_image_file_path}: {read_error_message_text}",
                )
        if len(local_image_read_failure_message_list) > 0:
            raise VideoCommandError(" ".join(local_image_read_failure_message_list))
        else:
            return local_image_bytes_by_path  # one byte string per local path


def image_middle_keyframe_timestamp_list_validate(
    middle_keyframe_list: list[tuple[float, str]],
    duration_seconds: int,
) -> None:
    """Check that middle frames sit inside the clip and far enough apart.

    Inputs are the (timestamp, source) pairs and the clip length in seconds.
    Returns None when every timestamp is allowed.
    Raises VideoCommandError when a time is outside the clip or two times are closer than 1/3 second.
    """
    ordered_middle_keyframe_timestamp_list: list[float] = []
    for middle_keyframe_timestamp_seconds, image_source_text in middle_keyframe_list:
        if image_source_text == "":
            raise VideoCommandError("A middle frame is missing a file path or URL.")
        else:
            ordered_middle_keyframe_timestamp_list.append(middle_keyframe_timestamp_seconds)
    ordered_middle_keyframe_timestamp_list.sort()
    previous_middle_keyframe_timestamp_seconds: float | None = None
    for middle_keyframe_timestamp_seconds in ordered_middle_keyframe_timestamp_list:
        if (
            middle_keyframe_timestamp_seconds <= 0
            or middle_keyframe_timestamp_seconds >= duration_seconds
        ):
            raise VideoCommandError(
                f"A middle frame at {middle_keyframe_timestamp_seconds:g} seconds sits outside the clip. "
                f"Use a time greater than 0 and less than the duration of {duration_seconds} seconds."
            )
        else:
            if previous_middle_keyframe_timestamp_seconds is not None:
                timestamp_gap_seconds = (
                    middle_keyframe_timestamp_seconds - previous_middle_keyframe_timestamp_seconds
                )
                if timestamp_gap_seconds + 1e-9 < minimum_keyframe_spacing_seconds:
                    raise VideoCommandError("Middle frames must be at least 1/3 second apart.")
                else:
                    previous_middle_keyframe_timestamp_seconds = middle_keyframe_timestamp_seconds
            else:
                previous_middle_keyframe_timestamp_seconds = middle_keyframe_timestamp_seconds
    return None  # every middle frame fits in the clip and the gaps are wide enough


def image_middle_keyframe_timestamp_seconds_read(
    middle_keyframe_pair: tuple[float, str],
) -> float:
    """Return the timestamp stored in one middle-frame pair.

    Input is (timestamp seconds, source text).
    Returns the timestamp so the middle frames can be ordered in time.
    The source text stays in the pair and is not part of the sort.
    """
    middle_keyframe_timestamp_seconds = middle_keyframe_pair[0]
    return middle_keyframe_timestamp_seconds  # order middle frames by the time they appear


def image_source_mime_type_for_local_path(local_image_file_path: str) -> str:
    """Return the data-URL content type for a local image path.

    Input is a filesystem path. The type comes from the suffix.
    Returns a mime type such as image/png.
    Raises VideoCommandError when the suffix is not a supported image type.
    """
    local_image_suffix = ""
    suffix_character_list: list[str] = []
    suffix_started = False
    for path_character in local_image_file_path:
        if path_character == "/":
            suffix_character_list = []
            suffix_started = False
        elif path_character == ".":
            suffix_character_list = ["."]
            suffix_started = True
        elif suffix_started:
            suffix_character_list.append(path_character)
        else:
            pass
    local_image_suffix = "".join(suffix_character_list).lower()
    if local_image_suffix in local_image_suffix_to_mime_type:
        return local_image_suffix_to_mime_type[local_image_suffix]  # content type for the data URL
    else:
        supported_suffix_text = ", ".join(sorted(local_image_suffix_to_mime_type))
        raise VideoCommandError(
            f"Unsupported image type for {local_image_file_path}. Use one of: {supported_suffix_text}."
        )


def image_source_remote_url_check(image_source_text: str) -> bool:
    """Return whether image_source_text is an http(s) URL.

    Input is one path or URL.
    Returns True for http and https, so the value is sent unchanged.
    Returns False for a local path, which must be read from disk.
    """
    image_source_text_lower = image_source_text.lower()
    if image_source_text_lower.startswith("https://") or image_source_text_lower.startswith("http://"):
        return True  # the API fetches this URL
    else:
        return False  # this is a local file


def image_source_request_url_for_generation(
    image_source_text: str,
    local_image_bytes_by_path: dict[str, bytes],
) -> str:
    """Turn one image source into the string the video API expects.

    Inputs are the path or URL, and the bytes already read for local files.
    A remote URL is returned unchanged.
    A local file becomes a data:image/...;base64,... URL.
    """
    if image_source_remote_url_check(image_source_text):
        return image_source_text  # remote URLs are sent unchanged
    else:
        mime_type = image_source_mime_type_for_local_path(image_source_text)
        if image_source_text not in local_image_bytes_by_path:
            raise VideoCommandError(f"The image file was not read: {image_source_text}")
        else:
            local_image_file_bytes = local_image_bytes_by_path[image_source_text]
            encoded_image_text = base64.standard_b64encode(local_image_file_bytes).decode("ascii")
            return f"data:{mime_type};base64,{encoded_image_text}"  # local file as a data URL


def image_source_text_list_from_role_assignment(
    image_role_assignment: VideoImageRoleAssignment,
) -> list[str]:
    """List every path or URL stored on a role assignment, including repeats.

    Input is the parsed image roles.
    Returns the sources in a stable order: first, last, loop, references, then middle frames.
    The caller uses this list to decide which local files to open.
    """
    image_source_text_list: list[str] = []
    if image_role_assignment.first_frame_source_text is not None:
        image_source_text_list.append(image_role_assignment.first_frame_source_text)
    else:
        pass
    if image_role_assignment.last_frame_source_text is not None:
        image_source_text_list.append(image_role_assignment.last_frame_source_text)
    else:
        pass
    if image_role_assignment.loop_frame_source_text is not None:
        image_source_text_list.append(image_role_assignment.loop_frame_source_text)
    else:
        pass
    for image_source_text in image_role_assignment.reference_image_source_text_list:
        image_source_text_list.append(image_source_text)
    for middle_keyframe_pair in image_role_assignment.middle_keyframe_list:
        image_source_text = middle_keyframe_pair[1]
        image_source_text_list.append(image_source_text)
    return image_source_text_list  # every source that may be a local file or a URL


def image_url_set_resolved_build(
    image_role_assignment: VideoImageRoleAssignment,
    local_image_bytes_by_path: dict[str, bytes],
) -> ResolvedVideoImageUrls:
    """Resolve every role to a URL or a data URL.

    Inputs are the parsed roles and the local file bytes.
    loop: copies one resolved image onto both the first frame and the last frame.
    Middle frames are ordered by timestamp.
    Returns the URL set passed to the video API.
    """
    if image_role_assignment.loop_frame_source_text is not None:
        loop_frame_url = image_source_request_url_for_generation(
            image_role_assignment.loop_frame_source_text,
            local_image_bytes_by_path,
        )
        first_frame_url = loop_frame_url
        last_frame_url = loop_frame_url
    else:
        if image_role_assignment.first_frame_source_text is not None:
            first_frame_url = image_source_request_url_for_generation(
                image_role_assignment.first_frame_source_text,
                local_image_bytes_by_path,
            )
        else:
            first_frame_url = None
        if image_role_assignment.last_frame_source_text is not None:
            last_frame_url = image_source_request_url_for_generation(
                image_role_assignment.last_frame_source_text,
                local_image_bytes_by_path,
            )
        else:
            last_frame_url = None
    reference_image_url_list = [
        image_source_request_url_for_generation(image_source_text, local_image_bytes_by_path)
        for image_source_text in image_role_assignment.reference_image_source_text_list
    ]
    ordered_middle_keyframe_list = sorted(
        image_role_assignment.middle_keyframe_list,
        key=image_middle_keyframe_timestamp_seconds_read,
    )
    middle_keyframe_url_list = [
        (
            middle_keyframe_timestamp_seconds,
            image_source_request_url_for_generation(image_source_text, local_image_bytes_by_path),
        )
        for middle_keyframe_timestamp_seconds, image_source_text in ordered_middle_keyframe_list
    ]
    return ResolvedVideoImageUrls(
        first_frame_url=first_frame_url,
        last_frame_url=last_frame_url,
        reference_image_url_list=reference_image_url_list,
        middle_keyframe_url_list=middle_keyframe_url_list,
    )  # URLs and data URLs for client.video.generate


# =============================================================================
# SAVED VIDEO URLS
# Reads and writes generated-videos.json through one queue. The worker thread
# is the only code that opens that file.
# =============================================================================

# Jobs are ("read", None, result queue) or ("write", record list, result queue).
generated_video_catalog_job_queue: queue.Queue = queue.Queue()


def generated_video_catalog_absolute_time_from_utc_text(generated_at_utc: str) -> datetime:
    """Parse a stored UTC timestamp.

    Input is the generated_at_utc text from one catalog record.
    Returns that time as an aware UTC datetime.
    Raises VideoCommandError when the text is not YYYY-MM-DDTHH:MM:SSZ.
    """
    try:
        parsed_generated_at = datetime.strptime(generated_at_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise VideoCommandError(
            f"{generated_video_catalog_file_name} has a record with an unreadable time. "
            "The file was left unchanged."
        )
    else:
        return parsed_generated_at.replace(tzinfo=timezone.utc)  # catalog times are UTC


def generated_video_catalog_anchor_path() -> Path:
    """Return the catalog path beside this script.

    There is no input.
    Returns generated-videos.json in the script directory, independent of the working directory.
    """
    return Path(__file__).resolve().with_name(generated_video_catalog_file_name)  # stable list location


def generated_video_catalog_assemble_record_list(catalog_payload: object) -> list[GeneratedVideoRecord]:
    """Turn parsed JSON into catalog records.

    Input is the object returned by json.loads.
    Returns the records in file order.
    Raises VideoCommandError when the shape is wrong, and does not write the file.
    """
    if not isinstance(catalog_payload, dict) or "videos" not in catalog_payload:
        raise VideoCommandError(
            f"{generated_video_catalog_file_name} is missing the videos list. The file was left unchanged."
        )
    else:
        video_entry_list = catalog_payload["videos"]
        if not isinstance(video_entry_list, list):
            raise VideoCommandError(
                f"{generated_video_catalog_file_name} is missing the videos list. The file was left unchanged."
            )
        else:
            generated_video_record_list: list[GeneratedVideoRecord] = []
            for video_entry in video_entry_list:
                if not isinstance(video_entry, dict):
                    raise VideoCommandError(
                        f"{generated_video_catalog_file_name} has a record that is not an object. "
                        "The file was left unchanged."
                    )
                else:
                    video_url = video_entry.get("video_url")
                    generated_at_utc = video_entry.get("generated_at_utc")
                    prompt_text = video_entry.get("prompt")
                    video_generation_model_name = video_entry.get("model")
                    aspect_ratio_text = video_entry.get("aspect_ratio")
                    duration_seconds = video_entry.get("duration_seconds")
                    if not isinstance(video_url, str) or video_url.strip() == "":
                        raise VideoCommandError(
                            f"{generated_video_catalog_file_name} has a record without a video URL. "
                            "The file was left unchanged."
                        )
                    elif not isinstance(generated_at_utc, str):
                        raise VideoCommandError(
                            f"{generated_video_catalog_file_name} has a record with an unreadable time. "
                            "The file was left unchanged."
                        )
                    elif not isinstance(prompt_text, str) or not isinstance(video_generation_model_name, str):
                        raise VideoCommandError(
                            f"{generated_video_catalog_file_name} has a record with an unreadable prompt or model. "
                            "The file was left unchanged."
                        )
                    elif aspect_ratio_text is not None and not isinstance(aspect_ratio_text, str):
                        raise VideoCommandError(
                            f"{generated_video_catalog_file_name} has a record with an unreadable aspect ratio. "
                            "The file was left unchanged."
                        )
                    elif duration_seconds is not None and (
                        isinstance(duration_seconds, bool) or not isinstance(duration_seconds, int)
                    ):
                        raise VideoCommandError(
                            f"{generated_video_catalog_file_name} has a record with an unreadable duration. "
                            "The file was left unchanged."
                        )
                    else:
                        generated_video_catalog_absolute_time_from_utc_text(generated_at_utc)
                        generated_video_record_list.append(
                            GeneratedVideoRecord(
                                video_url=video_url,
                                generated_at_utc=generated_at_utc,
                                prompt_text=prompt_text,
                                video_generation_model_name=video_generation_model_name,
                                aspect_ratio_text=aspect_ratio_text,
                                duration_seconds=duration_seconds,
                            )
                        )
            return generated_video_record_list  # records in the order stored on disk


def generated_video_catalog_bytes_read() -> list[GeneratedVideoRecord]:
    """Read the catalog file.

    There is no input. A missing file means no videos have been saved.
    Returns the stored records. Raises VideoCommandError when the file cannot be parsed.
    This function is called only from the catalog worker.
    """
    catalog_path = generated_video_catalog_anchor_path()
    if not catalog_path.exists():
        return []  # nothing has been saved yet
    else:
        try:
            catalog_text = catalog_path.read_text(encoding="utf-8")
        except OSError as catalog_read_error:
            raise VideoCommandError(
                f"{generated_video_catalog_file_name} could not be read. "
                f"The file was left unchanged. {catalog_read_error}"
            )
        else:
            try:
                catalog_payload = json.loads(catalog_text)
            except json.JSONDecodeError:
                raise VideoCommandError(
                    f"{generated_video_catalog_file_name} is not valid JSON. The file was left unchanged."
                )
            else:
                return generated_video_catalog_assemble_record_list(catalog_payload)  # parsed records


def generated_video_catalog_bytes_write(generated_video_record_list: list[GeneratedVideoRecord]) -> None:
    """Replace the catalog file with generated_video_record_list.

    Input is the full list to store, oldest first.
    Writes a temporary file beside the catalog, then replaces the catalog.
    Raises VideoCommandError when the write fails. A failed write leaves the previous file in place.
    """
    catalog_path = generated_video_catalog_anchor_path()
    catalog_payload = {
        "videos": [
            {
                "video_url": generated_video_record.video_url,
                "generated_at_utc": generated_video_record.generated_at_utc,
                "prompt": generated_video_record.prompt_text,
                "model": generated_video_record.video_generation_model_name,
                "aspect_ratio": generated_video_record.aspect_ratio_text,
                "duration_seconds": generated_video_record.duration_seconds,
            }
            for generated_video_record in generated_video_record_list
        ]
    }
    catalog_text = json.dumps(catalog_payload, indent=2) + "\n"
    temporary_catalog_path = catalog_path.with_suffix(".json.tmp")
    try:
        temporary_catalog_path.write_text(catalog_text, encoding="utf-8")
        os.replace(temporary_catalog_path, catalog_path)
    except OSError as catalog_write_error:
        raise VideoCommandError(
            f"{generated_video_catalog_file_name} could not be written. {catalog_write_error}"
        )
    else:
        return None  # the catalog on disk matches generated_video_record_list


def generated_video_catalog_file_job_perform() -> None:
    """Perform catalog reads and writes queued by the main thread.

    Each job is ("read", None, result queue) or ("write", record list, result queue).
    The result queue receives (True, payload) or (False, error text).
    This worker is the only place the catalog file is opened.
    """
    while True:
        catalog_job = generated_video_catalog_job_queue.get()
        catalog_job_name = catalog_job[0]
        catalog_job_result_queue = catalog_job[2]
        try:
            if catalog_job_name == "read":
                catalog_record_list = generated_video_catalog_bytes_read()
                catalog_job_result_queue.put((True, catalog_record_list))
            elif catalog_job_name == "write":
                generated_video_catalog_bytes_write(catalog_job[1])
                catalog_job_result_queue.put((True, None))
            else:
                catalog_job_result_queue.put((False, "Unknown catalog job."))
        except VideoCommandError as catalog_error:
            catalog_job_result_queue.put((False, catalog_error.message_text))
        except Exception as catalog_error:
            catalog_job_result_queue.put((False, str(catalog_error)))
        else:
            pass
        finally:
            generated_video_catalog_job_queue.task_done()


generated_video_catalog_thread = threading.Thread(
    target=generated_video_catalog_file_job_perform,
    name="generated_video_catalog_file_job_perform",
    daemon=True,
)
generated_video_catalog_thread.start()


def generated_video_catalog_load() -> list[GeneratedVideoRecord]:
    """Load the catalog through the file queue.

    There is no input.
    Returns the stored records, or an empty list when the file is missing.
    Raises VideoCommandError when the file is unreadable, and does not change it.
    """
    catalog_job_result_queue: queue.Queue = queue.Queue()
    generated_video_catalog_job_queue.put(("read", None, catalog_job_result_queue))
    catalog_job_succeeded, catalog_job_payload = catalog_job_result_queue.get()
    if catalog_job_succeeded:
        return catalog_job_payload  # records read by the worker
    else:
        raise VideoCommandError(str(catalog_job_payload))


def generated_video_catalog_prune_expired(
    generated_video_record_list: list[GeneratedVideoRecord],
) -> tuple[list[GeneratedVideoRecord], int]:
    """Drop records that are 24 hours old or older.

    Input is the catalog in file order.
    Returns the records still inside the 24 hour window, and how many were dropped.
    The clock is the local machine's UTC time. The API is not contacted.
    """
    kept_generated_video_record_list: list[GeneratedVideoRecord] = []
    expired_generated_video_record_count = 0
    current_utc_time = datetime.now(timezone.utc)
    generated_video_url_lifetime = timedelta(hours=generated_video_url_lifetime_hours)
    for generated_video_record in generated_video_record_list:
        generated_at_utc_time = generated_video_catalog_absolute_time_from_utc_text(
            generated_video_record.generated_at_utc,
        )
        record_age = current_utc_time - generated_at_utc_time
        if record_age >= generated_video_url_lifetime:
            expired_generated_video_record_count += 1
        else:
            kept_generated_video_record_list.append(generated_video_record)
    return (
        kept_generated_video_record_list,
        expired_generated_video_record_count,
    )  # survivors stay in their original order


def generated_video_catalog_save(generated_video_record_list: list[GeneratedVideoRecord]) -> None:
    """Store the catalog through the file queue.

    Input is the full record list, oldest first.
    Returns None after the worker has replaced the file.
    Raises VideoCommandError when the write fails.
    """
    catalog_job_result_queue: queue.Queue = queue.Queue()
    generated_video_catalog_job_queue.put(("write", generated_video_record_list, catalog_job_result_queue))
    catalog_job_succeeded, catalog_job_payload = catalog_job_result_queue.get()
    if catalog_job_succeeded:
        return None  # the worker finished the write
    else:
        raise VideoCommandError(str(catalog_job_payload))


def generated_video_catalog_text_for_list() -> str:
    """Delete expired URLs, then build the text for -l.

    There is no input. The catalog file is the source.
    Records 24 hours old or older are removed from the file before the text is built.
    Returns the list text, newest first, or the empty-list message.
    """
    generated_video_record_list = generated_video_catalog_load()
    kept_generated_video_record_list, expired_generated_video_record_count = (
        generated_video_catalog_prune_expired(generated_video_record_list)
    )
    if expired_generated_video_record_count > 0:
        generated_video_catalog_save(kept_generated_video_record_list)
    else:
        pass
    if len(kept_generated_video_record_list) == 0:
        return generated_video_catalog_empty_message  # nothing remains inside the 24 hour window
    else:
        generated_video_catalog_block_list: list[str] = []
        for generated_video_record in reversed(kept_generated_video_record_list):
            if generated_video_record.aspect_ratio_text is None:
                aspect_ratio_display_text = "aspect default"
            else:
                aspect_ratio_display_text = generated_video_record.aspect_ratio_text
            if generated_video_record.duration_seconds is None:
                duration_display_text = "duration default"
            else:
                duration_display_text = f"{generated_video_record.duration_seconds}s"
            generated_video_catalog_block_list.append(
                f"{generated_video_record.generated_at_utc}  {aspect_ratio_display_text}  "
                f"{duration_display_text}\n"
                f"{generated_video_record.video_url}\n"
                f"{generated_video_record.prompt_text}"
            )
        return "\n\n".join(generated_video_catalog_block_list)  # newest saved video first


def generated_video_catalog_url_append(generated_video_record: GeneratedVideoRecord) -> None:
    """Append one finished video to the catalog.

    Input is the new record. Older records stay in front of it.
    Returns None after the file is replaced.
    Raises VideoCommandError when the current file cannot be read or written, and does not replace a corrupt file.
    """
    generated_video_record_list = generated_video_catalog_load()
    generated_video_record_list.append(generated_video_record)
    generated_video_catalog_save(generated_video_record_list)
    return None  # the new URL is the last record in the file


# =============================================================================
# COMMAND LINE PARSING
# =============================================================================


def cli_argument_parser_build() -> argparse.ArgumentParser:
    """Build the flag parser.

    There is no input. Defaults come from the configuration section.
    Returns a parser whose flags may appear in any order.
    -h and --help are provided by argparse and exit 0.
    """
    argument_parser = argparse.ArgumentParser(
        prog="grok-imagine-video.py",
        description=command_help_description,
        epilog=command_help_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    argument_parser.add_argument(
        "-p",
        "--prompt",
        dest="prompt_argument",
        default=None,
        help="Prompt text. Required unless a frame is pinned with first:, last:, loop:, or a timestamp.",
    )
    argument_parser.add_argument(
        "-m",
        "--model",
        dest="video_generation_model_name",
        default=None,
        help=f"Video model. Default: {default_video_generation_model_name}.",
    )
    argument_parser.add_argument(
        "-i",
        "--images",
        dest="image_argument_token_list",
        nargs="+",
        default=None,
        metavar="IMAGE",
        help="Local files or URLs. Optional role prefixes: ref:, first:, last:, loop:, or seconds such as 3:.",
    )
    argument_parser.add_argument(
        "-d",
        "--duration",
        dest="duration_argument",
        type=int,
        default=None,
        metavar="SECONDS",
        help=(
            f"Clip length in whole seconds, from {minimum_video_duration_seconds} "
            f"through {maximum_video_duration_seconds}. "
            f"Omit it to let the API use {default_video_duration_seconds_when_omitted} seconds."
        ),
    )
    argument_parser.add_argument(
        "-ar",
        "--aspect-ratio",
        dest="aspect_ratio_argument",
        default=None,
        metavar="RATIO",
        help="One of 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3. Omit it to let the API use 16:9.",
    )
    argument_parser.add_argument(
        "-s",
        "--silent",
        dest="silent_video_requested",
        action="store_true",
        help="Build a video with no audio track. Do not pass -a with this flag.",
    )
    argument_parser.add_argument(
        "-a",
        "--audio",
        dest="audio_argument",
        nargs="+",
        action="append",
        default=None,
        metavar="VOICE",
        help="Preset voice id, up to 3. Example: -a eve or -a eve leo. Overrides nothing about --silent; the two cannot combine.",
    )
    argument_parser.add_argument(
        "--api-key",
        dest="api_key_argument",
        default=None,
        metavar="KEY",
        help="API key for this run. Overrides XAI_API_KEY when both are set. No short flag.",
    )
    argument_parser.add_argument(
        "-l",
        "--list-generated-videos",
        dest="list_generated_videos_requested",
        action="store_true",
        help=(
            "List saved video URLs and delete any link 24 hours old or older. "
            "Use this flag alone."
        ),
    )
    return argument_parser  # flags may be written in any order


def cli_aspect_ratio_resolve(aspect_ratio_argument: str | None) -> VideoAspectRatio | None:
    """Map -ar to a VideoAspectRatio, or None when the flag was omitted.

    Input is the raw flag text.
    Returns one of the seven ratios the SDK accepts.
    Raises VideoCommandError when the text is not one of those ratios.
    """
    if aspect_ratio_argument is None:
        return None  # omit aspect_ratio and keep the API default of 16:9
    else:
        stripped_aspect_ratio_argument = aspect_ratio_argument.strip()
        if stripped_aspect_ratio_argument in allowed_video_aspect_ratio_by_text:
            return allowed_video_aspect_ratio_by_text[stripped_aspect_ratio_argument]  # typed SDK ratio
        else:
            allowed_aspect_ratio_text = ", ".join(allowed_video_aspect_ratio_by_text)
            raise VideoCommandError(f"Aspect ratio must be one of: {allowed_aspect_ratio_text}.")


def cli_audio_voice_id_list_flatten(audio_argument: list[list[str]] | None) -> list[str]:
    """Flatten repeated -a groups into one voice list.

    Input is argparse's append result, or None when -a was omitted.
    -a eve leo and -a eve -a leo both become ["eve", "leo"].
    Returns the voice ids in order. Raises VideoCommandError when a id is blank or there are more than 3.
    """
    if audio_argument is None:
        return []  # no narration voices
    else:
        reference_voice_id_list: list[str] = []
        for audio_argument_group in audio_argument:
            for reference_voice_id in audio_argument_group:
                stripped_reference_voice_id = reference_voice_id.strip()
                if stripped_reference_voice_id == "":
                    raise VideoCommandError("A voice id passed to -a/--audio was blank.")
                else:
                    reference_voice_id_list.append(stripped_reference_voice_id)
        if len(reference_voice_id_list) > maximum_reference_voice_count:
            raise VideoCommandError(
                f"At most {maximum_reference_voice_count} voices can be passed to -a/--audio."
            )
        else:
            return reference_voice_id_list  # voice ids in the order given


def cli_duration_seconds_resolve(
    duration_argument: int | None,
    middle_keyframe_present: bool,
) -> int | None:
    """Choose the duration value that will be sent, or None to omit it.

    Inputs are the -d value and whether any middle frame was set.
    When -d is omitted and a middle frame exists, returns 8 so the timestamps
    are checked against the clip length the API would otherwise assume.
    Returns None when -d is omitted and there is no middle frame.
    """
    if duration_argument is None:
        if middle_keyframe_present:
            return default_video_duration_seconds_when_omitted  # middle frames need a known clip length
        else:
            return None  # omit duration and keep the API default
    else:
        if (
            duration_argument < minimum_video_duration_seconds
            or duration_argument > maximum_video_duration_seconds
        ):
            raise VideoCommandError(
                "Duration must be a whole number of seconds from "
                f"{minimum_video_duration_seconds} through {maximum_video_duration_seconds}."
            )
        else:
            return duration_argument  # the clip length from -d


def cli_list_generated_videos_conflict_check(parsed_arguments: argparse.Namespace) -> None:
    """Reject -l when it is combined with a generation flag.

    Input is the parsed arguments. -l is already known to be set.
    Returns None when no generation flag is present.
    Raises VideoCommandError when a generation flag is also set, so listing cannot start a video.
    """
    generation_flag_present = (
        parsed_arguments.prompt_argument is not None
        or parsed_arguments.video_generation_model_name is not None
        or parsed_arguments.image_argument_token_list is not None
        or parsed_arguments.duration_argument is not None
        or parsed_arguments.silent_video_requested
        or parsed_arguments.audio_argument is not None
        or parsed_arguments.aspect_ratio_argument is not None
        or parsed_arguments.api_key_argument is not None
    )
    if generation_flag_present:
        raise VideoCommandError(
            "Use -l/--list-generated-videos by itself. It lists saved videos and does not generate one."
        )
    else:
        return None  # -l is the only requested action


def cli_parsed_arguments_validate(parsed_arguments: argparse.Namespace) -> VideoCommandRequest:
    """Check parsed flags and return the request, still using local paths.

    Input is the argparse namespace. This does not read files and does not call the API.
    A prompt is required unless first:, last:, loop:, or a timestamp pins a frame.
    Returns the checked request. Raises VideoCommandError when the flags conflict.
    """
    prompt_argument = parsed_arguments.prompt_argument
    video_generation_model_name = parsed_arguments.video_generation_model_name
    image_argument_token_list = parsed_arguments.image_argument_token_list
    duration_argument = parsed_arguments.duration_argument
    silent_video_requested = parsed_arguments.silent_video_requested
    audio_argument = parsed_arguments.audio_argument
    api_key_argument = parsed_arguments.api_key_argument
    aspect_ratio = cli_aspect_ratio_resolve(parsed_arguments.aspect_ratio_argument)
    image_role_assignment = image_argument_resolved_role_assignment_build(image_argument_token_list)
    middle_keyframe_present = len(image_role_assignment.middle_keyframe_list) > 0
    duration_seconds = cli_duration_seconds_resolve(duration_argument, middle_keyframe_present)
    if middle_keyframe_present:
        if duration_seconds is None:
            raise VideoCommandError("A middle frame needs a clip length.")
        else:
            image_middle_keyframe_timestamp_list_validate(
                image_role_assignment.middle_keyframe_list,
                duration_seconds,
            )
    else:
        pass
    reference_voice_id_list = cli_audio_voice_id_list_flatten(audio_argument)
    if silent_video_requested and len(reference_voice_id_list) > 0:
        raise VideoCommandError(
            "Use -s/--silent only when -a/--audio is omitted. A silent video has no narration."
        )
    else:
        pass
    frame_is_pinned = (
        image_role_assignment.first_frame_source_text is not None
        or image_role_assignment.last_frame_source_text is not None
        or image_role_assignment.loop_frame_source_text is not None
        or middle_keyframe_present
    )
    if prompt_argument is None or prompt_argument.strip() == "":
        if frame_is_pinned:
            prompt_text = ""
        else:
            raise VideoCommandError(
                "Pass -p or --prompt. A prompt can be omitted only when a frame is pinned "
                "with first:, last:, loop:, or a timestamp."
            )
    else:
        prompt_text = prompt_argument
    if video_generation_model_name is None:
        resolved_video_generation_model_name = default_video_generation_model_name
    elif str(video_generation_model_name).strip() == "":
        raise VideoCommandError("Pass -m or --model with a model name.")
    else:
        resolved_video_generation_model_name = str(video_generation_model_name).strip()
    return VideoCommandRequest(
        prompt_text=prompt_text,
        video_generation_model_name=resolved_video_generation_model_name,
        duration_seconds=duration_seconds,
        silent_video_requested=silent_video_requested,
        reference_voice_id_list=reference_voice_id_list,
        api_key_argument=api_key_argument,
        aspect_ratio=aspect_ratio,
        image_role_assignment=image_role_assignment,
    )  # flags are consistent; local files are still paths


# =============================================================================
# REMOTE DATA FETCHING
# The video generation call is the remote fetch. It runs only from main,
# after the flags have been checked, and a one-line note is written to stderr first.
# The note names the model and never includes the API key or the prompt.
# =============================================================================


def video_client_api_key_choice_resolve(api_key_argument: str | None) -> str | None:
    """Decide which key source to use, without creating a client.

    Input is the --api-key value, or None when the flag was omitted.
    Returns the flag value when it is present, including when XAI_API_KEY is also set.
    Returns None when the flag is omitted and XAI_API_KEY is set, so the SDK reads that variable.
    Raises VideoCommandError when the flag is blank, or when neither source is set.
    The error text does not include a key.
    """
    if api_key_argument is not None:
        if api_key_argument.strip() == "":
            raise VideoCommandError(
                "Pass a key to --api-key, or omit the flag and set "
                f"{environment_variable_name_for_api_key}."
            )
        else:
            return api_key_argument  # the flag overrides XAI_API_KEY for this run
    else:
        environment_api_key = os.environ.get(environment_variable_name_for_api_key)
        if environment_api_key is None or environment_api_key.strip() == "":
            raise VideoCommandError(
                "Pass --api-key, or set the "
                f"{environment_variable_name_for_api_key} environment variable."
            )
        else:
            return None  # Client() with no key lets the SDK read XAI_API_KEY


def video_client_create(api_key_argument: str | None) -> xai_sdk.Client:
    """Create the xAI client from --api-key or from XAI_API_KEY.

    Input is the --api-key value, or None when the flag was omitted.
    When the flag is set, that value is passed into the client and overrides the environment.
    When the flag is omitted, the client is built with no key argument.
    Returns the client. Raises VideoCommandError when no key is available.
    """
    chosen_api_key = video_client_api_key_choice_resolve(api_key_argument)
    if chosen_api_key is None:
        return xai_sdk.Client()  # the SDK reads XAI_API_KEY
    else:
        return xai_sdk.Client(api_key=chosen_api_key)  # --api-key overrides the environment


def video_generation_start(video_command_request: VideoCommandRequest) -> str:
    """Read local images, call the video API, and return the video URL.

    Input is the checked command. Local files are read here, through the file queue.
    A stderr line names the model before the request. It does not include the API key.
    Returns the URL from the finished video response.
    """
    local_image_file_path_list: list[str] = []
    for image_source_text in image_source_text_list_from_role_assignment(
        video_command_request.image_role_assignment,
    ):
        if image_source_remote_url_check(image_source_text):
            pass
        else:
            image_source_mime_type_for_local_path(image_source_text)
            local_image_file_path_list.append(image_source_text)
    local_image_bytes_by_path = image_file_paths_read_together(local_image_file_path_list)
    resolved_video_image_urls = image_url_set_resolved_build(
        video_command_request.image_role_assignment,
        local_image_bytes_by_path,
    )
    # Resolve the key before announcing the request, so a missing key stops first.
    video_client_api_key_choice_resolve(video_command_request.api_key_argument)
    standard_error_message_write(
        "Generating video with model "
        f"{video_command_request.video_generation_model_name}.",
    )
    xai_client = video_client_create(video_command_request.api_key_argument)
    keyframe_list: list[UrlKeyframe] | None
    if len(resolved_video_image_urls.middle_keyframe_url_list) == 0:
        keyframe_list = None
    else:
        keyframe_list = []
        for middle_keyframe_timestamp_seconds, middle_keyframe_url in (
            resolved_video_image_urls.middle_keyframe_url_list
        ):
            middle_keyframe: UrlKeyframe = {
                "image_url": middle_keyframe_url,
                "timestamp": middle_keyframe_timestamp_seconds,
            }
            keyframe_list.append(middle_keyframe)
    reference_audio_list: list[VoiceAudioRef] | None
    if len(video_command_request.reference_voice_id_list) == 0:
        reference_audio_list = None
    else:
        reference_audio_list = []
        for reference_voice_id in video_command_request.reference_voice_id_list:
            reference_audio: VoiceAudioRef = {"voice_id": reference_voice_id}
            reference_audio_list.append(reference_audio)
    reference_image_url_list: list[str] | None
    if len(resolved_video_image_urls.reference_image_url_list) == 0:
        reference_image_url_list = None
    else:
        reference_image_url_list = resolved_video_image_urls.reference_image_url_list
    generate_audio: bool | None
    if video_command_request.silent_video_requested:
        generate_audio = False
    else:
        generate_audio = None
    video_response = xai_client.video.generate(
        prompt=video_command_request.prompt_text,
        model=video_command_request.video_generation_model_name,
        image_url=resolved_video_image_urls.first_frame_url,
        last_frame_url=resolved_video_image_urls.last_frame_url,
        keyframes=keyframe_list,
        duration=video_command_request.duration_seconds,
        aspect_ratio=video_command_request.aspect_ratio,
        reference_image_urls=reference_image_url_list,
        reference_audios=reference_audio_list,
        generate_audio=generate_audio,
    )
    video_url = video_response.url
    generated_video_record = GeneratedVideoRecord(
        video_url=video_url,
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        prompt_text=video_command_request.prompt_text,
        video_generation_model_name=video_command_request.video_generation_model_name,
        aspect_ratio_text=video_command_request.aspect_ratio,
        duration_seconds=video_command_request.duration_seconds,
    )
    try:
        generated_video_catalog_url_append(generated_video_record)
    except VideoCommandError as catalog_error:
        standard_error_message_write(catalog_error.message_text)
    else:
        pass
    return video_url  # response.url, valid for 24 hours according to the SDK


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
