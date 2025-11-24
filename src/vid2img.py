import cv2
import os
import argparse


def video_to_images(video_path, output_folder, fps_out):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Get video information
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps_in

    print(f"Input video FPS: {fps_in:.2f}")
    print(f"Target output FPS: {fps_out}")
    print(f"Video duration: {duration:.2f} seconds")

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Determine frame interval for output
    frame_interval = int(round(fps_in / fps_out)) if fps_out < fps_in else 1

    frame_index = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every N-th frame
        if frame_index % frame_interval == 0:
            frame_name = f"frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    print(f"Done! Saved {saved_count} frames to '{output_folder}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a video into an image sequence."
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", required=True, help="Output folder for image sequence")
    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Number of output images per second of video",
    )

    args = parser.parse_args()

    video_to_images(args.video, args.out, args.fps)
