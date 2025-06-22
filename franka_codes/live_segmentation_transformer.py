# main.py
#
# Description:
# This script performs real-time object segmentation on a live video stream from an
# Intel RealSense camera. It uses the `CLIPSeg` model, a transformer-based
# architecture that can segment objects in an image based on a text query.
#
# Requirements:
# pip install torch transformers Pillow opencv-python pyrealsense2
#
# Note:
# - This script requires a connected Intel RealSense camera.
# - The model will be downloaded from the Hugging Face Hub on the first run.
# - For better performance, run this on a machine with a CUDA-enabled GPU.
#   The script will automatically detect and use a GPU if available.

import torch
import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def main():
    # --- 1. Initialize Model and Processor ---
    # Load the CLIPSeg model and processor from Hugging Face.
    # `CLIPSeg` is designed for zero-shot image segmentation.
    print("Loading CLIPSeg model and processor...")
    try:
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection to download the model.")
        return

    # Automatically select device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded successfully on '{device}'.")

    # --- 2. Configure and Start RealSense Pipeline ---
    print("Configuring RealSense camera pipeline...")
    pipeline = rs.pipeline()
    config = rs.config()

    # Define camera stream parameters
    live_cam_W, live_cam_H = 1280, 720
    fps = 30

    # Enable color and depth streams
    # Using bgr8 format as it's directly compatible with OpenCV
    config.enable_stream(rs.stream.color, live_cam_W, live_cam_H, rs.format.bgr8, fps)
    # The depth stream isn't used for segmentation here, but is enabled as per the request.
    # It could be used for 3D applications with the segmentation mask.
    config.enable_stream(rs.stream.depth, live_cam_W, live_cam_H, rs.format.z16, fps)

    # Start streaming
    try:
        profile = pipeline.start(config)
        print("Camera pipeline started.")
    except RuntimeError as e:
        print(f"Error starting RealSense pipeline: {e}")
        print("Please check if the RealSense camera is connected properly.")
        return

    # --- 3. Main Processing Loop ---
    # Define the object queries to segment.
    # You can modify this list or prompt the user for input inside the loop.
    queries = ["an oreo blue package", "a red candy ", "a green candy bar"]
    query_index = 0
    current_prompt = queries[query_index]

    print("\nStarting segmentation loop...")
    print("Press 'n' to cycle to the next object query.")
    print("Press 'q' to quit.")

    try:
        while True:
            # --- 3a. Get Frames from Camera ---
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert the color frame to a NumPy array
            color_image_np = np.asanyarray(color_frame.get_data())

            # Convert BGR (from RealSense) to RGB for the model and PIL
            color_image_rgb = cv2.cvtColor(color_image_np, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(color_image_rgb)

            # --- 3b. Prepare Inputs and Run Inference ---
            # The processor prepares the image and text for the model
            inputs = processor(
                text=current_prompt,
                images=input_image,
                padding=True,
                return_tensors="pt"
            ).to(device)

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)

            # The raw output is in 'logits'. We need to process it.
            # The shape is (batch_size, 1, height, width)
            mask_logits = outputs.logits.cpu()

            # --- 3c. Post-process the Mask ---
            # Apply sigmoid to get probabilities, then resize to original image size
            mask_probs = torch.sigmoid(mask_logits).squeeze()
            mask_pil = Image.fromarray(mask_probs.numpy())
            mask_resized = mask_pil.resize((live_cam_W, live_cam_H))
            mask_np = np.array(mask_resized)

            # Threshold the probability map to get a binary mask
            threshold = 0.5
            binary_mask = (mask_np > threshold).astype(np.uint8) * 255

            # --- 3d. Visualize the Output ---
            # Create a color overlay for the segmented area
            # Here we are making the mask red (in BGR format for OpenCV)
            overlay_color = np.array([0, 0, 255]) # Red
            overlay = np.zeros_like(color_image_np, dtype=np.uint8)
            overlay[binary_mask == 255] = overlay_color

            # Blend the original image with the overlay
            alpha = 0.5 # Transparency of the overlay
            blended_image = cv2.addWeighted(color_image_np, 1, overlay, alpha, 0)

            # Add text to the display showing the current prompt
            cv2.putText(
                blended_image,
                f"Query: '{current_prompt}' (Press 'n' for next)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255), # White color
                2,
                cv2.LINE_AA,
            )

            # Display the result
            cv2.imshow("RealSense CLIP Segmentation", blended_image)

            # --- 3e. Handle User Input ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' pressed. Exiting...")
                break
            elif key == ord('n'):
                query_index = (query_index + 1) % len(queries)
                current_prompt = queries[query_index]
                print(f"Switched to new query: '{current_prompt}'")


    finally:
        # --- 4. Cleanup ---
        print("Stopping pipeline and closing windows.")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()