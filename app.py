import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from main_pipeline import run as run_pipeline

st.set_page_config(page_title="Floor Plan Generator", layout="wide")

st.title("Floor Plan Generator from Video")
st.markdown("""
Upload a video of a room scan to generate a 2D floor plan with detected furniture.
""")

uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()

    st.video(video_path)

    if st.button("Generate Floor Plan"):
        with st.spinner("Processing... This may take a while (Reconstruction + Detection)"):
            # Create a temporary output directory
            output_dir = tempfile.mkdtemp()
            
            # Construct arguments for the pipeline
            # We use subprocess to run it to avoid state issues, or we can import and run.
            # Importing is better for feedback but might be tricky with global args.
            # Let's try running via subprocess for isolation first, or just call the function if we can mock args.
            
            # Actually, calling the function directly is cleaner if we can set sys.argv
            # But let's just use subprocess to be safe and capture output
            
            cmd = [
                sys.executable, "src/main_pipeline.py",
                "--reconstruct",
                "--input_video", video_path,
                "--output_dir", output_dir
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream logs
            log_container = st.empty()
            logs = []
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    logs.append(line.strip())
                    # Update logs every few lines to avoid UI freezing
                    if len(logs) % 5 == 0:
                        log_container.code("\n".join(logs[-20:])) # Show last 20 lines
            
            if process.returncode == 0:
                st.success("Processing Complete!")
                
                # Display results
                floor_plan_path = os.path.join(output_dir, "floor_plan.png")
                if os.path.exists(floor_plan_path):
                    st.image(floor_plan_path, caption="Generated Floor Plan", use_column_width=True)
                    
                    with open(floor_plan_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Floor Plan",
                            data=file,
                            file_name="floor_plan.png",
                            mime="image/png"
                        )
                else:
                    st.error("Floor plan image was not generated.")
            else:
                st.error("Pipeline failed. Check logs above.")
            
            # Cleanup
            # shutil.rmtree(output_dir) # Keep for debugging for now
            os.unlink(video_path)

st.sidebar.header("About")
st.sidebar.info(
    "This application uses 3D reconstruction (COLMAP + OpenMVS) and YOLO object detection "
    "to create a floor plan from a video scan."
)
