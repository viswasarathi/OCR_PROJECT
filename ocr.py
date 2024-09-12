import streamlit as st
import torch
import easyocr
import cv2
import time
import psutil
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import tempfile
from PIL import Image

# Function to compute text similarity
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Function to process video with OCR model
def process_video(video_path, reader, device, frame_skip=5, display=False, save_output=False, output_path='output.avi', resize_factor=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=None)
    start_memory = psutil.virtual_memory().used

    frame_count = 0
    ocr_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            # Resize the frame to reduce processing time
            frame = cv2.resize(frame, (width, height))

            # Convert the frame to RGB format (required by EasyOCR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # OCR Inference
            result = reader.readtext(rgb_frame)
            ocr_results.append(result)

            if display:
                for (bbox, text, prob) in result:
                    # Draw bounding box and text on the frame
                    top_left = tuple([int(val) for val in bbox[0]])
                    bottom_right = tuple([int(val) for val in bbox[2]])
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display the processed frame
                cv2.imshow(f'OCR on {device}', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_output:
                out.write(frame)  # Save the frame with OCR results

        frame_count += 1

    end_time = time.time()
    end_cpu = psutil.cpu_percent(interval=None)
    end_memory = psutil.virtual_memory().used

    processing_time = end_time - start_time
    avg_fps = frame_count / processing_time
    cpu_usage = (end_cpu - start_cpu) / frame_count
    memory_usage = (end_memory - start_memory) / (1024 ** 2)  # Convert to MB

    cap.release()
    if display:
        cv2.destroyAllWindows()
    if save_output:
        out.release()

    return avg_fps, ocr_results, cpu_usage, memory_usage

# Streamlit App
st.title('Video OCR Processing with GPU and CPU')

uploaded_video = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
if uploaded_video is not None:
    # Save uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_video.read())
        temp_file_path = temp_file.name

    device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')

    st.write("Loading GPU model...")
    reader_gpu = easyocr.Reader(['en'], gpu=True)

    st.write("Processing video with GPU...")
    gpu_fps, gpu_results, gpu_cpu_usage, gpu_memory_usage = process_video(temp_file_path, reader_gpu, 'GPU', frame_skip=5, display=False, save_output=True, output_path='output_gpu.avi', resize_factor=0.5)

    st.write("Loading CPU model...")
    reader_cpu = easyocr.Reader(['en'], gpu=False)

    st.write("Processing video with CPU...")
    cpu_fps, cpu_results, cpu_cpu_usage, cpu_memory_usage = process_video(temp_file_path, reader_cpu, 'CPU', frame_skip=5, display=False, save_output=True, output_path='output_cpu.avi', resize_factor=0.5)

    # FPS Comparison and Display
    if gpu_fps is not None and cpu_fps is not None:
        st.write(f"GPU FPS: {gpu_fps:.2f}")
        st.write(f"CPU FPS: {cpu_fps:.2f}")

        fig, ax = plt.subplots()
        labels = ['GPU', 'CPU']
        fps_values = [gpu_fps, cpu_fps]
        ax.bar(labels, fps_values, color=['blue', 'green'])
        ax.set_ylabel('Frames Per Second (FPS)')
        ax.set_title('GPU vs CPU FPS Comparison')
        ax.set_ylim(0, max(fps_values) + 2)
        st.pyplot(fig)
    else:
        st.write("Could not calculate FPS for one or both runs.")

    # CPU and Memory Usage Comparison
    if gpu_cpu_usage is not None and cpu_cpu_usage is not None:
        st.write(f"GPU CPU Usage: {gpu_cpu_usage:.2f}%")
        st.write(f"CPU CPU Usage: {cpu_cpu_usage:.2f}%")

        fig, ax = plt.subplots()
        labels = ['GPU', 'CPU']
        usage_values = [gpu_cpu_usage, cpu_cpu_usage]
        ax.bar(labels, usage_values, color=['blue', 'green'])
        ax.set_ylabel('CPU Usage')
        ax.set_title('GPU vs CPU CPU Usage Comparison')
        ax.set_ylim(0, max(usage_values) + 10)
        st.pyplot(fig)

    if gpu_memory_usage is not None and cpu_memory_usage is not None:
        st.write(f"GPU Memory Usage: {gpu_memory_usage:.2f} MB")
        st.write(f"CPU Memory Usage: {cpu_memory_usage:.2f} MB")

        fig, ax = plt.subplots()
        labels = ['GPU', 'CPU']
        usage_values = [gpu_memory_usage, cpu_memory_usage]
        ax.bar(labels, usage_values, color=['blue', 'green'])
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('GPU vs CPU Memory Usage Comparison')
        ax.set_ylim(0, max(usage_values) + 10)
        st.pyplot(fig)

    # OCR Accuracy Comparison
    if gpu_results is not None and cpu_results is not None:
        total_similarity = 0
        comparisons = 0
        st.write("Accuracy comparison (sample results):")
        for i in range(min(5, len(gpu_results))):
            gpu_text = ' '.join([item[1] for item in gpu_results[i]])
            cpu_text = ' '.join([item[1] for item in cpu_results[i]])
            sim_score = similarity(gpu_text, cpu_text)
            total_similarity += sim_score
            comparisons += 1
            st.write(f"\nFrame {i + 1}:")
            st.write(f"GPU OCR: {gpu_text}")
            st.write(f"CPU OCR: {cpu_text}")
            st.write(f"Similarity: {sim_score:.2f}")

        avg_similarity = total_similarity / comparisons if comparisons > 0 else 0
        st.write(f"\nAverage Similarity Score: {avg_similarity:.2f}")
    else:
        st.write("Could not perform OCR accuracy comparison due to missing results.")

    # Save a sample frame from both the GPU and CPU processed videos
    def save_sample_frame(output_path, image_path, frame_number=0):
        cap = cv2.VideoCapture(output_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(image_path, frame)
            cap.release()
            return Image.open(image_path)
        else:
            cap.release()
            return None

    gpu_frame_img = save_sample_frame('output_gpu.avi', 'gpu_frame.jpg')
    cpu_frame_img = save_sample_frame('output_cpu.avi', 'cpu_frame.jpg')

    if gpu_frame_img:
        st.image(gpu_frame_img, caption='Sample Frame from GPU Processed Video')
    if cpu_frame_img:
        st.image(cpu_frame_img, caption='Sample Frame from CPU Processed Video')
