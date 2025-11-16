from importlib.metadata import files
import os
import re
import cv2
import numpy as np
from PIL import Image
import io
import yt_dlp
import streamlit as st
import streamlit as st
from PyPDF2 import PdfMerger
from skimage.metrics import structural_similarity as ssim
from fpdf import FPDF
import tempfile
from dotenv import load_dotenv
load_dotenv() ##load all the environment variables
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from huggingface_hub import InferenceClient
from huggingface_hub import HfApi
client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)

# ---------------------- Helper Functions ----------------------

def get_video_id(url):
    # Match YouTube Shorts URLs
    video_id_match = re.search(r"shorts\/(\w+)", url)
    if video_id_match:
        return video_id_match.group(1)

    # Match youtube.be shortened URLs
    video_id_match = re.search(r"youtu\.be\/([\w\-_]+)(\?.*)?", url)
    if video_id_match:
        return video_id_match.group(1)

    # Match regular YouTube URLs
    video_id_match = re.search(r"v=([\w\-_]+)", url)
    if video_id_match:
        return video_id_match.group(1)

    # Match YouTube live stream URLs
    video_id_match = re.search(r"live\/(\w+)", url)
    if video_id_match:
        return video_id_match.group(1)

    return None

def get_playlist_videos(playlist_url):
    ydl_opts = {
        'ignoreerrors': True,
        'playlistend': 1000,  # Maximum number of videos to fetch
        'extract_flat': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_info = ydl.extract_info(playlist_url, download=False)
        return [entry['url'] for entry in playlist_info['entries']]
    
    
def download_video(url, output_path):
    ydl_opts = {
        "outtmpl": output_path,
        "format": "best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # Add custom headers to avoid 403 errors
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-us,en;q=0.5",
            "Sec-Fetch-Mode": "navigate",
        },
        # Add retries for resilience
        "retries": 5,
        "fragment_retries": 5,
        "ignoreerrors": True,
        # Add rate limiting to avoid blocks
        "ratelimit": 1000000,  # 1M bytes per sec
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def extract_unique_frames(video_path, output_folder, frame_skip = 100, similarity_threshold=1.0):
    print(frame_skip)
    print(similarity_threshold)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    frame_count = 0
    saved_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is None:
                saved_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(saved_path, frame)
                saved_frames.append(saved_path)
                prev_frame = gray
            else:
                score, _ = ssim(prev_frame, gray, full=True)
                if score < similarity_threshold:
                    saved_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(saved_path, frame)
                    saved_frames.append(saved_path)
                    prev_frame = gray

        frame_count += 1

    cap.release()
    return saved_frames

def frames_to_pdf(frames_folder, output_pdf, frame_files):
    pdf = FPDF()
    pdf.set_auto_page_break(0)
    for frame in frame_files:
        pdf.add_page()
        pdf.image(frame, x=10, y=10, w=180)
    pdf.output(output_pdf, "F")

def get_video_title(url):
    ydl_opts = {
        'skip_download': True,
        'ignoreerrors': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(url, download=False)
        title = video_info['title'].replace('/', '-').replace('\\', '-').replace(':', '-').replace('*', '-').replace('?', '-').replace('<', '-').replace('>', '-').replace('|', '-').replace('"', '-').strip('.')
        return title



# ----------------------Summarize Yt Video----------------------
## getting the transcript data from yt videos
def extract_transcript_details(youtube_video_url):
    try:
        video_id=get_video_id(youtube_video_url)
        yyt_api = YouTubeTranscriptApi()
        transcript_text=yyt_api.fetch(video_id).to_raw_data()
        print(transcript_text)
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        print(f"Error extracting transcript: {e}")
        return None

def summarize_text(transcript_text,prompt):
    
    num_iters = int(len(transcript_text) / 1000)
    summarized_text = []
    for i in range(0, num_iters + 1):
        start = 0
        start = i * 1000
        end = (i + 1) * 1000
        out = client.summarization(transcript_text[start:end],model="facebook/bart-large-cnn",)
        print(out)
        out = out["summary_text"]
        summarized_text.append(out)
    return summarized_text
        
def summarize_yt_video():
    prompt="""You are a helpful assistant that converts YouTube video transcripts into detailed notes.
    Your task is to read the provided transcript and generate comprehensive notes that capture the key points, concepts, and ideas presented in the video.
    Use clear headings, bullet points, and concise language to organize the information effectively.
    Ensure that the notes are easy to read and understand, making them suitable for study or reference purposes.
    Focus on accuracy and clarity, avoiding unnecessary jargon or complex language.
    The goal is to create a valuable resource that summarizes the video's content in a way that is accessible
    and informative for the reader.
    """
    st.set_page_config(
    page_title="SnapScribe - Notes Generator",
    page_icon="📝",
    layout="wide"
    )

    # Enhanced CSS with modern student-friendly design
    st.markdown("""
    <style>
        /* Modern gradient background */
        .stApp {
            background: linear-gradient(120deg, #f8f9fa 0%, #e9ecef 100%);
            color: #2d3436;
        }
        
        /* Animated logo container */
        .logo-container {
            text-align: center;
            margin-bottom: 2rem;
            animation: float 6s ease-in-out infinite;
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        /* Modern card design */
        .modern-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.4);
        }
        
        .modern-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        }
        
        /* Animated button */
        .stButton button {
            background: linear-gradient(45deg, #4776E6, #8E54E9) !important;
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 5px 15px rgba(71, 118, 230, 0.2) !important;
        }
        
        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(71, 118, 230, 0.3) !important;
        }
        
        /* Input field styling */
        .stTextInput input {
            border-radius: 12px !important;
            padding: 15px !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;  
            box-shadow: 0 0 0 3px white !important;
            background: white !important;
            color: black !important;
            placeholder-color: black !important;
        }
        
        .stTextInput input:focus { 
            box-shadow: 0 0 0 3px white !important;
            background: white !important;
            color: black !important;
        }
        
        /* Feature pills */
        .feature-pill {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            background: rgba(71, 118, 230, 0.1);
            color: #4776E6;
            margin: 5px;
            font-size: 14px;
        }
        
        /* Download button */
        .stDownloadButton button {
            background: linear-gradient(45deg, #00b09b, #96c93d) !important;
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 12px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Enhanced Header with Animation
    st.markdown("""
        <div class='logo-container'>
            <h1 style='font-size: 3rem; font-weight: 800; background: linear-gradient(45deg, #4776E6, #8E54E9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                📚 SnapScribe
            </h1>
            <p style='font-size: 1.2rem; color: #666; margin-top: 10px;'>
                Transform Youtube Video Learning into Smart Summarized Notes
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Feature Pills
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <div class='feature-pill'>✨ Smart Summarization</div>
            <div class='feature-pill'>📝 Detailed Notes</div>
            <div class='feature-pill'>💾 PDF Export</div>
            <div class='feature-pill'>🎯 Key Points</div>
        </div>
    """, unsafe_allow_html=True)

    # Main Content in Modern Card
    # st.markdown("<div class='modern-card'>", unsafe_allow_html=True)

    # Two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        youtube_link = st.text_input(" Paste YouTube Video Link:", placeholder="https://www.youtube.com/watch?v=...")
        if youtube_link:
            video_id = get_video_id(youtube_link)
            if video_id:
                st.video(f"https://www.youtube.com/watch?v={video_id}")

    with col2:
        st.markdown("""
            <div style='background: rgba(71, 118, 230, 0.05); padding: 20px; border-radius: 12px;'>
                <h3 style='color: #4776E6; font-size: 1.2rem; margin-bottom: 10px;'>✨ How it works</h3>
                <ol style='color: #666; font-size: 0.9rem; margin-left: 20px;'>
                    <li>Paste your YouTube video link</li>
                    <li>Click "Generate Smart Notes"</li>
                    <li>Get instant notes & summary</li>
                    <li>Download as PDF for offline use</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

    # Process Button
    if youtube_link:
        if st.button("🎯 Generate Smart Notes"):
            with st.spinner("🤓 Processing your video..."):
                transcript_text = extract_transcript_details(youtube_link)
                if transcript_text:
                pdf = FPDF("P", 'mm', 'A4')
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_margins(15, 15, 15)
                pdf.cell(0, 10, "Full Transcript", ln=True, align='C')

                
                # Create directory if it doesn't exist
                os.makedirs("output_pdf", exist_ok=True)
                
                # Split text into lines that fit on page
                lines = [transcript_text[i:i+90] for i in range(0, len(transcript_text), 90)]
                
                # Add each line to PDF
                for line in lines:
                    pdf.cell(0, 10, txt=line, ln=True)
                    
                # Save PDF
                pdf.output("output_pdf/output_transcript.pdf")
            if transcript_text:
                # Show progress
                progress_bar = st.progress(0)
                for i in range(100):
                    # Simulate progress
                    progress_bar.progress(i + 1)
                
                summary = summarize_text(transcript_text, prompt)
                pdf = FPDF("P", "mm", "A4")
                pdf.add_page()
                pdf.set_font("Arial", "B", size=8)
                pdf.cell(0, 10, "Detailed Notes", ln=True, align='C')
                pdf.ln(10)
                
                pdf.set_font("Arial", size=12)
                full_text = "\n\n".join(summary)
                pdf.multi_cell(0, 8, full_text)
                pdf.ln(5)

                
                # Create directory if it doesn't exist
                os.makedirs("output_pdf", exist_ok=True)
                
                # Save PDF
                pdf.output("output_pdf/output_summary.pdf")
                # Results in tabs
                tab1, tab2 = st.tabs(["📝 Summary", "📚 Full Transcript"])
                
                with tab1:
                    st.markdown("<div style='background: white; padding: 20px; border-radius: 12px;color :black'>", unsafe_allow_html=True)
                    st.write(summary)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with tab2:
                    st.markdown("<div style='background: white; padding: 20px; border-radius: 12px;color :black'>", unsafe_allow_html=True)
                    st.write(transcript_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Download buttons in columns
                col1, col2 = st.columns(2)
                if summary:
                    with col1:
                        st.download_button(
                            "📥 Download Summary PDF",
                            data=open("output_pdf/output_summary.pdf", "rb").read(),
                            file_name="summary_notes.pdf",
                            mime="application/pdf"
                        )
                        st.download_button(
                            "📥 Download Transcript PDF",
                            data=open("output_pdf/output_transcript.pdf", "rb").read(),
                            file_name="transcript.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.error("❌ Video does not have a transcript. Please check the video link.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div style='text-align: center; margin-top: 2rem; padding: 20px; color: #666;'>
            <p>Made with ❤️ - SnapScribe</p>
            <p style='font-size: 0.8rem;'>Use SnapScribe to enhance your learning experience</p>
        </div>
    """, unsafe_allow_html=True)

       
# ---------------------- Video to Visual Summaries ----------------------

def Upload_video_to_pdf():
    # PAGE CONFIG 
    st.set_page_config(
        page_title="Local Video Processor | SnapScribe",
        page_icon="🎬",
        layout="centered"
    )
    st.title("🎬 Local Video Processor: Generate PDF from Video Frames")

    upload_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    frame_skip = int(st.text_input("Frame Skip Value (Higher value means faster processing but may miss some frames):", value=200))
    similarity_threshold = float(st.text_input("Similarity Threshold (0.0 to 1.0, lower means more frames):", value=0.9))


    if st.button("Process Video") and upload_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload_file.name)[1]) as temp_video_file:
            temp_video_file.write(upload_file.read())
            temp_video_file_path = temp_video_file.name

        st.video(temp_video_file_path)

        with tempfile.TemporaryDirectory() as temp_folder:
            with st.spinner("🤓 Processing your video..."):
                frames = extract_unique_frames(temp_video_file_path, temp_folder, frame_skip, similarity_threshold)
    
                output_pdf_path = os.path.join("output_pdfs", os.path.splitext(upload_file.name)[0])
                output_pdf_path = f"{output_pdf_path}.pdf"
                frames_to_pdf(temp_folder, output_pdf_path, frames)
    
                
                st.balloons()
        st.download_button(
            label="⬇️ Download Generated PDF",
            data=open(output_pdf_path, "rb").read(),
            file_name=os.path.basename(output_pdf_path),
        )
        os.remove(temp_video_file_path)

# ----------------------Pdf Merger----------------------
def merge_pdfs():
    # PAGE CONFIG 
    st.set_page_config(
        page_title="PDF Merger | SnapScribe",
        page_icon="📄",
        layout="centered"
    )

    # CUSTOM STYLES 
    st.markdown("""
        <style>
            body {
                background: linear-gradient(135deg, #3a7bd5, #3a6073);
                color: white;
            }
            .glass-box {
                background: rgba(255, 255, 255, 0.18);
                border-radius: 24px;
                box-shadow: 0 8px 40px rgba(0, 0, 0, 0.25);
                backdrop-filter: blur(12px);
                border: 1.5px solid rgba(255, 255, 255, 0.35);
                padding: 48px 32px;
                text-align: center;
                max-width: 750px;
                margin: 48px auto;
            }
            div.stButton > button:first-child {
                background: linear-gradient(135deg, #4F8BF9, #1C67E3);
                color: white;
                border-radius: 12px;
                height: 3.2em;
                font-weight: 700;
                font-size: 1.1em;
                border: none;
                box-shadow: 0 2px 8px rgba(79,139,249,0.15);
                transition: all 0.3s ease;
            }
            div.stButton > button:first-child:hover {
                background: linear-gradient(135deg, #5AA2FF, #4F8BF9);
                transform: scale(1.07);
                box-shadow: 0 4px 16px rgba(79,139,249,0.22);
            }
            .uploadedFile {
                color: white !important;
            }
            @keyframes fadeInDown {
                from { opacity: 0; transform: translateY(-20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .animated-title {
                animation: fadeInDown 1.2s ease-out;
                font-size: 2.6em;
                font-weight: 800;
                letter-spacing: 1px;
                margin-bottom: 0.2em;
            }
            .subtitle {
                font-size: 1.25em;
                opacity: 0.92;
                margin-bottom: 1.2em;
            }
            .step {
                background: rgba(255,255,255,0.09);
                border-radius: 8px;
                padding: 10px 18px;
                margin: 10px 0;
                font-size: 1.05em;
                color: #e3e3e3;
                box-shadow: 0 1px 4px rgba(79,139,249,0.08);
            }
            .download-btn {
                margin-top: 18px;
            }
        </style>
    """, unsafe_allow_html=True)

    # ---------------------- HEADER ----------------------
    st.markdown("""
        <div class="glass-box" style = "color:black">
            <h1 class="animated-title">📄 PDF Merger</h1>
            <div class="subtitle">Effortlessly combine multiple PDF files into one beautiful document.<br>
            <span style="font-size:16px;opacity:0.9;">Upload, arrange, and merge with a single click.</span></div>
            <div class="step" style="color:black;">1️⃣ <b>Upload</b> your PDF files</div>
            <div class="step" style="color:black;">2️⃣ <b>Arrange</b> them in your preferred order</div>
            <div class="step" style="color:black;">3️⃣ <b>Merge</b> and <b>Download</b> your new PDF</div>
        </div>
    """, unsafe_allow_html=True)

    # ---------------------- MAIN APP ----------------------
    # st.markdown('<div class="glass-box">', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "**📂 Upload your PDF files**",
        type="pdf",
        accept_multiple_files=True,
        help="**Select two or more PDFs to merge.**",
    )

    if uploaded_files and len(uploaded_files) >= 2:
        filenames = [file.name for file in uploaded_files]
        order = st.multiselect(
            "**🧩 Arrange your PDFs in the order you want:**",
            options=filenames,
            default=filenames,
            help="Drag or select to reorder your PDFs before merging.",
        )

        if st.button("**✨ Merge PDFs**"):
            file_map = {file.name: file for file in uploaded_files}
            merger = PdfMerger()
            temp_files = []

            for fname in order:
                pdf_file = file_map[fname]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_file.read())
                    tmp.flush()
                    temp_files.append(tmp.name)
                merger.append(tmp.name)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as out_tmp:
                merger.write(out_tmp.name)
                merger.close()
                out_tmp.seek(0)
                st.success("✅ Your PDFs have been merged successfully!")
                st.download_button(
                    label="⬇️ Download Merged PDF",
                    data=out_tmp.read(),
                    file_name="merged.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download-btn"
                )
                out_tmp.close()
                os.unlink(out_tmp.name)

            # Clean up temporary files
            for temp_file in temp_files:
                os.unlink(temp_file)

    else:
        st.info("Upload at least two PDF files to start merging.", icon="📘")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------- FOOTER ----------------------
    st.markdown("""
        <div style="text-align:center; margin-top:38px; color:black; font-size:15px;">
            Made with ❤️ by <b>SnapScribe</b> • Secure • Fast • Beautiful<br>
            <span style="font-size:13px;opacity:0.8;">Your files are never stored. All processing is done securely.</span>
        </div>
    """, unsafe_allow_html=True)

# ----------------------Image to Pdf -----------------------
def images_to_pdf(images):
    pil_images = []
    for img_file in images:
        img = Image.open(img_file)
        if img.mode != "RGB":
            img = img.convert("RGB")
        pil_images.append(img)
    pdf_bytes = io.BytesIO()
    pil_images[0].save(pdf_bytes, format="PDF", save_all=True, append_images=pil_images[1:], quality=95)
    pdf_bytes.seek(0)
    return pdf_bytes

def image_to_pdf_converter():
    # ---------------------- PAGE CONFIG ----------------------
    st.set_page_config(
        page_title="Image to PDF | SnapScribe",
        page_icon="🖼️",
        layout="centered"
    )

    # ---------------------- CUSTOM STYLES ----------------------
    st.markdown("""
        <style>
            body {
                background: linear-gradient(135deg, #ffecd2, #fcb69f);
                color: white;
            }
            .glass-box {
                background: rgba(255, 255, 255, 0.18);
                border-radius: 24px;
                box-shadow: 0 8px 40px rgba(0, 0, 0, 0.18);
                backdrop-filter: blur(12px);
                border: 1.5px solid rgba(255, 255, 255, 0.35);
                padding: 48px 32px;
                text-align: center;
                max-width: 750px;
                margin: 48px auto;
            }
            div.stButton > button:first-child {
                background: linear-gradient(135deg, #ffb347, #ffcc33);
                color: #222;
                border-radius: 12px;
                height: 3.2em;
                font-weight: 700;
                font-size: 1.1em;
                border: none;
                box-shadow: 0 2px 8px rgba(255,179,71,0.15);
                transition: all 0.3s ease;
            }
            div.stButton > button:first-child:hover {
                background: linear-gradient(135deg, #ffe29f, #ffb347);
                transform: scale(1.07);
                box-shadow: 0 4px 16px rgba(255,179,71,0.22);
            }
            .uploadedFile {
                color: #222 !important;
            }
            @keyframes fadeInDown {
                from { opacity: 0; transform: translateY(-20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .animated-title {
                animation: fadeInDown 1.2s ease-out;
                font-size: 2.6em;
                font-weight: 800;
                letter-spacing: 1px;
                margin-bottom: 0.2em;
            }
            .subtitle {
                font-size: 1.25em;
                opacity: 0.92;
                margin-bottom: 1.2em;
            }
            .step {
                background: rgba(255,255,255,0.09);
                border-radius: 8px;
                padding: 10px 18px;
                margin: 10px 0;
                font-size: 1.05em;
                color: white;
                box-shadow: 0 1px 4px rgba(255,179,71,0.08);
            }
            .download-btn {
                margin-top: 18px;
            }
        </style>
    """, unsafe_allow_html=True)

    # ---------------------- HEADER ----------------------
    st.markdown("""
        <div class="glass-box" style = "color: black">
            <h1 class="animated-title">🖼️ Image to PDF Converter</h1>
            <div class="subtitle">Transform your images into a single, high-quality PDF document.<br>
            <span style="font-size:16px;opacity:0.9;">Upload, arrange, and convert with ease!</span></div>
            <div class="step" style="color:black;">1️⃣ <b>Upload</b> your images (PNG, JPG, JPEG)</div>
            <div class="step" style="color:black;">2️⃣ <b>Arrange</b> them in your preferred order</div>
            <div class="step" style="color:black;">3️⃣ <b>Convert</b> and <b>Download</b> your PDF</div>
        </div>
    """, unsafe_allow_html=True)

    # ---------------------- MAIN APP ----------------------
    # st.markdown('<div class="glass-box">', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "🖼️ **Upload your images**",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Select two or more images to convert into a PDF.",
    )

    if uploaded_files and len(uploaded_files) >= 1:
        filenames = [file.name for file in uploaded_files]
        order = st.multiselect(
            "🔀 **Arrange your images in the order you want:**",
            options=filenames,
            default=filenames,
            help="**Drag or select to reorder your images before converting.**",
        )

        if st.button("✨ Generate PDF"):
            file_map = {file.name: file for file in uploaded_files}
            ordered_files = [file_map[fname] for fname in order]
            pdf_bytes = images_to_pdf(ordered_files)
            st.success("✅ Your images have been converted to PDF successfully!")
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name="images.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download-btn"
            )
    else:
        st.info("**Upload at least one image to start converting.**", icon="🖼️")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------- FOOTER ----------------------
    st.markdown("""
        <div style="text-align:center; margin-top:38px; color:black; font-size:15px;">
            Made with ❤️ by <b>SnapScribe</b> • Secure • Fast • Beautiful<br>
            <span style="font-size:13px;opacity:0.8;color:black;">Your files are never stored. All processing is done securely.</span>
        </div>
    """, unsafe_allow_html=True)
       
# ----------------------Home Page---------------------- 
def home_page():
     # PAGE CONFIG 
    st.set_page_config(
        page_title="SnapScribe",
        page_icon="Snapscribe_logo.png",
        layout="centered"
    )
    # CUSTOM STYLES
    st.markdown("""<div style="text-align:center;">
             <br><br><br><br><br>    
             <h3>📚<span style = "color:lightblue">Snap</span>Scribe:Your Ultimate Learning Companion</h3>
             </div>""", unsafe_allow_html=True)
   
    st.markdown ("""
    <p>
    Welcome to SnapScribe, the all-in-one platform designed to enhance your learning experience for both Students,Educators and Professionals! Whether you're looking to convert videos into summarized notes, extract images from videos, merge PDFs, or convert images to PDFs, SnapScribe has got you covered.
    </p>
    """,unsafe_allow_html=True)
     # ---------------------- FOOTER ----------------------
    st.markdown("""
        <div style="text-align:center; margin-top:38px; color:black; font-size:15px;">
            <br><br><br><br><br><br><br><br>
            Made with ❤️ by <b>SnapScribe</b> • Secure • Fast • Beautiful<br>
            <span style="font-size:13px;opacity:0.8;color:black;">Your files are never stored. All processing is done securely.</span>
        </div>
    """, unsafe_allow_html=True)
# ---------------------- Main Application ----------------------
if __name__ == "__main__":
    if not os.path.exists("output_pdfs"):
        os.makedirs("output_pdfs", exist_ok=True)
    app_mode = st.sidebar.selectbox("Choose the feature you want to use:", 
                            ["Home",
                            "Summarize YouTube Video", 
                             "Video to Visual Summaries (PDF)", 
                             "Merge PDFs", 
                             "Image to PDF Converter"])
    if app_mode == "Home":
        home_page()
    elif app_mode == "Summarize YouTube Video":
        summarize_yt_video()    
    elif app_mode == "Video to Visual Summaries (PDF)":
        Upload_video_to_pdf()   
    elif app_mode == "Merge PDFs":
        merge_pdfs()
    elif app_mode == "Image to PDF Converter":
        image_to_pdf_converter()    
        
    

