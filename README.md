# 📚 SnapScribe – AI-Powered Productivity & Learning Suite

SnapScribe is an end-to-end AI productivity platform built for **students, educators, and professionals**.  
It combines **video summarization, transcript extraction, PDF generation, image-to-PDF tools, frame extraction, and PDF utilities** into a single streamlined Streamlit application.

---

# 🚀 Features

### 🎥 1. DeepRead 
- Extracts YouTube transcripts automatically  
- Uses **HuggingFace BART model** to generate detailed notes  
- Includes video preview, real-time progress, tabs for summary & transcript  
- Export summary as PDF  

### 🖼️ 2. Video → Visual Summary (Frames to PDF)
- Upload local videos  
- Extract unique frames using **SSIM similarity check**  
- Control frame skip & similarity threshold  
- Generate a clean PDF containing visually distinct frames  

### 📄 3. PDF Merger
- Upload multiple PDFs  
- Rearrange order  
- Merge into a single combined PDF  
- Glassmorphism-style UI  

### 🖼️ 4. Image to PDF Converter
- Upload multiple images  
- Reorder images  
- Generate a high-quality multi-page PDF  
- Supports JPG / PNG / JPEG  

---

# 🛠️ Tech Stack

### **Frontend / UI**
- **Streamlit** (Primary UI framework)
- Custom CSS animations & styling

### **Backend Processing**
- **yt-dlp** – video & playlist extraction  
- **YouTubeTranscriptApi** – transcript extraction  
- **OpenCV + SSIM** – frame extraction  
- **Pillow** – image processing  
- **FPDF** – PDF generation  
- **PyPDF2** – merging PDFs  
- **Transformers + HuggingFace** – summarization model  

---

# 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/snapscribe.git
cd snapscribe
I suggest download it.
```

2.Install dependencies:
```
pip install -r requirements.txt

```
3. Add your HuggingFace Token:
```
HF_TOKEN=your_hf_api_key_here in the app.py file

```
4.Run the Streamlit app:
```
streamlit run main.py
```

5.Open the local link in your browser (usually ```http://localhost:8501/```).

📁 Project Structure
```
Snapscribe/
│── Snapscribe.py
│── requirements.txt
│── .env
│── output_pdfs/ (auto-generated)
│── README.md

```
🧠 How It Works 

1.🎬 YouTube → Notes

  Extract transcript

  Chunk transcript into 1000-char segments

  Summarize using facebook/bart-large-cnn

  Display in UI + provide PDF download

2.🎥 Video → Visual Summary

  Read video using OpenCV

  Every frame_skip frames → check similarity using SSIM

  Save unique frames → convert to PDF

3.📝 PDF Tools

  PyPDF2 merges documents

  PIL creates PDF from images with compression

📖 Usage

Paste a YouTube video or playlist URL into the input box.

Adjust frame sampling options:

Interval (seconds) → how often frames are captured.

Duplicate filtering → avoids saving nearly identical frames.

Click Generate Notes.

Download the generated PDF notes directly

✅ Example URLs

Single Video:
https://www.youtube.com/watch?v=VIDEO_ID

Playlist:
https://www.youtube.com/playlist?list=PLAYLIST_ID

Shorts:
https://www.youtube.com/shorts/VIDEO_ID

Livestream:
https://www.youtube.com/live/VIDEO_ID

⚠️ Notes

Private or geo-restricted videos may not download.

Large playlists may take more processing time.

For best results, use clean YouTube links (without extra tracking parameters).

📜 License

This project is for educational purposes.
You are free to use, modify, and extend it for personal or academic projects.

# 📘 SnapScribe – Next-Gen Video to Notes Converter  

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)  
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)  

> 📝 SnapScribe: Turn YouTube videos, playlists, Shorts, and livestreams into **high-quality PDF notes** with just one click. Perfect for students, educators, and lifelong learners.  
