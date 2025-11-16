# 📘 SnapScribe – Next-Gen Video to Notes Converter

SnapScribe is an innovative tool that converts YouTube videos, playlists, Shorts, and livestreams into **high-quality PDF notes**.  
It downloads the video, captures meaningful frames at regular intervals, skips duplicates, and compiles everything into a clean PDF named after the video title.  
This makes it easy for students, educators, and professionals to save time, revise faster, and keep study material handy offline.

---

## ✨ Features
- 🎥 Works with **single videos, playlists, Shorts, and livestreams**  
- 🖼 Captures **unique frames only** (avoids duplicates with SSIM filtering)  
- 📑 Saves each video as a **PDF with the video title as filename**  
- 🌐 Simple and interactive **Streamlit web interface**  
- 💾 Download and store PDFs **offline for future use**  

---

## 🛠 Requirements

- Python **3.8 or higher**  
- Virtual environment (recommended)  

### Install dependencies

Create a `requirements.txt` with the following:
- yt-dlp
- opencv-python
- opencv-python-headless
- numpy
- pillow
- streamlit
- scikit-image
- fpdf or fpdf2

🚀 How to Run Locally

1.Clone or download this project:
```
git clone https://github.com/shivanshumaurya11/snapscribe.git
cd snapscribe
```
I suggest download it.

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
