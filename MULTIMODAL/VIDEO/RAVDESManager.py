import os
import subprocess
import cv2
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from dataclasses import dataclass, field

@dataclass
class RAVDESSManager:
    base_path: str
    add_visualization: bool = True
    download_url: str = "https://zenodo.org/api/records/1188976/files-archive"
    output_csv_file: str = "/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/VIDEO/RAVDESS.csv"
    df_videos: pd.DataFrame = field(default=None, init=False)

    def __post_init__(self):
        """
        Initializes paths based on the base directory.
        """
        self.zip_folder = os.path.join(self.base_path, "ziped")
        self.unzip_folder = os.path.join(self.base_path, "unziped")
        self.video_folder = os.path.join(self.base_path, "videos")
        os.makedirs(self.zip_folder, exist_ok=True)
        os.makedirs(self.unzip_folder, exist_ok=True)
        os.makedirs(self.video_folder, exist_ok=True)

    def download_ravdess_dataset(self):
        """
        Downloads the full RAVDESS dataset from Zenodo.
        """
        zip_path = os.path.join(self.zip_folder, "RAVDESS_full.zip")
        print(f"📥 Downloading RAVDESS dataset to: {zip_path}")
        subprocess.run(["wget", "-O", zip_path, self.download_url], check=True)
        return zip_path

    def extract_ravdess_dataset(self, zip_path):
        """
        Extracts the downloaded RAVDESS dataset.
        """
        print(f"📂 Extracting RAVDESS dataset to: {self.unzip_folder}")
        subprocess.run(["unzip", zip_path, "-d", self.unzip_folder], check=True)

    def extract_video_files(self):
        """
        Extracts only the video files from the unzipped dataset.
        """
        print(f"🎬 Extracting video files to: {self.video_folder}")
        for file in os.listdir(self.unzip_folder):
            if file.startswith("Video_") and file.endswith(".zip"):
                zip_file_path = os.path.join(self.unzip_folder, file)
                subprocess.run(["unzip", "-o", zip_file_path, "-d", self.video_folder], check=True)

    def extract_video_metadata(self):
        """
        Extracts metadata from all videos in the video directory.
        """
        emotion_dict = {
            "01": "Neutral", "02": "Calm", "03": "Happy", "04": "Sad",
            "05": "Angry", "06": "Disgust", "07": "Fearful", "08": "Surprised"
        }
        data = []

        for root, _, files in os.walk(self.video_folder):
            for file in files:
                if file.endswith(".mp4"):
                    file_path = os.path.join(root, file)
                    cap = cv2.VideoCapture(file_path)
                    if not cap.isOpened():
                        continue
                    
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()

                    emotion_code = file.split("-")[2] if "-" in file else "00"
                    emotion = emotion_dict.get(emotion_code, "Unknown")

                    data.append({
                        "File": file, "Path": file_path, "Duration (s)": round(duration, 2),
                        "Total Frames": frame_count, "FPS": round(fps, 2),
                        "Height": height, "Width": width, "Emotion": emotion
                    })

        self.df_videos = pd.DataFrame(data)

    def display_video_from_df_index(self, video_index):
        """
        Displays a selected video with its corresponding emotion.
        """
        if self.df_videos is None or video_index < 0 or video_index >= len(self.df_videos):
            print("⚠️ Invalid index. Please choose a valid video index.")
            return

        video_path = self.df_videos.iloc[video_index]['Path']
        width = self.df_videos.iloc[video_index]['Width']
        height = self.df_videos.iloc[video_index]['Height']
        emotion = self.df_videos.iloc[video_index]['Emotion']

        title_widget = widgets.HTML(
            value=f"<h2 style='text-align:center; color:#333;'>🎬 Emotion: <b>{emotion}</b></h2>",
            layout=widgets.Layout(width="100%")
        )

        video_widget = widgets.HTML(
            value=f"<video width='{min(500, width)}' height='{min(400, height)}' controls>"
                  f"<source src='{video_path}' type='video/mp4'>"
                  "Your browser does not support the video tag.</video>",
            layout=widgets.Layout(display="flex", justify_content="center")
        )

        display(widgets.VBox([title_widget, video_widget], layout=widgets.Layout(align_items="center")))

    def display_one_video_per_emotion(self, save_path="ravdess_videos.html", df_videos=None):
        """
        Saves and displays the first occurrence of each emotion as an HTML file containing embedded videos.
        
        Args:
            df_videos (pd.DataFrame): DataFrame containing video metadata.
            save_path (str): Path to save the generated HTML file.
        """

        if df_videos is not None:
            self.df_videos = df_videos

        if self.df_videos is None:
            print("⚠️ No video metadata available. Please run extract_video_metadata() first.")
            return

        emotion_indices = {}

        # Find the first occurrence of each emotion
        for idx, row in self.df_videos.iterrows():
            emotion = row["Emotion"]
            if emotion not in emotion_indices:
                emotion_indices[emotion] = idx

        # Generate HTML content with embedded videos
        html_content = "<html><head><title>RAVDESS Video Display</title></head><body>"
        html_content += "<h1 style='text-align:center;'>RAVDESS Emotion Videos</h1>"

        for emotion, index in emotion_indices.items():
            video_path = df_videos.iloc[index]['Path']
            width = min(500, df_videos.iloc[index]['Width'])
            height = min(400, df_videos.iloc[index]['Height'])

            html_content += f"""
            <div style='text-align:center; margin-bottom:20px;'>
                <h2>🎬 Emotion: <b>{emotion}</b></h2>
                <video width='{width}' height='{height}' controls>
                    <source src='{video_path}' type='video/mp4'>
                    Your browser does not support the video tag.
                </video>
            </div>
            """

        html_content += "</body></html>"

        # Save the HTML file
        with open(save_path, "w") as f:
            f.write(html_content)

        print(f"✅ Video file saved: {save_path}")

        # Display the HTML file inside Jupyter Notebook
        from IPython.display import display, HTML
        display(HTML(filename=save_path))

    def full_pipeline(self):
        """
        Executes the entire pipeline: downloading, extracting, processing metadata, and displaying videos.
        """
        zip_file = self.download_ravdess_dataset()
        self.extract_ravdess_dataset(zip_file)
        self.extract_video_files()
        self.extract_video_metadata()
        self.df_videos.to_csv(self.output_csv_file)
        if self.add_visualization:
            self.display_one_video_per_emotion()