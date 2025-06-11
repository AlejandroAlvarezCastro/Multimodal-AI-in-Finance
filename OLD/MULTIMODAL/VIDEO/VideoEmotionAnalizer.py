import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from abc import ABC, abstractmethod
from facenet_pytorch import MTCNN
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig
# from moviepy.video.io.VideoFileClip import VideoFileClip
from fer import FER
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from dataclasses import dataclass, field
from typing import List



@dataclass
class FaceDetector:
    """
    Detects and crops faces from input images or video frames using MTCNN.

    Attributes:
        device (str): Computation device ("cuda" or "cpu").
        mtcnn (MTCNN): Pretrained face detection model.
    """
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn: MTCNN = field(default_factory=lambda: MTCNN(keep_all=False, post_process=True, device='cuda' if torch.cuda.is_available() else 'cpu'))

    def detect_faces(self, image: Image.Image):
        """
        Detects a single face in the given PIL image.

        Args:
            image (Image.Image): Input PIL image.

        Returns:
            Image.Image or None: Cropped face image or None if no face is found.
        """
        sample = self.mtcnn.detect(image)
        if sample[0] is not None:
            box = sample[0][0]
            face = image.crop(box)
            return face
        return None

    def recognize_faces(self, frame: np.ndarray) -> List[np.array]:
        """
        Detects multiple faces in a video frame and returns cropped facial regions.

        Args:
            frame (np.ndarray): Input frame (BGR or RGB format).

        Returns:
            List[np.ndarray]: List of cropped face arrays.
        """
        def detect_face(frame: np.ndarray):
            bounding_boxes, probs = self.mtcnn.detect(frame, landmarks=False)
            if probs is None or probs[0] == None:
                return []
            bounding_boxes = bounding_boxes[probs > 0.9]
            return bounding_boxes

        bounding_boxes = detect_face(frame)
        facial_images = []
        for bbox in bounding_boxes:
            box = bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            facial_images.append(frame[y1:y2, x1:x2, :])
        return facial_images


@dataclass
class EmotionRecognizer(ABC):
    """
    Abstract base class for facial emotion recognition models.

    Attributes:
        device (str): Computation device ("cuda" or "cpu").
    """
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    @abstractmethod
    def predict_emotion(self, face: Image.Image):
        """
        Predict emotion scores for a given facial image.

        Args:
            face (Image.Image): Input facial image.

        Returns:
            dict or None: Emotion probabilities or None if detection failed.
        """
        pass



@dataclass
class VITRecognizer(EmotionRecognizer):
    """
    Emotion recognizer using a ViT-based face expression classification model.

    Attributes:
        extractor (AutoFeatureExtractor): Feature extractor for the model.
        model (AutoModelForImageClassification): Loaded ViT model.
        config (AutoConfig): Model configuration with label mapping.
        id2label (dict): Mapping from class indices to emotion labels.
    """
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor: AutoFeatureExtractor = field(default_factory=lambda: AutoFeatureExtractor.from_pretrained("trpakov/vit-face-expression"))
    model: AutoModelForImageClassification = field(default_factory=lambda: AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression").to('cuda' if torch.cuda.is_available() else 'cpu'))
    config: AutoConfig = field(default_factory=lambda: AutoConfig.from_pretrained("trpakov/vit-face-expression"))
    # id2label: dict = field(init=False)

    def __post_init__(self):
        self.id2label = self.config.id2label

    def predict_emotion(self, face: Image.Image):
        """
        Predicts emotion scores from a facial image using ViT.

        Args:
            face (Image.Image): Cropped face image.

        Returns:
            dict: Emotion label -> probability.
        """
        inputs = self.extractor(images=face, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probabilities = probabilities.detach().cpu().numpy().tolist()[0]
        class_probabilities = {self.id2label[i]: prob for i, prob in enumerate(probabilities)}
        return class_probabilities

@dataclass
class FERRecognizer(EmotionRecognizer):
    """
    Emotion recognizer using the FER library and OpenCV images.

    Attributes:
        fer_detector (FER): FER emotion detection model.
    """
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fer_detector: FER = field(default_factory=lambda: FER(mtcnn=True))
    
    def predict_emotion(self, face: Image.Image):
        img_cv = np.array(face)[:, :, ::-1]  # Convert PIL to OpenCV
        detections = self.fer_detector.detect_emotions(img_cv)
        return detections[0]['emotions'] if detections else None

@dataclass
class EmotiEffRecognizer(EmotionRecognizer):
    """
    Emotion recognizer using the EmotiEff model.

    Attributes:
        model (str): EmotiEff model name.
        emotion_mapping (dict): Mapping from raw to standard emotion labels.
        emotion_labels (list): Ordered list of possible emotion labels.
    """
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model: str = "enet_b0_8_best_afew"
    emotion_mapping: dict = field(default_factory=lambda: {"anger": "angry", "happiness": "happy", "sadness": "sad", "fear": "fear", "surprise": "surprise", "disgust": "disgust", "neutral": "neutral"})
    emotion_labels: list = field(default_factory=lambda: ["angry", "happy", "sad", "fear", "surprise", "disgust", "neutral"])

    def __post_init__(self):
        self.recognizer = EmotiEffLibRecognizer(engine="torch", model_name=self.model, device=self.device)

    def map_emotion(self, emotion: str) -> str:
        """
        Maps raw EmotiEff emotion label to a standardized label.

        Args:
            emotion (str): Original emotion name.

        Returns:
            str: Mapped emotion label.
        """
        return self.emotion_mapping.get(emotion, emotion)

    def predict_emotion(self, facial_images):
        if not facial_images:
            return {emotion: 0.0 for emotion in self.emotion_labels}
        
        _, scores = self.recognizer.predict_emotions(facial_images[0])
        softmax_scores = torch.nn.functional.softmax(torch.tensor(scores), dim=1)
        emotion_probs = {self.emotion_labels[i]: float(softmax_scores[0, i]) for i in range(len(self.emotion_labels))}
    
        return emotion_probs


@dataclass
class VideoProcessor:
    """
    Handles video processing by extracting and reducing frames from a video file.

    Attributes:
        skips (float): Proportion of frames to keep from the total video (range: 0 to 1).
    """
    skips: float = 0.1
    
    def extract_frames(self, video_path: str):
        """
        Loads a video file, removes audio, and extracts its frames.

        Args:
            video_path (str): Path to the video file.

        Returns:
            list: A reduced list of frames based on the 'skips' parameter.
        """
        # clip = VideoFileClip(video_path)
        # video = clip.without_audio()
        # video_data = np.array(list(video.iter_frames()))
        # return self.reduce_video_frames(video_data)

    def reduce_video_frames(self, video_data: list):
        """
        Reduces the number of frames in a video according to the 'skips' proportion.

        Args:
            video_data (list): A list of all frames extracted from the video.

        Returns:
            list: A subset of the original frames sampled uniformly.

        Raises:
            ValueError: If 'skips' is not between 0 and 1 or leads to no selected frames.
        """
        if not (0 < self.skips <= 1):
            raise ValueError("The 'skips' value must be a proportion between 0 and 1.")
        
        total_frames = len(video_data)
        num_selected_frames = int(total_frames * self.skips)
        
        if num_selected_frames < 1:
            raise ValueError("The selected proportion is too small, leading to zero frames.")
        
        return [video_data[i] for i in np.linspace(0, total_frames - 1, num_selected_frames, dtype=int)]

@dataclass
class EmotionVideoAnalyzer:
    """
    Analyzes a video by detecting faces and recognizing emotions frame by frame.

    Attributes:
        recognizer (EmotionRecognizer): An instance of a facial emotion recognizer.
        face_detector (FaceDetector): Responsible for face detection in frames.
        processor (VideoProcessor): Handles frame extraction and reduction.
    """
    recognizer: EmotionRecognizer
    face_detector: FaceDetector
    processor: VideoProcessor
    
    def analyze_video(self, video_path: str):
        """
        Processes a video to predict emotions in the detected faces of selected frames.

        Args:
            video_path (str): Path to the video file.

        Returns:
            pd.DataFrame: A DataFrame containing emotion predictions for each frame.
        """
        frames = self.processor.extract_frames(video_path)
        predictions = []

        # For EmotiEff, multiple faces may be returned
        if isinstance(self.recognizer, EmotiEffRecognizer):
            for frame in frames:
                facial_images = self.face_detector.recognize_faces(frame)
                emotion = self.recognizer.predict_emotion(facial_images)
                if emotion is not None:
                    predictions.append(emotion)
            return pd.DataFrame(predictions)
        
        # For FER and VIT, a single cropped face is used
        if isinstance(self.recognizer, FERRecognizer) or isinstance(self.recognizer, VITRecognizer):
            for frame in frames:
                face = self.face_detector.detect_faces(Image.fromarray(frame))
                if face:
                    emotion = self.recognizer.predict_emotion(face)
                    if emotion is not None:
                        predictions.append(emotion)
            return pd.DataFrame(predictions)
        


@dataclass
class VideoEmotionAnalysis:
    """
    Orchestrates the complete video emotion analysis pipeline, from face detection to final classification.

    Attributes:
        mode (str): Emotion recognition model to use ('vit', 'fer', or 'emotieff').
        skips (float): Proportion of frames to process.
        method (str): Aggregation method for final prediction ('mode', 'mean', or 'abs').
        device (str): Device to run models on ('cuda' or 'cpu').
        emotieff_model (str): Model name to use for EmotiEffRecognizer.
    """
    mode: str
    skips: float = 0.1
    method: str = "mode"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    emotieff_model: str = "enet_b0_8_best_afew"
    
    def __post_init__(self):
        """
        Initializes internal components based on the selected mode.
        """
        self.face_detector = FaceDetector(self.device)
        self.video_processor = VideoProcessor(self.skips)
        
        match self.mode:
            case "vit":
                self.recognizer = VITRecognizer(self.device)
            case "fer":
                self.recognizer = FERRecognizer(self.device)
            case "emotieff":
                self.recognizer = EmotiEffRecognizer(self.device, model=self.emotieff_model)
            case _:
                raise ValueError("Unsupported mode. Choose 'vit', 'fer', or 'emotieff'.")
        
        self.analyzer = EmotionVideoAnalyzer(self.recognizer, self.face_detector, self.video_processor)
    
    def analyze_video(self, video_path: str) -> str:
        """
        Performs emotion recognition on a video and returns the dominant emotion.

        Args:
            video_path (str): Path to the input video.

        Returns:
            str: The predicted emotion label.
        """
        df = self.analyzer.analyze_video(video_path)
        if self.mode != "emotieff":
            pred = self.change_disgust_fear(self.get_prediction(df)) if not df.empty else 'unknown'
        else:
            pred = self.get_prediction(df) if not df.empty else 'unknown'
        print(f"Predicted Emotion: {pred}")
        return pred 
    
    def get_prediction(self, df: pd.DataFrame):
        """
        Aggregates frame-level emotion predictions to obtain a single label.

        Args:
            df (pd.DataFrame): DataFrame with emotion probabilities per frame.

        Returns:
            str: The final aggregated emotion label.
        """
        match self.method:
            case "mode":
                return df.idxmax(axis=1).mode()[0] 
            case "mean":
                return df.mean().idxmax()
            case "abs":
                return df.max().idxmax() 
            case _:
                raise ValueError(f"Unknown method: {self.method}")
            
    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies video emotion classification to a DataFrame of video paths.

        Args:
            df (pd.DataFrame): A DataFrame with a 'Path' column containing video file paths.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'classification' column.
        """
        if "Path" not in df.columns:
            raise ValueError("DataFrame must contain a 'Path' column with video file paths.")
        
        df["classification"] = df["Path"].apply(self.analyze_video)
        return df
    
    def change_disgust_fear(self, prediction: str) -> str:
        """
        Swaps 'disgust' and 'fear' labels, useful for post-hoc corrections.

        Args:
            prediction (str): The predicted emotion label.

        Returns:
            str: Possibly corrected emotion label.
        """
        if self.mode in ["vit", "fer", "emotieff"]:
            if prediction == "disgust":
                return "fear"
            elif prediction == "fear":
                return "disgust"
        return prediction
    
    def get_embeddings(self, video):
        pass