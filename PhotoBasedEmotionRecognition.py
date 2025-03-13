import cv2
from deepface import DeepFace

# Define emotion mapping
emotion_map = {
    "neutral": "Neutral",
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fear": "Sad",     # Map fear to Sad
    "disgust": "Angry",  # Map disgust to Angry
    "surprise": "Happy"  # Map surprise to Happy
}

# Load image
image_path = "./photo dataset/happy2.jpg"  # Change this to your image path
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("Error: Could not load image. Check the file path.")
else:
    try:
        # Analyze the image for emotions
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        
        # Get detected emotion and map it
        detected_emotion = result[0]['dominant_emotion']
        emotion_label = emotion_map.get(detected_emotion, "Neutral")  # Default to Neutral

        # Print the detected emotion
        print(f"Detected Emotion: {emotion_label}")
    
    except Exception as e:
        print(f"Error: {e}")
