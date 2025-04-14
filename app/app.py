import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import os
import tempfile

def transcribe_audio(audio_file, language='en-US'):
    """
    Transcribe audio file to text using Google's speech recognition.
    
    Parameters:
    audio_file (str): Path to the audio file
    language (str): Language code for recognition (default: en-US)
    
    Returns:
    str: Transcribed text
    """
    # Create a recognizer instance
    recognizer = sr.Recognizer()
    
    # Convert to WAV if not already in that format
    file_extension = os.path.splitext(audio_file)[1].lower()
    
    if file_extension != '.wav':
        try:
            audio = AudioSegment.from_file(audio_file)
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            audio.export(temp_wav.name, format='wav')
            audio_file = temp_wav.name
        except Exception as e:
            return f"Error converting audio: {str(e)}"
    
    # Load the audio file
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        
        # Use Google's speech recognition
        text = recognizer.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from speech recognition service: {str(e)}"
    except Exception as e:
        return f"Error during transcription: {str(e)}"
    finally:
        # Clean up temporary file if created
        if file_extension != '.wav' and 'temp_wav' in locals():
            os.unlink(temp_wav.name)

def main():
    st.set_page_config(page_title="Audio to Text Converter", layout="wide")
    
    st.title("🎙️ Audio to Text Converter")
    st.write("Upload an audio file and convert it to text!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", 
                                     type=["wav", "mp3", "ogg", "flac", "m4a"])
    
    # Language selection
    languages = {
        "English (US)": "en-US",
        "English (UK)": "en-GB",
        "Spanish": "es-ES",
        "French": "fr-FR",
        "German": "de-DE",
        "Italian": "it-IT",
        "Japanese": "ja-JP",
        "Korean": "ko-KR",
        "Chinese (Mandarin)": "zh-CN",
        "Russian": "ru-RU",
        "Hindi": "hi-IN"
    }
    
    selected_language = st.selectbox(
        "Select language",
        options=list(languages.keys()),
        index=0
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        # Add a button to trigger transcription
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing..."):
                try:
                    # Call the transcribe function
                    transcription = transcribe_audio(
                        temp_file_path, 
                        language=languages[selected_language]
                    )
                    
                    # Display the transcription
                    st.subheader("Transcription Result:")
                    st.write(transcription)
                    
                    # Add a download button for the transcription
                    st.download_button(
                        "Download Transcription",
                        transcription,
                        file_name="transcription.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    # Clean up the temporary file
                    os.unlink(temp_file_path)
    
    # Add information about requirements
    with st.expander("Requirements and Setup Information"):
        st.markdown("""
        ### Required Libraries
        To run this application, you need to install the following packages:
        ```
        pip install streamlit speech_recognition pydub
        ```
        
        ### Additional System Requirements
        - For MP3 support, you need to install FFmpeg on your system
        - For audio recognition to work properly, make sure your microphone is set up correctly
        """)

if __name__ == "__main__":
    main()