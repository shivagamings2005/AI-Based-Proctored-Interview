from flask import Flask, request, jsonify 
from flask_cors import CORS
import cv2
import numpy as np
import librosa
import webrtcvad
import torch
import os
import base64
from werkzeug.utils import secure_filename
import math
import time
import logging
from uuid import uuid4
import asyncio
import websockets
import json
import io
import soundfile as sf
import noisereduce as nr
import nemo.collections.asr.models as asr_models
from pydub import AudioSegment
import threading
import gc
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from datetime import datetime
from dotenv import load_dotenv
import pymongo
import google.generativeai as genai
from l2cs import Pipeline
from tzlocal import get_localzone
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
load_dotenv()

# Access environment variables
mongo_uri = os.getenv("MONGO_URI")
google_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Connect to MongoDB
mongo_client = pymongo.MongoClient(mongo_uri)
db = mongo_client['proctoring']
proctoring_collection = db['proctoring']

# Configure Google Gemini
genai.configure(api_key=google_api_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Configure Groq
groq_client = Groq(api_key=groq_api_key)

class MalpracticeDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        try:
            self.gaze_pipeline = Pipeline(
                weights='L2CSNet_gaze360.pkl',
                arch='ResNet50',
                device=self.device
            )
            logging.info("Gaze pipeline initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize gaze pipeline: {str(e)}")
            raise e
        self.yaw_correction_factor = 0.9
        self.pitch_correction_factor = 0.8
        self.sessions = {}  # {session_id: {noFaceStartTime, multipleFacesTime, notLookingTime, warningCount}}
        self.session_lock = threading.Lock()  # Thread-safe access to sessions
        self.websocket_sessions = {}  # Track WebSocket connections by session ID
        self.pdf_evidence = {}  # {session_id: {'doc': doc, 'story': [], 'evidence_folder': path}}
        self.pdf_lock = threading.Lock() 
        self.session_privileges = {} 
        
        # Initialize diarization model
        print("Loading diarization model...")
        try:
            self.diar_model = asr_models.SortformerEncLabelModel.restore_from(
                restore_path="/home/shiva_subhan_s/.cache/huggingface/hub/models--nvidia--diar_sortformer_4spk-v1/snapshots/4cb5954e59a1a6527e6ec061a0568b61efa8babd/diar_sortformer_4spk-v1.nemo",
                map_location='cuda',
                strict=False
            )
            self.diar_model.eval()
            print("Diarization model loaded successfully!")
        except Exception as e:
            logging.error(f"Failed to initialize diarization model: {str(e)}")
            raise e
        

    def initialize_session(self, session_id):
        """Initialize session state with thread safety"""
        with self.session_lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    'noFaceStartTime': None,
                    'multipleFacesTime': None,
                    'notLookingTime': None,
                    'warningCount': 0,
                    'terminate': False,
                    'terminateReason': '',
                    'last_activity': time.time(),
                    'created_at': time.time()
                }
                logging.info(f"Initialized new session: {session_id}")

    def format_timestamp_with_timezone(self, dt):
        """Format a datetime object with local timezone abbreviation."""
        local_tz = get_localzone()  # Get the local timezone
        local_dt = dt.astimezone(local_tz)  # Convert to local timezone
        return local_dt.strftime('%B %d, %Y at %H:%M:%S %Z')  # Include %Z for timezone abbreviation

    def cleanup_session(self, session_id):
        """Clean up session data and finalize PDF evidence"""
        # Finalize PDF before cleanup
        pdf_path = self.finalize_pdf_evidence(session_id)
        if session_id in self.pdf_evidence:
            evidence_folder = self.pdf_evidence[session_id]['evidence_folder']
            for filename in os.listdir(evidence_folder):
                if filename.endswith('.jpg'):
                    file_path = os.path.join(evidence_folder, filename)
                    try:
                        os.remove(file_path)
                        logging.info(f"Removed evidence image: {file_path}")
                    except Exception as e:
                        logging.error(f"Failed to remove image {file_path}: {str(e)}")
        with self.session_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
            if session_id in self.session_privileges:
                del self.session_privileges[session_id]
                logging.info(f"Cleaned up session: {session_id}")
                
            # Also remove from websocket sessions if exists
            if session_id in self.websocket_sessions:
                del self.websocket_sessions[session_id]
                logging.info(f"Cleaned up WebSocket session: {session_id}")
        
        # Clean up PDF evidence data
        with self.pdf_lock:
            if session_id in self.pdf_evidence:
                del self.pdf_evidence[session_id]
                logging.info(f"Cleaned up PDF evidence for session: {session_id}")

        
                
        # Force garbage collection to free memory
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.debug("Cleared CUDA cache")
        
        return pdf_path

    def update_session_activity(self, session_id):
        """Update last activity timestamp for session"""
        with self.session_lock:
            if session_id in self.sessions:
                self.sessions[session_id]['last_activity'] = time.time()

    def cleanup_inactive_sessions(self, max_inactive_time=300):  # 5 minutes
        """Clean up sessions that have been inactive for too long"""
        current_time = time.time()
        inactive_sessions = []
        
        with self.session_lock:
            for session_id, session_data in self.sessions.items():
                if current_time - session_data.get('last_activity', 0) > max_inactive_time:
                    inactive_sessions.append(session_id)
        
        # Clean up inactive sessions
        for session_id in inactive_sessions:
            logging.info(f"Cleaning up inactive session: {session_id}")
            self.cleanup_session(session_id)

    def get_session_count(self):
        """Get current number of active sessions"""
        with self.session_lock:
            return len(self.sessions)

    def register_websocket_session(self, session_id, websocket):
        """Register a WebSocket connection for a session"""
        self.websocket_sessions[session_id] = websocket
        logging.info(f"Registered WebSocket for session: {session_id}")

    def unregister_websocket_session(self, session_id):
        """Unregister a WebSocket connection and cleanup session"""
        if session_id in self.websocket_sessions:
            del self.websocket_sessions[session_id]
            logging.info(f"Unregistered WebSocket for session: {session_id}")
        
        # Cleanup the session data as well
        self.cleanup_session(session_id)

    def analyze_gaze_from_frame(self, frame, session_id,exam_name):
        """Analyze gaze direction from a single frame and update session state"""
        privileges = self.session_privileges.get(session_id, {})
        if privileges.get('screenGaze', 1) == 0:
            return {
                'success': True,
                'multiple_faces': False,
                'no_face': False,
                'not_looking_at_screen': False,
                'yaw': 0,
                'pitch': 0,
                'warning': False
            }
        try:
            self.initialize_session(session_id)
            self.update_session_activity(session_id)
            
            with self.session_lock:
                session = self.sessions[session_id]
            
            logging.debug(f"Analyzing frame for session {session_id}")

            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            enhanced_rgb = cv2.GaussianBlur(enhanced_rgb, (3, 3), 0)
            
            # Process with gaze pipeline
            try:
                results = self.gaze_pipeline.step(enhanced_rgb)
            except ValueError as e:
                if "need at least one array to stack" in str(e):
                    logging.info(f"No faces detected in frame for session {session_id}: {str(e)}")
                    if session['noFaceStartTime'] is None:
                        session['noFaceStartTime'] = time.time()
                        logging.info(f"No face detected, starting timer for session {session_id}")
                    elif time.time() - session['noFaceStartTime'] >= 10:
                        session['terminate'] = True
                        session['terminateReason'] = 'No person detected for 10 seconds'
                        logging.warning(f"Terminating session {session_id}: No face for 10 seconds")
                    return {
                        'status': 'no_face',
                        'message': 'No face detected',
                        'warningCount': session['warningCount'],
                        'noFaceDuration': time.time() - session['noFaceStartTime'] if session['noFaceStartTime'] else 0,
                        'terminate': session['terminate'],
                        'terminateReason': session['terminateReason']
                    }
                else:
                    raise e

            # Extract faces
            num_faces = len(results.bboxes) if hasattr(results, 'bboxes') else 0
            logging.debug(f"Detected {num_faces} faces")

            if num_faces == 0:
                if session['noFaceStartTime'] is None:
                    session['noFaceStartTime'] = time.time()
                    logging.info(f"No face detected, starting timer for session {session_id}")
                elif time.time() - session['noFaceStartTime'] >= 10:
                    session['terminate'] = True
                    session['terminateReason'] = 'No person detected for 10 seconds'
                    logging.warning(f"Terminating session {session_id}: No face for 10 seconds")
                return {
                    'status': 'no_face',
                    'message': 'No face detected',
                    'warningCount': session['warningCount'],
                    'noFaceDuration': time.time() - session['noFaceStartTime'] if session['noFaceStartTime'] else 0,
                    'terminate': session['terminate'],
                    'terminateReason': session['terminateReason']
                }
            else:
                if session['noFaceStartTime'] is not None:
                    logging.info(f"Face detected, resetting no face timer for session {session_id}")
                session['noFaceStartTime'] = None

            if num_faces > 1:
                if session['multipleFacesTime'] is None:
                    session['multipleFacesTime'] = time.time()
                    logging.info(f"Multiple faces detected, starting timer for session {session_id}")
                elif time.time() - session['multipleFacesTime'] >= 3:
                    if session['warningCount'] == 0:
                        self.initialize_pdf_evidence(session_id,exam_name)
                    session['warningCount'] += 1
                    logging.warning(f"Issuing warning {session['warningCount']}/5 for session {session_id}")
                    session['multipleFacesTime'] = None
                    self.add_frame_evidence(session_id, frame, "Multiple Faces", 
                                          f"Warning {session['warningCount']}/5: Multiple faces detected",
                                          {'faces_detected': num_faces})
                    if session['warningCount'] >= 5:
                        session['terminate'] = True
                        session['terminateReason'] = 'Too many warnings for multiple faces detected'
                        logging.warning(f"Terminating session {session_id}: Too many warnings")
                    return {
                        'status': 'multiple_faces',
                        'message': f"Warning {session['warningCount']}/5: Multiple faces detected",
                        'warningCount': session['warningCount'],
                        'noFaceDuration': 0,
                        'terminate': session['terminate'],
                        'terminateReason': session['terminateReason']
                    }
                return {
                    'status': 'multiple_faces',
                    'message': 'Multiple faces detected',
                    'warningCount': session['warningCount'],
                    'noFaceDuration': 0,
                    'terminate': session['terminate'],
                    'terminateReason': session['terminateReason']
                }
            else:
                session['multipleFacesTime'] = None
            
            # Extract gaze data
            faces_data = self._extract_faces_data(results)
            if faces_data:
                yaw, pitch, success = self._extract_gaze_data(faces_data[0])
                if success:
                    yaw *= self.yaw_correction_factor
                    pitch *= self.pitch_correction_factor
                    
                    # Analyze gaze direction
                    yaw_deg = math.degrees(yaw) if abs(yaw) <= math.pi else yaw
                    pitch_deg = math.degrees(pitch) if abs(pitch) <= math.pi else pitch
                    
                    looking_outside = (abs(yaw_deg) > 18 or abs(pitch_deg) > 18)
                    
                    if looking_outside:
                        if session['notLookingTime'] is None:
                            session['notLookingTime'] = time.time()
                            logging.info(f"Not looking at screen, starting timer for session {session_id}")
                        elif time.time() - session['notLookingTime'] >= 3:
                            if session['warningCount'] == 0:
                                self.initialize_pdf_evidence(session_id,exam_name)
                            session['warningCount'] += 1
                            logging.warning(f"Issuing warning {session['warningCount']}/5 for session {session_id}")
                            session['notLookingTime'] = None  # Reset after warning
                            self.add_frame_evidence(session_id, frame, "Not Looking at Screen", 
                                                  f"Warning {session['warningCount']}/5: Not looking at screen",
                                                  {'yaw_degrees': yaw_deg, 'pitch_degrees': pitch_deg})
                            if session['warningCount'] >= 5:
                                session['terminate'] = True
                                session['terminateReason'] = 'Too many warnings for not looking at screen'
                                logging.warning(f"Terminating session {session_id}: Too many warnings")
                            else:
                                return {
                                    'status': 'not_looking',
                                    'message': f"Warning {session['warningCount']}/5: Not looking at screen",
                                    'warningCount': session['warningCount'],
                                    'noFaceDuration': 0,
                                    'terminate': session['terminate'],
                                    'terminateReason': session['terminateReason']
                                }
                        return {
                            'status': 'not_looking',
                            'message': 'Not looking at screen',
                            'warningCount': session['warningCount'],
                            'noFaceDuration': 0,
                            'terminate': session['terminate'],
                            'terminateReason': session['terminateReason']
                        }
                    else:
                        session['notLookingTime'] = None
                        return {
                            'status': 'looking',
                            'message': 'Looking at screen',
                            'warningCount': session['warningCount'],
                            'noFaceDuration': 0,
                            'terminate': session['terminate'],
                            'terminateReason': session['terminateReason']
                        }
            
            logging.info(f"No face data extracted for session {session_id}")
            if session['noFaceStartTime'] is None:
                session['noFaceStartTime'] = time.time()
                logging.info(f"No face detected, starting timer for session {session_id}")
            elif time.time() - session['noFaceStartTime'] >= 10:
                session['terminate'] = True
                session['terminateReason'] = 'No person detected for 10 seconds'
                logging.warning(f"Terminating session {session_id}: No face for 10 seconds")
            return {
                'status': 'no_face',
                'message': 'No face detected',
                'warningCount': session['warningCount'],
                'noFaceDuration': time.time() - session['noFaceStartTime'] if session['noFaceStartTime'] else 0,
                'terminate': session['terminate'],
                'terminateReason': session['terminateReason']
            }
            
        except Exception as e:
            logging.error(f"Gaze analysis error for session {session_id}: {str(e)}")
            session = self.sessions.get(session_id, {})
            return {
                'status': 'error',
                'message': f"Gaze analysis error: {str(e)}",
                'warningCount': session.get('warningCount', 0),
                'noFaceDuration': time.time() - session.get('noFaceStartTime', time.time()) if session.get('noFaceStartTime') else 0,
                'terminate': session.get('terminate', False),
                'terminateReason': session.get('terminateReason', '')
            }
    
    def _extract_faces_data(self, results):
        try:
            if results is None:
                return []
            if hasattr(results, 'faces') and results.faces is not None:
                return results.faces
            elif hasattr(results, 'pitch') and hasattr(results, 'yaw'):
                return [results]
            return []
        except Exception as e:
            logging.error(f"Error extracting faces data: {str(e)}")
            return []
    
    def _extract_gaze_data(self, face_result):
        try:
            if hasattr(face_result, 'yaw') and hasattr(face_result, 'pitch'):
                yaw = self._convert_to_float(face_result.yaw)
                pitch = self._convert_to_float(face_result.pitch)
                return yaw, pitch, True
            elif isinstance(face_result, dict):
                yaw = self._convert_to_float(face_result.get('yaw', 0))
                pitch = self._convert_to_float(face_result.get('pitch', 0))
                return yaw, pitch, True
            elif hasattr(face_result, 'gaze') and len(face_result.gaze) >= 2:
                yaw = self._convert_to_float(face_result.gaze[0])
                pitch = self._convert_to_float(face_result.gaze[1])
                return yaw, pitch, True
        except (AttributeError, TypeError, ValueError, IndexError) as e:
            logging.error(f"Error extracting gaze data: {str(e)}")
        return 0, 0, False
    
    def _convert_to_float(self, value):
        try:
            if isinstance(value, np.ndarray):
                return float(value.item() if value.size == 1 else value.flatten()[0])
            elif hasattr(value, 'item'):
                return float(value.item())
            elif hasattr(value, '__len__') and len(value) > 0:
                return float(value[0])
            else:
                return float(value)
        except Exception as e:
            logging.error(f"Error converting to float: {str(e)}")
            return 0.0
    
    def analyze_audio(self, audio_file_path, session_id):
        privileges = self.session_privileges.get(session_id, {})
        if privileges.get('multiSpeaker', 1) == 0:
            # Skip multi-speaker check if multiSpeaker is 0
            return {
                'success': True,
                'multiple_speakers': False,
                'warning': False
            }
        """Analyze audio for multiple speakers and background noise using librosa and VAD"""
        try:
            logging.debug(f"Analyzing audio file: {audio_file_path}")
            audio, sr = librosa.load(audio_file_path, mono=True, sr=16000)
            
            vad = webrtcvad.Vad(3)
            
            frames = self._frame_audio(audio, frame_duration_ms=30, sample_rate=sr)
            
            speech_frames = []
            non_speech_frames = []
            
            for frame in frames:
                if vad.is_speech(frame, sr):
                    speech_frames.append(frame)
                else:
                    non_speech_frames.append(frame)
            
            speech_segments = len(speech_frames)
            
            background_noise_detected = False
            for segment in non_speech_frames[:10]:
                segment_array = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32767
                if len(segment_array) > 0:
                    mfcc = librosa.feature.mfcc(y=segment_array, sr=sr, n_mfcc=13)
                    if np.mean(mfcc) > -25:
                        background_noise_detected = True
                        break
            
            if len(speech_frames) > 10:
                energies = []
                for frame in speech_frames:
                    frame_array = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32767
                    energy = np.mean(frame_array ** 2)
                    energies.append(energy)
                
                energy_var = np.var(energies)
                multiple_speakers = energy_var > 0.01
            else:
                multiple_speakers = False
            
            if background_noise_detected:
                return {"status": "background_noise", "message": "Background noise detected"}
            elif multiple_speakers:
                return {"status": "multiple_speakers", "message": "Multiple speakers detected"}
            else:
                return {"status": "clean_audio", "message": "Audio is clean"}
                
        except Exception as e:
            logging.error(f"Audio analysis error: {str(e)}")
            return {"status": "error", "message": f"Audio analysis error: {str(e)}"}
    
    def _frame_audio(self, audio, frame_duration_ms, sample_rate):
        """Frame audio for webrtcvad"""
        try:
            frame_size = int(sample_rate * frame_duration_ms / 1000.0)
            audio = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio * 32767).astype(np.int16)
            num_frames = (len(audio_int16) + frame_size - 1) // frame_size
            audio_int16 = np.pad(audio_int16, (0, frame_size * num_frames - len(audio_int16)), mode='constant')
            frames = [audio_int16[i * frame_size:(i + 1) * frame_size].tobytes() for i in range(num_frames)]
            return frames
        except Exception as e:
            logging.error(f"Error framing audio: {str(e)}")
            return []

    def process_diarization_audio(self, audio_data, session_id, exam_name):
        """Process audio data for diarization and return results"""
        try:
            # Update session activity if session_id provided
            if session_id:
                self.initialize_session(session_id)
                self.update_session_activity(session_id)
                with self.session_lock:
                    session = self.sessions[session_id]
            
            # Convert bytes to audio array
            audio_segment = AudioSegment.from_raw(
                io.BytesIO(audio_data),
                sample_width=2,  # 16-bit
                frame_rate=16000,  # 16kHz
                channels=1
            )
            
            # Convert to numpy array
            audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
            
            # Apply noise reduction
            reduced_noise_audio = nr.reduce_noise(
                y=audio_array, 
                sr=16000, 
                stationary=False,
                prop_decrease=0.8 
            )
            
            # Create temporary wav data in memory
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, reduced_noise_audio, 16000, format='WAV')
            wav_buffer.seek(0)
            
            # Save to temporary file for diarization (model requires file path)
            temp_path = f"temp_audio_{uuid4()}.wav"
            with open(temp_path, 'wb') as f:
                f.write(wav_buffer.getvalue())
            
            try:
                # Perform diarization
                predicted_segments = self.diar_model.diarize(audio=temp_path, batch_size=1)
                
                # Extract unique speakers
                unique_speakers = set()
                segments_info = []
                
                if predicted_segments and len(predicted_segments) > 0:
                    for seg in predicted_segments[0]:
                        parts = seg.split()
                        if len(parts) >= 3:
                            speaker = parts[2]
                            unique_speakers.add(speaker)
                            segments_info.append({
                                'start': parts[0],
                                'end': parts[1], 
                                'speaker': speaker
                            })
                
                # Handle multiple speakers warning
                total_speakers = len(unique_speakers)
                result = {
                    'success': True,
                    'total_speakers': total_speakers,
                    'speakers': list(unique_speakers),
                    'segments': segments_info
                }
                if session_id and total_speakers > 1:
                    if session['warningCount'] == 0:
                        folder_path, session_id = self.initialize_pdf_evidence(session_id,exam_name)
                    else:
                        with self.pdf_lock:
                            if session_id in self.pdf_evidence:
                                folder_path = self.pdf_evidence[session_id]['evidence_folder']
                            else:
                                logging.error(f"No evidence folder found for session {session_id}")
                                folder_path = os.path.join('evidence', f'session_{session_id}_{int(time.time())}')
                                os.makedirs(folder_path, exist_ok=True)
                    session['warningCount'] += 1
                    timestamp = datetime.now()
                    #timestamp to show state like IST alone
                    audio_file_path = os.path.join(folder_path, f"violation_{session['warningCount']}_{timestamp.strftime('%Y%m%d_%H%M%S_%Z')}.wav")
                    logging.warning(f"Issuing warning {session['warningCount']}/5 for session {session_id}: Multiple speakers detected")
                    self.add_audio_evidence(session_id, result, audio_file_path, "Multiple Speakers",f"Warning {session['warningCount']}/5: Multiple speakers detected")
                    result['warningCount'] = session['warningCount']
                    result['message'] = f"Warning {session['warningCount']}/5: Multiple speakers detected"
                    try:
                        sf.write(audio_file_path, audio_array, 16000, format='WAV')
                        logging.info(f"Saved audio evidence: {audio_file_path}")
                    except Exception as e:
                        logging.error(f"Failed to save audio file {audio_file_path}: {str(e)}")
                    if session['warningCount'] >= 5:
                        session['terminate'] = True
                        session['terminateReason'] = 'Too many warnings for multiple speakers detected'
                        logging.warning(f"Terminating session {session_id}: Too many warnings")
                        result['terminate'] = True
                        result['terminateReason'] = session['terminateReason']
                    else:
                        result['terminate'] = False
                        result['terminateReason'] = ''
                elif session_id:
                    result['warningCount'] = session['warningCount']
                    result['message'] = 'Audio processed successfully'
                    result['terminate'] = session['terminate']
                    result['terminateReason'] = session['terminateReason']
                
                return result
                
            finally:
                # Always clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        except Exception as e:
            logging.error(f"Diarization error for session {session_id}: {str(e)}")
            session = self.sessions.get(session_id, {})
            return {
                'success': False,
                'error': str(e),
                'warningCount': session.get('warningCount', 0),
                'terminate': session.get('terminate', False),
                'terminateReason': session.get('terminateReason', '')
            }
    def initialize_pdf_evidence(self, session_id,exam_name):
        """Initialize PDF evidence collection for a session"""
        with self.pdf_lock:
            if session_id not in self.pdf_evidence:
                # Create evidence folder for this session
                evidence_folder = os.path.join('evidence', f'session_{session_id}_{int(time.time())}')
                os.makedirs(evidence_folder, exist_ok=True)
                
                # Create PDF document
                pdf_path = os.path.join(evidence_folder, f'malpractice_evidence_{session_id}.pdf')
                doc = SimpleDocTemplate(
                    pdf_path, 
                    pagesize=A4,
                    topMargin=0.8*inch,
                    bottomMargin=0.8*inch,
                    leftMargin=0.8*inch,
                    rightMargin=0.8*inch
                )
                
                # Initialize story (content) list
                story = []
                styles = getSampleStyleSheet()
                
                # Define custom color palette
                primary_color = colors.HexColor('#1a365d')      # Deep blue
                secondary_color = colors.HexColor('#2d3748')    # Dark gray
                accent_color = colors.HexColor('#e53e3e')       # Professional red
                light_bg = colors.HexColor('#f7fafc')          # Light gray background
                warning_bg = colors.HexColor('#fed7d7')        # Light red background
                
                # Add company/system header with line
                header_table = Table([[
                    Paragraph("INTERVIEW MONITORING SYSTEM", ParagraphStyle(
                        'HeaderCompany',
                        parent=styles['Normal'],
                        fontSize=10,
                        textColor=secondary_color,
                        fontName='Helvetica-Bold'
                    )),
                    Paragraph("CONFIDENTIAL REPORT", ParagraphStyle(
                        'HeaderConfidential',
                        parent=styles['Normal'],
                        fontSize=10,
                        textColor=accent_color,
                        fontName='Helvetica-Bold',
                        alignment=TA_RIGHT
                    ))
                ]], colWidths=[3*inch, 3*inch])
                
                header_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LINEBELOW', (0, 0), (-1, -1), 2, primary_color),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ]))
                
                story.append(header_table)
                story.append(Spacer(1, 20))
                
                # Main title with enhanced styling
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    spaceAfter=30,
                    alignment=TA_CENTER,
                    textColor=primary_color,
                    fontName='Helvetica-Bold',
                    leading=28
                )
                
                story.append(Paragraph("MALPRACTICE EVIDENCE REPORT", title_style))
                
                # Add decorative line under title
                line_table = Table([['']], colWidths=[6*inch])
                line_table.setStyle(TableStyle([
                    ('LINEBELOW', (0, 0), (-1, -1), 1, primary_color),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
                ]))
                story.append(line_table)
                
                # Session information in a professional card format
                current_time = datetime.now()
                info_card_data = [
                    ['Session Information', ''],
                    ['Gmail', session_id],
                    ['Exam Name', exam_name],
                    ['Report Generated', self.format_timestamp_with_timezone(current_time)],
                    ['Monitoring Started', self.format_timestamp_with_timezone(current_time)]
                ]
                
                info_table = Table(info_card_data, colWidths=[2.2*inch, 3.8*inch])
                info_table.setStyle(TableStyle([
                    # Header row styling
                    ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('SPAN', (0, 0), (-1, 0)),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    
                    # Data row styling
                    ('BACKGROUND', (0, 1), (0, -1), light_bg),
                    ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('TEXTCOLOR', (0, 1), (-1, -1), secondary_color),
                    ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('LEFTPADDING', (0, 0), (-1, -1), 15),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                    
                    # Borders
                    ('BOX', (0, 0), (-1, -1), 1, primary_color),
                    ('INNERGRID', (0, 1), (-1, -1), 0.5, colors.lightgrey),
                ]))
                
                story.append(info_table)
                story.append(Spacer(1, 30))
                
                # Violations section header with icon-like styling
                violations_header = Table([[
                    Paragraph("⚠", ParagraphStyle('IconStyle', fontSize=16, textColor=accent_color)),
                    Paragraph("VIOLATION SUMMARY", ParagraphStyle(
                        'ViolationsHeader',
                        parent=styles['Heading2'],
                        fontSize=16,
                        textColor=primary_color,
                        fontName='Helvetica-Bold',
                        leftIndent=10
                    ))
                ]], colWidths=[0.5*inch, 5.5*inch])
                
                violations_header.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                    ('LINEBELOW', (0, 0), (-1, -1), 2, accent_color),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ]))
                
                story.append(violations_header)
                story.append(Spacer(1, 15))
                
                self.pdf_evidence[session_id] = {
                    'doc': doc,
                    'story': story,
                    'evidence_folder': evidence_folder,
                    'pdf_path': pdf_path,
                    'violation_count': 0,
                    'start_time': current_time,
                    'colors': {
                        'primary': primary_color,
                        'secondary': secondary_color,
                        'accent': accent_color,
                        'light_bg': light_bg,
                        'warning_bg': warning_bg
                    }
                }
                
                logging.info(f"Initialized PDF evidence collection for session {session_id}")
                return evidence_folder, session_id

    def add_frame_evidence(self, session_id, frame, violation_type, message, additional_info=None):
        """Add frame evidence to PDF report"""
        try:
            with self.pdf_lock:
                if session_id not in self.pdf_evidence:
                    logging.debug(f"No PDF evidence initialized for session {session_id}, skipping evidence addition")
                    return
                
                evidence = self.pdf_evidence[session_id]
                evidence['violation_count'] += 1
                colors_dict = evidence['colors']
                
                # Save frame image
                timestamp = datetime.now()
                image_filename = f"violation_{evidence['violation_count']}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                image_path = os.path.join(evidence['evidence_folder'], image_filename)
                cv2.imwrite(image_path, frame)
                
                # Add to PDF story
                styles = getSampleStyleSheet()
                
                # Violation card header
                violation_header_data = [[
                    Paragraph("🚨", ParagraphStyle('ViolationIcon', fontSize=14)),
                    Paragraph(f"VIOLATION #{evidence['violation_count']}: {violation_type.upper()}", 
                            ParagraphStyle(
                                'ViolationTitle',
                                parent=styles['Heading3'],
                                fontSize=14,
                                fontName='Helvetica-Bold',
                                textColor=colors_dict['accent'],
                                leftIndent=10
                            ))
                ]]
                
                violation_header_table = Table(violation_header_data, colWidths=[0.5*inch, 5.5*inch])
                violation_header_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors_dict['warning_bg']),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('LEFTPADDING', (0, 0), (-1, -1), 15),
                    ('BOX', (0, 0), (-1, -1), 1, colors_dict['accent']),
                ]))
                
                evidence['story'].append(violation_header_table)
                evidence['story'].append(Spacer(1, 10))
                
                # Details in a professional card
                table_data = [
                    ['Violation Details', ''],
                    ['Timestamp', self.format_timestamp_with_timezone(timestamp)],
                    ['Violation Type', violation_type.title()],
                    ['Description', message],
                    ['Session Time', f"{time.time() - evidence['start_time'].timestamp():.1f} seconds"]
                ]
                
                if additional_info:
                    for key, value in additional_info.items():
                        table_data.append([key.replace('_', ' ').title(), str(value)])
                
                details_table = Table(table_data, colWidths=[2.2*inch, 3.8*inch])
                details_table.setStyle(TableStyle([
                    # Header row
                    ('BACKGROUND', (0, 0), (-1, 0), colors_dict['primary']),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('SPAN', (0, 0), (-1, 0)),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    
                    # Data rows
                    ('BACKGROUND', (0, 1), (0, -1), colors_dict['light_bg']),
                    ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors_dict['secondary']),
                    ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 12),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                    
                    # Borders
                    ('BOX', (0, 0), (-1, -1), 1, colors_dict['primary']),
                    ('INNERGRID', (0, 1), (-1, -1), 0.25, colors.lightgrey),
                ]))
                
                evidence['story'].append(details_table)
                evidence['story'].append(Spacer(1, 15))
                
                # Add captured frame with professional styling
                if os.path.exists(image_path):
                    # Image header
                    img_header = Table([[
                        Paragraph("📸 CAPTURED EVIDENCE", ParagraphStyle(
                            'ImageHeader',
                            fontSize=11,
                            fontName='Helvetica-Bold',
                            textColor=colors_dict['secondary']
                        ))
                    ]], colWidths=[6*inch])
                    
                    img_header.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors_dict['light_bg']),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('LEFTPADDING', (0, 0), (-1, -1), 12),
                        ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
                    ]))
                    
                    evidence['story'].append(img_header)
                    evidence['story'].append(Spacer(1, 5))
                    
                    # Image with border
                    img_table = Table([[Image(image_path, width=4*inch, height=3*inch)]], colWidths=[6*inch])
                    img_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('BOX', (0, 0), (-1, -1), 2, colors_dict['primary']),
                        ('TOPPADDING', (0, 0), (-1, -1), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ]))
                    
                    evidence['story'].append(img_table)
                    evidence['story'].append(Spacer(1, 25))
                
                logging.info(f"Added frame evidence for session {session_id}: {violation_type}")
                
        except Exception as e:
            logging.error(f"Error adding frame evidence for session {session_id}: {str(e)}")

    def add_audio_evidence(self, session_id, audio_result, audio_path, violation_type, message):
        """Add audio evidence to PDF report"""
        try:
            with self.pdf_lock:
                if session_id not in self.pdf_evidence:
                    logging.debug(f"No PDF evidence initialized for session {session_id}, skipping evidence addition")
                    return
                
                evidence = self.pdf_evidence[session_id]
                evidence['violation_count'] += 1
                colors_dict = evidence['colors']
                
                styles = getSampleStyleSheet()
                timestamp = datetime.now()
                
                # Violation card header
                violation_header_data = [[
                    Paragraph("🎤", ParagraphStyle('AudioIcon', fontSize=14)),
                    Paragraph(f"VIOLATION #{evidence['violation_count']}: {violation_type.upper()}", 
                            ParagraphStyle(
                                'ViolationTitle',
                                parent=styles['Heading3'],
                                fontSize=14,
                                fontName='Helvetica-Bold',
                                textColor=colors_dict['accent'],
                                leftIndent=10
                            ))
                ]]
                
                violation_header_table = Table(violation_header_data, colWidths=[0.5*inch, 5.5*inch])
                violation_header_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors_dict['warning_bg']),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('LEFTPADDING', (0, 0), (-1, -1), 15),
                    ('BOX', (0, 0), (-1, -1), 1, colors_dict['accent']),
                ]))
                
                evidence['story'].append(violation_header_table)
                evidence['story'].append(Spacer(1, 10))
                
                # Audio details table
                table_data = [
                    ['Audio Violation Details', ''],
                    ['Timestamp', self.format_timestamp_with_timezone(timestamp)],
                    ['Violation Type', violation_type.title()],
                    ['Description', message],
                    ['Session Time', f"{time.time() - evidence['start_time'].timestamp():.1f} seconds"]
                ]
                
                if 'total_speakers' in audio_result:
                    table_data.append(['Speakers Detected', str(audio_result['total_speakers'])])
                    
                if 'speakers' in audio_result:
                    table_data.append(['Speaker IDs', ', '.join(audio_result['speakers'])])
                
                details_table = Table(table_data, colWidths=[2.2*inch, 3.8*inch])
                details_table.setStyle(TableStyle([
                    # Header row
                    ('BACKGROUND', (0, 0), (-1, 0), colors_dict['primary']),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('SPAN', (0, 0), (-1, 0)),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    
                    # Data rows
                    ('BACKGROUND', (0, 1), (0, -1), colors_dict['light_bg']),
                    ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors_dict['secondary']),
                    ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 12),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                    
                    # Borders
                    ('BOX', (0, 0), (-1, -1), 1, colors_dict['primary']),
                    ('INNERGRID', (0, 1), (-1, -1), 0.25, colors.lightgrey),
                ]))
                
                evidence['story'].append(details_table)
                evidence['story'].append(Spacer(1, 15))
                
                # Add speaker timeline if available
                if 'segments' in audio_result and audio_result['segments']:
                    # Timeline header
                    timeline_header = Table([[
                        Paragraph("🕐 SPEAKER TIMELINE", ParagraphStyle(
                            'TimelineHeader',
                            fontSize=11,
                            fontName='Helvetica-Bold',
                            textColor=colors_dict['secondary']
                        ))
                    ]], colWidths=[6*inch])
                    
                    timeline_header.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors_dict['light_bg']),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('LEFTPADDING', (0, 0), (-1, -1), 12),
                        ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
                    ]))
                    
                    evidence['story'].append(timeline_header)
                    evidence['story'].append(Spacer(1, 5))
                    
                    timeline_data = [['Start Time', 'End Time', 'Speaker ID']]
                    for segment in audio_result['segments']:
                        timeline_data.append([
                            segment.get('start', 'N/A'),
                            segment.get('end', 'N/A'),
                            segment.get('speaker', 'Unknown')
                        ])
                    
                    timeline_table = Table(timeline_data, colWidths=[2*inch, 2*inch, 2*inch])
                    timeline_table.setStyle(TableStyle([
                        # Header row
                        ('BACKGROUND', (0, 0), (-1, 0), colors_dict['secondary']),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        
                        # Data rows
                        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors_dict['secondary']),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                        
                        # Alternating row colors
                        *[('BACKGROUND', (0, i), (-1, i), colors_dict['light_bg']) 
                        for i in range(2, len(timeline_data), 2)],
                        
                        # Borders
                        ('BOX', (0, 0), (-1, -1), 1, colors_dict['primary']),
                        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ]))
                    
                    evidence['story'].append(timeline_table)
                
                evidence['story'].append(Spacer(1, 15))
                
                # Audio file reference with styling
                audio_ref = Table([[
                    Paragraph(f"🔊 Audio Evidence: {os.path.basename(audio_path)}", ParagraphStyle(
                        'AudioRef',
                        fontSize=10,
                        fontName='Helvetica-Oblique',
                        textColor=colors_dict['secondary']
                    ))
                ]], colWidths=[6*inch])
                
                audio_ref.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors_dict['light_bg']),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 12),
                    ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
                ]))
                
                evidence['story'].append(audio_ref)
                evidence['story'].append(Spacer(1, 25))
                
                logging.info(f"Added audio evidence for session {session_id}: {violation_type}")
                
        except Exception as e:
            logging.error(f"Error adding audio evidence for session {session_id}: {str(e)}")

    def finalize_pdf_evidence(self, session_id):
        """Finalize and save PDF evidence report"""
        try:
            with self.pdf_lock:
                if session_id not in self.pdf_evidence:
                    logging.info(f"No PDF evidence to finalize for session {session_id} (no warnings issued)")
                    return None
                
                evidence = self.pdf_evidence[session_id]
                colors_dict = evidence['colors']
                styles = getSampleStyleSheet()
                
                # Add summary footer
                end_time = datetime.now()
                duration = end_time - evidence['start_time']
                
                # Page break for summary
                if evidence['violation_count'] > 0:
                    evidence['story'].append(PageBreak())
                
                # Summary header with icon
                summary_header = Table([[
                    Paragraph("📋", ParagraphStyle('SummaryIcon', fontSize=16)),
                    Paragraph("SESSION SUMMARY", ParagraphStyle(
                        'SummaryTitle',
                        parent=styles['Heading2'],
                        fontSize=18,
                        textColor=colors_dict['primary'],
                        fontName='Helvetica-Bold',
                        leftIndent=10
                    ))
                ]], colWidths=[0.5*inch, 5.5*inch])
                
                summary_header.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                    ('LINEBELOW', (0, 0), (-1, -1), 2, colors_dict['primary']),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
                ]))
                
                evidence['story'].append(summary_header)
                evidence['story'].append(Spacer(1, 15))
                
                # Summary data in professional card
                summary_data = [
                    ['Final Report Summary', ''],
                    ['Session Start', self.format_timestamp_with_timezone(evidence['start_time'])],
                    ['Session End', self.format_timestamp_with_timezone(end_time)],
                    ['Total Duration', str(duration).split('.')[0]],
                    ['Total Violations', str(evidence['violation_count'])],
                    ['Evidence Files', f"{evidence['violation_count']} files captured"],
                    ['Report Status', 'COMPLETED ✓']
                ]
                
                summary_table = Table(summary_data, colWidths=[2.2*inch, 3.8*inch])
                summary_table.setStyle(TableStyle([
                    # Header row
                    ('BACKGROUND', (0, 0), (-1, 0), colors_dict['primary']),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('SPAN', (0, 0), (-1, 0)),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    
                    # Data rows
                    ('BACKGROUND', (0, 1), (0, -1), colors_dict['light_bg']),
                    ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors_dict['secondary']),
                    ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('LEFTPADDING', (0, 0), (-1, -1), 15),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                    
                    # Special styling for completion status
                    ('TEXTCOLOR', (1, -1), (1, -1), colors.darkgreen),
                    ('FONTNAME', (1, -1), (1, -1), 'Helvetica-Bold'),
                    
                    # Borders
                    ('BOX', (0, 0), (-1, -1), 1, colors_dict['primary']),
                    ('INNERGRID', (0, 1), (-1, -1), 0.25, colors.lightgrey),
                ]))
                
                evidence['story'].append(summary_table)
                evidence['story'].append(Spacer(1, 30))
                
                # Professional disclaimer section
                disclaimer_header = Table([[
                    Paragraph("⚖️ LEGAL DISCLAIMER", ParagraphStyle(
                        'DisclaimerHeader',
                        fontSize=12,
                        fontName='Helvetica-Bold',
                        textColor=colors_dict['secondary']
                    ))
                ]], colWidths=[6*inch])
                
                disclaimer_header.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors_dict['light_bg']),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('LEFTPADDING', (0, 0), (-1, -1), 15),
                    ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
                ]))
                
                evidence['story'].append(disclaimer_header)
                evidence['story'].append(Spacer(1, 10))
                
                disclaimer_style = ParagraphStyle(
                    'DisclaimerStyle',
                    parent=styles['Normal'],
                    fontSize=9,
                    alignment=TA_JUSTIFY,
                    textColor=colors_dict['secondary'],
                    leading=12,
                    leftIndent=15,
                    rightIndent=15,
                    spaceBefore=5,
                    spaceAfter=5
                )
                
                disclaimer_text = """
                This report was automatically generated by the Interview Monitoring System for examination 
                security purposes. All timestamps reflect local system time at the point of capture. 
                The captured images, audio recordings, and behavioral data contained within this document 
                serve as official evidence of potential policy violations during the monitored session."""
                
                disclaimer_table = Table([[
                    Paragraph(disclaimer_text, disclaimer_style)
                ]], colWidths=[6*inch])
                
                disclaimer_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                    ('TOPPADDING', (0, 0), (-1, -1), 15),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                    ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
                ]))
                
                evidence['story'].append(disclaimer_table)
                evidence['story'].append(Spacer(1, 20))
                
                # Footer with generation info
                footer_data = [[
                    Paragraph("Report Generated by Interview Monitoring System", ParagraphStyle(
                        'FooterLeft',
                        fontSize=8,
                        textColor=colors.grey,
                        fontName='Helvetica-Oblique'
                    )),
                    Paragraph(self.format_timestamp_with_timezone(end_time), ParagraphStyle(
                        'FooterRight',
                        fontSize=8,
                        textColor=colors.grey,
                        fontName='Helvetica-Oblique',
                        alignment=TA_RIGHT
                    ))
                ]]
                
                footer_table = Table(footer_data, colWidths=[3*inch, 3*inch])
                footer_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LINEABOVE', (0, 0), (-1, -1), 1, colors.lightgrey),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                ]))
                
                evidence['story'].append(footer_table)
                
                # Build PDF
                evidence['doc'].build(evidence['story'])
                
                pdf_path = evidence['pdf_path']
                logging.info(f"Professional PDF evidence report generated: {pdf_path}")
                
                return pdf_path
                    
        except Exception as e:
            logging.error(f"Error finalizing PDF evidence for session {session_id}: {str(e)}")
            return None

    def authenticate_user(self, email, exam_name):
        """Authenticate user and fetch exam privileges from MongoDB"""
        try:
            exam = proctoring_collection.find_one({"examName": exam_name})
            if exam and email in exam.get('emails', []):
                self.session_privileges[email] = exam.get('privileges', {})
                return {
                    "success": True,
                    "privileges": exam.get('privileges', {}),
                    "sessionMode": exam.get('sessionMode', 'Virtual Interview'),
                    "examTimeSeconds": exam.get('examTimeSeconds', 600) # Default to 600 if not found
                }
            else:
                return {"success": False, "error": "Invalid email or exam name"}
        except Exception as e:
            logging.error(f"Authentication error: {str(e)}")
            return {"success": False, "error": str(e)}
        
    def get_gemini_question(self, session_id, exam_name, chat_history):
        """Generate a question using Google Gemini with chat history"""
        try:
            exam = proctoring_collection.find_one({"examName": exam_name})
            if not exam:
                return {"success": False, "error": "Exam not found"}
            
            # Format chat history for prompt
            history_text = "\n".join([f"{msg['message']} (at {msg['timestamp']})" for msg in chat_history]) if chat_history else "No previous interactions."
            prompt = f"""
            You are an AI interviewer ask relative question to hire interviwee for AI role. 
            Previous interactions:
            {history_text}
            
            Generate a relevant interview question for the exam, considering the previous interactions to ensure continuity.
            """
            response = gemini_model.generate_content(prompt)
            question = response.text.strip()
            logging.info(f"Generated question for session {session_id}: {question}")
            return {"success": True, "question": question}
        except Exception as e:
            logging.error(f"Error generating question for session {session_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def evaluate_answer(self, session_id, exam_name, question, answer, chat_history):
        """Evaluate an answer using Google Gemini with chat history"""
        try:
            exam = proctoring_collection.find_one({"examName": exam_name})
            if not exam:
                return {"success": False, "error": "Exam not found"}
            
            # Format chat history for prompt
            history_text = "\n".join([f"{msg['message']} (at {msg['timestamp']})" for msg in chat_history]) if chat_history else "No previous interactions."
            prompt = f"""
            You are an AI interviewer for the exam '{exam_name}'. 
            Previous interactions:
            {history_text}
            
            Evaluate the following answer to the question:
            Question: {question}
            Answer: {answer}
            
            Provide a concise evaluation (max 100 words) and a score out of 10.
            If applicable, generate a follow-up question related to the same topic.
            Return the response in JSON format with fields: evaluation, score, nextQuestion.
            Ensure the output is valid JSON.
            """
            response = gemini_model.generate_content(prompt).text
            # Ensure response is valid JSON
            try:
                result = json.loads(response[response.find("{"):response.find("}")+1])
                if not all(key in result for key in ['evaluation', 'score', 'nextQuestion']):
                    raise ValueError("Incomplete JSON response from Gemini")
                logging.info(f"Evaluated answer for session {session_id}: {result}")
                return {"success": True, **result}
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON from Gemini: {response}")
                return {"success": False, "error": f"Invalid JSON response: {str(e)}"}
            except ValueError as e:
                logging.error(f"Invalid response format: {str(e)}")
                return {"success": False, "error": str(e)}
        except Exception as e:
            logging.error(f"Error evaluating answer for session {session_id}: {str(e)}")
            return {"success": False, "error": str(e)}
        
# Initialize detector
detector = MalpracticeDetector()
if not os.path.exists('evidence'):
    os.makedirs('evidence')

# Start a background thread to clean up inactive sessions
def cleanup_inactive_sessions_periodically():
    while True:
        time.sleep(60)  # Check every minute
        detector.cleanup_inactive_sessions()
        logging.debug(f"Active sessions: {detector.get_session_count()}")

cleanup_thread = threading.Thread(target=cleanup_inactive_sessions_periodically, daemon=True)
cleanup_thread.start()

async def handle_client(websocket, path):
    """Handle WebSocket client connections for audio diarization"""
    client_address = websocket.remote_address
    session_id = None
    print(f"Client connected to WebSocket: {client_address}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data['type'] == 'init':
                    # Initialize session
                    session_id = data.get('session_id', str(uuid4()))
                    detector.register_websocket_session(session_id, websocket)
                    await websocket.send(json.dumps({
                        'type': 'init_response',
                        'session_id': session_id,
                        'success': True
                    }))
                    
                elif data['type'] == 'audio':
                    # Decode base64 audio data
                    audio_bytes = base64.b64decode(data['audio'])
                    current_session_id = data.get('session_id', session_id)
                    
                    # Process audio
                    result = detector.process_diarization_audio(audio_bytes, current_session_id,data['examName'])
                    result['type'] = 'audio_response'
                    
                    # Send result back to client
                    await websocket.send(json.dumps(result))
                    
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'success': False,
                    'error': 'Invalid JSON format'
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'success': False,
                    'error': str(e)
                }))
                
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected from WebSocket: {client_address}")
    except Exception as e:
        logging.error(f"WebSocket error for {client_address}: {str(e)}")
    finally:
        # Clean up session when connection is closed
        if session_id:
            print(f"Cleaning up session {session_id} for disconnected client {client_address}")
            detector.unregister_websocket_session(session_id)

def start_websocket_server():
    """Start the WebSocket server in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print("Starting WebSocket server on ws://localhost:8765")
    start_server = websockets.serve(handle_client, '0.0.0.0', 8765)
    loop.run_until_complete(start_server)
    loop.run_forever()

@app.route('/api/analyze_frame', methods=['POST'])
def analyze_frame():
    try:
        data = request.get_json()
        if not data or 'frame' not in data or 'session_id' not in data:
            logging.error("Missing frame data or session_id in request")
            return jsonify({"error": "Missing frame data or session_id"}), 400
        
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logging.error("Could not decode frame")
            return jsonify({"error": "Could not decode frame"}), 400
        
        session_id = data['session_id']
        exam_name = data['examName']
        result = detector.analyze_gaze_from_frame(frame, session_id,exam_name)
        logging.debug(f"Frame analysis result for session {session_id}: {result}")
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Frame analysis endpoint error: {str(e)}")
        return jsonify({"error": f"Frame analysis failed: {str(e)}"}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'examName' not in data:
            logging.error("Missing email or examName in request")
            return jsonify({"error": "Missing email or examName"}), 400
        
        email = data['email']
        exam_name = data['examName']
        result = detector.authenticate_user(email, exam_name)
        # print("\n\n\n\n\njsonify(result)\n\n\n\n\n")
        return jsonify(result)
    except Exception as e:
        logging.error(f"Login endpoint error: {str(e)}")
        return jsonify({"error": f"Login failed: {str(e)}"}), 500
    
@app.route('/api/analyze_audio', methods=['POST'])
def analyze_audio():
    try:
        if 'audio' not in request.files:
            logging.error("No audio file provided")
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            logging.error("No audio file selected")
            return jsonify({"error": "No audio file selected"}), 400
        
        filename = secure_filename(audio_file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(temp_path)
        
        try:
            result = detector.analyze_audio(temp_path,request.get_json()['session_id'])
            logging.debug(f"Audio analysis result: {result}")
            return jsonify(result)
        finally:
            # Always clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        logging.error(f"Audio analysis endpoint error: {str(e)}")
        return jsonify({"error": f"Audio analysis failed: {str(e)}"}), 500

@app.route('/api/cleanup_session', methods=['POST'])
def cleanup_session():
    """Endpoint to manually cleanup a session"""
    try:
        data = request.get_json()
        if not data or 'session_id' not in data:
            return jsonify({"error": "Missing session_id"}), 400
        
        session_id = data['session_id']
        detector.cleanup_session(session_id)
        
        return jsonify({"message": f"Session {session_id} cleaned up successfully"})
        
    except Exception as e:
        logging.error(f"Session cleanup endpoint error: {str(e)}")
        return jsonify({"error": f"Session cleanup failed: {str(e)}"}), 500

@app.route('/api/session_stats', methods=['GET'])
def session_stats():
    """Get current session statistics"""
    try:
        return jsonify({
            "active_sessions": detector.get_session_count(),
            "websocket_sessions": len(detector.websocket_sessions)
        })
    except Exception as e:
        logging.error(f"Session stats endpoint error: {str(e)}")
        return jsonify({"error": f"Failed to get session stats: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Backend is running",
        "active_sessions": detector.get_session_count()
    })

@app.route('/api/get_ai_question', methods=['POST'])
def get_ai_question():
    try:
        data = request.get_json()
        if not data or 'session_id' not in data or 'examName' not in data:
            logging.error("Missing session_id or examName in request")
            return jsonify({"error": "Missing session_id or examName"}), 400
        
        session_id = data['session_id']
        exam_name = data['examName']
        chat_history = data.get('chat_history', [])
        result = detector.get_gemini_question(session_id, exam_name, chat_history)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Get AI question endpoint error: {str(e)}")
        return jsonify({"error": f"Failed to get AI question: {str(e)}"}), 500

@app.route('/api/evaluate_answer', methods=['POST'])
def evaluate_answer():
    try:
        data = request.get_json()
        if not data or 'session_id' not in data or 'examName' not in data or 'question' not in data or 'answer' not in data:
            logging.error("Missing required fields in request")
            return jsonify({"error": "Missing required fields"}), 400
        
        session_id = data['session_id']
        exam_name = data['examName']
        question = data['question']
        answer = data['answer']
        chat_history = data.get('chat_history', [])
        result = detector.evaluate_answer(session_id, exam_name, question, answer, chat_history)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Evaluate answer endpoint error: {str(e)}")
        return jsonify({"error": f"Failed to evaluate answer: {str(e)}"}), 500
    
@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using Groq API"""
    try:
        if 'audio' not in request.files:
            logging.error("No audio file provided")
            return jsonify({"success": False, "error": "No audio file provided"}), 400
        audio_file = request.files['audio']
        filename = secure_filename(audio_file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(temp_path)
        try:
            with open(temp_path, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(filename, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                    language="en",
                )
                return jsonify({"success": True, "text": transcription.text})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logging.info(f"Removed temporary audio file: {temp_path}")
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Update available endpoints in main
if __name__ == '__main__':
    print("Starting Interview Monitoring Backend...")
    print("Backend running on: http://localhost:5000")
    print("WebSocket server will run on: ws://localhost:8765")
    print("Make sure your React app is running on: http://localhost:3000")
    
    # Start WebSocket server in a separate thread
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()
    
    print(f"Background cleanup thread started - checking every 60 seconds")
    print("Available endpoints:")
    print("  - POST /api/analyze_frame - Analyze video frame")
    print("  - POST /api/analyze_audio - Analyze audio file")
    print("  - POST /api/cleanup_session - Manual session cleanup")
    print("  - GET /api/session_stats - Get session statistics")
    print("  - GET /api/health - Health check")
    print("  - POST /api/get_ai_question - Get AI interview question")
    print("  - POST /api/evaluate_answer - Evaluate AI interview answer")
    
    app.run(debug=False, host='0.0.0.0', port=5000)