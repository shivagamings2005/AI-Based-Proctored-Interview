import React, { useState, useEffect, useRef } from 'react';
import ControlPanel from './ControlPanel';
import ViolationsLog from './ViolationsLog';
import './MainMeetArea.css';

const MainMeetArea = ({
  isMonitoring,
  startMonitoring,
  stopMonitoring,
  backendStatus,
  audioStatus,
  isFullscreen,
  warningCount,
  violations,
  sessionMode,
  aiQuestion,
  userAnswer,
  setUserAnswer,
  aiResponse,
  submitAnswer,
  fetchNextQuestion,
}) => {
  const [isTextareaLocked, setIsTextareaLocked] = useState(false);
  const [isMicOn, setIsMicOn] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [liveCaption, setLiveCaption] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const textareaRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingIntervalRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const streamRef = useRef(null);

  useEffect(() => {
  if (isMicOn && !isRecording) {
    startRecording();
  } else if (!isMicOn && isRecording) {
    stopRecording();
  }

  return () => {
    stopRecording();
  };
}, [isMicOn, sessionMode]);

  useEffect(() => {
    if (transcript && sessionMode === 'AI Interview' && !isTextareaLocked) {
      const textarea = textareaRef.current;
      const startPos = textarea.selectionStart;
      const endPos = textarea.selectionEnd;
      const currentValue = userAnswer || '';
      const newValue =
        currentValue.substring(0, startPos) +
        transcript +
        currentValue.substring(endPos);
      setUserAnswer(newValue);
      setTranscript('');
      const newCursorPos = startPos + transcript.length;
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = newCursorPos;
      }, 0);
    }
  }, [transcript, userAnswer, setUserAnswer, isTextareaLocked, sessionMode]);

  const startRecording = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;
    setIsRecording(true);
    
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    
    mediaRecorder.ondataavailable = (event) => {
      audioChunksRef.current.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      if (audioChunksRef.current.length > 0) {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/mp3' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.mp3');
        
        try {
          const response = await fetch('http://localhost:5000/api/transcribe', {
            method: 'POST',
            body: formData,
          });
          const data = await response.json();
          if (data.success) {
            const transcribedText = data.text.trim();
            if (transcribedText) {
              const timestamp = new Date().toLocaleString();
              if (sessionMode === 'AI Interview') {
                setTranscript(transcribedText);
              } else {
                setLiveCaption(transcribedText);
                setChatHistory((prev) => [
                  ...prev,
                  { timestamp, text: transcribedText },
                ]);
              }
            }
          }
        } catch (error) {
          console.error('Error sending audio:', error);
        }
      }
      
      audioChunksRef.current = [];
      
      // Restart recording if still supposed to be on
      if (isMicOn && streamRef.current && streamRef.current.active) {
        setTimeout(() => {
          if (mediaRecorderRef.current && isMicOn) {
            mediaRecorderRef.current.start();
          }
        }, 100);
      }
    };

    mediaRecorder.start();
    
    if (sessionMode !== 'AI Interview') {
      recordingIntervalRef.current = setInterval(() => {
        if (mediaRecorder.state === 'recording' && isMicOn) {
          mediaRecorder.stop();
          // Recording will restart automatically in onstop handler
        }
      }, 3000); // Reduced to 3 seconds for more responsive captions
    }
  } catch (error) {
    console.error('Error accessing microphone:', error);
    setIsMicOn(false);
    setIsRecording(false);
    alert('Microphone access denied or not available.');
  }
};

const stopRecording = () => {
  setIsRecording(false);
  
  if (recordingIntervalRef.current) {
    clearInterval(recordingIntervalRef.current);
    recordingIntervalRef.current = null;
  }
  
  if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
    mediaRecorderRef.current.stop();
  }
  
  if (streamRef.current) {
    streamRef.current.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }
};

  const toggleMic = () => {
  setIsMicOn(!isMicOn);
};

  const handleSubmitAnswer = () => {
    submitAnswer();
    setIsTextareaLocked(true);
  };

  const handleNextQuestion = () => {
    fetchNextQuestion();
    setIsTextareaLocked(false);
  };

  return (
    <div className="main-meet-area">
      <div className="controls-column">
        <ControlPanel
          isMonitoring={isMonitoring}
          startMonitoring={startMonitoring}
          stopMonitoring={stopMonitoring}
          backendStatus={backendStatus}
          audioStatus={audioStatus}
          isFullscreen={isFullscreen}
          warningCount={warningCount}
        />
        <ViolationsLog violations={violations} />
      </div>
      <div className="meet-content">
        {sessionMode === 'AI Interview' ? (
          <div className="ai-interview-container">
            <h3>AI Interview</h3>
            {aiQuestion && <p><strong>Question:</strong> {aiQuestion}</p>}
            <div className="textarea-container">
              <textarea
                ref={textareaRef}
                value={userAnswer}
                onChange={(e) => setUserAnswer(e.target.value)}
                placeholder="Type your answer here..."
                className="answer-textarea"
                disabled={isTextareaLocked}
              />
              <button
                onClick={toggleMic}
                className={`mic-btn ${isMicOn ? 'mic-on' : 'mic-off'}`}
                title={isMicOn ? 'Turn off microphone' : 'Turn on microphone'}
                disabled={isTextareaLocked}
              >
                {isMicOn ? '🎤' : '🔇'}
              </button>
            </div>
            <button
              onClick={handleSubmitAnswer}
              disabled={!userAnswer?.trim() || isTextareaLocked}
              className="submit-answer-btn"
            >
              Submit Answer
            </button>
            {aiResponse && (
              <div className="ai-response">
                <p><strong>Evaluation:</strong> {aiResponse.evaluation}</p>
                <p><strong>Score:</strong> {aiResponse.score}/10</p>
                <button
                  onClick={handleNextQuestion}
                  className="next-question-btn"
                >
                  Next Question
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="non-ai-interview-container">
            <div className="image-container">
              <img
                src="https://media.istockphoto.com/id/1440329437/vector/meeting-of-politicians-or-corporate-employees-at-round-table.jpg?s=612x612&w=0&k=20&c=tahK2wRelUb6CvBcA4moyts0Lf03smWfCBroRToyDkA="
                alt="Interview Mock"
                className="mock-meet-image"
              />
              <button
                onClick={toggleMic}
                className={`mic-btn ${isMicOn ? 'mic-on' : 'mic-off'}`}
                title={isMicOn ? 'Turn off captions' : 'Turn on captions'}
              >
                {isMicOn ? '💬' : '🚫'}
              </button>
            </div>
            <div className="chat-history">
              <h4>Chat History</h4>
              <div className="chat-history-content">
                {chatHistory.length > 0 ? (
                  chatHistory.map((entry, index) => (
                    <div key={index} className="chat-entry">
                      <span className="chat-timestamp">{entry.timestamp}</span>: {entry.text}
                    </div>
                  ))
                ) : (
                  <p>No captions recorded yet.</p>
                )}
              </div>
            </div>
            {liveCaption && (
              <div className="live-caption">
                <strong>Live Caption:</strong> {liveCaption}
              </div>
            )}
          </div>
        )}
        
      </div>
    </div>
  );
};

export default MainMeetArea;