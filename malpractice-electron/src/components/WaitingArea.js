import { useState, useEffect, useRef } from 'react';
import './WaitingArea.css'; 
const WaitingArea = ({ onJoinMeet, startMonitoring, screenGaze, multiSpeaker }) => {
  const [noiseLevel, setNoiseLevel] = useState(0);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isMicOn, setIsMicOn] = useState(false);
  const [quietStartTime, setQuietStartTime] = useState(null);
  const [continuousQuietSeconds, setContinuousQuietSeconds] = useState(0);
  const videoRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const animationFrameRef = useRef(null);
  const NOISE_THRESHOLD = 0.6;

  useEffect(() => {
    startMedia();
    return () => stopMedia();
  }, []);

  useEffect(() => {
    if (!isMicOn) {
      setQuietStartTime(null);
      setContinuousQuietSeconds(0);
    }
  }, [isMicOn]);

  const startMedia = async () => {
    try {
      const constraints = {
        video: screenGaze !== 0 ? true : false,
        audio: multiSpeaker !== 0 ? true : false,
      };
      if (constraints.video || constraints.audio) {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;
        if (videoRef.current && screenGaze !== 0) {
          videoRef.current.srcObject = stream;
        }
        if (multiSpeaker !== 0) {
          startNoiseMonitoring(stream);
        }
      }
      setIsCameraOn(constraints.video);
      setIsMicOn(constraints.audio);
    }
     catch (error) {
      console.error('Media access failed:', error);
    }
  };

  const stopMedia = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
  };

  const startNoiseMonitoring = (stream) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    audioContextRef.current = audioContext;
    const analyser = audioContext.createAnalyser();
    analyserRef.current = analyser;
    analyser.fftSize = 2048;
    const microphone = audioContext.createMediaStreamSource(stream);
    microphone.connect(analyser);

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    const updateNoiseLevel = () => {
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        return;
      }

      analyser.getByteTimeDomainData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const amplitude = (dataArray[i] - 128) / 128;
        sum += amplitude * amplitude;
      }
      const rms = Math.sqrt(sum / dataArray.length);
      setNoiseLevel(rms);

      setIsMicOn(currentIsMicOn => {
        setQuietStartTime(currentQuietStartTime => {
          setContinuousQuietSeconds(currentContinuousQuietSeconds => {
            if (currentIsMicOn) {
              const currentTime = Date.now();
              if (rms < NOISE_THRESHOLD) {
                if (currentQuietStartTime === null) {
                  setQuietStartTime(currentTime);
                  return 0;
                } else {
                  const quietDuration = Math.floor((currentTime - currentQuietStartTime) / 1000);
                  return Math.min(quietDuration, 3);
                }
              } else {
                setQuietStartTime(null);
                return 0;
              }
            } else {
              setQuietStartTime(null);
              return 0;
            }
          });
          return currentQuietStartTime;
        });
        return currentIsMicOn;
      });

      animationFrameRef.current = requestAnimationFrame(updateNoiseLevel);
    };

    updateNoiseLevel();
  };

  const toggleCamera = () => {
    if (screenGaze === 0) return; // Prevent toggling if screenGaze is 0
    if (streamRef.current) {
      const videoTrack = streamRef.current.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        setIsCameraOn(videoTrack.enabled);
      }
    }
  };

  const toggleMic = () => {
    if (multiSpeaker === 0) return; // Prevent toggling if multiSpeaker is 0
    if (streamRef.current) {
      const audioTrack = streamRef.current.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setIsMicOn(audioTrack.enabled);
        if (!audioTrack.enabled) {
          setQuietStartTime(null);
          setContinuousQuietSeconds(0);
        }
      }
    }
  };

  const canJoinMeet = (multiSpeaker === 0 || continuousQuietSeconds >= 3) && (screenGaze === 0 || isCameraOn) && (multiSpeaker === 0 || isMicOn);

  return (
    <div className="waiting-area">
      <div className="waiting-video-container">
        {screenGaze !== 0 && (
          <>
            <video ref={videoRef} autoPlay muted playsInline className="waiting-video" />
            {!isCameraOn && (
              <div className="video-off-overlay">
                <span>Camera Off</span>
              </div>
            )}
          </>
        )}
        <div className="waiting-controls">
          {screenGaze !== 0 && (
            <button
              onClick={toggleCamera}
              className={`control-btn ${isCameraOn ? 'active' : ''}`}
              title={isCameraOn ? 'Turn off camera' : 'Turn on camera'}
            >
              📹
            </button>
          )}
          {multiSpeaker !== 0 && (
            <button
              onClick={toggleMic}
              className={`control-btn ${isMicOn ? 'active' : ''}`}
              title={isMicOn ? 'Turn off microphone' : 'Turn on microphone'}
            >
              🎤
            </button>
          )}
        </div>
      </div>
      <div className="waiting-info">
        <h2>Ready to Join?</h2>
        {multiSpeaker !== 0 && (
          <div className="noise-meter">
            <div className="noise-bar" style={{ width: `${Math.min(noiseLevel * 1000, 100)}%` }} />
            <span>Noise Level: {noiseLevel < NOISE_THRESHOLD ? 'Acceptable' : 'Too High'}</span>
          </div>
        )}
        <button
          onClick={() => {
            onJoinMeet();
            window.electronAPI.startExam();
            startMonitoring();
          }}
          className="join-btn"
          disabled={!canJoinMeet}
        >
          {multiSpeaker === 0 || continuousQuietSeconds >= 3
            ? 'Join Interview'
            : `Wait ${3 - continuousQuietSeconds}s for quiet audio`}
        </button>
      </div>
    </div>
  );
};

export default WaitingArea;