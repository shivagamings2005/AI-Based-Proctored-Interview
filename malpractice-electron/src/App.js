import { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import WaitingArea from './components/WaitingArea';
import Header from './components/Header';
import VideoSection from './components/VideoSection';
import MainMeetArea from './components/MainMeetArea';
import Footer from './components/Footer';
import MalpracticePopup from './components/MalpracticePopup';

const App = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [violations, setViolations] = useState([]);
  const [currentStatus, setCurrentStatus] = useState('Ready to start');
  const [showMalpracticePopup, setShowMalpracticePopup] = useState(false);
  const [malpracticeMessage, setMalpracticeMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [warningCount, setWarningCount] = useState(0);
  const [isInWaitingArea, setIsInWaitingArea] = useState(true);
  const [chatMessages, setChatMessages] = useState([]);
  const [audioStatus, setAudioStatus] = useState('Not recording');
  const [timeLeft, setTimeLeft] = useState(null);
  const [userEmail, setUserEmail] = useState('');
  const [examName, setExamName] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [privileges, setPrivileges] = useState({});
  const [sessionMode, setSessionMode] = useState('');
  const [loginError, setLoginError] = useState('');  
  const [showLogin, setShowLogin] = useState(true);
  const [aiQuestion, setAiQuestion] = useState('');
  const [userAnswer, setUserAnswer] = useState('');
  const [aiResponse, setAiResponse] = useState(null);
  const [nextQuestion, setNextQuestion] = useState('');
  const sessionId = useRef(userEmail);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const websocketRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const recordingIntervalRef = useRef(null);
  const frameAnalysisInterval = useRef(null);
  const timerIntervalRef = useRef(null);
  const lastWarningCountRef = useRef(0); // Track last warning count

  useEffect(() => {
    console.log('Checking backend connection');
    checkBackendConnection();
    return () => cleanup();
  }, []);
  useEffect(() => {
    const authenticateUser = async () => {
      if (userEmail && examName) {
        try {
          const response = await fetch('http://localhost:5000/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: userEmail, examName }),
          });
          const result = await response.json();
          if (result.success) {
            setPrivileges({
              multiFace: result.privileges.multiFace,
              noFace: result.privileges.noFace,
              screenGaze: result.privileges.screenGaze,
              multiSpeaker: result.privileges.multiSpeaker,
            });
            setSessionMode(result.sessionMode);
            setIsAuthenticated(true);
            setShowLogin(false);
            setTimeLeft(result.examTimeSeconds);
            sessionId.current = userEmail; // Set sessionId to email
          } else {
            setLoginError(result.error || 'Invalid email or exam name');
          }
        } catch (error) {
          console.error('Login failed:', error);
          setLoginError('Failed to authenticate');
        }
      }
    };

    authenticateUser();
  }, [userEmail, examName]);

  useEffect(() => {
  if (sessionMode === 'AI Interview' && isAuthenticated && !isInWaitingArea) {
    const fetchAiQuestion = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/get_ai_question', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId.current, examName, chat_history: chatMessages }),
        });
        const result = await response.json();
        if (result.success) {
          setAiQuestion(result.question);
        } else {
          console.error('Failed to fetch AI question:', result.error);
        }
      } catch (error) {
        console.error('Error fetching AI question:', error);
      }
    };
    fetchAiQuestion();
  }
}, [sessionMode, isAuthenticated, isInWaitingArea, examName]);

  const checkBackendConnection = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/health');
      if (response.ok) {
        console.log('Backend connected');
        setBackendStatus('connected');
      } else {
        console.warn('Backend health check failed, status:', response.status);
        setBackendStatus('error');
      }
    } catch (error) {
      console.error('Backend connection failed:', error);
      setBackendStatus('error');
    }
  };

  useEffect(() => {
    const preventKeyboardShortcuts = (e) => {
      if (isMonitoring) {
        if (e.ctrlKey || e.altKey || e.metaKey || e.key === 'Tab') {
          e.preventDefault();
          addViolation('Keyboard shortcut blocked');
        }
      }
    };

    const preventContextMenu = (e) => {
      if (isMonitoring) {
        e.preventDefault();
        addViolation('Right-click blocked');
      }
    };

    document.addEventListener('keydown', preventKeyboardShortcuts);
    document.addEventListener('contextmenu', preventContextMenu);

    return () => {
      document.removeEventListener('keydown', preventKeyboardShortcuts);
      document.removeEventListener('contextmenu', preventContextMenu);
    };
  }, [isMonitoring]);

  useEffect(() => {
    const handleFullscreenChange = async () => {
      const isCurrentlyFullscreen = !!(
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.mozFullScreenElement ||
        document.msFullscreenElement
      );
      console.log('Fullscreen change detected, isFullscreen:', isCurrentlyFullscreen, 'isMonitoring:', isMonitoring);
      setIsFullscreen(isCurrentlyFullscreen);
      if (isMonitoring && !isCurrentlyFullscreen) {
        console.log('Fullscreen exited during monitoring, logging violation');
        addViolation('Exited fullscreen mode');
        setWarningCount((prev) => {
          const newCount = prev + 1;
          console.log('Updated warningCount:', newCount);
          return newCount;
        });
        await showMalpracticeAlert('Fullscreen mode required during interview');
        if (warningCount + 1 >= 5) {
          await showMalpracticeAlert('Too many warnings: Exiting interview due to repeated violations');
          await exitExam();
        }
        if (window.electron) {
          console.log('Sending start-monitoring IPC to re-enter fullscreen');
          window.electron.send('start-monitoring');
        }
      }
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
      document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
    };
  }, [isMonitoring, warningCount]);

  useEffect(() => {
    let fullscreenPollInterval;
    if (isMonitoring && window.electron) {
      fullscreenPollInterval = setInterval(() => {
        const isFullScreen = document.fullscreenElement !== null;
        console.log('Polling fullscreen state, isFullScreen:', isFullScreen);
        if (!isFullScreen) {
          console.log('Fullscreen exited during monitoring, sending start-monitoring IPC');
          window.electron.send('start-monitoring');
        }
      }, 1000);
    }
    return () => {
      if (fullscreenPollInterval) {
        console.log('Clearing fullscreen poll interval');
        clearInterval(fullscreenPollInterval);
      }
    };
  }, [isMonitoring]);

  useEffect(() => {
    if (isMonitoring && timeLeft > 0) {
      timerIntervalRef.current = setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 1) {
            stopMonitoring();
            window.electronAPI.exitApp();
            window.close();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
    };
  }, [isMonitoring, timeLeft]);

  const enterFullscreen = async () => {
    console.log('Entering fullscreen, electron available:', !!window.electron);
    try {
      if (window.electron) {
        window.electron.send('start-monitoring');
        console.log('Sent start-monitoring IPC');
      } else {
        console.log('Using browser fullscreen API');
        const element = document.documentElement;
        if (element.requestFullscreen) {
          await element.requestFullscreen();
        } else if (element.webkitRequestFullscreen) {
          await element.webkitRequestFullscreen();
        } else if (element.mozRequestFullScreen) {
          await element.mozRequestFullScreen();
        } else if (element.msRequestFullscreen) {
          await element.msRequestFullscreen();
        }
      }
    } catch (error) {
      console.error('Failed to enter fullscreen:', error);
    }
  };

  const exitFullscreen = async () => {
    console.log('Exiting fullscreen, electron available:', !!window.electron);
    try {
      if (window.electron) {
        window.electron.send('stop-monitoring');
        console.log('Sent stop-monitoring IPC');
      } else {
        console.log('Using browser exit fullscreen API');
        if (document.exitFullscreen) {
          await document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
          await document.webkitExitFullscreen();
        } else if (document.mozCancelFullScreen) {
          await document.mozCancelFullScreen();
        } else if (document.msExitFullscreen) {
          await document.msExitFullscreen();
        }
      }
    } catch (error) {
      console.error('Failed to exit fullscreen:', error);
    }
  };

  const startCamera = async () => {
    console.log('Starting camera');
    try {
      if (privileges.screenGaze === 0) {
        console.log('Camera disabled due to screenGaze privilege');
        return null;
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: privileges.multiSpeaker !== 0, // Enable audio only if multiSpeaker is not 0
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      return stream;
    } catch (error) {
      console.error('Camera access failed:', error);
      throw error;
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const startAudioRecording = async (stream) => {
    if (privileges.multiSpeaker === 0) {
      console.log('Audio recording disabled due to multiSpeaker privilege');
      setIsRecording(false);
      setAudioStatus('Audio disabled');
      return () => {};
    }
    console.log('Starting audio recording');
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      const chunks = [];
      let isRecording = true;

      processor.onaudioprocess = (e) => {
        if (!isRecording) return;
        const inputData = e.inputBuffer.getChannelData(0);
        const int16Array = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          int16Array[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
        }
        chunks.push(int16Array.buffer);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      const sendAudioChunk = async () => {
        if (chunks.length > 0 && isRecording) {
          const audioData = chunks.splice(0, chunks.length);
          const audioBuffer = new Blob(audioData, { type: 'audio/raw' });
          const arrayBuffer = await audioBuffer.arrayBuffer();
          const base64Audio = btoa(
            new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), '')
          );

          if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
            websocketRef.current.send(
              JSON.stringify({
                type: 'audio',
                audio: base64Audio,
                session_id: sessionId.current,
                examName: examName,
              })
            );
          }
        }
      };

      recordingIntervalRef.current = setInterval(sendAudioChunk, 10000);
      setIsRecording(true);
      setAudioStatus('Recording');

      return () => {
        isRecording = false;
        processor.disconnect();
        source.disconnect();
        audioContext.close();
      };
    } catch (error) {
      console.error('Audio recording failed:', error);
      setAudioStatus('Error');
    }
  };

  const stopAudioRecording = () => {
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setIsRecording(false);
    setAudioStatus('Stopped');
  };

  const connectWebSocket = async () => {
    console.log('Connecting WebSocket');
    try {
      websocketRef.current = new WebSocket('ws://localhost:8765');
      websocketRef.current.onopen = () => {
        console.log('Connected to WebSocket server');
        setAudioStatus('Connected to server');
        websocketRef.current.send(
          JSON.stringify({
            type: 'init',
            session_id: sessionId.current,
          })
        );
      };
      websocketRef.current.onmessage = async (event) => {
        const result = JSON.parse(event.data);
        if (result.type === 'audio_response' && result.success) {
          setAudioStatus(`Speakers: ${result.total_speakers}`);
          if (result.total_speakers > 1) {
            addViolation(result.message || 'Multiple speakers detected');
            setWarningCount(result.warningCount);
            if (result.message && result.warningCount > lastWarningCountRef.current) {
              console.log(`New audio warning detected: warningCount increased from ${lastWarningCountRef.current} to ${result.warningCount}`);
              lastWarningCountRef.current = result.warningCount;
              await showMalpracticeAlert(result.message);
            }
            if (result.terminate) {
              await showMalpracticeAlert(result.terminateReason);
              await exitExam();
            }
          }
        } else if (result.type === 'error') {
          setAudioStatus(`Error: ${result.error}`);
          addViolation(`Audio processing error: ${result.error}`);
        }
      };
      websocketRef.current.onclose = () => {
        console.log('WebSocket connection closed');
        setAudioStatus('Disconnected from server');
      };
      websocketRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setAudioStatus('WebSocket error');
      };
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      setAudioStatus('WebSocket connection failed');
    }
  };

  const cleanup = () => {
    stopAudioRecording();
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
    }
    fetch('http://localhost:5000/api/cleanup_session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId.current }),
    }).catch((error) => console.error('Session cleanup failed:', error));
  };

  const showMalpracticeAlert = (message) => {
    return new Promise((resolve) => {
      console.log('Showing malpractice alert:', message);
      setMalpracticeMessage(message);
      setShowMalpracticePopup(true);

      if (window.electron) {
        console.log('Sending show-popup IPC to Electron');
        window.electron.send('show-popup');
      }

      window.malpracticePopupResolve = () => {
        console.log('Malpractice popup closed');
        setShowMalpracticePopup(false);
        resolve();
      };
    });
  };

  const analyzeFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.drawImage(videoRef.current, 0, 0);
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    try {
      const response = await fetch('http://localhost:5000/api/analyze_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: frameData, session_id: sessionId.current , examName : examName }),
      });
      const result = await response.json();
      setWarningCount(result.warningCount);

      // Throttle popups for non-warning detections (10 seconds), but allow warnings immediately
      const now = Date.now();
      const lastAlertTimes = window.lastAlertTimes || {
        no_face: 0,
        multiple_faces: 0,
        not_looking: 0,
        error: 0,
      };
      window.lastAlertTimes = lastAlertTimes;

      // Check if a new warning was issued
      const isNewWarning = result.warningCount > lastWarningCountRef.current;
      if (isNewWarning) {
        console.log(`New warning detected: warningCount increased from ${lastWarningCountRef.current} to ${result.warningCount}`);
        lastWarningCountRef.current = result.warningCount;
      }

      if (result.status === 'no_face') {
        addViolation(`No face detected (Duration: ${result.noFaceDuration || 0}s)`);
        setCurrentStatus('⚠ No face detected');
        if (isNewWarning) {
          await showMalpracticeAlert(result.message || 'No face detected in the webcam');
          lastAlertTimes.no_face = now;
        }
      } else if (result.status === 'multiple_faces') {
        addViolation(result.message || 'Multiple faces detected');
        setCurrentStatus('🚨 Multiple faces detected');
        if (isNewWarning) {
          await showMalpracticeAlert(result.message || 'Multiple faces detected in the webcam');
          lastAlertTimes.multiple_faces = now;
        }
      } else if (result.status === 'not_looking') {
        addViolation(result.message || 'Not looking at screen');
        setCurrentStatus('👁️ Not looking');
        if (isNewWarning) {
          await showMalpracticeAlert(result.message || 'Please look at the screen during the interview');
          lastAlertTimes.not_looking = now;
        }
      } else if (result.status === 'looking') {
        setCurrentStatus('✅ Looking at screen');
      } else {
        setCurrentStatus('❌ Analysis failed');
        if (now - lastAlertTimes.error > 10000) {
          await showMalpracticeAlert('Frame analysis failed');
          lastAlertTimes.error = now;
        }
      }

      // Handle termination
      if (result.terminate) {
        await showMalpracticeAlert(result.terminateReason || 'Session terminated due to violation');
        await exitExam();
      }
    } catch (error) {
      console.error('Frame analysis failed:', error);
      setCurrentStatus('❌ Analysis error');
      const now = Date.now();
      const lastAlertTimes = window.lastAlertTimes || { error: 0 };
      if (now - lastAlertTimes.error > 10000) {
        await showMalpracticeAlert('Frame analysis failed due to an error');
        lastAlertTimes.error = now;
      }
    }
  };

  const addViolation = (violation) => {
    const timestamp = new Date().toLocaleTimeString();
    setViolations((prev) => [...prev, { violation, timestamp }]);
  };

  const startMonitoring = async () => {
    console.log('Starting monitoring...');
    if (backendStatus !== 'connected') {
      console.warn('Backend not connected');
      alert('Backend server is not responding. Please ensure the Python server is running on localhost:5000');
      return;
    }
    try {
      await enterFullscreen();
      const stream = await startCamera();
      setIsMonitoring(true);
      setCurrentStatus('🔄 Initializing monitoring...');
      frameAnalysisInterval.current = setInterval(analyzeFrame, 2000);
      await connectWebSocket();
      await startAudioRecording(stream);
      setCurrentStatus('✅ Monitoring active');
    } catch (error) {
      console.error('Failed to start monitoring:', error);
      alert(`Failed to start monitoring: ${error.message || 'Unknown error'}`);
    }
  };

  const stopMonitoring = async () => {
    setIsMonitoring(false);
    cleanup();
    if (frameAnalysisInterval.current) {
      clearInterval(frameAnalysisInterval.current);
    }
    stopCamera();
    await exitFullscreen();
    setCurrentStatus('🛑 Monitoring stopped');
  };

  const exitExam = async () => {
    await stopMonitoring();
    setShowMalpracticePopup(false);
    if (window.electron) {
      window.electron.send('end-exam');
    }
    setCurrentStatus('🛑 Exam terminated due to violation');
    setTimeout(() => {
      window.close();
    }, 2000);
  };

  const handleExit = useCallback(async () => {
    await stopMonitoring();
    if (window.electron) {
      window.electron.send('end-exam');
    }
    window.close();
  }, []);

  const handleJoinMeet = () => {
    setIsInWaitingArea(false);
  };


  const submitAnswer = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/evaluate_answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId.current,
          examName,
          question: aiQuestion,
          answer: userAnswer,
          chat_history: chatMessages,
        }),
      });
      const result = await response.json();
      if (result.success) {
        setAiResponse({ evaluation: result.evaluation, score: result.score }); // Store evaluation and score
        setNextQuestion(result.nextQuestion || ''); // Store next question
        setChatMessages((prev) => [
          ...prev,
          { message: `Question: ${aiQuestion}`, timestamp: new Date().toLocaleTimeString() },
          { message: `Answer: ${userAnswer}`, timestamp: new Date().toLocaleTimeString() },
          { message: `Evaluation: ${result.evaluation}, Score: ${result.score}/10`, timestamp: new Date().toLocaleTimeString() },
        ]);
      } else {
        console.error('Failed to evaluate answer:', result.error);
      }
    } catch (error) {
      console.error('Error evaluating answer:', error);
    }
  };
  const fetchNextQuestion = async () => {
    try {
      setAiResponse(null); // Clear previous evaluation and score
      setAiQuestion(nextQuestion || ''); // Set the next question if available
      setNextQuestion(''); // Clear stored next question
      setUserAnswer(''); // Clear user answer
      if (!nextQuestion) {
        // Fetch a new question if no nextQuestion is available
        const response = await fetch('http://localhost:5000/api/get_ai_question', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId.current, examName, chat_history: chatMessages }),
        });
        const result = await response.json();
        if (result.success) {
          setAiQuestion(result.question);
        } else {
          console.error('Failed to fetch next AI question:', result.error);
        }
      }
    } catch (error) {
      console.error('Error fetching next AI question:', error);
    }
  };

  if (
    showLogin) {
    return (
      <div className="login-screen">
        <h2>Login to Interview</h2>
        <input
          type="email"
          placeholder="Enter your email"
          value={userEmail}
          onChange={(e) => setUserEmail(e.target.value)}
        />
        <input
          type="text"
          placeholder="Enter exam name"
          value={examName}
          onChange={(e) => setExamName(e.target.value)}
        />
        {loginError && <p className="error">{loginError}</p>}
      </div>
    );
  }

  if (backendStatus === 'checking') {
    return (
      <div className="loading-screen">
        <div className="spinner"></div>
        <h2>Checking backend connection...</h2>
      </div>
    );
  }

  if (backendStatus === 'error') {
    return (
      <div className="error-screen">
        <h2>❌ Backend Connection Failed</h2>
        <p>Please ensure the Python server is running on localhost:5000</p>
        <button onClick={checkBackendConnection} className="retry-btn">
          🔄 Retry Connection
        </button>
      </div>
    );
  }

  if (isInWaitingArea) {
    return (
      <WaitingArea
        onJoinMeet={handleJoinMeet}
        startMonitoring={startMonitoring}
        screenGaze={privileges.screenGaze}
        multiSpeaker={privileges.multiSpeaker}
      />
    );
  }

  return (
    <div className={`app ${isFullscreen ? 'fullscreen' : ''}`}>
      <MalpracticePopup
        showMalpracticePopup={showMalpracticePopup}
        setShowMalpracticePopup={setShowMalpracticePopup}
        malpracticeMessage={malpracticeMessage}
      />
      <Header
        currentStatus={currentStatus}
        isMonitoring={isMonitoring}
        timeLeft={timeLeft}
        handleExit={handleExit}
      />
      <main className="app-main">
        <div className="meet-container">
          <VideoSection
            videoRef={videoRef}
            canvasRef={canvasRef}
            isMonitoring={isMonitoring}
            isRecording={isRecording}
          />
          <MainMeetArea
            isMonitoring={isMonitoring}
            startMonitoring={startMonitoring}
            stopMonitoring={stopMonitoring}
            backendStatus={backendStatus}
            audioStatus={audioStatus}
            isFullscreen={isFullscreen}
            warningCount={warningCount}
            violations={violations}
            sessionMode={sessionMode}
            aiQuestion={aiQuestion}
            userAnswer={userAnswer}
            setUserAnswer={setUserAnswer}
            aiResponse={aiResponse}
            submitAnswer={submitAnswer}
            fetchNextQuestion={fetchNextQuestion}
          />
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default App;
