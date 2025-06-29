import './VideoSection.css';
const VideoSection = ({ videoRef, canvasRef, isMonitoring, isRecording }) => {
  return (
    <div className="video-section">
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="video-feed"
        />
        <canvas
          ref={canvasRef}
          style={{ display: 'none' }}
        />
        {!isMonitoring && (
          <div className="video-overlay">
            <h3>📹 Camera Feed</h3>
            <p>Click "Start Monitoring" to begin</p>
          </div>
        )}
        {isRecording && (
          <div className="recording-indicator">
            <div className="recording-dot"></div>
            <span>Recording</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoSection;