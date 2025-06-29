import './ControlPanel.css';
const ControlPanel = ({ isMonitoring, startMonitoring, stopMonitoring, backendStatus, audioStatus, isFullscreen, warningCount }) => {

  return (
    <div className="control-panel">
      <div className="control-section">
        <h3>🎛️ Control Panel</h3>
        <div className="monitoring-info">
          <div className="info-item">
            <span className="info-icon">🔍</span>
            <span>Gaze Tracking: {isMonitoring ? 'Active' : 'Inactive'}</span>
          </div>
          <div className="info-item">
            <span className="info-icon">🎤</span>
            <span>Audio Analysis: {audioStatus}</span>
          </div>
          <div className="info-item">
            <span className="info-icon">🖥️</span>
            <span>Fullscreen: {isFullscreen ? 'Enabled' : 'Disabled'}</span>
          </div>
          <div className="info-item">
            <span className="info-icon">🌐</span>
            <span>Backend: {backendStatus === 'connected' ? 'Connected' : 'Disconnected'}</span>
          </div>
          <div className="info-item">
            <span className="info-icon">⚠️</span>
            <span>Warnings: {warningCount}/5</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;