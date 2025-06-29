import './Header.css';

const Header = ({ currentStatus, isMonitoring, timeLeft, handleExit }) => {
  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <header className="app-header">
      <div className="header-left">
        <h1>🎯 Interview Monitoring System</h1>
        <div className="status-indicator">
          Status: <span className={`status ${isMonitoring ? 'active' : 'inactive'}`}>
            {currentStatus}
          </span>
        </div>
      </div>
      <div className="header-right">
        <div className="timer">
          Time Left: {formatTime(timeLeft)}
        </div>
        <button
          onClick={handleExit}
          className="exit-btn"
          title="Exit Interview"
        >
          🚪 Exit
        </button>
      </div>
    </header>
  );
};

export default Header;