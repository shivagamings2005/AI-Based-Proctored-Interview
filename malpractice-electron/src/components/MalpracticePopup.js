import './MalpracticePopup.css'; 
const MalpracticePopup = ({ showMalpracticePopup, setShowMalpracticePopup, malpracticeMessage }) => {
  const handleClose = () => {
    setShowMalpracticePopup(false);
    // Notify Electron to restore exam mode
    if (window.electron) {
      console.log('Sending hide-popup IPC to Electron');
      window.electron.send('hide-popup');
    }
    // Resolve the Promise
    if (window.malpracticePopupResolve) {
      console.log('Resolving malpractice popup Promise');
      window.malpracticePopupResolve();
      delete window.malpracticePopupResolve; // Clean up
    }
  };

  return (
    showMalpracticePopup && (
      <div className="malpractice-popup">
        <div className="popup-content">
          <h2>🚨 MALPRACTICE DETECTED</h2>
          <p>{malpracticeMessage}</p>
          <p>This incident has been recorded.</p>
          <button
            onClick={handleClose}
            className="popup-close-btn"
          >
            Acknowledge
          </button>
        </div>
      </div>
    )
  );
};

export default MalpracticePopup;