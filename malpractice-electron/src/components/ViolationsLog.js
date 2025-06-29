import './ViolationsLog.css';
const ViolationsLog = ({ violations }) => {
  return (
    <div className="violations-section">
      <h3>⚠️ Violations Log ({violations.length})</h3>
      <div className="violations-list">
        {violations.length === 0 ? (
          <p className="no-violations">No violations detected</p>
        ) : (
          violations.slice(-5).map((violation, index) => (
            <div key={index} className="violation-item">
              <span className="violation-time">{violation.timestamp}</span>
              <span className="violation-text">{violation.violation}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default ViolationsLog;