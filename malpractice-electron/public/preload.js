const { contextBridge, ipcRenderer } = require('electron');

// Expose safe IPC methods to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  send: (channel, data) => {
    // Whitelist safe channels
    const validChannels = [
      'start-monitoring',
      'stop-monitoring',
      'start-exam',
      'end-exam',
      'emergency-exit',
      'show-popup', // Add new channel
      'hide-popup', // Add new channel
    ];
    if (validChannels.includes(channel)) {
      console.log(`Preload: Sending IPC message on channel ${channel}`);
      ipcRenderer.send(channel, data);
    } else {
      console.warn(`Preload: Blocked unsafe IPC channel ${channel}`);
    }
  },
  // Exam control methods
  startExam: () => {
    console.log('Preload: Starting exam mode');
    ipcRenderer.send('start-exam');
  },
  
  endExam: () => {
    console.log('Preload: Ending exam mode');
    ipcRenderer.send('end-exam');
  },
  
  // Legacy methods for compatibility
  startMonitoring: () => {
    console.log('Preload: Starting monitoring (legacy)');
    ipcRenderer.send('start-monitoring');
  },
  
  stopMonitoring: () => {
    console.log('Preload: Stopping monitoring (legacy)');
    ipcRenderer.send('stop-monitoring');
  },
  
  // Emergency exit for development/testing
  emergencyExit: () => {
    console.log('Preload: Emergency exit requested');
    ipcRenderer.send('emergency-exit');
  },
  
  // Safe exit method
  exitApp: () => {
    console.log('Preload: Safe exit requested');
    ipcRenderer.send('end-exam');
    // Small delay to ensure cleanup happens before closing
    setTimeout(() => {
      ipcRenderer.send('emergency-exit');
    }, 500);
  },
  
  // Listen for exam status updates (if needed)
  onExamStatusChange: (callback) => {
    ipcRenderer.on('exam-status-changed', (event, status) => {
      callback(status);
    });
  },
  
  // Remove listeners
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  }
});

// Block some dangerous APIs in the renderer process
window.addEventListener('DOMContentLoaded', () => {
  // Disable right-click context menu during exam
    document.addEventListener('contextmenu', (e) => {
    // You can add exam mode check here if needed
    e.preventDefault();
  });
  
  // Disable text selection during exam (optional)
  document.addEventListener('selectstart', (e) => {
    // You can add exam mode check here if needed
    // e.preventDefault();
  });
  
  // Disable drag and drop (optional)
  document.addEventListener('dragstart', (e) => {
    e.preventDefault();
  });
  
  document.addEventListener('drop', (e) => {
    e.preventDefault();
  });
  
  // Block common keyboard shortcuts in the web page
  document.addEventListener('keydown', (e) => {
    // Block Ctrl+U (view source)
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'u') {
      e.preventDefault();
      return false;
    }
    
    // Block Ctrl+Shift+J (console)
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'j') {
      e.preventDefault();
      return false;
    }
    
    // Block Ctrl+Shift+C (inspect element)
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'c') {
      e.preventDefault();
      return false;
    }
    
    // You can add more shortcuts to block here
  });
});