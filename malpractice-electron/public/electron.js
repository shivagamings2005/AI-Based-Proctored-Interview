const { app, BrowserWindow, ipcMain, globalShortcut } = require('electron');
const path = require('path');

const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

let mainWindow;
let isExamMode = false;
let registeredShortcuts = [];
let windowWatcher = null; // For polling window state

// List of shortcuts to block during exam - only those that can be registered
const BLOCKED_SHORTCUTS = [
  // Note: Alt+Tab cannot be blocked via globalShortcut on Windows
  // It will be handled via before-input-event and window focus management
  'CommandOrControl+Tab',
  'CommandOrControl+Shift+Tab',
  'F11', // Fullscreen toggle
  'CommandOrControl+W', // Close tab/window
  'CommandOrControl+Q', // Quit application (Mac)
  'CommandOrControl+Shift+I', // DevTools
  'CommandOrControl+R', // Refresh
  'CommandOrControl+Shift+R', // Hard refresh
  'CommandOrControl+T', // New tab
  'CommandOrControl+N', // New window
  'Escape', // Exit fullscreen
  'CommandOrControl+Shift+C', // DevTools inspect
  'CommandOrControl+U', // View source
  'CommandOrControl+Shift+J', // Console
];

// Windows key combinations that need special handling
const WINDOWS_SHORTCUTS = [
  'Meta+D', // Show desktop
  'Meta+L', // Lock screen  
  'Meta+R', // Run dialog
  'Meta+E', // File explorer
  'Meta+M', // Minimize all
  'Meta+Tab', // Task view
  'Meta+Up', // Maximize
  'Meta+Down', // Minimize/Restore
  'Meta+Left', // Snap left
  'Meta+Right', // Snap right
];

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    titleBarStyle: 'hidden', // Hide title bar but keep window controls accessible
    // Alternative: use frame: false for completely frameless window
    // frame: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      preload: path.join(__dirname, 'preload.js'),
      // Disable some web security features that could be exploited
      webSecurity: true,
      allowRunningInsecureContent: false,
      experimentalFeatures: false
    },
  });

  if (isDev) {
    console.log('Development mode: Loading http://localhost:3000');
    mainWindow.loadURL('http://localhost:3000');
  } else {
    console.log('Production mode: Loading build/index.html');
    mainWindow.loadFile(path.join(__dirname, '../build/index.html'));
  }

  mainWindow.maximize();

  // Handle fullscreen events only during exam mode
  mainWindow.on('enter-full-screen', () => {
    console.log('Entered fullscreen, exam mode:', isExamMode);
  });

  mainWindow.on('leave-full-screen', () => {
    console.log('Leave fullscreen attempted, exam mode:', isExamMode);
    if (isExamMode) {
      console.log('Re-entering fullscreen due to exam mode');
      setTimeout(() => {
        if (mainWindow && isExamMode) {
          mainWindow.setFullScreen(true);
        }
      }, 100);
    }
  });

  // Block certain input events during exam mode
  mainWindow.webContents.on('before-input-event', (event, input) => {
    if (!isExamMode) return;

    const { key, alt, control, meta, shift, type } = input;
    
    // Only process keyDown events to avoid double processing
    if (type !== 'keyDown') return;
    
    // Block Alt+Tab and Alt+Shift+Tab (most important for task switching)
    if (alt && key.toLowerCase() === 'tab') {
      console.log('Blocked Alt+Tab during exam');
      event.preventDefault();
      return;
    }

    // Block ALL Windows key (Meta key) combinations
    if (meta) {
      console.log('Blocked Windows key combination during exam:', key);
      event.preventDefault();
      return;
    }

    // Block Ctrl+Tab (browser tab switching)
    if (control && key.toLowerCase() === 'tab') {
      console.log('Blocked Ctrl+Tab during exam');
      event.preventDefault();
      return;
    }

    // Block F11 (fullscreen toggle)
    if (key === 'F11') {
      console.log('Blocked F11 during exam');
      event.preventDefault();
      return;
    }

    // Block Escape (exit fullscreen)
    if (key === 'Escape') {
      console.log('Blocked Escape during exam');
      event.preventDefault();
      return;
    }

    // Block Alt+F4 (close window)
    if (alt && key === 'F4') {
      console.log('Blocked Alt+F4 during exam');
      event.preventDefault();
      return;
    }

    // Block DevTools shortcuts
    if ((control || meta) && key.toLowerCase() === 'i' && shift) {
      console.log('Blocked DevTools shortcut during exam');
      event.preventDefault();
      return;
    }

    if (key === 'F12') {
      console.log('Blocked F12 DevTools during exam');
      event.preventDefault();
      return;
    }

    // Block refresh shortcuts
    if ((control || meta) && key.toLowerCase() === 'r') {
      console.log('Blocked refresh during exam');
      event.preventDefault();
      return;
    }

    // Block view source
    if ((control || meta) && key.toLowerCase() === 'u') {
      console.log('Blocked view source during exam');
      event.preventDefault();
      return;
    }

    // Block console shortcuts
    if ((control || meta) && shift && key.toLowerCase() === 'j') {
      console.log('Blocked console shortcut during exam');
      event.preventDefault();
      return;
    }
  });

  // Disable context menu during exam mode
  mainWindow.webContents.on('context-menu', (event) => {
    if (isExamMode) {
      console.log('Blocked context menu during exam');
      event.preventDefault();
    }
  });

  // Prevent DevTools from opening during exam
  mainWindow.webContents.on('devtools-opened', () => {
    if (isExamMode) {
      console.log('Closing DevTools during exam');
      mainWindow.webContents.closeDevTools();
    }
  });

  mainWindow.on('closed', () => {
    console.log('Main window closed');
    // Only call stopExamMode if it hasn't been called yet
    if (isExamMode) {
      stopExamMode();
    }
    mainWindow = null;
  });
}

function startExamMode() {
  console.log('Starting exam mode - blocking shortcuts and enabling fullscreen');
  isExamMode = true;
  
  if (mainWindow) {
    // Set window to be always on top and focused
    mainWindow.setAlwaysOnTop(true);
    mainWindow.focus();
    
    // Enter fullscreen
    mainWindow.setFullScreen(true);
    
    // Register global shortcuts to block them (only those that can be registered)
    BLOCKED_SHORTCUTS.forEach(shortcut => {
      try {
        const success = globalShortcut.register(shortcut, () => {
          console.log(`Blocked shortcut during exam: ${shortcut}`);
          // Refocus the main window when blocked shortcuts are attempted
          if (mainWindow) {
            mainWindow.focus();
            mainWindow.setFullScreen(true);
          }
        });
        
        if (success) {
          registeredShortcuts.push(shortcut);
        } else {
          console.log(`Could not register shortcut: ${shortcut} (may be system-reserved)`);
        }
      } catch (error) {
        console.log(`Error registering shortcut ${shortcut}:`, error.message);
      }
    });

    // Try to register some Windows key combinations with Meta syntax
    WINDOWS_SHORTCUTS.forEach(shortcut => {
      try {
        const success = globalShortcut.register(shortcut, () => {
          console.log(`Blocked Windows shortcut during exam: ${shortcut}`);
          if (mainWindow) {
            mainWindow.focus();
            mainWindow.setFullScreen(true);
          }
        });
        
        if (success) {
          registeredShortcuts.push(shortcut);
        }
      } catch (error) {
        // Windows shortcuts often can't be globally registered, that's normal
        console.log(`Windows shortcut ${shortcut} not registrable (handled by input events)`);
      }
    });

    console.log(`Successfully registered ${registeredShortcuts.length} shortcuts for blocking`);
    
    // Prevent window from being minimized during exam
    mainWindow.on('minimize', () => {
      if (isExamMode) {
        console.log('Window minimize blocked during exam - restoring and refocusing');
        mainWindow.restore();
        mainWindow.focus();
        mainWindow.setFullScreen(true);
        mainWindow.setAlwaysOnTop(true);
      }
    });
    
    // Additional focus management
    mainWindow.on('blur', () => {
      if (isExamMode) {
        console.log('Window lost focus during exam, refocusing...');
        setTimeout(() => {
          if (mainWindow && isExamMode && !mainWindow.isDestroyed()) {
            mainWindow.focus();
            mainWindow.setFullScreen(true);
            mainWindow.setAlwaysOnTop(true);
          }
        }, 100);
      }
    });

    // Handle show/hide events (for Show Desktop functionality)
    mainWindow.on('hide', () => {
      if (isExamMode) {
        console.log('Window hide blocked during exam - showing and refocusing');
        mainWindow.show();
        mainWindow.focus();
        mainWindow.setFullScreen(true);
        mainWindow.setAlwaysOnTop(true);
      }
    });

    // Prevent window state changes during exam
    mainWindow.on('restore', () => {
      if (isExamMode) {
        console.log('Ensuring fullscreen mode during exam');
        setTimeout(() => {
          if (mainWindow && isExamMode && !mainWindow.isDestroyed()) {
            mainWindow.setFullScreen(true);
            mainWindow.setAlwaysOnTop(true);
          }
        }, 50);
      }
    });

    // Aggressive window state monitoring to counter 3-finger gestures
    if (windowWatcher) {
      clearInterval(windowWatcher);
    }
    
    windowWatcher = setInterval(() => {
      if (isExamMode && mainWindow && !mainWindow.isDestroyed()) {
        try {
          // Check if window is minimized or hidden and restore it
          if (mainWindow.isMinimized()) {
            console.log('Window was minimized - restoring immediately');
            mainWindow.restore();
            mainWindow.focus();
            mainWindow.setFullScreen(true);
            mainWindow.setAlwaysOnTop(true);
          }
          
          // Ensure window is always visible and focused
          if (!mainWindow.isVisible()) {
            console.log('Window was hidden - showing immediately');
            mainWindow.show();
            mainWindow.focus();
            mainWindow.setFullScreen(true);
            mainWindow.setAlwaysOnTop(true);
          }
          
          // Ensure fullscreen mode
          if (!mainWindow.isFullScreen()) {
            console.log('Window lost fullscreen - restoring fullscreen');
            mainWindow.setFullScreen(true);
          }
          
          // Ensure always on top
          if (!mainWindow.isAlwaysOnTop()) {
            console.log('Window lost always-on-top - restoring');
            mainWindow.setAlwaysOnTop(true);
          }
        } catch (error) {
          console.log('Error in window watcher:', error.message);
        }
      }
    }, 250); // Check every 250ms for aggressive protection
  }
}

function stopExamMode() {
  // Prevent multiple calls
  if (!isExamMode) {
    console.log('Exam mode already stopped, skipping cleanup');
    return;
  }
  
  console.log('Stopping exam mode - unblocking shortcuts and exiting fullscreen');
  isExamMode = false;
  
  // Stop the window watcher
  if (windowWatcher) {
    clearInterval(windowWatcher);
    windowWatcher = null;
    console.log('Window watcher stopped');
  }
  
  // Safely handle window operations
  if (mainWindow && !mainWindow.isDestroyed()) {
    try {
      // Remove always on top
      mainWindow.setAlwaysOnTop(false);
      
      // Exit fullscreen
      mainWindow.setFullScreen(false);
      
      // Remove all event listeners we added
      mainWindow.removeAllListeners('blur');
      mainWindow.removeAllListeners('minimize');
      mainWindow.removeAllListeners('hide');
      mainWindow.removeAllListeners('restore');
    } catch (error) {
      console.log('Error during window cleanup:', error.message);
    }
  }
  
  // Unregister all blocked shortcuts
  registeredShortcuts.forEach(shortcut => {
    try {
      globalShortcut.unregister(shortcut);
    } catch (error) {
      console.log(`Error unregistering shortcut ${shortcut}:`, error.message);
    }
  });
  
  registeredShortcuts = [];
  console.log('All exam mode shortcuts unregistered and normal window behavior restored');
}

app.whenReady().then(() => {
  console.log('App ready, creating window');
  createWindow();
  
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      console.log('Activating new window');
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  console.log('All windows closed');
  // Only call stopExamMode if it hasn't been called yet
  if (isExamMode) {
    stopExamMode();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  console.log('App will quit - cleaning up shortcuts');
  // Only call stopExamMode if it hasn't been called yet
  if (isExamMode) {
    stopExamMode();
  }
});

// In electron.js, add this IPC handler inside the app.whenReady() block or below existing IPC handlers:
// In electron.js, add these IPC handlers (if not already present):
ipcMain.on('show-popup', () => {
  console.log('Received show-popup IPC - adjusting window for popup');
  if (mainWindow && !mainWindow.isDestroyed() && isExamMode) {
    try {
      // Temporarily disable always-on-top to allow popup rendering
      mainWindow.setAlwaysOnTop(false);
      // Ensure window is focused
      mainWindow.focus();
    } catch (error) {
      console.log('Error adjusting window for popup:', error.message);
    }
  }
});

ipcMain.on('hide-popup', () => {
  console.log('Received hide-popup IPC - restoring exam mode');
  if (mainWindow && !mainWindow.isDestroyed() && isExamMode) {
    try {
      // Restore always-on-top and fullscreen
      mainWindow.setAlwaysOnTop(true);
      mainWindow.setFullScreen(true);
      mainWindow.focus();
    } catch (error) {
      console.log('Error restoring exam mode after popup:', error.message);
    }
  }
});

// IPC handlers for exam mode control
ipcMain.on('start-monitoring', () => {
  console.log('Received start-monitoring IPC - entering exam mode');
  startExamMode();
});

ipcMain.on('stop-monitoring', () => {
  console.log('Received stop-monitoring IPC - exiting exam mode');
  stopExamMode();
});

// Additional IPC handlers for exam control
ipcMain.on('start-exam', () => {
  console.log('Received start-exam IPC');
  startExamMode();
});

ipcMain.on('end-exam', () => {
  console.log('Received end-exam IPC');
  stopExamMode();
});

// Handle emergency exit (for development/testing)
ipcMain.on('emergency-exit', () => {
  console.log('Emergency exit requested');
  stopExamMode();
  setTimeout(() => {
    app.quit();
  }, 500);
});

// Export functions for testing (if needed)
module.exports = {
  startExamMode,
  stopExamMode,
  isExamMode: () => isExamMode
};
//npm run electron:start