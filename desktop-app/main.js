const { app, BrowserWindow, Menu, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let flaskProcess;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        },
        icon: path.join(__dirname, 'icon.png'),
        title: 'NeuroVision AI - Advanced Brain MRI Analysis'
    });

    // Start Flask backend
    startFlaskBackend();

    // Load the app
    setTimeout(() => {
        mainWindow.loadURL('http://localhost:5000');
    }, 3000);

    // Create custom menu
    const menu = Menu.buildFromTemplate([
        {
            label: 'File',
            submenu: [
                {
                    label: 'Open Study',
                    accelerator: 'CmdOrCtrl+O',
                    click: () => {
                        dialog.showOpenDialog({
                            properties: ['openFile', 'multiSelections'],
                            filters: [
                                { name: 'NIfTI Files', extensions: ['nii', 'nii.gz'] },
                                { name: 'All Files', extensions: ['*'] }
                            ]
                        }).then(result => {
                            if (!result.canceled && mainWindow) {
                                mainWindow.webContents.send('load-files', result.filePaths);
                            }
                        });
                    }
                },
                { type: 'separator' },
                {
                    label: 'Export Report',
                    accelerator: 'CmdOrCtrl+E',
                    click: () => {
                        mainWindow.webContents.send('export-report');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Exit',
                    accelerator: 'CmdOrCtrl+Q',
                    click: () => {
                        app.quit();
                    }
                }
            ]
        },
        {
            label: 'View',
            submenu: [
                { label: 'Toggle Full Screen', role: 'togglefullscreen' },
                { label: 'Zoom In', role: 'zoomin' },
                { label: 'Zoom Out', role: 'zoomout' },
                { label: 'Reset Zoom', role: 'resetzoom' }
            ]
        },
        {
            label: 'Tools',
            submenu: [
                {
                    label: 'Developer Tools',
                    accelerator: 'CmdOrCtrl+I',
                    click: () => {
                        mainWindow.webContents.openDevTools();
                    }
                },
                {
                    label: 'Clear Cache',
                    click: () => {
                        mainWindow.webContents.session.clearCache();
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            message: 'Cache cleared successfully!'
                        });
                    }
                }
            ]
        },
        {
            label: 'Help',
            submenu: [
                {
                    label: 'Documentation',
                    click: () => {
                        require('electron').shell.openExternal('https://github.com/your-repo/neurovision');
                    }
                },
                {
                    label: 'About',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'About NeuroVision AI',
                            message: 'NeuroVision AI v2.0\n\nAdvanced Brain MRI Analysis Platform\n\n© 2024 NeuroVision AI Team',
                            buttons: ['OK']
                        });
                    }
                }
            ]
        }
    ]);

    Menu.setApplicationMenu(menu);

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

function startFlaskBackend() {
    const pythonPath = process.platform === 'win32' ? 'python' : 'python3';
    
    flaskProcess = spawn(pythonPath, ['app.py'], {
        cwd: path.join(__dirname, '..'), // Parent directory where app.py is
        stdio: 'pipe'
    });

    flaskProcess.stdout.on('data', (data) => {
        console.log(`Flask: ${data}`);
    });

    flaskProcess.stderr.on('data', (data) => {
        console.error(`Flask Error: ${data}`);
    });
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (flaskProcess) {
        flaskProcess.kill();
    }
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// IPC handlers for native features
ipcMain.handle('save-file', async (event, { content, filename }) => {
    const { filePath } = await dialog.showSaveDialog({
        defaultPath: filename,
        filters: [
            { name: 'Text Files', extensions: ['txt'] },
            { name: 'PDF Files', extensions: ['pdf'] },
            { name: 'All Files', extensions: ['*'] }
        ]
    });
    
    if (filePath) {
        fs.writeFileSync(filePath, content);
        return { success: true, path: filePath };
    }
    return { success: false };
});

ipcMain.handle('show-notification', (event, { title, message }) => {
    dialog.showMessageBox({
        type: 'info',
        title: title,
        message: message
    });
});