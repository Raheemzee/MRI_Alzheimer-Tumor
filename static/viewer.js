// Global variables
let views = {};
let aiData = null;
let alzData = null;
let lastReport = "";
let playing = false;
let animationFrameId = null;
let idx = { axial: 0, sagittal: 0, coronal: 0 };

// --------------------
// UPLOAD FUNCTION
// --------------------
function upload() {
    let files = document.getElementById("fileInput").files;
    if (files.length === 0) {
        showNotification("Please select at least one file", "warning");
        return;
    }
    
    let formData = new FormData();
    for (let f of files) {
        formData.append("files", f);
    }
    
    showNotification("Analyzing brain MRI...", "info");
    
    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            showNotification(data.error, "error");
            return;
        }
        
        views = data;
        aiData = data.tumor;
        alzData = data.alzheimers;
        
        // Reset indices
        idx = { axial: 0, sagittal: 0, coronal: 0 };
        
        updateAI();
        updateAlzheimersPanel();
        renderAll();
        updateSliceCounters();
        
        showNotification("Analysis complete!", "success");
    })
    .catch(err => {
        showNotification("Error analyzing study: " + err.message, "error");
        console.error(err);
    });
}

// --------------------
// NOTIFICATION SYSTEM
// --------------------
function showNotification(message, type) {
    const reportBox = document.getElementById("reportBox");
    const icons = {
        success: "✅",
        error: "❌",
        warning: "⚠️",
        info: "ℹ️"
    };
    const colors = {
        success: "text-green-400",
        error: "text-red-400",
        warning: "text-yellow-400",
        info: "text-cyan-400"
    };
    reportBox.innerHTML = `<i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'} mr-2 ${colors[type]}"></i>${icons[type]} ${message}`;
    setTimeout(() => {
        if (!lastReport && reportBox.innerHTML.includes(message)) {
            reportBox.innerHTML = '<i class="fas fa-info-circle text-cyan-400 mr-2"></i>Ready for analysis. Upload brain MRI to begin.<div class="mt-2 text-gray-500"><i class="fas fa-lightbulb"></i> Tip: Upload FLAIR sequences for best Alzheimer\'s detection</div>';
        }
    }, 3000);
}

// --------------------
// UPDATE ALZHEIMER'S PANEL
// --------------------
function updateAlzheimersPanel() {
    if (!alzData) return;
    
    // Risk score and level
    document.getElementById("alzScore").innerText = alzData.atrophy_score + "%";
    const riskBadge = document.getElementById("alzRiskBadge");
    riskBadge.innerText = alzData.risk_level;
    
    // Set color for risk level
    if (alzData.risk_level === "High") {
        riskBadge.className = "text-xs px-2 py-1 rounded-full bg-red-900/50 text-red-300 font-mono";
    } else if (alzData.risk_level === "Moderate") {
        riskBadge.className = "text-xs px-2 py-1 rounded-full bg-yellow-900/50 text-yellow-300 font-mono";
    } else {
        riskBadge.className = "text-xs px-2 py-1 rounded-full bg-green-900/50 text-green-300 font-mono";
    }
    
    // Hippocampal volume
    const hippoStatus = alzData.biomarkers.hippocampal_atrophy ? '⚠️' : '✓';
    document.getElementById("hippocampalVol").innerHTML = `${alzData.hippocampal_volume_mm3.toLocaleString()} mm³ ${hippoStatus}`;
    
    // WMH
    const wmhStatus = alzData.biomarkers.white_matter_disease ? '⚠️' : '✓';
    document.getElementById("wmhCount").innerHTML = `${alzData.wmh_count} lesions (${alzData.wmh_volume_mm3.toLocaleString()} mm³) ${wmhStatus}`;
    
    // VBR
    document.getElementById("vbr").innerHTML = alzData.vbr + "%";
}

// --------------------
// UPDATE TUMOR PANEL
// --------------------
function updateAI() {
    if (!aiData) return;
    
    const statusBadge = document.getElementById("tumorStatusBadge");
    if (aiData.detected) {
        statusBadge.innerHTML = '<span class="status-badge status-critical"></span>Detected';
        statusBadge.className = "text-xs px-2 py-1 rounded-full bg-red-900/50 text-red-300 font-mono";
    } else {
        statusBadge.innerHTML = '<span class="status-badge status-normal"></span>Not Detected';
        statusBadge.className = "text-xs px-2 py-1 rounded-full bg-green-900/50 text-green-300 font-mono";
    }
    
    document.getElementById("tumorVolume").innerHTML = `<span class="text-xl font-bold text-white">${aiData.estimated_volume_cm3}</span> <span class="text-xs text-gray-400">cm³</span>`;
    document.getElementById("tumorConfidence").innerHTML = `${aiData.confidence}%`;
    document.getElementById("largestSlice").innerHTML = `Slice ${aiData.largest_slice}`;
    document.getElementById("tumorPixels").innerHTML = aiData.pixels.toLocaleString();
}

// --------------------
// SLICE COUNTERS
// --------------------
function updateSliceCounters() {
    if (!views.axial) return;
    
    document.getElementById("axialIndex").innerHTML = `<i class="fas fa-layer-group mr-1"></i>Slice ${idx.axial + 1} / ${views.axial.plain.length}`;
    document.getElementById("sagittalIndex").innerHTML = `<i class="fas fa-layer-group mr-1"></i>Slice ${idx.sagittal + 1} / ${views.sagittal.plain.length}`;
    document.getElementById("coronalIndex").innerHTML = `<i class="fas fa-layer-group mr-1"></i>Slice ${idx.coronal + 1} / ${views.coronal.plain.length}`;
}

// --------------------
// GET SLICE DATA
// --------------------
function getSlice(v) {
    let showSeg = document.getElementById("toggleSeg").checked;
    let showAlz = document.getElementById("toggleAlzheimers")?.checked;
    
    if (showAlz && views[v].alzheimers_overlay && views[v].alzheimers_overlay[idx[v]]) {
        return views[v].alzheimers_overlay[idx[v]];
    } else if (showSeg) {
        return views[v].overlay[idx[v]];
    } else {
        return views[v].plain[idx[v]];
    }
}

// --------------------
// DRAW CANVAS
// --------------------
function drawCanvas(id, src) {
    let c = document.getElementById(id);
    if (!c) return;
    
    let ctx = c.getContext("2d");
    
    let img = new Image();
    img.onload = () => {
        c.width = img.width;
        c.height = img.height;
        const brightness = document.getElementById("brightness").value;
        ctx.filter = `brightness(${brightness}%)`;
        ctx.drawImage(img, 0, 0);
    };
    img.onerror = () => {
        console.error(`Failed to load image for ${id}`);
    };
    img.src = src;
}

// --------------------
// RENDER ALL VIEWS
// --------------------
function renderAll() {
    if (!views.axial) return;
    
    drawCanvas("axialCanvas", getSlice("axial"));
    drawCanvas("sagittalCanvas", getSlice("sagittal"));
    drawCanvas("coronalCanvas", getSlice("coronal"));
    
    updateSliceCounters();
}

// --------------------
// NEXT SLICE FUNCTION
// --------------------
function nextSlice() {
    if (!views.axial) return;
    
    ["axial", "sagittal", "coronal"].forEach(v => {
        idx[v]++;
        if (idx[v] >= views[v].plain.length) {
            idx[v] = 0;
        }
    });
    
    renderAll();
}

// --------------------
// PREVIOUS SLICE FUNCTION
// --------------------
function previousSlice() {
    if (!views.axial) return;
    
    ["axial", "sagittal", "coronal"].forEach(v => {
        idx[v]--;
        if (idx[v] < 0) {
            idx[v] = views[v].plain.length - 1;
        }
    });
    
    renderAll();
}

// --------------------
// CINE LOOP FUNCTIONS
// --------------------
function togglePlay() {
    if (!views.axial) {
        showNotification("Please load a study first", "warning");
        return;
    }
    
    playing = !playing;
    const playBtn = document.getElementById("playText");
    const playIcon = document.getElementById("playIcon");
    
    if (playing) {
        if (playBtn) playBtn.innerText = 'Pause';
        if (playIcon) {
            playIcon.className = 'fas fa-pause';
        }
        startCineLoop();
    } else {
        if (playBtn) playBtn.innerText = 'Play Cine Loop';
        if (playIcon) {
            playIcon.className = 'fas fa-play';
        }
        stopCineLoop();
    }
}

function startCineLoop() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    
    let lastTimestamp = 0;
    
    function animate(currentTime) {
        if (!playing) return;
        
        let speed = parseInt(document.getElementById("speed").value);
        // Convert speed (0-200) to delay (300ms to 50ms)
        let delay = Math.max(50, 300 - speed);
        
        if (currentTime - lastTimestamp >= delay) {
            nextSlice();
            lastTimestamp = currentTime;
        }
        
        animationFrameId = requestAnimationFrame(animate);
    }
    
    animationFrameId = requestAnimationFrame(animate);
}

function stopCineLoop() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// --------------------
// KEYBOARD CONTROLS
// --------------------
document.addEventListener('keydown', function(e) {
    if (!views.axial) return;
    
    // Prevent default scrolling with arrow keys
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'ArrowDown' || e.key === ' ' || e.key === 'Space') {
        e.preventDefault();
    }
    
    switch(e.key) {
        case 'ArrowRight':
        case 'ArrowDown':
            nextSlice();
            break;
        case 'ArrowLeft':
        case 'ArrowUp':
            previousSlice();
            break;
        case ' ':
        case 'Space':
            togglePlay();
            break;
    }
});

// --------------------
// GENERATE REPORT
// --------------------
function generateReport() {
    if (!aiData || !alzData) {
        showNotification("Please upload and analyze a study first", "warning");
        return;
    }
    
    const date = new Date().toLocaleString();
    
    lastReport = `
╔═══════════════════════════════════════════════════════════════╗
║                 NEUROVISION AI CLINICAL REPORT                ║
║                         ${date}                               ║
╚═══════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TUMOR ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Status:              ${aiData.detected ? "⚠️  TUMOR DETECTED" : "✓  No tumor detected"}
  Estimated Volume:    ${aiData.estimated_volume_cm3} cm³
  Largest Slice:       Slice ${aiData.largest_slice}
  Segmentation Pixels: ${aiData.pixels.toLocaleString()}
  Confidence:          ${aiData.confidence}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ALZHEIMER'S DISEASE ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Risk Score:          ${alzData.atrophy_score}%
  Risk Level:          ${alzData.risk_level.toUpperCase()}
  
  Key Biomarkers:
  • Hippocampal Volume:    ${alzData.hippocampal_volume_mm3.toLocaleString()} mm³ ${alzData.biomarkers.hippocampal_atrophy ? "(ATROPHY DETECTED)" : "(Normal range)"}
  • Ventricle Volume:      ${alzData.ventricle_volume_mm3.toLocaleString()} mm³ ${alzData.biomarkers.ventricle_enlargement ? "(ENLARGED)" : "(Normal range)"}
  • White Matter Lesions:  ${alzData.wmh_count} lesions (${alzData.wmh_volume_mm3.toLocaleString()} mm³) ${alzData.biomarkers.white_matter_disease ? "(ELEVATED)" : "(Normal range)"}
  • Ventricle/Brain Ratio: ${alzData.vbr}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CLINICAL IMPRESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ${alzData.risk_level === "High" ? "🔴 HIGH LIKELIHOOD of Alzheimer's disease pathology. Multiple abnormal biomarkers detected. Immediate clinical correlation recommended." :
    alzData.risk_level === "Moderate" ? "🟡 MODERATE RISK for Alzheimer's disease. Further clinical evaluation and longitudinal follow-up recommended." :
    "🟢 LOW RISK for Alzheimer's disease based on current imaging biomarkers."}

  ${aiData.detected ? "🔴 INTRACRANIAL LESION detected. Neurosurgical consultation and further characterization recommended." : "✓ No significant intracranial mass lesion detected."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ${alzData.risk_level !== "Low" ? "1. Neurological consultation for cognitive assessment\n  2. Consider neuropsychological testing\n  3. Follow-up imaging in 6-12 months to assess progression" : "1. Routine clinical follow-up as indicated\n  2. Monitor for any cognitive changes"}

  ${aiData.detected ? "\n  ⚠️  Immediate neurosurgical referral recommended for lesion characterization" : ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  This is an automated analysis using artificial intelligence.
  All findings should be reviewed by a qualified radiologist
  or neurologist before making clinical decisions.

  Report generated by NeuroVision AI v2.0
╔═══════════════════════════════════════════════════════════════╗
║                   END OF CLINICAL REPORT                      ║
╚═══════════════════════════════════════════════════════════════╝
`;
    
    const reportBox = document.getElementById("reportBox");
    reportBox.innerHTML = `<pre class="font-mono text-xs" style="white-space: pre-wrap; overflow-x: auto;">${lastReport}</pre>`;
    showNotification("Clinical report generated", "success");
}

// --------------------
// DOWNLOAD REPORT
// --------------------
function downloadReport() {
    if (!lastReport) {
        showNotification("Please generate a report first", "warning");
        return;
    }
    
    const blob = new Blob([lastReport], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `NeuroVision_Report_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    showNotification("Report downloaded", "success");
}

// --------------------
// MOUSE MOVE FOR CROSSHAIRS
// --------------------
document.querySelectorAll('.viewer').forEach(viewer => {
    viewer.addEventListener('mousemove', (e) => {
        const rect = viewer.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        
        const crosshairH = viewer.querySelector('.crosshairH');
        const crosshairV = viewer.querySelector('.crosshairV');
        
        if (crosshairH) crosshairH.style.top = `${y}%`;
        if (crosshairV) crosshairV.style.left = `${x}%`;
    });
});

// --------------------
// INITIALIZATION
// --------------------
console.log("NeuroVision AI Ready - Cine loop controls active");
console.log("Controls: Arrow keys to navigate, Space to play/pause");