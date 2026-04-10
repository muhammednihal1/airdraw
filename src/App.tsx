import { useEffect, useRef, useState, useCallback } from 'react';
import { Camera, CameraOff, Download, Trash2, Undo2, GripHorizontal } from 'lucide-react';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import './App.css';

// Canvas can't resolve CSS variables, so use raw hex values
const COLORS = [
  { name: 'cyan', value: '#00f0ff' },
  { name: 'magenta', value: '#ff00e5' },
  { name: 'lime', value: '#39ff14' },
  { name: 'blue', value: '#4d6dff' },
  { name: 'pink', value: '#ff2d6b' },
  { name: 'gold', value: '#ffd700' },
  { name: 'purple', value: '#b400ff' },
  { name: 'white', value: '#ffffff' },
];

type Point = { x: number; y: number };
type Stroke = { points: Point[]; color: string; thickness: number; glow: number };
type GestureMode = 'DRAWING' | 'ERASING' | 'IDLE' | 'THUMBSUP' | 'HEART' | 'PEACE' | 'NONE';

// Big emoji reaction (macOS FaceTime style)
type ActiveReaction = {
  id: number;
  emoji: string;
  timestamp: number;
};

// ─── Gesture Detection ───
function detectGesture(landmarks: { x: number; y: number; z: number }[]): GestureMode {
  const wrist = landmarks[0];
  const thumbTip = landmarks[4];
  const thumbIp = landmarks[3];
  const indexPip = landmarks[6];
  const indexTip = landmarks[8];
  const middlePip = landmarks[10];
  const middleTip = landmarks[12];
  const ringPip = landmarks[14];
  const ringTip = landmarks[16];
  const pinkyPip = landmarks[18];
  const pinkyTip = landmarks[20];

  const indexExtended = dist2d(indexTip, wrist) > dist2d(indexPip, wrist);
  const middleExtended = dist2d(middleTip, wrist) > dist2d(middlePip, wrist);
  const ringExtended = dist2d(ringTip, wrist) > dist2d(ringPip, wrist);
  const pinkyExtended = dist2d(pinkyTip, wrist) > dist2d(pinkyPip, wrist);

  // Thumb up: thumb extended upward, all others curled
  const thumbUp = thumbTip.y < thumbIp.y && thumbTip.y < wrist.y
    && !indexExtended && !middleExtended && !ringExtended && !pinkyExtended;
  if (thumbUp) return 'THUMBSUP';

  // Peace sign: index + middle extended, others curled
  const peace = indexExtended && middleExtended && !ringExtended && !pinkyExtended;
  if (peace) return 'PEACE';

  // Heart shape: check if index tip and thumb tip are close (forming a heart-like pinch at top)
  const pinchDist = dist2d(indexTip, thumbTip);
  const isHeart = pinchDist < 0.06 && !middleExtended && !ringExtended && !pinkyExtended
    && indexTip.y < wrist.y; // tips above wrist
  if (isHeart) return 'HEART';

  // Index pointing (draw)
  const indexPointing = indexExtended && !middleExtended && !ringExtended && !pinkyExtended;
  if (indexPointing) return 'DRAWING';

  // Open palm (erase)
  const openPalm = indexExtended && middleExtended && ringExtended && pinkyExtended;
  if (openPalm) return 'ERASING';

  // Fist (idle)
  const fist = !indexExtended && !middleExtended && !ringExtended && !pinkyExtended;
  if (fist) return 'IDLE';

  return 'NONE';
}

function dist2d(a: { x: number; y: number }, b: { x: number; y: number }) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// ─── Web Audio Synth Sounds ───
let audioCtx: AudioContext | null = null;
function getAudioCtx() {
  if (!audioCtx) audioCtx = new AudioContext();
  return audioCtx;
}

function playBeep(freq: number, duration: number, type: OscillatorType = 'sine', vol = 0.08) {
  try {
    const ctx = getAudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(vol, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (e) { /* ignore audio errors */ }
}

function playStartupSound() {
  setTimeout(() => playBeep(440, 0.15, 'sine', 0.06), 0);
  setTimeout(() => playBeep(554, 0.15, 'sine', 0.06), 120);
  setTimeout(() => playBeep(659, 0.15, 'sine', 0.06), 240);
  setTimeout(() => playBeep(880, 0.3, 'sine', 0.08), 400);
}

function playGestureSound(gesture: GestureMode) {
  switch (gesture) {
    case 'THUMBSUP': playBeep(784, 0.2, 'triangle', 0.06); break;
    case 'HEART': playBeep(523, 0.3, 'sine', 0.05); break;
    case 'PEACE': playBeep(698, 0.15, 'square', 0.04); break;
    case 'DRAWING': playBeep(1047, 0.05, 'sine', 0.03); break;
  }
}

// ─── JARVIS Speech ───
function jarvisSpeak(text: string) {
  if ('speechSynthesis' in window) {
    const synth = window.speechSynthesis;
    synth.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 0.95;
    utter.pitch = 0.8;
    utter.volume = 0.7;
    // Try to find a suitable voice
    const voices = synth.getVoices();
    const preferred = voices.find(v => v.name.includes('Daniel') || v.name.includes('Google UK English Male') || v.name.includes('Alex'));
    if (preferred) utter.voice = preferred;
    synth.speak(utter);
  }
}

// ─── Neon Hand Drawing ───
function drawNeonHand(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number; z: number }[][],
  w: number,
  h: number,
  connections: { start: number; end: number }[] | undefined
) {
  ctx.save();
  for (const lm of landmarks) {
    // Draw neon connections - triple pass for glow
    if (connections) {
      for (const conn of connections) {
        const s = lm[conn.start];
        const e = lm[conn.end];
        const sx = s.x * w, sy = s.y * h;
        const ex = e.x * w, ey = e.y * h;

        // Outer glow
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.15)';
        ctx.lineWidth = 8;
        ctx.shadowBlur = 25;
        ctx.shadowColor = '#00f0ff';
        ctx.stroke();

        // Middle glow
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.5)';
        ctx.lineWidth = 3;
        ctx.shadowBlur = 12;
        ctx.shadowColor = '#00f0ff';
        ctx.stroke();

        // Core bright line
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.strokeStyle = 'rgba(200, 255, 255, 0.9)';
        ctx.lineWidth = 1.2;
        ctx.shadowBlur = 6;
        ctx.shadowColor = '#ffffff';
        ctx.stroke();
      }
    }

    // Draw neon landmark dots
    for (const point of lm) {
      const px = point.x * w, py = point.y * h;

      // Outer glow
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 0, 229, 0.15)';
      ctx.shadowBlur = 20;
      ctx.shadowColor = '#ff00e5';
      ctx.fill();

      // Inner dot
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 100, 255, 0.8)';
      ctx.shadowBlur = 8;
      ctx.shadowColor = '#ff00e5';
      ctx.fill();

      // Core white
      ctx.beginPath();
      ctx.arc(px, py, 1.2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.shadowBlur = 0;
      ctx.fill();
    }
  }
  ctx.restore();
}

// ─── Draw fingertip cursor ───
function drawFingerCursor(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  color: string,
  gesture: GestureMode
) {
  ctx.save();
  if (gesture === 'DRAWING') {
    // Pulsing neon ring
    const time = Date.now() / 300;
    const pulse = 8 + Math.sin(time) * 3;

    ctx.beginPath();
    ctx.arc(x, y, pulse, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.shadowBlur = 20;
    ctx.shadowColor = color;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.shadowBlur = 10;
    ctx.shadowColor = color;
    ctx.fill();
  } else if (gesture === 'ERASING') {
    ctx.beginPath();
    ctx.arc(x, y, 50, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 45, 107, 0.5)';
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 8]);
    ctx.shadowBlur = 15;
    ctx.shadowColor = '#ff2d6b';
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

// ─── Main App ───
function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const drawCanvasRef = useRef<HTMLCanvasElement>(null);
  const handCanvasRef = useRef<HTMLCanvasElement>(null);
  const hudCanvasRef = useRef<HTMLCanvasElement>(null);

  const [cameraOn, setCameraOn] = useState(true);
  const [gesture, setGesture] = useState<GestureMode>('NONE');
  const [color, setColor] = useState(COLORS[0].value);
  const [thickness, setThickness] = useState(6);
  const [glow, setGlow] = useState(60);
  const [showTutorial, setShowTutorial] = useState(true);
  const [modelReady, setModelReady] = useState(false);
  const [jarvisText, setJarvisText] = useState('');
  const [showJarvis, setShowJarvis] = useState(false);

  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const currentStroke = useRef<Stroke | null>(null);
  const isDrawing = useRef(false);

  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const requestRef = useRef<number>(0);
  const lastVideoTimeRef = useRef(-1);
  const lastGestureRef = useRef<GestureMode>('NONE');
  const gestureHoldTime = useRef(0);
  const [activeReaction, setActiveReaction] = useState<ActiveReaction | null>(null);
  const reactionIdRef = useRef(0);

  // Refs for latest values (avoids stale closures in rAF)
  const colorRef = useRef(color);
  const thicknessRef = useRef(thickness);
  const glowRef = useRef(glow);
  const strokesRef = useRef(strokes);

  useEffect(() => { colorRef.current = color; }, [color]);
  useEffect(() => { thicknessRef.current = thickness; }, [thickness]);
  useEffect(() => { glowRef.current = glow; }, [glow]);
  useEffect(() => { strokesRef.current = strokes; }, [strokes]);

  // Initialize MediaPipe
  useEffect(() => {
    let cancelled = false;
    const init = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        if (cancelled) return;
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2,
        });
        if (cancelled) return;
        handLandmarkerRef.current = landmarker;
        setModelReady(true);
      } catch (e) {
        console.error("Failed to load hand landmarker:", e);
      }
    };
    init();
    return () => { cancelled = true; };
  }, []);

  // Camera start/stop
  useEffect(() => {
    if (!cameraOn) {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
      return;
    }
    let stream: MediaStream | null = null;
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }
    }).then(s => {
      stream = s;
      if (videoRef.current) videoRef.current.srcObject = stream;
    }).catch(err => console.error("Camera error:", err));
    return () => { if (stream) stream.getTracks().forEach(t => t.stop()); };
  }, [cameraOn]);

  // JARVIS startup greeting
  const handleStart = useCallback(() => {
    setShowTutorial(false);
    playStartupSound();
    setShowJarvis(true);
    const greeting = "Systems online. Hand tracking initialized. Ready to create.";
    setJarvisText(greeting);
    jarvisSpeak(greeting);
    setTimeout(() => setShowJarvis(false), 4000);
  }, []);

  // Trigger a single big emoji reaction (macOS FaceTime style)
  const triggerReaction = useCallback((emoji: string) => {
    const id = reactionIdRef.current++;
    setActiveReaction({ id, emoji, timestamp: Date.now() });
    // Auto-clear after animation completes (2s)
    setTimeout(() => {
      setActiveReaction(prev => (prev && prev.id === id ? null : prev));
    }, 2000);
  }, []);

  // Render strokes
  const renderStroke = useCallback((ctx: CanvasRenderingContext2D, stroke: Stroke) => {
    if (!stroke || !stroke.points || stroke.points.length < 2) return;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
    for (let i = 1; i < stroke.points.length - 1; i++) {
      const midX = (stroke.points[i].x + stroke.points[i + 1].x) / 2;
      const midY = (stroke.points[i].y + stroke.points[i + 1].y) / 2;
      ctx.quadraticCurveTo(stroke.points[i].x, stroke.points[i].y, midX, midY);
    }
    const last = stroke.points[stroke.points.length - 1];
    ctx.lineTo(last.x, last.y);
    ctx.strokeStyle = stroke.color;
    ctx.lineWidth = stroke.thickness;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (stroke.glow > 0) {
      ctx.shadowBlur = stroke.glow;
      ctx.shadowColor = stroke.color;
    }
    ctx.stroke();

    // Second pass for extra glow
    if (stroke.glow > 20) {
      ctx.globalAlpha = 0.4;
      ctx.lineWidth = stroke.thickness + 4;
      ctx.shadowBlur = stroke.glow * 1.5;
      ctx.stroke();
    }
    ctx.restore();
  }, []);

  const renderAllStrokes = useCallback(() => {
    const canvas = drawCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    strokesRef.current.forEach(s => renderStroke(ctx, s));
    if (currentStroke.current) renderStroke(ctx, currentStroke.current);
  }, [renderStroke]);

  // Eraser
  const eraseNear = useCallback((ex: number, ey: number, radius: number) => {
    setStrokes(prev => prev.filter(stroke =>
      !stroke.points.some(p => {
        const dx = p.x - ex;
        const dy = p.y - ey;
        return Math.sqrt(dx * dx + dy * dy) < radius;
      })
    ));
  }, []);

  // Draw HUD overlay (Iron Man style)
  const drawHUD = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number) => {
    ctx.save();
    const time = Date.now() / 1000;

    // Corner brackets
    const bLen = 40;
    const bPad = 30;
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.shadowBlur = 8;
    ctx.shadowColor = '#00f0ff';

    // Top-left
    ctx.beginPath();
    ctx.moveTo(bPad, bPad + bLen);
    ctx.lineTo(bPad, bPad);
    ctx.lineTo(bPad + bLen, bPad);
    ctx.stroke();

    // Top-right
    ctx.beginPath();
    ctx.moveTo(w - bPad - bLen, bPad);
    ctx.lineTo(w - bPad, bPad);
    ctx.lineTo(w - bPad, bPad + bLen);
    ctx.stroke();

    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(bPad, h - bPad - bLen);
    ctx.lineTo(bPad, h - bPad);
    ctx.lineTo(bPad + bLen, h - bPad);
    ctx.stroke();

    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(w - bPad - bLen, h - bPad);
    ctx.lineTo(w - bPad, h - bPad);
    ctx.lineTo(w - bPad, h - bPad - bLen);
    ctx.stroke();

    // Scanning line
    const scanY = ((time * 50) % h);
    ctx.beginPath();
    ctx.moveTo(0, scanY);
    ctx.lineTo(w, scanY);
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.06)';
    ctx.lineWidth = 1;
    ctx.shadowBlur = 0;
    ctx.stroke();

    // Grid dots
    ctx.fillStyle = 'rgba(0, 240, 255, 0.04)';
    const gridSize = 60;
    for (let x = gridSize; x < w; x += gridSize) {
      for (let y = gridSize; y < h; y += gridSize) {
        ctx.fillRect(x, y, 1, 1);
      }
    }

    // Status text top-right
    ctx.font = '10px "Inter", monospace';
    ctx.fillStyle = 'rgba(0, 240, 255, 0.35)';
    ctx.textAlign = 'right';
    const now = new Date();
    ctx.fillText(`SYS: ACTIVE`, w - 40, 70);
    ctx.fillText(`${now.toLocaleTimeString()}`, w - 40, 84);
    ctx.fillText(`FPS: 60`, w - 40, 98);

    ctx.restore();
  }, []);



  // ─── Main Detection Loop ───
  useEffect(() => {
    if (showTutorial) return;

    const loop = () => {
      const video = videoRef.current;
      const drawCanvas = drawCanvasRef.current;
      const handCanvas = handCanvasRef.current;
      const hudCanvas = hudCanvasRef.current;

      if (!video || !drawCanvas || !handCanvas || !hudCanvas || video.readyState < 3) {
        requestRef.current = requestAnimationFrame(loop);
        return;
      }

      // Match canvas sizes
      if (drawCanvas.width !== video.videoWidth && video.videoWidth > 0) {
        drawCanvas.width = video.videoWidth;
        drawCanvas.height = video.videoHeight;
        handCanvas.width = video.videoWidth;
        handCanvas.height = video.videoHeight;
        hudCanvas.width = video.videoWidth;
        hudCanvas.height = video.videoHeight;
      }

      const handCtx = handCanvas.getContext('2d');
      if (handCtx) handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);

      const hudCtx = hudCanvas.getContext('2d');
      if (hudCtx) {
        hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
        drawHUD(hudCtx, hudCanvas.width, hudCanvas.height);
      }

      if (handLandmarkerRef.current && video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;

        try {
          const results = handLandmarkerRef.current.detectForVideo(video, performance.now());

          if (results.landmarks && results.landmarks.length > 0) {
            const landmarks = results.landmarks[0];
            const connections = HandLandmarker.HAND_CONNECTIONS;

            // Draw neon hand skeleton
            if (handCtx) {
              drawNeonHand(handCtx, results.landmarks, handCanvas.width, handCanvas.height, connections);
            }

            const mode = detectGesture(landmarks);
            setGesture(mode);

            const indexTip = landmarks[8];
            const px = (1 - indexTip.x) * drawCanvas.width;
            const py = indexTip.y * drawCanvas.height;

            // Draw fingertip cursor on HUD
            if (hudCtx) {
              drawFingerCursor(hudCtx, px, py, colorRef.current, mode);
            }

            // Handle gesture transitions (sound + big emoji reaction + JARVIS)
            if (mode !== lastGestureRef.current) {
              gestureHoldTime.current = 0;
              if (mode === 'THUMBSUP') {
                playGestureSound('THUMBSUP');
                triggerReaction('👍');
                setJarvisText("Affirmative, sir.");
                setShowJarvis(true);
                setTimeout(() => setShowJarvis(false), 2000);
              } else if (mode === 'HEART') {
                playGestureSound('HEART');
                triggerReaction('❤️');
                setJarvisText("Signal received. Systems glowing.");
                setShowJarvis(true);
                setTimeout(() => setShowJarvis(false), 2000);
              } else if (mode === 'PEACE') {
                playGestureSound('PEACE');
                triggerReaction('✌️');
                setJarvisText("Peace protocol activated.");
                setShowJarvis(true);
                setTimeout(() => setShowJarvis(false), 2000);
              }
              lastGestureRef.current = mode;
            } else {
              gestureHoldTime.current++;
            }

            if (mode === 'DRAWING') {
              if (!isDrawing.current) {
                isDrawing.current = true;
                currentStroke.current = {
                  points: [{ x: px, y: py }],
                  color: colorRef.current,
                  thickness: thicknessRef.current,
                  glow: glowRef.current,
                };
              } else if (currentStroke.current) {
                currentStroke.current.points.push({ x: px, y: py });
              }
            } else {
              if (isDrawing.current && currentStroke.current) {
                const finished = currentStroke.current;
                currentStroke.current = null;
                isDrawing.current = false;
                setStrokes(prev => [...prev, finished]);
              }
              isDrawing.current = false;

              if (mode === 'ERASING') {
                const palmX = (1 - landmarks[9].x) * drawCanvas.width;
                const palmY = landmarks[9].y * drawCanvas.height;
                eraseNear(palmX, palmY, 50);
              }
            }
          } else {
            setGesture('NONE');
            if (isDrawing.current && currentStroke.current) {
              const finished = currentStroke.current;
              currentStroke.current = null;
              isDrawing.current = false;
              setStrokes(prev => [...prev, finished]);
            }
            lastGestureRef.current = 'NONE';
          }
        } catch (_) { /* skip frame */ }
      }

      renderAllStrokes();
      requestRef.current = requestAnimationFrame(loop);
    };

    requestRef.current = requestAnimationFrame(loop);
    return () => { if (requestRef.current) cancelAnimationFrame(requestRef.current); };
  }, [showTutorial, renderAllStrokes, eraseNear, drawHUD, triggerReaction]);

  const handleUndo = () => setStrokes(prev => prev.slice(0, -1));
  const handleClear = () => { setStrokes([]); currentStroke.current = null; isDrawing.current = false; };

  const handleDownload = () => {
    if (!drawCanvasRef.current || !videoRef.current) return;
    const w = drawCanvasRef.current.width;
    const h = drawCanvasRef.current.height;
    const dl = document.createElement("canvas");
    dl.width = w; dl.height = h;
    const ctx = dl.getContext("2d");
    if (!ctx) return;
    ctx.save(); ctx.scale(-1, 1); ctx.translate(-w, 0);
    ctx.drawImage(videoRef.current, 0, 0, w, h);
    ctx.restore();
    strokes.forEach(s => renderStroke(ctx, s));
    const link = document.createElement("a");
    link.href = dl.toDataURL("image/png");
    link.download = `airdraw_${Date.now()}.png`;
    link.click();
    playBeep(880, 0.15, 'sine', 0.05);

    setJarvisText("Capture saved to local archives.");
    setShowJarvis(true);
    setTimeout(() => setShowJarvis(false), 3000);
    jarvisSpeak("Capture successful.");
  };

  const gestureEmoji: Record<GestureMode, string> = {
    'DRAWING': '☝️', 'ERASING': '🖐️', 'IDLE': '✊',
    'THUMBSUP': '👍', 'HEART': '❤️', 'PEACE': '✌️', 'NONE': '👋',
  };
  const gestureLabel: Record<GestureMode, string> = {
    'DRAWING': 'DRAWING', 'ERASING': 'ERASING', 'IDLE': 'IDLE',
    'THUMBSUP': 'THUMBS UP!', 'HEART': 'LOVE!', 'PEACE': 'PEACE!', 'NONE': 'SHOW HAND',
  };

  return (
    <div className="app-container">
      <h1 style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', border: 0 }}>toobz — AI Air Drawing & Hand Tracking Canvas</h1>
      <video ref={videoRef} autoPlay playsInline muted className="background-video" />
      <canvas ref={handCanvasRef} className="hand-overlay-canvas" />
      <canvas ref={drawCanvasRef} className="drawing-canvas" />
      <canvas ref={hudCanvasRef} className="hud-canvas" />

      {/* Big Emoji Reaction (macOS FaceTime style) */}
      {activeReaction && (
        <div className="reaction-overlay" key={activeReaction.id}>
          <div className="reaction-emoji">{activeReaction.emoji}</div>
        </div>
      )}

      {/* JARVIS Text */}
      {showJarvis && (
        <div className="jarvis-text">
          <div className="jarvis-label">J.A.R.V.I.S</div>
          <div className="jarvis-message">{jarvisText}</div>
        </div>
      )}

      {/* Tutorial Modal */}
      {showTutorial && (
        <div className="tutorial-overlay">
          <div className="tutorial-modal glass-panel">
            <div className="tutorial-logo">
              <img src="/favicon.svg" alt="toobz - The AI-powered Air Drawing Canvas" style={{ width: '64px', height: '64px', borderRadius: '16px', boxShadow: '0 8px 32px rgba(170, 59, 255, 0.3)' }} />
            </div>
            <h2>How to Play</h2>
            <div className="tutorial-grid">
              <div className="tutorial-item">
                <span className="tutorial-emoji">☝️</span>
                <div><strong>Draw</strong><p>Point index finger to draw</p></div>
              </div>
              <div className="tutorial-item">
                <span className="tutorial-emoji">🖐️</span>
                <div><strong>Erase</strong><p>Sweep open palm to erase</p></div>
              </div>
              <div className="tutorial-item">
                <span className="tutorial-emoji">👍</span>
                <div><strong>Thumbs Up</strong><p>Show thumbs up for emoji burst</p></div>
              </div>
              <div className="tutorial-item">
                <span className="tutorial-emoji">❤️</span>
                <div><strong>Heart</strong><p>Pinch fingers for heart rain</p></div>
              </div>
              <div className="tutorial-item">
                <span className="tutorial-emoji">✌️</span>
                <div><strong>Peace</strong><p>Peace sign for celebration</p></div>
              </div>
              <div className="tutorial-item">
                <span className="tutorial-emoji">✊</span>
                <div><strong>Idle</strong><p>Close fist to pause</p></div>
              </div>
            </div>
            <button className="tutorial-btn" onClick={handleStart} disabled={!modelReady}>
              {modelReady ? "Let's Go!" : "⏳ Loading AI Model..."}
            </button>
          </div>
        </div>
      )}

      {/* Top Left */}
      {!showTutorial && (
        <div className="top-left-chip" onClick={() => setCameraOn(!cameraOn)}>
          {cameraOn ? <Camera size={16} /> : <CameraOff size={16} />}
          <span>Camera {cameraOn ? 'ON' : 'OFF'}</span>
        </div>
      )}

      {/* Bottom Center */}
      {!showTutorial && (
        <div className={`bottom-chip gesture-${gesture.toLowerCase()}`}>
          <span>{gestureEmoji[gesture]} {gestureLabel[gesture]}</span>
        </div>
      )}

      {/* Right Toolbar */}
      {!showTutorial && (
        <div className="toolbar glass-panel">
          <div className="toolbar-section">
            <span className="section-title">COLORS</span>
            <div className="color-grid">
              {COLORS.map(c => (
                <button key={c.name}
                  className={`color-btn ${color === c.value ? 'active' : ''}`}
                  style={{ backgroundColor: c.value, boxShadow: color === c.value ? `0 0 12px ${c.value}` : 'none' }}
                  onClick={() => { setColor(c.value); playBeep(600, 0.05, 'sine', 0.03); }}
                />
              ))}
            </div>
          </div>
          <div className="toolbar-section">
            <span className="section-title">THICKNESS</span>
            <div className="slider-container vertical">
              <input type="range" className="vertical-slider" min="1" max="30"
                value={thickness} onChange={e => setThickness(parseInt(e.target.value))} />
            </div>
            <span className="slider-value">{thickness}px</span>
          </div>
          <div className="toolbar-section">
            <span className="section-title">GLOW</span>
            <div className="slider-container vertical">
              <input type="range" className="vertical-slider" min="0" max="100"
                value={glow} onChange={e => setGlow(parseInt(e.target.value))} />
            </div>
            <span className="slider-value">{glow}%</span>
          </div>
          <div className="toolbar-divider" />
          <div className="toolbar-actions">
            <button className="action-btn" onClick={handleUndo} title="Undo"><Undo2 size={18} /></button>
            <button className="action-btn" onClick={handleClear} title="Clear"><Trash2 size={18} /></button>
            <button className="action-btn" title="Options"><GripHorizontal size={18} /></button>
            <button className="action-btn" onClick={handleDownload} title="Download"><Download size={18} /></button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
