import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from './ui/badge';
import { Progress } from '../components/ui/progress';
import { Camera, CameraOff, RotateCcw, Play, Pause, AlertCircle, Info, Volume2, VolumeX } from 'lucide-react';
import { TemporalBuffer } from '../ml/TemporalBuffer';

interface RecognizedSign {
  sign: string;
  confidence: number;
  timestamp: Date;
  classifierUsed?: 'static' | 'temporal';
}

// Motion threshold: below this, use static classifier; above, use temporal
const MOTION_THRESHOLD = 8;

interface OverlayMetrics {
  fps: number;
  latencyMs: number;
  classifierUsed: 'static' | 'temporal' | 'none';
  topConfidence: number;
  topCandidates: { gesture: string; confidence: number }[];
  activeHands: number;
  handsDetected: ('left' | 'right')[];
}

// Sign language gesture patterns based on hand landmarks
// Uses finger extension bitmask patterns: [Thumb, Index, Middle, Ring, Pinky]
// In production, the trained CustomMLP replaces these heuristics.

/**
 * Returns a bitmask array [thumb, index, middle, ring, pinky] where 1 = extended.
 */
function getFingerBitmask(landmarks: number[][]): [number, number, number, number, number] {
  if (!landmarks || landmarks.length < 21) return [0, 0, 0, 0, 0];

  // Finger tip and MCP indices
  const tips = [4, 8, 12, 16, 20];
  const mcps = [2, 5, 9, 13, 17];
  const mask: [number, number, number, number, number] = [0, 0, 0, 0, 0];

  // Thumb: check horizontal extension (x distance from MCP)
  const thumbExt = Math.abs(landmarks[tips[0]][0] - landmarks[mcps[0]][0]) > 30;
  mask[0] = thumbExt ? 1 : 0;

  // Other fingers: check if tip is above MCP (y-axis, lower y = higher on screen)
  for (let i = 1; i < 5; i++) {
    mask[i] = landmarks[tips[i]][1] < landmarks[mcps[i]][1] - 20 ? 1 : 0;
  }
  return mask;
}

function matchBitmask(
  landmarks: number[][],
  expected: [number, number, number, number, number],
  tolerance: number = 0,
): number {
  const actual = getFingerBitmask(landmarks);
  let matches = 0;
  for (let i = 0; i < 5; i++) {
    if (actual[i] === expected[i]) matches++;
  }
  const score = matches / 5;
  return score >= (5 - tolerance) / 5 ? score * 0.85 : 0;
}

const signPatterns: Record<string, (landmarks: number[][]) => number> = {
  // ──────── A–Z Alphabet (finger extension bitmasks) ────────
  // Bitmask: [Thumb, Index, Middle, Ring, Pinky]
  'A': (lm) => matchBitmask(lm, [0, 0, 0, 0, 0]),     // Fist, thumb alongside
  'B': (lm) => matchBitmask(lm, [0, 1, 1, 1, 1]),     // Flat hand, thumb tucked
  'C': (lm) => {                                        // Curved C shape — all spread
    const ext = checkFingersExtended(lm);
    return ext >= 3 ? 0.70 : 0;
  },
  'D': (lm) => matchBitmask(lm, [0, 1, 0, 0, 0]),     // Index up only
  'E': (lm) => matchBitmask(lm, [0, 0, 0, 0, 0]),     // All curled, tips down
  'F': (lm) => matchBitmask(lm, [0, 0, 1, 1, 1]),     // Thumb+index touch, 3 up
  'G': (lm) => matchBitmask(lm, [1, 1, 0, 0, 0]),     // Thumb+index horizontal
  'H': (lm) => matchBitmask(lm, [0, 1, 1, 0, 0]),     // Index+middle horizontal
  'I': (lm) => matchBitmask(lm, [0, 0, 0, 0, 1]),     // Pinky only
  'J': (lm) => matchBitmask(lm, [0, 0, 0, 0, 1]),     // Pinky + J motion (static approx)
  'K': (lm) => matchBitmask(lm, [1, 1, 1, 0, 0]),     // Thumb between index+middle
  'L': (lm) => matchBitmask(lm, [1, 1, 0, 0, 0]),     // L-shape
  'M': (lm) => matchBitmask(lm, [0, 0, 0, 0, 0]),     // 3 fingers over thumb
  'N': (lm) => matchBitmask(lm, [0, 0, 0, 0, 0]),     // 2 fingers over thumb
  'O': (lm) => {                                        // Fingertips touching, O shape
    const ext = checkFingersExtended(lm);
    return ext <= 1 ? 0.72 : 0;
  },
  'P': (lm) => matchBitmask(lm, [1, 1, 1, 0, 0]),     // Like K, tilted down
  'Q': (lm) => matchBitmask(lm, [1, 1, 0, 0, 0]),     // Like G, pointing down
  'R': (lm) => matchBitmask(lm, [0, 1, 1, 0, 0]),     // Index+middle crossed
  'S': (lm) => matchBitmask(lm, [0, 0, 0, 0, 0]),     // Fist, thumb over
  'T': (lm) => matchBitmask(lm, [0, 0, 0, 0, 0]),     // Fist, thumb between
  'U': (lm) => matchBitmask(lm, [0, 1, 1, 0, 0]),     // Index+middle together up
  'V': (lm) => matchBitmask(lm, [0, 1, 1, 0, 0]),     // Index+middle spread (V)
  'W': (lm) => matchBitmask(lm, [0, 1, 1, 1, 0]),     // 3 fingers spread
  'X': (lm) => matchBitmask(lm, [0, 1, 0, 0, 0]),     // Index hooked
  'Y': (lm) => matchBitmask(lm, [1, 0, 0, 0, 1]),     // Thumb+pinky (hang loose)
  'Z': (lm) => matchBitmask(lm, [0, 1, 0, 0, 0]),     // Index traces Z (static approx)

  // ──────── Common Signs ────────
  'Hello': (landmarks) => {
    const fingersExtended = checkFingersExtended(landmarks);
    return fingersExtended >= 4 ? 0.85 : 0;
  },
  'Thank You': (landmarks) => {
    const palmFacingUp = checkPalmOrientation(landmarks, 'up');
    const fingersExtended = checkFingersExtended(landmarks);
    return (palmFacingUp && fingersExtended >= 3) ? 0.80 : 0;
  },
  'Please': (landmarks) => {
    const palmOpen = checkFingersExtended(landmarks) >= 4;
    return palmOpen ? 0.75 : 0;
  },
  'Yes': (landmarks) => {
    const fistClosed = checkFingersExtended(landmarks) <= 1;
    return fistClosed ? 0.82 : 0;
  },
  'No': (landmarks) => {
    const twoFingersExtended = checkFingersExtended(landmarks) === 2;
    return twoFingersExtended ? 0.78 : 0;
  },
  'Good': (landmarks) => {
    const thumbUp = checkThumbPosition(landmarks, 'up');
    const fingersDown = checkFingersExtended(landmarks) <= 1;
    return (thumbUp && fingersDown) ? 0.88 : 0;
  },
  'Bad': (landmarks) => {
    const thumbDown = checkThumbPosition(landmarks, 'down');
    const fingersDown = checkFingersExtended(landmarks) <= 1;
    return (thumbDown && fingersDown) ? 0.83 : 0;
  },
  'Help': (landmarks) => {
    const fingersExtended = checkFingersExtended(landmarks);
    return fingersExtended >= 3 ? 0.76 : 0;
  },
  'Sorry': (landmarks) => {
    const fistClosed = checkFingersExtended(landmarks) <= 1;
    return fistClosed ? 0.79 : 0;
  },
  'More': (landmarks) => {
    // Both hands: fingertips together — approximate with closed hand
    const fist = checkFingersExtended(landmarks) <= 1;
    return fist ? 0.74 : 0;
  },
  'Stop': (landmarks) => {
    // Flat hand — all fingers extended
    return checkFingersExtended(landmarks) >= 4 ? 0.77 : 0;
  },
  'Water': (landmarks) => {
    // W-hand taps chin — check W bitmask
    return matchBitmask(landmarks, [0, 1, 1, 1, 0]) > 0 ? 0.73 : 0;
  },
  'Eat': (landmarks) => {
    // Flat O taps mouth — fingers slightly curled
    return checkFingersExtended(landmarks) <= 2 ? 0.72 : 0;
  },
  'Home': (landmarks) => {
    // Flat O from cheek to ear
    return checkFingersExtended(landmarks) <= 2 ? 0.71 : 0;
  },
  'Love': (landmarks) => {
    const specialPattern = checkLoveSign(landmarks);
    return specialPattern ? 0.90 : 0;
  },
};

// Helper functions for gesture recognition
function checkFingersExtended(landmarks: number[][]): number {
  if (!landmarks || landmarks.length < 21) return 0;

  let extendedCount = 0;
  const fingerTips = [8, 12, 16, 20];
  const fingerMCPs = [5, 9, 13, 17];

  fingerTips.forEach((tip, i) => {
    const tipY = landmarks[tip][1];
    const mcpY = landmarks[fingerMCPs[i]][1];
    if (tipY < mcpY - 20) {
      extendedCount++;
    }
  });

  const thumbTip = landmarks[4];
  const thumbMCP = landmarks[2];
  if (Math.abs(thumbTip[0] - thumbMCP[0]) > 30) {
    extendedCount++;
  }

  return extendedCount;
}

function checkPalmOrientation(landmarks: number[][], direction: string): boolean {
  if (!landmarks || landmarks.length < 21) return false;
  const wrist = landmarks[0];
  const middleMCP = landmarks[9];
  if (direction === 'up') {
    return middleMCP[1] < wrist[1] - 30;
  }
  return false;
}

function checkThumbPosition(landmarks: number[][], direction: string): boolean {
  if (!landmarks || landmarks.length < 21) return false;
  const thumbTip = landmarks[4];
  const indexMCP = landmarks[5];
  if (direction === 'up') {
    return thumbTip[1] < indexMCP[1] - 40;
  } else if (direction === 'down') {
    return thumbTip[1] > indexMCP[1] + 40;
  }
  return false;
}

function checkLoveSign(landmarks: number[][]): boolean {
  if (!landmarks || landmarks.length < 21) return false;
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const pinkyTip = landmarks[20];
  const middleTip = landmarks[12];
  const thumbMCP = landmarks[2];
  const indexMCP = landmarks[5];
  const middleMCP = landmarks[9];

  const thumbExtended = Math.abs(thumbTip[0] - thumbMCP[0]) > 30;
  const indexExtended = indexTip[1] < indexMCP[1] - 20;
  const middleDown = middleTip[1] > middleMCP[1];
  const pinkyExtended = pinkyTip[1] < middleMCP[1] - 10;

  return thumbExtended && indexExtended && pinkyExtended && middleDown;
}

function recognizeSign(landmarks: number[][]): RecognizedSign | null {
  let bestMatch: RecognizedSign | null = null;
  let highestConfidence = 0.70;

  for (const [signName, detector] of Object.entries(signPatterns)) {
    const confidence = detector(landmarks);
    if (confidence > highestConfidence) {
      highestConfidence = confidence;
      bestMatch = {
        sign: signName,
        confidence: confidence,
        timestamp: new Date()
      };
    }
  }
  return bestMatch;
}

export function SignLanguageRecognizer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [currentSign, setCurrentSign] = useState<RecognizedSign | null>(null);
  const [recognitionHistory, setRecognitionHistory] = useState<RecognizedSign[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [model, setModel] = useState<any>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const recognitionIntervalRef = useRef<number | null>(null);

  // === Upgrade 3: Temporal buffer ===
  const temporalBufferRef = useRef(new TemporalBuffer(20));

  // === Upgrade 5: Two-handed state ===
  const [detectedHands, setDetectedHands] = useState<('left' | 'right')[]>([]);

  // === Upgrade 6: Research overlay ===
  const [showOverlay, setShowOverlay] = useState(false);
  const [overlayMetrics, setOverlayMetrics] = useState<OverlayMetrics>({
    fps: 0, latencyMs: 0, classifierUsed: 'none',
    topConfidence: 0, topCandidates: [], activeHands: 0, handsDetected: [],
  });
  const fpsFrames = useRef<number[]>([]);
  const latencyFrames = useRef<number[]>([]);

  // Keyboard shortcut: Shift+D toggles overlay
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.shiftKey && e.key === 'D') {
        setShowOverlay(prev => !prev);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // Ensure video element always receives the stream
  useEffect(() => {
    if (cameraActive && stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current?.play().catch(e => console.error('Play error:', e));
        setVideoError(null);
      };
      videoRef.current.onerror = () => {
        setVideoError('Video playback failed.');
      };
      videoRef.current.onplay = () => {
        setVideoError(null);
      };
    }
  }, [stream, cameraActive]);

  // Load the handpose model
  useEffect(() => {
    const loadModel = async () => {
      try {
        setModelLoading(true);
        // Dynamically import TensorFlow.js and handpose
        const tf = await import('@tensorflow/tfjs');
        const handpose = await import('@tensorflow-models/handpose');
        await tf.ready();
        const loadedModel = await handpose.load({ maxNumHands: 2 } as any);
        setModel(loadedModel);
        setModelLoading(false);
      } catch (error) {
        setCameraError('Error loading AI model. Please refresh and try again.');
        setModelLoading(false);
        console.error('Error loading model:', error);
      }
    };
    loadModel();
  }, []);

  const speakText = useCallback((text: string) => {
    if (!audioEnabled) return;
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 1;
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);
      window.speechSynthesis.speak(utterance);
    }
  }, [audioEnabled]);

  const replayAudio = () => {
    if (currentSign) {
      speakText(currentSign.sign);
    }
  };

  const startCamera = useCallback(async () => {
    try {
      setIsLoading(true);
      setCameraError(null);

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera not supported by this browser');
      }

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          facingMode: 'user'
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        setStream(mediaStream);
        setCameraActive(true);

        // Ensure video playback starts
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(e => console.error('Play error:', e));
          setVideoError(null);
        };
        videoRef.current.onerror = () => {
          setVideoError('Video playback failed.');
        };
        videoRef.current.onplay = () => {
          setVideoError(null);
        };
      } else {
        // If videoRef is null, just set the stream state and let the useEffect handle it
        setStream(mediaStream);
        setCameraActive(true);
      }
    } catch (error: any) {
      if (error.name !== 'NotAllowedError') {
        console.error('Error accessing camera:', error);
      }

      let errorMessage = 'Could not access camera. ';

      if (error.name === 'NotAllowedError') {
        errorMessage += 'Camera permission was denied. Please allow camera access in your browser settings and try again.';
        setTimeout(() => {
          setDemoMode(true);
          setCameraError(null);
          setIsRecognizing(true);
        }, 500);
      } else if (error.name === 'NotFoundError') {
        errorMessage += 'No camera found on this device.';
      } else if (error.name === 'NotReadableError') {
        errorMessage += 'Camera is already in use by another application.';
      } else if (error.name === 'OverconstrainedError') {
        errorMessage += 'Camera does not meet the required specifications.';
      } else if (error.message === 'Camera not supported by this browser') {
        errorMessage += 'Your browser does not support camera access.';
      } else {
        errorMessage += 'Please check your camera and permissions.';
      }

      setCameraError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      setCameraActive(false);
      setIsRecognizing(false);
    }
    setCameraError(null);
    setDemoMode(false);
  }, [stream]);

  const startDemoMode = useCallback(() => {
    setDemoMode(true);
    setCameraError(null);
    setIsRecognizing(true);

    const interval = setInterval(() => {
      const mockSigns = ['Hello', 'Thank You', 'Please', 'Yes', 'No', 'Good', 'Bad', 'Help', 'Sorry', 'Love'];
      const randomSign = mockSigns[Math.floor(Math.random() * mockSigns.length)];
      const recognizedSign: RecognizedSign = {
        sign: randomSign,
        confidence: 0.75 + Math.random() * 0.2,
        timestamp: new Date()
      };

      setCurrentSign(recognizedSign);
      setRecognitionHistory(prev => [recognizedSign, ...prev].slice(0, 10));
    }, 2000);

    recognitionIntervalRef.current = interval as unknown as number;
  }, []);

  const startRecognition = useCallback(async () => {
    if (!model && !demoMode) {
      setCameraError('Model is still loading. Please wait...');
      return;
    }

    setIsRecognizing(true);

    if (demoMode) {
      const interval = setInterval(() => {
        const mockSigns = ['Hello', 'Thank you', 'Please', 'Yes', 'No', 'Good', 'Bad', 'Help', 'Sorry', 'Love'];
        const randomSign = mockSigns[Math.floor(Math.random() * mockSigns.length)];
        const recognizedSign: RecognizedSign = {
          sign: randomSign,
          confidence: 0.75 + Math.random() * 0.2,
          timestamp: new Date()
        };

        setCurrentSign(recognizedSign);
        setRecognitionHistory(prev => [recognizedSign, ...prev].slice(0, 10));
      }, 2000);

      recognitionIntervalRef.current = interval as unknown as number;
      return;
    }

    let isActive = true;

    const detectHands = async () => {
      if (videoRef.current && model && isActive) {
        const frameStart = performance.now();
        try {
          const predictions = await model.estimateHands(videoRef.current);
          const inferenceTime = performance.now() - frameStart;

          // === Upgrade 5: Detect handedness ===
          const hands: ('left' | 'right')[] = [];
          const allLandmarks: number[][] = [];

          if (predictions.length > 0) {
            // Sort by x-centroid to determine left/right
            const sorted = [...predictions].sort((a: any, b: any) => {
              const aCx = a.landmarks.reduce((s: number, l: number[]) => s + l[0], 0) / a.landmarks.length;
              const bCx = b.landmarks.reduce((s: number, l: number[]) => s + l[0], 0) / b.landmarks.length;
              return aCx - bCx;
            });

            for (let hi = 0; hi < sorted.length; hi++) {
              // Since camera is mirrored: leftmost in frame = right hand
              hands.push(hi === 0 ? 'right' : 'left');
              allLandmarks.push(...sorted[hi].landmarks);
            }
            setDetectedHands(hands);

            // Use primary hand for recognition
            const primaryLandmarks = sorted[0].landmarks;

            // Draw landmarks on canvas
            if (canvasRef.current) {
              const ctx = canvasRef.current.getContext('2d');
              if (ctx && videoRef.current) {
                canvasRef.current.width = videoRef.current.videoWidth;
                canvasRef.current.height = videoRef.current.videoHeight;
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

                const connections = [
                  [0, 1], [1, 2], [2, 3], [3, 4],
                  [0, 5], [5, 6], [6, 7], [7, 8],
                  [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16],
                  [0, 17], [17, 18], [18, 19], [19, 20],
                  [5, 9], [9, 13], [13, 17]
                ];

                // Draw each detected hand with different colors
                const handColors = ['#3b82f6', '#f97316']; // blue, orange
                for (let hi = 0; hi < sorted.length; hi++) {
                  const lm = sorted[hi].landmarks;
                  const color = handColors[hi % handColors.length];

                  // Dots
                  ctx.fillStyle = color;
                  lm.forEach((landmark: number[]) => {
                    ctx.beginPath();
                    ctx.arc(landmark[0], landmark[1], 5, 0, 2 * Math.PI);
                    ctx.fill();
                  });

                  // Connections
                  ctx.strokeStyle = color;
                  ctx.lineWidth = 2;
                  connections.forEach(([start, end]) => {
                    ctx.beginPath();
                    ctx.moveTo(lm[start][0], lm[start][1]);
                    ctx.lineTo(lm[end][0], lm[end][1]);
                    ctx.stroke();
                  });

                  // === Upgrade 5: Hand label badge ===
                  const handLabel = hands[hi]?.toUpperCase() || '';
                  const wrist = lm[0];
                  ctx.font = 'bold 14px sans-serif';
                  ctx.fillStyle = 'white';
                  ctx.strokeStyle = color;
                  ctx.lineWidth = 3;
                  ctx.strokeText(handLabel, wrist[0] - 15, wrist[1] + 25);
                  ctx.fillText(handLabel, wrist[0] - 15, wrist[1] + 25);
                }
              }
            }

            // === Upgrade 3: Temporal routing ===
            temporalBufferRef.current.push(primaryLandmarks);
            const motionMag = temporalBufferRef.current.getMotionVector();
            let classifierUsed: 'static' | 'temporal' = 'static';

            let recognizedSignResult: RecognizedSign | null = null;

            if (motionMag < MOTION_THRESHOLD) {
              // Static classifier
              classifierUsed = 'static';
              recognizedSignResult = recognizeSign(primaryLandmarks);
            } else {
              // Temporal/dynamic — fall back to static heuristics for now
              // (SequenceClassifier requires training data; using static as fallback)
              classifierUsed = 'temporal';
              recognizedSignResult = recognizeSign(primaryLandmarks);
            }

            // Build top-3 candidates
            const candidates: { gesture: string; confidence: number }[] = [];
            for (const [signName, detector] of Object.entries(signPatterns)) {
              const conf = detector(primaryLandmarks);
              if (conf > 0) candidates.push({ gesture: signName, confidence: conf });
            }
            candidates.sort((a, b) => b.confidence - a.confidence);
            const top3 = candidates.slice(0, 3);

            if (recognizedSignResult) {
              recognizedSignResult.classifierUsed = classifierUsed;
              setCurrentSign(recognizedSignResult);
              setRecognitionHistory(prev => {
                if (prev.length > 0 && prev[0].sign === recognizedSignResult!.sign) {
                  return prev;
                }
                return [recognizedSignResult!, ...prev].slice(0, 10);
              });
            }

            // === Upgrade 6: Update overlay metrics ===
            const now = performance.now();
            fpsFrames.current.push(now);
            fpsFrames.current = fpsFrames.current.filter(t => now - t < 1000);
            latencyFrames.current.push(inferenceTime);
            if (latencyFrames.current.length > 10) latencyFrames.current.shift();
            const avgLatency = latencyFrames.current.reduce((a, b) => a + b, 0) / latencyFrames.current.length;

            setOverlayMetrics({
              fps: fpsFrames.current.length,
              latencyMs: Math.round(avgLatency * 10) / 10,
              classifierUsed,
              topConfidence: recognizedSignResult?.confidence ?? 0,
              topCandidates: top3,
              activeHands: predictions.length,
              handsDetected: hands,
            });
          } else {
            setDetectedHands([]);
            if (canvasRef.current) {
              const ctx = canvasRef.current.getContext('2d');
              if (ctx) {
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
              }
            }
            setCurrentSign(null);
            setOverlayMetrics(prev => ({ ...prev, activeHands: 0, handsDetected: [], classifierUsed: 'none' }));
          }
        } catch (error) {
          setCameraError('Error detecting hands: ' + (error instanceof Error ? error.message : String(error)));
          console.error('Error detecting hands:', error);
        }
      }

      if (isActive) {
        requestAnimationFrame(detectHands);
      }
    };

    detectHands();

    return () => {
      isActive = false;
    };
  }, [model, demoMode]);

  const stopRecognition = useCallback(() => {
    setIsRecognizing(false);
    if (recognitionIntervalRef.current) {
      clearInterval(recognitionIntervalRef.current);
      recognitionIntervalRef.current = null;
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }

    setCurrentSign(null);
  }, []);

  const clearHistory = () => {
    setRecognitionHistory([]);
  };

  useEffect(() => {
    return () => {
      stopCamera();
      if (recognitionIntervalRef.current) {
        clearInterval(recognitionIntervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (currentSign && audioEnabled) {
      speakText(currentSign.sign);
    }
  }, [currentSign, audioEnabled, speakText]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Sign Language Recognizer
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Real-time ASL detection using AI-powered hand tracking
          </p>
        </div>

        {modelLoading && (
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg dark:bg-blue-950/20 dark:border-blue-800/30">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5 animate-pulse" />
              <div>
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  Loading AI model for hand detection... This may take a moment.
                </p>
              </div>
            </div>
          </div>
        )}

        {!cameraActive && !demoMode && !cameraError && (
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg dark:bg-blue-950/20 dark:border-blue-800/30">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div className="space-y-2">
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  New to sign language recognition?
                </p>
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  Try our Demo Mode to see how recognition works without needing camera access.
                  Perfect for exploring the app or if you're having camera issues.
                </p>
                <Button
                  onClick={startDemoMode}
                  size="sm"
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Start Demo Mode
                </Button>
              </div>
            </div>
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Camera Feed {demoMode && <Badge variant="secondary">Demo Mode</Badge>}
              <div className="flex gap-2">
                {!cameraActive && !demoMode ? (
                  <>
                    <Button
                      onClick={startCamera}
                      disabled={isLoading}
                      variant="outline"
                      size="sm"
                    >
                      <Camera className="w-4 h-4 mr-2" />
                      {isLoading ? 'Starting...' : 'Start Camera'}
                    </Button>
                    <Button
                      onClick={startDemoMode}
                      variant="secondary"
                      size="sm"
                    >
                      <Play className="w-4 h-4 mr-2" />
                      Demo Mode
                    </Button>
                  </>
                ) : cameraActive ? (
                  <Button
                    onClick={stopCamera}
                    variant="outline"
                    size="sm"
                  >
                    <CameraOff className="w-4 h-4 mr-2" />
                    Stop Camera
                  </Button>
                ) : (
                  <Button
                    onClick={() => setDemoMode(false)}
                    variant="outline"
                    size="sm"
                  >
                    Exit Demo
                  </Button>
                )}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative bg-black rounded-lg overflow-hidden">
              {cameraActive ? (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-80 object-cover z-10 relative"
                    style={{
                      background: 'black',
                      display: 'block',
                      width: '100%',
                      height: '320px',
                      objectFit: 'cover'
                    }}
                  >
                    Sorry, your browser does not support embedded videos.
                  </video>
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full z-20"
                    style={{
                      pointerEvents: 'none',
                      position: 'absolute',
                      top: 0,
                      left: 0
                    }}
                  />
                  {/* Fallback if video cannot play */}
                  {videoError && (
                    <div className="absolute inset-0 flex items-center justify-center text-white bg-black bg-opacity-80 z-30">
                      <p>{videoError}</p>
                    </div>
                  )}
                </>
              ) : demoMode ? (
                <div className="w-full h-80 flex items-center justify-center bg-gradient-to-br from-blue-900 to-purple-900 text-white">
                  <div className="text-center">
                    <Play className="w-16 h-16 mx-auto mb-4" />
                    <p className="text-lg mb-2">Demo Mode Active</p>
                    <p className="text-sm opacity-80">Simulating sign recognition without camera</p>
                  </div>
                </div>
              ) : (
                <div className="w-full h-80 flex items-center justify-center text-muted-foreground">
                  <div className="text-center space-y-4">
                    <div>
                      <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p>Click "Start Camera" to begin</p>
                    </div>
                    {!cameraError && (
                      <div className="text-center">
                        <p className="text-xs opacity-60 mb-3">No camera? No problem!</p>
                        <Button
                          onClick={startDemoMode}
                          variant="secondary"
                          size="sm"
                        >
                          <Play className="w-4 h-4 mr-2" />
                          Try Demo Mode
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {isRecognizing && (
                <div className="absolute top-4 left-4">
                  <Badge variant="destructive" className="animate-pulse">
                    ● {demoMode ? 'DEMO MODE' : 'RECOGNIZING'}
                  </Badge>
                </div>
              )}

              {/* === Upgrade 6: Research Metrics Overlay (Shift+D) === */}
              {showOverlay && (
                <div className="absolute top-4 right-0 bg-black/80 text-white text-xs p-3 rounded-lg"
                  style={{ minWidth: '200px', zIndex: 50, backdropFilter: 'blur(4px)' }}>
                  <div className="font-bold mb-2 text-green-400">⚡ Research Metrics</div>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-300">FPS</span>
                      <span className="font-mono">{overlayMetrics.fps}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Latency</span>
                      <span className="font-mono">{overlayMetrics.latencyMs}ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Classifier</span>
                      <span className={`font-mono ${overlayMetrics.classifierUsed === 'temporal' ? 'text-yellow-400' : 'text-blue-400'}`}>
                        {overlayMetrics.classifierUsed}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Confidence</span>
                      <span className="font-mono">{overlayMetrics.topConfidence.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Hands</span>
                      <span className="font-mono">
                        {overlayMetrics.activeHands} ({overlayMetrics.handsDetected.join(', ') || '—'})
                      </span>
                    </div>
                  </div>
                  {overlayMetrics.topCandidates.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-white/20">
                      <div className="text-gray-400 mb-1">Top Candidates</div>
                      {overlayMetrics.topCandidates.map((c, i) => (
                        <div key={i} className="flex items-center gap-2 mb-1">
                          <span className="w-8 truncate">{c.gesture}</span>
                          <div className="flex-1 bg-white/20 rounded-full h-1.5">
                            <div
                              className={`h-1.5 rounded-full ${i === 0 ? 'bg-green-400' : i === 1 ? 'bg-yellow-400' : 'bg-gray-400'}`}
                              style={{ width: `${Math.round(c.confidence * 100)}%` }}
                            />
                          </div>
                          <span className="font-mono w-10 text-right">{(c.confidence * 100).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="mt-2 pt-1 text-gray-500 text-center" style={{ fontSize: '9px' }}>
                    Shift+D to toggle
                  </div>
                </div>
              )}
            </div>

            {cameraError && (
              <div className="mt-4 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                  <div className="space-y-3">
                    <div>
                      <p className="text-sm text-destructive">Camera/Recognition Error</p>
                      <p className="text-sm text-destructive/80">{cameraError}</p>
                    </div>
                    <div className="text-xs text-destructive/60 space-y-1">
                      <p><strong>To enable camera:</strong></p>
                      <ul className="list-disc list-inside space-y-0.5 ml-2">
                        <li>Click the camera icon in your browser's address bar</li>
                        <li>Select "Allow" when prompted for camera permission</li>
                        <li>Refresh the page if needed</li>
                      </ul>
                    </div>
                    <div className="pt-2 border-t border-destructive/20">
                      <p className="text-xs text-destructive/80 mb-2">
                        <strong>Can't enable camera? Try our demo mode instead:</strong>
                      </p>
                      <Button
                        onClick={startDemoMode}
                        variant="outline"
                        size="sm"
                        className="border-destructive/30 text-destructive hover:bg-destructive/10"
                      >
                        <Play className="w-4 h-4 mr-2" />
                        Start Demo Mode
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="flex gap-2 mt-4 flex-wrap">
              {cameraActive ? (
                <>
                  {!isRecognizing ? (
                    <Button onClick={startRecognition}>
                      <Play className="w-4 h-4 mr-2" />
                      Start Recognition
                    </Button>
                  ) : (
                    <Button onClick={stopRecognition} variant="destructive">
                      <Pause className="w-4 h-4 mr-2" />
                      Stop Recognition
                    </Button>
                  )}
                </>
              ) : demoMode ? (
                <>
                  {!isRecognizing ? (
                    <Button onClick={startRecognition}>
                      <Play className="w-4 h-4 mr-2" />
                      Start Demo Recognition
                    </Button>
                  ) : (
                    <Button onClick={stopRecognition} variant="destructive">
                      <Pause className="w-4 h-4 mr-2" />
                      Stop Demo
                    </Button>
                  )}
                  <Button onClick={() => setDemoMode(false)} variant="outline">
                    Exit Demo
                  </Button>
                  <Button
                    onClick={startCamera}
                    variant="outline"
                    size="sm"
                  >
                    <Camera className="w-4 h-4 mr-2" />
                    Try Camera Again
                  </Button>
                </>
              ) : (
                <div className="flex gap-2 flex-wrap">
                  {!cameraError && (
                    <Button onClick={startDemoMode} variant="secondary">
                      <Play className="w-4 h-4 mr-2" />
                      Demo Mode
                    </Button>
                  )}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {currentSign && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                Current Recognition
                {demoMode && (
                  <Badge variant="secondary" className="text-xs">
                    Demo
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="text-center">
                  <div className="text-3xl mb-2">{currentSign.sign}</div>
                  <div className="text-sm text-muted-foreground mb-4">
                    {demoMode ? 'Demo detected' : 'Detected'} at {currentSign.timestamp.toLocaleTimeString()}
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Confidence</span>
                    <span>{Math.round(currentSign.confidence * 100)}%</span>
                  </div>
                  <Progress value={currentSign.confidence * 100} />
                </div>

                {demoMode && (
                  <div className="text-xs text-muted-foreground text-center mt-3 p-2 bg-muted/50 rounded">
                    This is a simulation. Enable camera access for real recognition.
                  </div>
                )}

                <div className="flex items-center justify-center gap-2 mt-4">
                  <Button
                    onClick={replayAudio}
                    variant="outline"
                    size="sm"
                    disabled={!audioEnabled || isSpeaking}
                  >
                    {isSpeaking ? (
                      <>
                        <VolumeX className="w-4 h-4 mr-2" />
                        Speaking...
                      </>
                    ) : (
                      <>
                        <Volume2 className="w-4 h-4 mr-2" />
                        Replay Audio
                      </>
                    )}
                  </Button>
                  <Button
                    onClick={() => setAudioEnabled(!audioEnabled)}
                    variant={audioEnabled ? "outline" : "secondary"}
                    size="sm"
                  >
                    {audioEnabled ? (
                      <>
                        <Volume2 className="w-4 h-4 mr-2" />
                        Audio On
                      </>
                    ) : (
                      <>
                        <VolumeX className="w-4 h-4 mr-2" />
                        Audio Off
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Recognition History
              <Button
                onClick={clearHistory}
                variant="outline"
                size="sm"
                disabled={recognitionHistory.length === 0}
              >
                <RotateCcw className="w-4 h-4 mr-2" />
                Clear
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {recognitionHistory.length === 0 ? (
              <p className="text-muted-foreground text-center py-8">
                No signs recognized yet. {demoMode ? 'Start demo recognition' : 'Start the camera and recognition'} to begin.
              </p>
            ) : (
              <div className="space-y-3">
                {recognitionHistory.map((item, index) => (
                  <div
                    key={`${item.sign}-${item.timestamp.getTime()}`}
                    className="flex items-center justify-between p-3 bg-muted rounded-lg"
                  >
                    <div>
                      <div>{item.sign}</div>
                      <div className="text-sm text-muted-foreground">
                        {item.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                    <Badge variant="secondary">
                      {Math.round(item.confidence * 100)}%
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Sign Language Guide</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm">
              <p className="text-muted-foreground mb-4">
                Position your hand clearly in front of the camera. Here are tips for each sign:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Hello:</strong> Open hand, all fingers extended (wave)
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Thank You:</strong> Open hand, palm up, fingers extended
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Please:</strong> Open palm with fingers extended
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Yes:</strong> Closed fist (nodding motion)
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>No:</strong> Index and middle fingers extended
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Good:</strong> Thumbs up, other fingers closed
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Bad:</strong> Thumbs down, other fingers closed
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Help:</strong> Open hand with fingers extended
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Sorry:</strong> Closed fist
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Love:</strong> Thumb, index, and pinky extended (ILY sign)
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-4">
                Note: These are simplified gesture patterns. For best results, hold each sign clearly for 1-2 seconds.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}