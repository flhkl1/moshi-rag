import { useCallback, useEffect, useRef, useState } from "react";
import { useMediaContext } from "../MediaContext";
import { getExtension, getMimeType } from "../getMimeType";
import fixWebmDuration from "webm-duration-fix";
import { colors } from "../../../theme/colors";

type UseRecordingArgs = {
  transcriptText: string;
  retrievalText: string;
};

const CANVAS_WIDTH = 1920;
const CANVAS_HEIGHT = 1080;

const drawWrappedText = (
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  lineHeight: number,
  maxLines: number,
  stickToBottom = false,
) => {
  const lines: string[] = [];
  const hasContent = text.trim().length > 0;
  const paragraphs = text.split("\n");
  for (const paragraph of paragraphs) {
    const words = paragraph.split(/\s+/).filter(Boolean);
    if (words.length === 0) {
      lines.push("");
      continue;
    }
    let currentLine = "";
    for (const word of words) {
      const testLine = currentLine ? `${currentLine} ${word}` : word;
      if (ctx.measureText(testLine).width <= maxWidth) {
        currentLine = testLine;
        continue;
      }
      if (currentLine) {
        lines.push(currentLine);
      }
      currentLine = word;
    }
    if (currentLine) {
      lines.push(currentLine);
    }
  }
  let visibleLines = stickToBottom ? lines.slice(-maxLines) : lines.slice(0, maxLines);
  if (lines.length > maxLines && hasContent) {
    if (stickToBottom) {
      visibleLines = [...visibleLines];
      visibleLines[0] = `... ${visibleLines[0]}`;
    } else {
      visibleLines = [...visibleLines];
      const lastIdx = visibleLines.length - 1;
      visibleLines[lastIdx] = `${visibleLines[lastIdx].slice(0, Math.max(0, visibleLines[lastIdx].length - 3))}...`;
    }
  }
  const baseY = y;
  visibleLines.forEach((line, idx) => {
    ctx.fillText(line, x, baseY + idx * lineHeight);
  });
};

export const useRecording = ({ transcriptText, retrievalText }: UseRecordingArgs) => {
  const { audioStreamDestination, visualizerCanvasRef } = useMediaContext();
  const audioRecorderRef = useRef<MediaRecorder | null>(null);
  const videoRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const videoChunksRef = useRef<Blob[]>([]);
  const videoStreamRef = useRef<MediaStream | null>(null);
  const compositorCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const compositorRafRef = useRef<number | null>(null);
  const transcriptTextRef = useRef("");
  const retrievalTextRef = useRef("");
  const [isAudioRecording, setIsAudioRecording] = useState(false);
  const [isVideoRecording, setIsVideoRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [videoBlob, setVideoBlob] = useState<Blob | null>(null);

  useEffect(() => {
    transcriptTextRef.current = transcriptText;
  }, [transcriptText]);

  useEffect(() => {
    retrievalTextRef.current = retrievalText;
  }, [retrievalText]);

  const downloadBlob = useCallback((blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, []);

  const startAudioRecording = useCallback(() => {
    if (audioRecorderRef.current?.state === "recording") {
      console.log("Audio recording already in progress");
      return;
    }

    const mimeType = getMimeType("audio");
    const recorder = new MediaRecorder(audioStreamDestination.current.stream, {
      mimeType,
    });

    audioChunksRef.current = [];
    setAudioBlob(null);
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };

    recorder.start();
    audioRecorderRef.current = recorder;
    setIsAudioRecording(true);
  }, [audioStreamDestination]);

  const stopAudioRecording = useCallback(async () => {
    const recorder = audioRecorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      return;
    }
    await new Promise<void>((resolve) => {
      recorder.onstop = async () => {
        const mimeType = getMimeType("audio");
        let blob: Blob;
        if (mimeType.includes("webm")) {
          blob = await fixWebmDuration(new Blob(audioChunksRef.current, { type: mimeType }));
        } else {
          blob = new Blob(audioChunksRef.current, { type: mimeType });
        }
        setAudioBlob(blob);
        audioChunksRef.current = [];
        audioRecorderRef.current = null;
        setIsAudioRecording(false);
        resolve();
      };
      recorder.stop();
    });
  }, []);

  const startCompositor = useCallback(() => {
    if (!compositorCanvasRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = CANVAS_WIDTH;
      canvas.height = CANVAS_HEIGHT;
      compositorCanvasRef.current = canvas;
    }
    const canvas = compositorCanvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    const draw = () => {
      // background
      ctx.fillStyle = colors.bgCanvas;
      ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
      ctx.strokeStyle = colors.border;
      ctx.lineWidth = 1;
      for (let x = 0; x < CANVAS_WIDTH; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, CANVAS_HEIGHT);
        ctx.stroke();
      }
      for (let y = 0; y < CANVAS_HEIGHT; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(CANVAS_WIDTH, y);
        ctx.stroke();
      }

      // header bar
      ctx.fillStyle = colors.bgHeader;
      ctx.fillRect(0, 0, CANVAS_WIDTH, 84);
      ctx.strokeStyle = colors.border;
      ctx.beginPath();
      ctx.moveTo(0, 84);
      ctx.lineTo(CANVAS_WIDTH, 84);
      ctx.stroke();

      ctx.font = "600 38px sans-serif";
      ctx.fillStyle = colors.accentGreen;
      ctx.fillText("Moshi", 72, 54);
      const moshiWidth = ctx.measureText("Moshi").width;
      ctx.fillStyle = colors.accentBlue;
      ctx.fillText("RAG", 72 + moshiWidth, 54);

      // Left pane
      ctx.fillStyle = colors.bgPanel;
      ctx.fillRect(56, 130, 760, 890);
      ctx.fillStyle = colors.textPrimary;
      ctx.font = "600 26px sans-serif";
      ctx.fillText("Voice Visualizer", 84, 182);

      const orbSource = visualizerCanvasRef.current;
      if (orbSource) {
        const orbSize = 560;
        const orbX = 156;
        const orbY = 240;
        ctx.drawImage(orbSource, orbX, orbY, orbSize, orbSize);
      } else {
        ctx.fillStyle = colors.border;
        ctx.beginPath();
        ctx.arc(436, 520, 220, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Right top retrieval pane
      ctx.fillStyle = colors.bgPanel;
      ctx.fillRect(860, 130, 1004, 360);
      ctx.fillStyle = colors.textTranscript;
      ctx.font = "600 26px sans-serif";
      ctx.fillText("Retrieval", 892, 182);
      ctx.fillStyle = colors.textTranscript;
      ctx.font = "400 24px sans-serif";
      drawWrappedText(
        ctx,
        retrievalTextRef.current || "",
        892,
        228,
        940,
        34,
        7,
      );

      // Right bottom transcript pane
      ctx.fillStyle = colors.bgPanel;
      ctx.fillRect(860, 520, 1004, 500);
      ctx.fillStyle = colors.textTranscript;
      ctx.font = "600 26px sans-serif";
      ctx.fillText("Transcript", 892, 572);
      ctx.fillStyle = colors.textTranscript;
      ctx.font = "400 24px sans-serif";
      drawWrappedText(
        ctx,
        transcriptTextRef.current || "",
        892,
        620,
        940,
        34,
        11,
        true,
      );

      compositorRafRef.current = window.requestAnimationFrame(draw);
    };
    draw();
  }, [visualizerCanvasRef]);

  const stopCompositor = useCallback(() => {
    if (compositorRafRef.current !== null) {
      window.cancelAnimationFrame(compositorRafRef.current);
      compositorRafRef.current = null;
    }
  }, []);

  const startVideoRecording = useCallback(() => {
    if (videoRecorderRef.current?.state === "recording") {
      console.log("Video recording already in progress");
      return;
    }
    startCompositor();
    const canvas = compositorCanvasRef.current;
    if (!canvas) {
      console.error("Compositor canvas unavailable");
      return;
    }
    const captureStream = canvas.captureStream(30);
    const mixedAudioTrack = audioStreamDestination.current.stream.getAudioTracks()[0];
    if (mixedAudioTrack) {
      captureStream.addTrack(mixedAudioTrack);
    }
    const mimeType = getMimeType("video");
    const recorder = new MediaRecorder(captureStream, {
      mimeType,
      videoBitsPerSecond: 1_000_000,
    });

    videoChunksRef.current = [];
    setVideoBlob(null);
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        videoChunksRef.current.push(event.data);
      }
    };

    recorder.start();
    videoRecorderRef.current = recorder;
    videoStreamRef.current = captureStream;
    setIsVideoRecording(true);
  }, [audioStreamDestination, startCompositor]);

  const downloadAudio = useCallback(() => {
    stopAudioRecording();
  }, [stopAudioRecording]);

  const stopVideoRecording = useCallback(async () => {
    const recorder = videoRecorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      return;
    }
    await new Promise<void>((resolve) => {
      recorder.onstop = async () => {
        const mimeType = getMimeType("video");
        let blob: Blob;
        if (mimeType.includes("webm")) {
          blob = await fixWebmDuration(new Blob(videoChunksRef.current, { type: mimeType }));
        } else {
          blob = new Blob(videoChunksRef.current, { type: mimeType });
        }
        setVideoBlob(blob);
        videoChunksRef.current = [];
        stopCompositor();
        if (videoStreamRef.current) {
          videoStreamRef.current.getTracks().forEach((track) => track.stop());
          videoStreamRef.current = null;
        }
        videoRecorderRef.current = null;
        setIsVideoRecording(false);
        resolve();
      };
      recorder.stop();
    });
  }, [stopCompositor]);

  const downloadVideo = useCallback(() => {
    stopVideoRecording();
  }, [stopVideoRecording]);

  const saveAudio = useCallback(() => {
    if (!audioBlob) {
      return;
    }
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    downloadBlob(audioBlob, `moshirag-audio-${timestamp}.${getExtension("audio")}`);
  }, [audioBlob, downloadBlob]);

  const saveVideo = useCallback(() => {
    if (!videoBlob) {
      return;
    }
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    downloadBlob(videoBlob, `moshirag-video-${timestamp}.${getExtension("video")}`);
  }, [videoBlob, downloadBlob]);

  useEffect(() => {
    return () => {
      if (audioRecorderRef.current && audioRecorderRef.current.state !== "inactive") {
        audioRecorderRef.current.stop();
      }
      if (videoRecorderRef.current && videoRecorderRef.current.state !== "inactive") {
        videoRecorderRef.current.stop();
      }
      stopCompositor();
      if (videoStreamRef.current) {
        videoStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stopCompositor]);

  return {
    startAudioRecording,
    stopAudioRecording,
    downloadAudio,
    saveAudio,
    startVideoRecording,
    stopVideoRecording,
    downloadVideo,
    saveVideo,
    isAudioRecording,
    isVideoRecording,
    hasAudioRecording: audioBlob !== null,
    hasVideoRecording: videoBlob !== null,
    isRecording: isAudioRecording || isVideoRecording,
  };
};
