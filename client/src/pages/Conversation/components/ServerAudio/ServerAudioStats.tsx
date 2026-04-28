import { useState, useEffect, useRef } from "react";
import { colors } from "../../../../theme/colors";

type ServerAudioStatsProps = {
  getAudioStats: React.MutableRefObject<
    () => {
      playedAudioDuration: number;
      missedAudioDuration: number;
      totalAudioMessages: number;
      delay: number;
      minPlaybackDelay: number;
      maxPlaybackDelay: number;
    }
  >;
};

export const ServerAudioStats = ({ getAudioStats }: ServerAudioStatsProps) => {
  const [audioStats, setAudioStats] = useState(getAudioStats.current());

  const movingAverageSum = useRef<number>(0.);
  const movingAverageCount = useRef<number>(0.);
  const movingBeta = 0.85;

  const convertMinSecs = (total_secs: number) => {
    if (!Number.isFinite(total_secs) || total_secs < 0) {
      total_secs = 0;
    }
    // convert secs to the format mm:ss.cc
    let mins = (Math.floor(total_secs / 60)).toString();
    let secs = (Math.floor(total_secs) % 60).toString();
    let cents = (Math.floor(100 * (total_secs - Math.floor(total_secs)))).toString();
    if (secs.length < 2) {
      secs = "0" + secs;
    }
    if (cents.length < 2) {
      cents = "0" + cents;
    }
    return mins + ":" + secs + "." + cents;
  };

  useEffect(() => {
    const interval = setInterval(() => {
      const newAudioStats = getAudioStats.current();
      setAudioStats(newAudioStats);
      movingAverageCount.current *= movingBeta;
      movingAverageCount.current += (1 - movingBeta) * 1;
      movingAverageSum.current *= movingBeta;
      movingAverageSum.current += (1 - movingBeta) * newAudioStats.delay;

    }, 141);
    return () => {
      clearInterval(interval);
    };
  }, []);

  const movingLatency =
    movingAverageCount.current > 0
      ? movingAverageSum.current / movingAverageCount.current
      : audioStats.delay;

  return (
    <div className="w-full text-sm" style={{ color: colors.textPrimary }}>
      <table>
        <tbody>
          <tr>
            <td className="text-sm pr-2">Audio played: </td>
            <td>{convertMinSecs(audioStats.playedAudioDuration)}</td>
          </tr>
          <tr>
            <td className="text-sm pr-2">Missed audio: </td>
            <td>{convertMinSecs(audioStats.missedAudioDuration)}</td>
          </tr>
          <tr>
            <td className="text-sm pr-2">Latency: </td>
            <td>{Number.isFinite(movingLatency) ? movingLatency.toFixed(3) : "0.000"}</td>
          </tr>
          <tr>
            <td className="text-sm pr-2">Min/Max buffer: </td>
            <td>{audioStats.minPlaybackDelay.toFixed(3)} / {audioStats.maxPlaybackDelay.toFixed(3)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};
