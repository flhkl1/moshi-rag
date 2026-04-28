import { FC, useEffect, useRef } from "react";
import { useServerText } from "../../hooks/useServerText";

export type TextDisplayProps = {
  containerRef: React.RefObject<HTMLDivElement>;
  displayColor: boolean | undefined;
};

// Palette 2: Purple to Green Moshi
// sns.diverging_palette(288, 145, s=90, l=72, n=11)
export const textDisplayColors = [
  "#d19bf7", "#d7acf6", "#debdf5", "#e4cef4",
  "#ebe0f3", "#eef2f0", "#c8ead9", "#a4e2c4",
  "#80d9af", "#5bd09a", "#38c886"]

export function clamp_color(v: number) {
  return v <= 0
    ? 0
    : v >= textDisplayColors.length
      ? textDisplayColors.length
      : v
}

export const TextDisplay: FC<TextDisplayProps> = ({
  containerRef, displayColor
}) => {
  const { text, textColor, textType } = useServerText();
  // Only show main chat lines (not reference)
  const textIndices = textType.map((type, i) => (type === "text" || type === "coloredtext") ? i : -1).filter(i => i !== -1);
  const filteredText = textIndices.map(i => text[i]);
  const filteredTextColor = textIndices.map(i => textColor[i]);
  const currentIndex = text.length - 1;
  const prevScrollTop = useRef(0);

  useEffect(() => {
    if (containerRef.current) {
      prevScrollTop.current = containerRef.current.scrollTop;
      containerRef.current.scroll({
        top: containerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [text]);

  return renderTextDisplay(filteredText, filteredTextColor, displayColor, currentIndex, containerRef);
};

export function renderTextDisplay(
  text: string[],
  textColor: number[],
  displayColor: boolean | undefined,
  currentIndex: number,
  containerRef: React.RefObject<HTMLDivElement>
) {
  if (displayColor && (textColor.length == text.length)) {
    return (
      <div className="h-full w-full max-w-full max-h-full  p-2 text-white whitespace-pre-line" ref={containerRef}>
        {text.map((t, i) => (
          <span
            key={i}
            className={`${i === currentIndex ? "font-bold" : "font-normal"}`}
            style={{ color: `${textDisplayColors[clamp_color(textColor[i])]}` }}
          >
            {t}
          </span>
        ))}
      </div>
    );
  } else {
    return (
      <div className="h-full w-full max-w-full max-h-full  p-2 text-white whitespace-pre-line" ref={containerRef}>
        {text.map((t, i) => (
          <span
            key={i}
            className={`${i === currentIndex ? "font-bold" : "font-normal"}`}
          >
            {t}
          </span>
        ))}
      </div>
    );
  }
}
