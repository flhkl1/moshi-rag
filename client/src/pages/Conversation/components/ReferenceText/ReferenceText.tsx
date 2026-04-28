import { FC, useEffect, useRef } from "react";
import { useServerText } from "../../hooks/useServerText";
import { TextDisplayProps, renderTextDisplay } from "../TextDisplay/TextDisplay";

export const ReferenceText: FC<TextDisplayProps> = ({ containerRef, displayColor }) => {
  const { text, textColor, textType } = useServerText();
  const textIndices = textType.map((type, i) => (type === "referencetext" || type === "coloredreferencetext") ? i : -1).filter(i => i !== -1);
  let filteredText = textIndices.map(i => text[i]);
  let filteredTextColor = textIndices.map(i => textColor[i]);
  const resetSymbol = "\0";
  const lastResetIdx = filteredText.lastIndexOf(resetSymbol);
  if (lastResetIdx !== -1) {
    filteredText = filteredText.slice(lastResetIdx + 1);
    filteredTextColor = filteredTextColor.slice(lastResetIdx + 1);
  }
  const currentIndex = filteredText.length - 1;
  const prevScrollTop = useRef(0);

  useEffect(() => {
    if (containerRef.current) {
      prevScrollTop.current = containerRef.current.scrollTop;
      containerRef.current.scroll({
        top: containerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [filteredText]);

  return renderTextDisplay(filteredText, filteredTextColor, displayColor, currentIndex, containerRef);
};
