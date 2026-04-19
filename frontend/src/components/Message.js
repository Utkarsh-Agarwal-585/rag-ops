import React from "react";

const BACKEND = "http://localhost:8000";

// Splits answer text into segments: plain text and image references.
// e.g. "Here is the diagram [Image: /storage/images/fig.png] as shown."
// → ["Here is the diagram ", {type:"image", path:"/storage/images/fig.png"}, " as shown."]
function parseContent(text) {
  const parts = [];
  const regex = /\[Image:\s*([^\]]+)\]/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: "text", value: text.slice(lastIndex, match.index) });
    }
    parts.push({ type: "image", path: match[1].trim() });
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push({ type: "text", value: text.slice(lastIndex) });
  }

  return parts;
}

export default function Message({ role, content, sources }) {
  const parts = parseContent(content || "");

  return (
    <div className={`message ${role}`}>
      <span className="message-role">{role}</span>
      <div className="bubble">
        {parts.map((part, i) =>
          part.type === "image" ? (
            <div key={i} className="inline-image-wrapper">
              <img
                src={`${BACKEND}${part.path}`}
                alt="Referenced diagram"
                loading="lazy"
                style={{ maxWidth: "100%", borderRadius: "8px", margin: "8px 0" }}
              />
            </div>
          ) : (
            <span key={i}>{part.value}</span>
          )
        )}
      </div>

      {role === "assistant" && sources && sources.length > 0 && (
        <div className="sources">
          <span className="sources-title">Sources</span>
          {sources.map((src, i) => (
            <div className="source-card" key={i}>
              {src.type === "image" && src.image_path ? (
                <>
                  <img
                    src={`${BACKEND}${src.image_path}`}
                    alt={src.content || "Source image"}
                    loading="lazy"
                  />
                  {src.content && (
                    <p className="image-caption">{src.content}</p>
                  )}
                </>
              ) : (
                <p className="source-content">
                  {src.content?.length > 250
                    ? src.content.slice(0, 250) + "…"
                    : src.content}
                </p>
              )}
              <div className="source-meta">
                <span>{src.source}</span>
                {src.score != null && (
                  <span>score: {src.score.toFixed(3)}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
