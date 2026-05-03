import React from "react";

const BACKEND = "http://localhost:8000";

/**
 * Tokenise answer text into renderable segments.
 *
 * Handles (in order of precedence):
 *   [Image: /path]  → inline <img>
 *   **text**        → <strong> (bold heading)
 *   *text*          → <em> (italic)
 *   \n              → <br> (line break)
 *   everything else → plain <span>
 */
function parseContent(text) {
  const parts = [];
  // Matches [Image:...], **bold**, *italic*, and newlines in one pass.
  const regex = /\[Image:\s*([^\]]+)\]|\*\*([^*]+)\*\*|\*([^*]+)\*|\n/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    // Flush any plain text before this match.
    if (match.index > lastIndex) {
      parts.push({ type: "text", value: text.slice(lastIndex, match.index) });
    }

    if (match[1] !== undefined) {
      // [Image: /path]
      parts.push({ type: "image", path: match[1].trim() });
    } else if (match[2] !== undefined) {
      // **bold**
      parts.push({ type: "bold", value: match[2] });
    } else if (match[3] !== undefined) {
      // *italic*
      parts.push({ type: "italic", value: match[3] });
    } else {
      // \n newline
      parts.push({ type: "br" });
    }

    lastIndex = regex.lastIndex;
  }

  // Flush any remaining plain text after the last match.
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
        {parts.map((part, i) => {
          switch (part.type) {
            case "image":
              return (
                <div key={i} className="inline-image-wrapper">
                  <img
                    src={`${BACKEND}${part.path}`}
                    alt="Referenced diagram"
                    loading="lazy"
                    style={{ maxWidth: "100%", borderRadius: "8px", margin: "8px 0" }}
                  />
                </div>
              );
            case "bold":
              return <strong key={i}>{part.value}</strong>;
            case "italic":
              return <em key={i}>{part.value}</em>;
            case "br":
              return <br key={i} />;
            default:
              return <span key={i}>{part.value}</span>;
          }
        })}
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
