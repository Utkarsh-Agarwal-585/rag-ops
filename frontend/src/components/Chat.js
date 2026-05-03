import React, { useState, useRef, useEffect } from "react";
import Message from "./Message";

export default function Chat({ messages, loading, onSend, disabled }) {
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    onSend(input.trim());
    setInput("");
  };

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 && !loading && (
          <p className="chat-empty">
            {disabled
              ? "Upload a document to unlock the chat."
              : "Ask a question about your uploaded documents."}
          </p>
        )}

        {messages.map((msg, i) => (
          <Message key={i} {...msg} />
        ))}

        {loading && (
          <div className="message assistant">
            <span className="message-role">assistant</span>
            <div className="bubble">
              Thinking<span className="loading-dots"></span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <form className="chat-input-bar" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder={disabled ? "Enter an API key first" : "Ask a question…"}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading || disabled}
          aria-label="Chat input"
        />
        <button
          className="btn btn-primary"
          type="submit"
          disabled={loading || disabled || !input.trim()}
        >
          Send
        </button>
      </form>
    </div>
  );
}
