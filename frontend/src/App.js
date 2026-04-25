import React, { useState, useEffect, useCallback } from "react";
import Upload from "./components/Upload";
import Chat from "./components/Chat";

const API_BASE = "http://localhost:8000/api/v1";

export default function App() {
  const [apiKey, setApiKey] = useState("");
  const [provider, setProvider] = useState("gemini");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploaded, setUploaded] = useState(false);
  const [contextLost, setContextLost] = useState(false);

  const checkServerContext = useCallback(async () => {
    if (!uploaded) return;
    try {
      const res = await fetch(`${API_BASE}/chunks/stats`);
      if (res.ok) {
        const { total_chunks } = await res.json();
        if (total_chunks === 0) {
          setUploaded(false);
          setContextLost(true);
        }
      }
    } catch (_) {}
  }, [uploaded]);

  useEffect(() => {
    window.addEventListener("focus", checkServerContext);
    return () => window.removeEventListener("focus", checkServerContext);
  }, [checkServerContext]);

  const handleSend = async (query) => {
    if (!query.trim() || loading) return;

    const userMsg = { role: "user", content: query };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setLoading(true);

    // Build history from the last 5 turns (10 messages) before this query.
    // Exclude error messages so they don't pollute the context.
    const history = messages
      .filter((m) => !m.content.startsWith("Error:"))
      .slice(-10)
      .map(({ role, content }) => ({ role, content }));

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          api_key: apiKey,
          provider,
          history,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer || "No answer found.",
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${err.message}`, sources: [] },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">RAG Assistant</header>
      {contextLost && (
        <div className="context-lost-banner">
          The server was restarted and your documents are no longer loaded.
          Please re-upload your file to continue.
        </div>
      )}
      <div className="app-body">
        <aside className="sidebar">
          <div>
            <h3>Settings</h3>
            <label htmlFor="api-key">API Key</label>
            <input
              id="api-key"
              type="password"
              placeholder="Enter your API key"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </div>

          <div>
            <label htmlFor="provider">Provider</label>
            <select
              id="provider"
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
            >
              <option value="gemini">Gemini</option>
              <option value="openai">OpenAI</option>
            </select>
          </div>

          <Upload
            apiBase={API_BASE}
            apiKey={apiKey}
            onUploadSuccess={() => { setUploaded(true); setContextLost(false); }}
          />
        </aside>

        <Chat
          messages={messages}
          loading={loading}
          onSend={handleSend}
          disabled={!uploaded}
        />
      </div>
    </div>
  );
}
