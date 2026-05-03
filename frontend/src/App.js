import React, { useState, useEffect, useCallback } from "react";
import Upload from "./components/Upload";
import Chat from "./components/Chat";
import DocList from "./components/DocList";

const API_BASE = "http://localhost:8000/api/v1";

export default function App() {
  const [apiKey, setApiKey] = useState("");
  const [provider, setProvider] = useState("gemini");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploaded, setUploaded] = useState(false);
  const [contextLost, setContextLost] = useState(false);
  const [uploadCount, setUploadCount] = useState(0);
  const [uploadResetKey, setUploadResetKey] = useState(0);

  // Sync the `uploaded` flag with the actual document count from the backend.
  // Runs on mount (handles persistence across restarts) and whenever
  // uploadCount changes (after upload or delete).
  const syncDocumentState = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (res.ok) {
        const { total_documents } = await res.json();
        setUploaded(total_documents > 0);
        if (total_documents === 0) setContextLost(false);
      }
    } catch (_) {}
  }, []);

  useEffect(() => {
    syncDocumentState();
  }, [syncDocumentState, uploadCount]);

  const checkServerContext = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (res.ok) {
        const { total_documents } = await res.json();
        const hasDocuments = total_documents > 0;
        if (!hasDocuments && uploaded) setContextLost(true);
        setUploaded(hasDocuments);
      }
    } catch (_) {}
  }, [uploaded]);

  useEffect(() => {
    window.addEventListener("focus", checkServerContext);
    return () => window.removeEventListener("focus", checkServerContext);
  }, [checkServerContext]);

  const handleSend = async (query) => {
    if (!query.trim() || loading) return;

    // Guard: require an API key before hitting the backend.
    if (!apiKey.trim()) {
      setMessages((prev) => [
        ...prev,
        { role: "user", content: query },
        {
          role: "assistant",
          content: "Please enter your API key in the sidebar before asking questions.",
          sources: [],
        },
      ]);
      return;
    }

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
        // Pydantic validation errors return detail as an array of objects.
        const detail = Array.isArray(err.detail)
          ? err.detail.map((e) => e.msg || JSON.stringify(e)).join(", ")
          : err.detail || `Request failed (${res.status})`;
        throw new Error(detail);
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
            resetKey={uploadResetKey}
            onUploadSuccess={() => {
              setUploaded(true);
              setContextLost(false);
              setUploadCount((c) => c + 1);
            }}
          />

          <DocList
            apiBase={API_BASE}
            refreshTrigger={uploadCount}
            onDocDeleted={() => setUploadCount((c) => c + 1)}
            onAllDocsDeleted={() => {
              setMessages([]);
              setUploaded(false);
              setUploadResetKey((k) => k + 1);
            }}
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
