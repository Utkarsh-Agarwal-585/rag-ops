import React, { useState, useEffect, useCallback } from "react";

/**
 * Collapsible list of all documents uploaded to the backend.
 * Each row has a delete button that removes the document and all its data.
 * Refreshes automatically after each upload via the `refreshTrigger` prop.
 */
export default function DocList({ apiBase, refreshTrigger, onDocDeleted, onAllDocsDeleted }) {
  const [docs, setDocs] = useState([]);
  const [open, setOpen] = useState(true);
  const [loading, setLoading] = useState(false);
  const [deletingDoc, setDeletingDoc] = useState(null); // filename being deleted

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocs(data.documents || []);
        return data.total_documents ?? (data.documents?.length ?? 0);
      }
    } catch (_) {
      // Silently ignore — backend may not be ready yet.
    } finally {
      setLoading(false);
    }
    return null;
  }, [apiBase]);

  // Fetch on mount and whenever a new upload completes.
  useEffect(() => {
    fetchDocs();
  }, [fetchDocs, refreshTrigger]);

  const handleDelete = async (docName) => {
    const confirmed = window.confirm(
      `Delete "${docName}"?\n\nThis will remove all chunks, embeddings, and extracted images for this document.`
    );
    if (!confirmed) return;

    setDeletingDoc(docName);
    try {
      const res = await fetch(
        `${apiBase}/documents/${encodeURIComponent(docName)}`,
        { method: "DELETE" }
      );

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert(`Delete failed: ${err.detail || res.status}`);
        return;
      }

      const result = await res.json();

      // Warn the user if any step had issues (non-fatal).
      if (result.warnings?.length > 0) {
        console.warn("Delete warnings:", result.warnings);
      }

      // Refresh the list; if no documents remain, notify parent to clear chat.
      const remaining = await fetchDocs();
      onDocDeleted?.(docName);
      if (remaining === 0) onAllDocsDeleted?.();
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    } finally {
      setDeletingDoc(null);
    }
  };

  return (
    <div className="doc-list">
      <button
        className="doc-list-toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="doc-list-toggle-icon">{open ? "▾" : "▸"}</span>
        Docs uploaded so far
        {docs.length > 0 && (
          <span className="doc-list-count">{docs.length}</span>
        )}
      </button>

      {open && (
        <ul className="doc-list-items">
          {loading && docs.length === 0 && (
            <li className="doc-list-empty">Loading…</li>
          )}
          {!loading && docs.length === 0 && (
            <li className="doc-list-empty">No documents uploaded yet.</li>
          )}
          {docs.map((doc) => (
            <li key={doc.name} className="doc-list-item">
              <span className="doc-list-name">{doc.name}</span>
              <span className="doc-list-meta">
                {doc.total} chunk{doc.total !== 1 ? "s" : ""}
              </span>
              <button
                className="doc-list-delete"
                onClick={() => handleDelete(doc.name)}
                disabled={deletingDoc === doc.name}
                aria-label={`Delete ${doc.name}`}
                title="Delete document"
              >
                {deletingDoc === doc.name ? "…" : "✕"}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
