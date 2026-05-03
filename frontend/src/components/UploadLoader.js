import React from "react";
import ReactDOM from "react-dom";

export default function UploadLoader({ filename }) {
  return ReactDOM.createPortal(
    <div className="loader-overlay" aria-modal="true" role="dialog" aria-label="Uploading file">
      <div className="loader-box">
        <div className="loader-spinner" />
        <p className="loader-title">Processing document…</p>
        {filename && <p className="loader-filename">{filename}</p>}
        <p className="loader-hint">Parsing, chunking &amp; indexing — please wait</p>
      </div>
    </div>,
    document.body
  );
}
