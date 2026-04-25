import React, { useState, useRef } from "react";
import UploadLoader from "./UploadLoader";

export default function Upload({ apiBase, apiKey, onUploadSuccess }) {
  const [status, setStatus] = useState({ text: "", type: "" });
  const [uploading, setUploading] = useState(false);
  const [fileSelected, setFileSelected] = useState(false);
  const [currentFile, setCurrentFile] = useState("");
  const fileRef = useRef(null);

  const handleUpload = async () => {
    const file = fileRef.current?.files[0];
    if (!file) {
      setStatus({ text: "Please select a file first.", type: "error" });
      return;
    }

    setUploading(true);
    setCurrentFile(file.name);
    setStatus({ text: "", type: "" });

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("api_key", apiKey || "");

      const res = await fetch(`${apiBase}/upload`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed (${res.status})`);
      }

      const data = await res.json();
      setStatus({
        text: `${data.source}: ${data.chunks_created} chunks created`,
        type: "success",
      });
      fileRef.current.value = "";
      setFileSelected(false);
      onUploadSuccess();
    } catch (err) {
      setStatus({ text: err.message, type: "error" });
    } finally {
      setUploading(false);
      setCurrentFile("");
    }
  };

  return (
    <>
      {uploading && <UploadLoader filename={currentFile} />}

      <div>
        <h3>Upload Document</h3>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt,.log"
          aria-label="Choose a file to upload"
          onChange={(e) => setFileSelected(!!e.target.files?.[0])}
        />
        <button
          className="btn btn-primary"
          style={{ marginTop: 8, width: "100%" }}
          onClick={handleUpload}
          disabled={uploading || !fileSelected}
        >
          Upload
        </button>
        {status.text && (
          <p className={`upload-status ${status.type}`}>{status.text}</p>
        )}
      </div>
    </>
  );
}
