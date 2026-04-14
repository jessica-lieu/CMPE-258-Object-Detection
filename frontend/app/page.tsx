"use client";
import React, { useState } from 'react';

export default function VideoProcessor() {
  const [engine, setEngine] = useState("yolo_trt");
  const [loading, setLoading] = useState(false);
  const [processedUrl, setProcessedUrl] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [annotationsFile, setAnnotationsFile] = useState<File | null>(null);
  const [history, setHistory] = useState<any[]>([]);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setSelectedFile(e.target.files[0]);
  };

  const runInference = async () => {
    if (!selectedFile) return alert("Select a video!");
    setLoading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);
    if (annotationsFile) {
      formData.append("annotations", annotationsFile);
    }

    try {
      const start = performance.now();
      const resp = await fetch(`http://localhost:8000/process?engine=${engine}`, {
        method: "POST",
        body: formData
      });

      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      setProcessedUrl(url);

      // Get latency from custom header or calculate it
      const serverLatency = resp.headers.get("X-Inference-Latency");
      const mapScore = resp.headers.get("X-Inference-mAP");
      const totalTime = (performance.now() - start).toFixed(2);

      setHistory(prev => [{
        id: Date.now(),
        engine: engine.toUpperCase(),
        latency: serverLatency || totalTime,
        mapScore: mapScore ? `${mapScore}%` : 'N/A'
      }, ...prev]);

    } catch (err) {
      console.error(err);
      alert("Error processing video.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-8 font-sans">
      <h1 className="text-3xl font-bold mb-6 text-center">Object Detection</h1>

      <div className="bg-gray-100 p-6 rounded-xl mb-8 flex gap-4 items-end justify-center">
        <div>
          <label className="block text-sm font-bold mb-1 text-black">Acceleration Engine</label>
          <select value={engine} onChange={(e) => setEngine(e.target.value)} className="border p-2 rounded bg-white text-black">
            <option value="yolo_trt">YOLO + TensorRT</option>
            <option value="yolo_onnx">YOLO + ONNX</option>
            <option value="detr_trt">RT-DETR + TensorRT</option>
            <option value="detr_onnx">RT-DETR + ONNX</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-bold mb-1 text-black px-1">Video File</label>
          <input type="file" accept="video/*" onChange={onFileChange} className="border p-1 bg-white rounded text-black block w-full max-w-xs" />
        </div>
        <div>
          <label className="block text-sm font-bold mb-1 text-black px-1">Annotations (.zip)</label>
          <input type="file" accept=".zip" onChange={(e) => setAnnotationsFile(e.target.files?.[0] || null)} className="border p-1 bg-white rounded text-black block w-full max-w-xs" />
        </div>
        <button 
          onClick={runInference} 
          disabled={loading}
          className="bg-blue-600 text-white px-6 py-2 rounded font-bold hover:bg-blue-700 disabled:bg-gray-400"
        >
          {loading ? "Processing (This takes a moment)..." : "Start Inference"}
        </button>
      </div>

      {/* RESULT VIEW */}
      <div className="bg-black rounded-2xl overflow-hidden shadow-2xl min-h-[400px] flex items-center justify-center">
        {processedUrl ? (
          <video 
            key={processedUrl}
            src={processedUrl} 
            controls 
            autoPlay 
            className="w-full h-auto" 
          />
        ) : (
          <div className="text-gray-500 italic">No video processed yet.</div>
        )}
      </div>

      {/* BENCHMARK TABLE */}
      <div className="mt-10">
        <h2 className="text-xl font-bold mb-4">Hardware Performance Metrics</h2>
        <table className="w-full border-collapse bg-white shadow-sm rounded-lg overflow-hidden">
          <thead className="bg-gray-800 text-white">
            <tr>
              <th className="p-3 text-left">Configuration</th>
              <th className="p-3 text-left">Inference Latency (Total)</th>
              <th className="p-3 text-left">mAP (50)</th>
            </tr>
          </thead>
          <tbody>
            {history.map(row => (
              <tr key={row.id} className="border-b bg-white text-black">
                <td className="p-3 font-mono">{row.engine}</td>
                <td className="p-3 text-blue-600 font-bold">{row.latency} ms</td>
                <td className="p-3 text-green-600 font-bold">{row.mapScore}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}