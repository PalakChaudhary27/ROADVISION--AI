import React, { useState, useRef, useEffect } from 'react';
import { 
  UploadCloud, 
  Image as ImageIcon, 
  Scan, 
  AlertCircle, 
  CheckCircle2, 
  Loader2, 
  Sparkles, 
  FileText,
  MapPin,
  Trash2,
  Cpu,
  Info,
  ArrowRight
} from 'lucide-react';
import { apiUrl } from '../services/api';

export default function NewInspectionPage({ onInferenceSuccess, backendOnline }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [title, setTitle] = useState('');
  const [roadClass, setRoadClass] = useState('Urban Arterial');
  const [surfaceType, setSurfaceType] = useState('Asphalt Concrete');
  const [notes, setNotes] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [progressMsg, setProgressMsg] = useState('');
  const [errorMsg, setErrorMsg] = useState(null);
  const [samples, setSamples] = useState([]);
  const fileInputRef = useRef(null);

  // Load sample gallery from backend
  useEffect(() => {
    fetch(apiUrl('/api/samples'))
      .then(res => res.ok ? res.json() : [])
      .then(data => setSamples(data))
      .catch(() => setSamples([]));
  }, []);

  const handleFileChange = (selectedFile) => {
    setErrorMsg(null);
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith('image/')) {
      setErrorMsg('Please select a valid image file (JPEG, PNG, or WEBP).');
      return;
    }

    if (selectedFile.size > 25 * 1024 * 1024) {
      setErrorMsg('Image size exceeds 25MB limit.');
      return;
    }

    setFile(selectedFile);
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    if (!title) {
      const cleanName = selectedFile.name.replace(/\.[^/.]+$/, "").replace(/[-_]/g, ' ');
      setTitle(`Survey of ${cleanName.charAt(0).toUpperCase() + cleanName.slice(1)}`);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleSelectSample = async (sample) => {
    try {
      setIsLoading(true);
      setProgressMsg(`Loading sample image: ${sample.title}...`);
      setErrorMsg(null);
      const res = await fetch(apiUrl(`/api/samples/${sample.filename}`));
      if (!res.ok) throw new Error("Could not load sample image file.");
      const blob = await res.blob();
      const sampleFile = new File([blob], sample.filename, { type: 'image/jpeg' });
      setTitle(sample.title);
      setNotes(sample.description || `Field survey sample from ${sample.title}`);
      handleFileChange(sampleFile);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setIsLoading(false);
      setProgressMsg('');
    }
  };

  const handleRunInference = async () => {
    if (!file) {
      setErrorMsg('Please select or upload a road image before analyzing.');
      return;
    }

    setIsLoading(true);
    setErrorMsg(null);

    const steps = [
      'Reading source roadway image...',
      'Preprocessing to 512×512 ImageNet tensor...',
      'Executing DeepLabV3-ResNet50 forward pass...',
      'Quantifying pothole & crack pixel areas...',
      'Generating visual overlay & condition assessment...'
    ];

    let stepIndex = 0;
    setProgressMsg(steps[0]);
    const interval = setInterval(() => {
      stepIndex = (stepIndex + 1) % steps.length;
      setProgressMsg(steps[stepIndex]);
    }, 900);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const resp = await fetch(apiUrl('/predict'), {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned error status ${resp.status}`);
      }

      const prediction = await resp.json();
      prediction.title = title || file.name;
      prediction.notes = notes 
        ? `${notes} [Class: ${roadClass}, Surface: ${surfaceType}]`
        : `[Class: ${roadClass}, Surface: ${surfaceType}]`;

      // Pass result to parent app
      onInferenceSuccess(prediction);

    } catch (err) {
      console.error("Inference failure:", err);
      setErrorMsg(err.message || 'Inference failed. Check backend connectivity.');
    } finally {
      clearInterval(interval);
      setIsLoading(false);
      setProgressMsg('');
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Workspace Header */}
      <div className="bg-white p-6 rounded-xl border border-[#E3E8EF] shadow-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold text-blue-600 mb-1">
            <Scan className="w-4 h-4" />
            <span>Autonomous Distress Inspection Workspace</span>
          </div>
          <h2 className="text-xl font-bold text-[#172435] tracking-tight">Inspect New Roadway</h2>
          <p className="text-xs text-slate-500 mt-1 max-w-xl">
            Upload dashcam captures, mobile inspection photos, or drone survey imagery.
            The trained DeepLabV3-ResNet50 model will evaluate pixels for pothole cavities and crack fractures.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-xs font-semibold text-slate-700">
          <Cpu className="w-4 h-4 text-blue-600" />
          <span>DeepLabV3-ResNet50</span>
        </div>
      </div>

      {/* Main Grid: Upload & Inspection Setup */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Drag & Drop Zone + Sample Gallery (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {/* Upload Dropzone */}
          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-6 text-center transition-all flex flex-col items-center justify-center min-h-[340px] relative ${
              previewUrl
                ? 'border-slate-300 bg-[#0B1320]'
                : 'border-slate-300 hover:border-blue-500 bg-white hover:bg-blue-50/20'
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => handleFileChange(e.target.files[0])}
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
            />

            {previewUrl ? (
              <div className="relative w-full h-full flex flex-col items-center">
                <img
                  src={previewUrl}
                  alt="Road Preview"
                  className="max-h-[300px] w-auto object-contain rounded-lg shadow-lg border border-slate-700"
                />
                <div className="mt-3 flex items-center gap-3 text-xs text-slate-300 bg-black/60 backdrop-blur-xs px-3 py-1.5 rounded-full border border-white/10">
                  <span className="font-semibold text-white truncate max-w-[200px]">{file?.name}</span>
                  <span>•</span>
                  <span className="font-mono">{(file?.size / (1024 * 1024)).toFixed(2)} MB</span>
                  <button
                    onClick={() => {
                      setFile(null);
                      setPreviewUrl(null);
                    }}
                    className="p-1 text-red-400 hover:text-red-300 hover:bg-white/10 rounded transition-colors"
                    title="Remove Image"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center space-y-3 py-6">
                <div className="w-14 h-14 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center border border-blue-100 shadow-xs">
                  <UploadCloud className="w-7 h-7" />
                </div>
                <div>
                  <p className="text-sm font-bold text-slate-800">
                    Drag and drop roadway image here
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Supports high-resolution JPEG, PNG, or WEBP (up to 25MB)
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-bold shadow-xs transition-all"
                >
                  Browse Roadway Image
                </button>
              </div>
            )}
          </div>

          {/* Quick Select Verified Sample Gallery */}
          {samples.length > 0 && (
            <div className="bg-white p-4 rounded-xl border border-[#E3E8EF] shadow-xs">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                  <ImageIcon className="w-3.5 h-3.5 text-blue-600" />
                  Or Select Verified Dataset Sample:
                </span>
                <span className="text-[10px] text-slate-400 font-semibold uppercase">1-Click Test</span>
              </div>
              
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                {samples.map((s, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSelectSample(s)}
                    disabled={isLoading}
                    className="p-2 rounded-lg border border-slate-200 hover:border-blue-500 hover:bg-blue-50/30 text-left transition-all group flex flex-col justify-between"
                  >
                    <div>
                      <div className="h-16 w-full rounded bg-slate-100 overflow-hidden mb-1.5 border border-slate-200">
                        <img 
                          src={`/api/samples/${s.filename}`} 
                          alt={s.title}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                          onError={(e) => {
                            e.target.style.display = 'none';
                          }}
                        />
                      </div>
                      <span className="text-[11px] font-bold text-slate-800 line-clamp-1 group-hover:text-blue-600">
                        {s.title}
                      </span>
                    </div>
                    <span className="text-[10px] text-slate-400 block mt-1">Test sample</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Survey Metadata & Action Card (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white p-5 rounded-xl border border-[#E3E8EF] shadow-xs space-y-4">
            <h3 className="text-sm font-bold text-[#172435] pb-2 border-b border-slate-100 flex items-center gap-1.5">
              <FileText className="w-4 h-4 text-blue-600" />
              <span>Survey Metadata</span>
            </h3>

            {/* Title field */}
            <div className="space-y-1">
              <label className="text-xs font-bold text-slate-700 block">
                Roadway / Survey Title <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. National Highway 44 - Milepost 12.4"
                className="w-full text-xs px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-slate-800"
              />
            </div>

            {/* Road Classification Dropdown */}
            <div className="space-y-1">
              <label className="text-xs font-bold text-slate-700 block">
                Functional Road Classification
              </label>
              <select
                value={roadClass}
                onChange={(e) => setRoadClass(e.target.value)}
                className="w-full text-xs px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-slate-800 bg-white"
              >
                <option value="National Highway">National Highway / Expressway</option>
                <option value="Urban Arterial">Urban Arterial Corridor</option>
                <option value="State Highway">State Highway</option>
                <option value="Local Collector">Local Collector / Municipal Road</option>
                <option value="Industrial Bypass">Industrial Logistics Corridor</option>
              </select>
            </div>

            {/* Surface Material */}
            <div className="space-y-1">
              <label className="text-xs font-bold text-slate-700 block">
                Pavement Surface Material
              </label>
              <select
                value={surfaceType}
                onChange={(e) => setSurfaceType(e.target.value)}
                className="w-full text-xs px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-slate-800 bg-white"
              >
                <option value="Asphalt Concrete">Dense Bituminous Asphalt Concrete (AC)</option>
                <option value="Open Graded Friction">Open-Graded Friction Course (OGFC)</option>
                <option value="Rigid Portland Concrete">Rigid Portland Cement Concrete (PCC)</option>
                <option value="Surface Treated Chip Seal">Bituminous Surface Treatment / Chip Seal</option>
              </select>
            </div>

            {/* Inspector Notes */}
            <div className="space-y-1">
              <label className="text-xs font-bold text-slate-700 block">
                Field Observations / Chainage
              </label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={3}
                placeholder="e.g. Surveyed outer driving lane under dry daylight conditions..."
                className="w-full text-xs px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-slate-800 resize-none"
              ></textarea>
            </div>

            {/* Error Message Box */}
            {errorMsg && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-800 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
                <span className="leading-tight">{errorMsg}</span>
              </div>
            )}

            {/* Progress Status Message */}
            {isLoading && (
              <div className="p-3.5 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-900 space-y-2">
                <div className="flex items-center gap-2 font-bold text-blue-800">
                  <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                  <span>Processing Neural Segmentation...</span>
                </div>
                <p className="text-[11px] text-blue-700 font-mono">
                  {progressMsg}
                </p>
                <div className="w-full bg-blue-200 rounded-full h-1.5 overflow-hidden">
                  <div className="bg-blue-600 h-1.5 rounded-full animate-pulse w-3/4"></div>
                </div>
              </div>
            )}

            {/* Main Inference Button */}
            <button
              type="button"
              onClick={handleRunInference}
              disabled={isLoading || !file}
              className={`w-full py-3 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-2 shadow-sm ${
                isLoading || !file
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-600/25'
              }`}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Analyzing Road Surface...</span>
                </>
              ) : (
                <>
                  <Scan className="w-4 h-4" />
                  <span>Execute Neural Segmentation</span>
                  <ArrowRight className="w-4 h-4 ml-1" />
                </>
              )}
            </button>
          </div>

          {/* Model Specification Card */}
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs text-slate-600 space-y-2">
            <div className="flex items-center justify-between text-slate-700 font-bold">
              <span>Model Architecture</span>
              <span className="font-mono text-blue-700">DeepLabV3-ResNet50</span>
            </div>
            <div className="flex items-center justify-between text-[11px]">
              <span>Inference Resolution</span>
              <span className="font-mono text-slate-800">512 × 512 px (Normalized)</span>
            </div>
            <div className="flex items-center justify-between text-[11px]">
              <span>Decision Space</span>
              <span className="font-mono text-slate-800">Argmax (Discrete Mask)</span>
            </div>
            <div className="flex items-center justify-between text-[11px]">
              <span>Output Classes</span>
              <span className="font-mono text-slate-800">Pothole (#FF3B30), Crack (#FFCC00)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
