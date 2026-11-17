import React, { useState } from 'react';
import ImageComparison from '../components/ImageComparison';
import SeverityBadge from '../components/SeverityBadge';
import EmptyState from '../components/EmptyState';
import { 
  Download, 
  Save, 
  CheckCircle2, 
  AlertTriangle, 
  Layers, 
  Scan, 
  FileText, 
  ChevronRight,
  ShieldAlert,
  ArrowLeft,
  Sparkles,
  Info,
  Check,
  CheckCheck
} from 'lucide-react';

export default function DetectionResultsPage({ 
  currentResult, 
  onSaveInspection, 
  onGenerateReport, 
  onNewInspection,
  isSaved,
  isSaving
}) {
  if (!currentResult) {
    return (
      <EmptyState
        icon={Scan}
        title="No Active Inspection"
        description="Upload an image in the New Inspection module or select a saved survey from the history registry."
        actionLabel="Start Inspection Now"
        onAction={onNewInspection}
      />
    );
  }

  const {
    title,
    filename,
    pothole_area_percentage,
    crack_area_percentage,
    total_damaged_area_percentage,
    background_area_percentage,
    severity,
    severity_description,
    image_dimensions,
    metrics,
    images,
    notes
  } = currentResult;

  const handleDownloadOverlay = () => {
    const link = document.createElement('a');
    link.href = images.overlay;
    link.download = `roadvision_${filename || 'inspection'}_overlay.jpg`;
    link.click();
  };

  const handleDownloadMask = () => {
    const link = document.createElement('a');
    link.href = images.mask;
    link.download = `roadvision_${filename || 'inspection'}_mask.png`;
    link.click();
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Top Banner with Actions */}
      <div className="bg-white p-5 rounded-xl border border-[#E3E8EF] shadow-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">
            <span>Inspection Output</span>
            <span>•</span>
            <span className="text-blue-600 font-mono font-semibold">{currentResult.id || 'LIVE-EVALUATION'}</span>
          </div>
          <h2 className="text-xl font-bold text-[#172435] tracking-tight">{title || filename}</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Model: DeepLabV3-ResNet50 • Input: {image_dimensions?.width}×{image_dimensions?.height} px • Status: Analysis Finalized
          </p>
        </div>

        {/* Action Buttons Group */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={onNewInspection}
            className="px-3 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-bold transition-colors flex items-center gap-1.5"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>New</span>
          </button>

          <button
            onClick={handleDownloadOverlay}
            className="px-3 py-2 bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 rounded-lg text-xs font-bold transition-colors flex items-center gap-1.5 shadow-xs"
            title="Download full-resolution blended distress overlay JPEG"
          >
            <Download className="w-3.5 h-3.5 text-blue-600" />
            <span>Overlay</span>
          </button>

          <button
            onClick={handleDownloadMask}
            className="px-3 py-2 bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 rounded-lg text-xs font-bold transition-colors flex items-center gap-1.5 shadow-xs"
            title="Download pure segmentation mask PNG"
          >
            <Download className="w-3.5 h-3.5 text-amber-600" />
            <span>Mask</span>
          </button>

          <button
            onClick={() => onGenerateReport(currentResult)}
            className="px-3.5 py-2 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 rounded-lg text-xs font-bold transition-colors flex items-center gap-1.5"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Generate PDF</span>
          </button>

          <button
            onClick={() => onSaveInspection(currentResult)}
            disabled={isSaved || isSaving}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 shadow-xs ${
              isSaved
                ? 'bg-emerald-600 text-white cursor-default'
                : 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-600/30'
            }`}
          >
            {isSaved ? (
              <>
                <Check className="w-3.5 h-3.5" />
                <span>Saved to Registry</span>
              </>
            ) : isSaving ? (
              <span>Saving...</span>
            ) : (
              <>
                <Save className="w-3.5 h-3.5" />
                <span>Save to History</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Grid: Multi-view Imagery (Left) + Detailed Metrics Panel (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Multi-view Image Comparison (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          <ImageComparison
            original={images.original || images.overlay}
            overlay={images.overlay}
            mask={images.mask}
            dimensions={image_dimensions}
          />

          {notes && (
            <div className="bg-white p-4 rounded-xl border border-[#E3E8EF] text-xs text-slate-700 shadow-xs">
              <span className="font-bold text-slate-900 block mb-1">Inspection & Observer Notes:</span>
              <p className="text-slate-600 leading-relaxed italic">{notes}</p>
            </div>
          )}
        </div>

        {/* Right: Damage Quantification & Metrics (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          {/* Severity Classification Card */}
          <div className="bg-white p-5 rounded-xl border border-[#E3E8EF] shadow-xs">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">
              Pavement Health Assessment
            </span>
            <div className="flex items-center justify-between">
              <span className="text-base font-bold text-slate-900">Condition Severity</span>
              <SeverityBadge severity={severity} size="lg" />
            </div>
            <p className="text-xs text-slate-600 mt-2.5 bg-slate-50 p-3 rounded-lg border border-slate-200/80 leading-relaxed">
              {severity_description}
            </p>
          </div>

          {/* Area Metrics Cards */}
          <div className="bg-white p-5 rounded-xl border border-[#E3E8EF] shadow-xs space-y-3.5">
            <h3 className="text-xs font-bold text-[#172435] uppercase tracking-wider pb-2 border-b border-slate-100 flex items-center justify-between">
              <span>Distress Quantification</span>
              <span className="text-[10px] text-slate-400 font-mono font-normal">Pixel Census</span>
            </h3>

            {/* Total Damaged Area Hero */}
            <div className="p-4 rounded-xl bg-[#101D30] text-white shadow-xs border border-slate-800">
              <div className="flex items-center justify-between text-xs text-slate-300 mb-1">
                <span className="font-semibold text-slate-300">Total Damaged Surface</span>
                <ShieldAlert className="w-4 h-4 text-amber-400" />
              </div>
              <div className="text-3xl font-extrabold tracking-tight text-white">
                {total_damaged_area_percentage}%
              </div>
              <p className="text-[10px] text-slate-400 mt-1">
                Combined pothole cavity & crack pixel footprint
              </p>
            </div>

            {/* Pothole Area Metric */}
            <div className="p-3.5 rounded-lg border border-red-200 bg-red-50/50">
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded bg-[#FF3B30] flex-shrink-0"></span>
                  <span className="text-xs font-bold text-red-950">Pothole Area (Class 1)</span>
                </div>
                <span className="text-sm font-extrabold text-[#FF3B30]">{pothole_area_percentage}%</span>
              </div>
              <div className="flex items-center justify-between text-[11px] text-slate-600 mt-1 pt-1 border-t border-red-200/50">
                <span>Pixel Footprint:</span>
                <span className="font-mono font-bold text-slate-800">
                  {metrics?.pixels?.pothole?.toLocaleString()} px
                </span>
              </div>
              <span className="text-[10px] text-red-700 font-medium block mt-1">
                Recommendation: Localized cold-patch or hot-mix filling
              </span>
            </div>

            {/* Crack Area Metric */}
            <div className="p-3.5 rounded-lg border border-amber-200 bg-amber-50/50">
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded bg-[#FFCC00] flex-shrink-0"></span>
                  <span className="text-xs font-bold text-amber-950">Road Crack Area (Class 2)</span>
                </div>
                <span className="text-sm font-extrabold text-amber-600">{crack_area_percentage}%</span>
              </div>
              <div className="flex items-center justify-between text-[11px] text-slate-600 mt-1 pt-1 border-t border-amber-200/50">
                <span>Pixel Footprint:</span>
                <span className="font-mono font-bold text-slate-800">
                  {metrics?.pixels?.crack?.toLocaleString()} px
                </span>
              </div>
              <span className="text-[10px] text-amber-800 font-medium block mt-1">
                Recommendation: Bituminous crack sealing or slurry seal
              </span>
            </div>

            {/* Healthy Road Surface */}
            <div className="p-3.5 rounded-lg border border-slate-200 bg-slate-50">
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded bg-[#14243A] flex-shrink-0"></span>
                  <span className="text-xs font-bold text-slate-800">Intact Road Surface (Class 0)</span>
                </div>
                <span className="text-sm font-bold text-slate-700">{background_area_percentage}%</span>
              </div>
              <div className="flex items-center justify-between text-[11px] text-slate-500 mt-1 pt-1 border-t border-slate-200">
                <span>Pixel Count:</span>
                <span className="font-mono text-slate-700">
                  {metrics?.pixels?.background?.toLocaleString()} px
                </span>
              </div>
            </div>
          </div>

          {/* Technical Metadata Card */}
          <div className="bg-slate-50 p-4 rounded-xl border border-[#E3E8EF] text-xs text-slate-600 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Total Pixels:</span>
              <span className="font-mono font-semibold text-slate-800">
                {metrics?.pixels?.total?.toLocaleString()} px
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Inference Engine:</span>
              <span className="font-semibold text-blue-700">DeepLabV3-ResNet50</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Quantification Standard:</span>
              <span className="font-semibold text-slate-700">Pixel Area % Ratio</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
