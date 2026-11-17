import React, { useState } from 'react';
import { 
  Eye, 
  Layers, 
  Image as ImageIcon, 
  Columns, 
  Maximize2, 
  Sliders, 
  Sparkles,
  Info
} from 'lucide-react';

export default function ImageComparison({ original, overlay, mask, dimensions }) {
  const [activeView, setActiveView] = useState('overlay'); // 'overlay' | 'mask' | 'original' | 'side-by-side'
  const [overlayOpacity, setOverlayOpacity] = useState(100); // 25, 50, 75, 100
  const [isFullscreen, setIsFullscreen] = useState(false);

  return (
    <div className={`bg-white rounded-xl border border-[#E3E8EF] overflow-hidden shadow-xs ${
      isFullscreen ? 'fixed inset-4 z-50 flex flex-col bg-white shadow-2xl border-slate-400' : ''
    }`}>
      {/* Control bar */}
      <div className="p-3 bg-slate-50/90 border-b border-[#E3E8EF] flex flex-wrap items-center justify-between gap-3">
        {/* View mode toggle pills */}
        <div className="flex items-center gap-1 bg-slate-200/60 p-1 rounded-lg">
          <button
            onClick={() => setActiveView('overlay')}
            className={`px-3 py-1.5 text-xs font-bold rounded-md flex items-center gap-1.5 transition-all ${
              activeView === 'overlay'
                ? 'bg-white text-blue-700 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-blue-600" />
            Distress Overlay
          </button>
          
          <button
            onClick={() => setActiveView('mask')}
            className={`px-3 py-1.5 text-xs font-bold rounded-md flex items-center gap-1.5 transition-all ${
              activeView === 'mask'
                ? 'bg-white text-blue-700 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Eye className="w-3.5 h-3.5 text-amber-500" />
            Class Mask
          </button>

          <button
            onClick={() => setActiveView('original')}
            className={`px-3 py-1.5 text-xs font-bold rounded-md flex items-center gap-1.5 transition-all ${
              activeView === 'original'
                ? 'bg-white text-blue-700 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <ImageIcon className="w-3.5 h-3.5 text-slate-500" />
            Source Image
          </button>

          <button
            onClick={() => setActiveView('side-by-side')}
            className={`px-3 py-1.5 text-xs font-bold rounded-md flex items-center gap-1.5 transition-all ${
              activeView === 'side-by-side'
                ? 'bg-white text-blue-700 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Columns className="w-3.5 h-3.5 text-indigo-600" />
            Side-by-Side
          </button>
        </div>

        {/* Right side controls */}
        <div className="flex items-center gap-2">
          {activeView === 'overlay' && (
            <div className="flex items-center gap-1.5 bg-white px-2.5 py-1 rounded-md border border-[#E3E8EF] text-xs font-medium text-slate-600">
              <Sliders className="w-3 h-3 text-slate-400" />
              <span className="text-[11px] text-slate-500">Opacity:</span>
              {[50, 75, 100].map(val => (
                <button
                  key={val}
                  onClick={() => setOverlayOpacity(val)}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                    overlayOpacity === val 
                      ? 'bg-blue-600 text-white' 
                      : 'text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  {val}%
                </button>
              ))}
            </div>
          )}

          {dimensions && (
            <div className="text-[11px] text-slate-600 font-mono bg-white px-2.5 py-1 rounded-md border border-[#E3E8EF]">
              {dimensions.width} × {dimensions.height} px
            </div>
          )}

          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 text-slate-500 hover:text-slate-800 hover:bg-white rounded-md border border-[#E3E8EF] transition-colors"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen Inspection"}
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Imagery Viewer Display */}
      <div className={`p-4 bg-[#0B1320] flex items-center justify-center overflow-hidden relative ${
        isFullscreen ? 'flex-1 min-h-[500px]' : 'min-h-[380px]'
      }`}>
        {activeView === 'side-by-side' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-6xl">
            {/* Original Left */}
            <div className="relative rounded-lg overflow-hidden border border-slate-800 bg-slate-900">
              <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded text-[10px] font-bold bg-black/75 text-slate-200 backdrop-blur-xs border border-white/10">
                Figure A: Source Road Surface
              </div>
              <img
                src={original}
                alt="Source Roadway"
                className="w-full h-auto object-contain max-h-[480px] mx-auto block"
              />
            </div>
            
            {/* Overlay Right */}
            <div className="relative rounded-lg overflow-hidden border border-slate-800 bg-slate-900">
              <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded text-[10px] font-bold bg-blue-900/80 text-blue-200 backdrop-blur-xs border border-blue-400/30">
                Figure B: DeepLabV3 Distress Overlay
              </div>
              <img
                src={overlay}
                alt="Distress Overlay"
                className="w-full h-auto object-contain max-h-[480px] mx-auto block"
              />
            </div>
          </div>
        ) : (
          <div className="relative max-w-full flex justify-center items-center">
            {activeView === 'overlay' && (
              <div className="relative">
                {/* Background base image for custom opacity blending if < 100% */}
                {overlayOpacity < 100 && (
                  <img
                    src={original}
                    alt="Source Base"
                    className="max-h-[520px] max-w-full object-contain rounded-lg border border-slate-800"
                  />
                )}
                <img
                  src={overlay}
                  alt="Road Distress Overlay"
                  style={{ opacity: overlayOpacity / 100 }}
                  className={`max-h-[520px] max-w-full object-contain rounded-lg shadow-xl border border-slate-800 ${
                    overlayOpacity < 100 ? 'absolute inset-0 w-full h-full' : ''
                  }`}
                />
              </div>
            )}

            {activeView === 'mask' && (
              <img
                src={mask}
                alt="DeepLabV3 Semantic Segmentation Mask"
                className="max-h-[520px] max-w-full object-contain rounded-lg shadow-xl border border-slate-800 bg-black"
              />
            )}

            {activeView === 'original' && (
              <img
                src={original}
                alt="Original Source RGB Road Image"
                className="max-h-[520px] max-w-full object-contain rounded-lg shadow-xl border border-slate-800"
              />
            )}

            {/* View description chip */}
            <div className="absolute bottom-3 right-3 px-2.5 py-1 rounded bg-black/80 backdrop-blur-md text-white text-[11px] font-semibold border border-white/10 shadow-sm flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse"></span>
              <span>
                {activeView === 'overlay' && `Distress Overlay (${overlayOpacity}% Opacity)`}
                {activeView === 'mask' && 'Pure Class Segmentation Mask (Raw Classes)'}
                {activeView === 'original' && 'Original High-Resolution RGB Image'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Engineering Classification Legend Bar */}
      <div className="p-3 bg-white border-t border-[#E3E8EF] flex flex-wrap items-center justify-between text-xs gap-3">
        <div className="flex items-center gap-2">
          <Info className="w-3.5 h-3.5 text-blue-600" />
          <span className="font-bold text-[#172435]">Class Assignment:</span>
        </div>
        
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-3.5 rounded-xs bg-[#FF3B30] border border-black/20 shadow-xs flex-shrink-0"></span>
            <span className="font-semibold text-slate-800">Pothole (Class 1)</span>
          </div>
          
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-3.5 rounded-xs bg-[#FFCC00] border border-black/20 shadow-xs flex-shrink-0"></span>
            <span className="font-semibold text-slate-800">Road Crack (Class 2)</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-3.5 rounded-xs bg-[#14243A] border border-slate-300 shadow-xs flex-shrink-0"></span>
            <span className="font-semibold text-slate-800">Roadway Surface (Class 0)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
