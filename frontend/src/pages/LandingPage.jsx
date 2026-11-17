import React, { useState } from "react";
import { 
  ArrowRight, 
  Layers, 
  Activity, 
  Scan, 
  BarChart3, 
  History, 
  FileText, 
  ShieldCheck, 
  Cpu, 
  Compass, 
  Eye, 
  CheckCircle2, 
  Sliders, 
  ChevronRight,
  Database,
  Terminal,
  AlertTriangle,
  Menu,
  X
} from "lucide-react";

export default function LandingPage({ onNavigate }) {
  const [activeTab, setActiveTab] = useState("all");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [sliderPosition, setSliderPosition] = useState(50);

  const scrollToSection = (e, sectionId) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    setMobileMenuOpen(false);
    if (sectionId === "home") {
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    const el = document.getElementById(sectionId);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const handleNavAction = (e, targetRoute) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    setMobileMenuOpen(false);
    onNavigate(targetRoute);
  };

  const modules = [
    {
      id: "dashboard",
      title: "Overview Dashboard",
      tag: "INFRASTRUCTURE HEALTH",
      desc: "Holistic asset metrics, critical roadway distress status, and fleet survey timelines.",
      icon: Activity,
      action: "dashboard"
    },
    {
      id: "inspection",
      title: "New Inspection Workspace",
      tag: "REAL-TIME SEGMENTATION",
      desc: "Drag-and-drop imagery or load pre-calibrated highway survey benchmarks in 1 click.",
      icon: Scan,
      action: "new-inspection"
    },
    {
      id: "results",
      title: "Detection Results & Spatial Viewer",
      tag: "PIXEL-LEVEL ANALYSIS",
      desc: "Multi-layer inspection viewer with dual-channel distress masks and opacity blends.",
      icon: Layers,
      action: "results"
    },
    {
      id: "analytics",
      title: "Pavement Analytics",
      tag: "TRENDS & SEVERITY",
      desc: "Aggregate damage distributions, pothole-to-crack ratios, and severity classifications.",
      icon: BarChart3,
      action: "analytics"
    },
    {
      id: "history",
      title: "Inspection Audit History",
      tag: "PERMANENT RECORDS",
      desc: "Searchable registry of all pavement scans with condition grades and inspection logs.",
      icon: History,
      action: "history"
    },
    {
      id: "reports",
      title: "Engineering Reports",
      tag: "A4 COMPLIANT PDF",
      desc: "Formal road authority audit documentation ready for print, dispatch and export.",
      icon: FileText,
      action: "reports"
    }
  ];

  return (
    <div className="min-h-screen bg-[#0a0a0b] text-[#f3f1ec] selection:bg-[#f5a524] selection:text-[#0a0a0b]">
      {/* =========================================================================
          SITE HEADER (PIVAI Style)
         ========================================================================= */}
      <header className="sticky top-0 z-50 bg-[#0a0a0b]/90 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 flex items-center justify-between">
          {/* Brand */}
          <button 
            type="button"
            className="flex items-center gap-3 cursor-pointer text-left" 
            onClick={(e) => handleNavAction(e, "landing")}
          >
            <div className="w-9 h-9 rounded-full bg-[#f5a524] flex items-center justify-center font-bold text-[#0a0a0b] shadow-sm">
              <Scan className="w-5 h-5" />
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-lg tracking-tight text-[#f3f1ec]">RoadVision<span className="text-[#f5a524]">.AI</span></span>
              <span className="font-mono text-[10px] text-white/50 tracking-wider uppercase">Pavement Intelligence</span>
            </div>
          </button>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-7 text-[14.5px] font-medium text-[#f3f1ec]/70">
            <a href="#home" onClick={(e) => scrollToSection(e, "home")} className="hover:text-[#f3f1ec] transition-colors">Home</a>
            <a href="#platform" onClick={(e) => scrollToSection(e, "platform")} className="hover:text-[#f3f1ec] transition-colors">Platform</a>
            <a href="#features" onClick={(e) => scrollToSection(e, "features")} className="hover:text-[#f3f1ec] transition-colors">Features</a>
            <a href="#technology" onClick={(e) => scrollToSection(e, "technology")} className="hover:text-[#f3f1ec] transition-colors">Technology</a>
            <a href="#about" onClick={(e) => scrollToSection(e, "about")} className="hover:text-[#f3f1ec] transition-colors">About</a>
          </nav>

          {/* Header Action Buttons */}
          <div className="hidden sm:flex items-center gap-3">
            <button 
              type="button"
              onClick={(e) => handleNavAction(e, "login")}
              className="btn-pivai-ghost text-xs uppercase tracking-wider font-mono h-9 px-4"
            >
              Sign In
            </button>
            <button 
              type="button"
              onClick={(e) => handleNavAction(e, "signup")}
              className="btn-pivai-amber text-xs uppercase tracking-wider font-mono h-9 px-5"
            >
              Get Started
              <ArrowRight className="w-3.5 h-3.5 ml-1" />
            </button>
          </div>

          {/* Mobile menu trigger */}
          <button 
            type="button"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)} 
            className="md:hidden p-2 rounded-lg border border-white/10 text-white/80"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        {/* Mobile dropdown */}
        {mobileMenuOpen && (
          <div className="md:hidden bg-[#121214] border-b border-white/10 px-4 py-6 space-y-4">
            <a 
              href="#home" 
              onClick={(e) => scrollToSection(e, "home")}
              className="block text-white/80 hover:text-white font-medium"
            >
              Home
            </a>
            <a 
              href="#platform" 
              onClick={(e) => scrollToSection(e, "platform")}
              className="block text-white/80 hover:text-white font-medium"
            >
              Platform
            </a>
            <a 
              href="#features" 
              onClick={(e) => scrollToSection(e, "features")}
              className="block text-white/80 hover:text-white font-medium"
            >
              Features
            </a>
            <a 
              href="#technology" 
              onClick={(e) => scrollToSection(e, "technology")}
              className="block text-white/80 hover:text-white font-medium"
            >
              Technology
            </a>
            <a 
              href="#about" 
              onClick={(e) => scrollToSection(e, "about")}
              className="block text-white/80 hover:text-white font-medium"
            >
              About
            </a>
            <div className="pt-4 border-t border-white/10 flex flex-col gap-2">
              <button 
                type="button"
                onClick={(e) => handleNavAction(e, "login")}
                className="btn-pivai-ghost w-full justify-center"
              >
                Sign In
              </button>
              <button 
                type="button"
                onClick={(e) => handleNavAction(e, "signup")}
                className="btn-pivai-amber w-full justify-center"
              >
                Get Started
                <ArrowRight className="w-4 h-4 ml-1" />
              </button>
            </div>
          </div>
        )}
      </header>

      {/* =========================================================================
          HERO SECTION (Inspired by PIVAI: Display headline, Amber CTA, Road visual)
         ========================================================================= */}
      <section id="home" className="relative pt-16 pb-24 md:pt-28 md:pb-36 overflow-hidden border-b border-white/10">
        {/* Subtle radial ambient glow */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[400px] bg-[#f5a524]/10 blur-[140px] pointer-events-none rounded-full" />
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="max-w-3xl">
            {/* Pill Label */}
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/15 text-[#f5a524] text-[11px] font-mono tracking-widest uppercase mb-6">
              <span className="w-1.5 h-1.5 rounded-full bg-[#f5a524] animate-pulse" />
              Perception of Road Infrastructure Based on Visual AI
            </div>

            {/* Main Display Headline */}
            <h1 className="text-4xl sm:text-6xl lg:text-7xl font-bold tracking-tight text-[#f3f1ec] leading-[1.08] mb-6">
              Intelligent Road Inspection.<br />
              <span className="text-white/60">Smarter Infrastructure.</span>
            </h1>

            {/* Supporting Lede */}
            <p className="text-lg sm:text-xl text-white/70 max-w-2xl leading-relaxed mb-8">
              RoadVision AI uses deep learning and semantic segmentation to identify potholes and road cracks, 
              quantify pavement damage at the pixel level, and support data-driven highway maintenance.
            </p>

            {/* CTAs */}
            <div className="flex flex-wrap items-center gap-4">
              <button 
                type="button"
                onClick={(e) => handleNavAction(e, "signup")}
                className="btn-pivai-amber text-base font-semibold px-7 h-12"
              >
                Get Started
                <ArrowRight className="w-4 h-4 ml-1" />
              </button>

              <button 
                type="button"
                onClick={(e) => scrollToSection(e, "platform")}
                className="btn-pivai-ghost text-base font-medium px-6 h-12"
              >
                Explore Platform
              </button>
            </div>
          </div>

          {/* Interactive Inspection Comparison Showcase */}
          <div className="mt-16 md:mt-20 rounded-2xl border border-white/15 bg-[#121214] p-3 sm:p-5 shadow-2xl">
            <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-white/10 mb-4 px-2">
              <div className="flex items-center gap-3">
                <span className="w-3 h-3 rounded-full bg-emerald-500" />
                <span className="font-mono text-xs text-white/80 uppercase tracking-wider">
                  Live Model Inference Visualizer · DeepLabV3-ResNet50
                </span>
              </div>
              <div className="flex items-center gap-4 font-mono text-xs">
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm bg-[#FF3B30]" />
                  <span className="text-white/80">Pothole (Class 1)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm bg-[#FFCC00]" />
                  <span className="text-white/80">Crack (Class 2)</span>
                </div>
              </div>
            </div>

            {/* Split Comparison Canvas */}
            <div className="relative h-[340px] sm:h-[480px] rounded-xl overflow-hidden bg-black group select-none">
              {/* Background Original Road */}
              <img 
                src="/api/samples/road_pothole_and_crack.jpg" 
                alt="Original Road Pavement" 
                className="absolute inset-0 w-full h-full object-cover"
                onError={(e) => {
                  e.target.src = "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?w=1200&auto=format&fit=crop&q=80";
                }}
              />

              {/* Segmentation Overlay with Slider */}
              <div 
                className="absolute inset-y-0 right-0 overflow-hidden"
                style={{ left: `${sliderPosition}%` }}
              >
                <div className="relative w-full h-full">
                  <img 
                    src="/api/samples/road_pothole_and_crack.jpg" 
                    alt="AI Overlay" 
                    className="absolute top-0 right-0 h-full max-w-none object-cover"
                    style={{ width: '100%', minWidth: '1000px', filter: 'contrast(1.1) brightness(0.95)' }}
                    onError={(e) => {
                      e.target.src = "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?w=1200&auto=format&fit=crop&q=80";
                    }}
                  />
                  {/* Artificial distress highlight mask layer for visual demonstration */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[#FF3B30]/30 to-[#FFCC00]/30 mix-blend-screen" />
                </div>
              </div>

              {/* Divider Line */}
              <div 
                className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_12px_rgba(255,255,255,0.8)] cursor-ew-resize z-20 flex items-center justify-center -ml-0.5"
                style={{ left: `${sliderPosition}%` }}
              >
                <div className="w-7 h-7 rounded-full bg-[#0a0a0b] border-2 border-white flex items-center justify-center text-white shadow-lg">
                  <Sliders className="w-3.5 h-3.5" />
                </div>
              </div>

              {/* Range input for drag interaction */}
              <input 
                type="range" 
                min="5" 
                max="95" 
                value={sliderPosition} 
                onChange={(e) => setSliderPosition(Number(e.target.value))}
                className="absolute inset-0 opacity-0 cursor-ew-resize z-30" 
              />

              {/* Floating Engineering Badges */}
              <div className="absolute bottom-4 left-4 z-10 px-3 py-1.5 rounded-lg bg-[#0a0a0b]/80 backdrop-blur-md border border-white/10 text-xs font-mono text-white/80">
                RAW FIELD CAPTURE (640×360)
              </div>
              <div className="absolute bottom-4 right-4 z-10 px-3 py-1.5 rounded-lg bg-[#0a0a0b]/80 backdrop-blur-md border border-white/10 text-xs font-mono text-[#f5a524]">
                QUANTIFIED DAMAGE: 10.63%
              </div>
            </div>

            {/* PIVAI-style Hero Metrics Strip */}
            <div className="mt-4 pt-4 border-t border-white/10 grid grid-cols-2 sm:grid-cols-4 gap-4 text-left font-mono">
              <div className="p-3 rounded-lg bg-white/[0.02]">
                <div className="text-[11px] text-[#f5a524] uppercase tracking-wider mb-1">Architecture</div>
                <div className="text-sm font-semibold text-white">DeepLabV3-ResNet50</div>
              </div>
              <div className="p-3 rounded-lg bg-white/[0.02]">
                <div className="text-[11px] text-[#f5a524] uppercase tracking-wider mb-1">Classes</div>
                <div className="text-sm font-semibold text-white">Road · Pothole · Crack</div>
              </div>
              <div className="p-3 rounded-lg bg-white/[0.02]">
                <div className="text-[11px] text-[#f5a524] uppercase tracking-wider mb-1">Inference Resolution</div>
                <div className="text-sm font-semibold text-white">512×512 Native Argmax</div>
              </div>
              <div className="p-3 rounded-lg bg-white/[0.02]">
                <div className="text-[11px] text-[#f5a524] uppercase tracking-wider mb-1">Audit Output</div>
                <div className="text-sm font-semibold text-white">A4 Engineering Cert</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          PLATFORM MODULES / FEATURES (Section 2)
         ========================================================================= */}
      <section id="platform" className="py-24 bg-[#121214] border-b border-white/10">
        <div id="features" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 scroll-mt-20">
          <div className="max-w-2xl mb-16">
            <div className="text-[11px] font-mono tracking-widest text-[#f5a524] uppercase mb-3">
              INTEGRATED ROAD WORKSPACE
            </div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-white mb-4">
              Six Specialized Inspection Modules
            </h2>
            <p className="text-white/60 text-base leading-relaxed">
              From roadside field acquisition to automated distress indexing and executive reporting, 
              RoadVision AI delivers end-to-end infrastructure oversight.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {modules.map((mod) => {
              const Icon = mod.icon;
              return (
                <div 
                  key={mod.id} 
                  className="rounded-xl border border-white/10 bg-[#0a0a0b] p-6 hover:border-[#f5a524]/40 transition-all duration-300 group flex flex-col justify-between"
                >
                  <div>
                    <div className="w-11 h-11 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center text-[#f5a524] mb-5 group-hover:scale-110 transition-transform">
                      <Icon className="w-5 h-5" />
                    </div>
                    <div className="text-[10px] font-mono tracking-widest text-white/50 uppercase mb-2">
                      {mod.tag}
                    </div>
                    <h3 className="text-lg font-bold text-white mb-2 group-hover:text-[#f5a524] transition-colors">
                      {mod.title}
                    </h3>
                    <p className="text-sm text-white/60 leading-relaxed mb-6">
                      {mod.desc}
                    </p>
                  </div>

                  <button 
                    type="button"
                    onClick={(e) => handleNavAction(e, "login")}
                    className="inline-flex items-center text-xs font-mono font-medium text-white/80 group-hover:text-[#f5a524] transition-colors"
                  >
                    Open Module <ChevronRight className="w-3.5 h-3.5 ml-1 group-hover:translate-x-1 transition-transform" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* =========================================================================
          TECHNOLOGY DEEP DIVE (DeepLabV3, ResNet50, PyTorch, Classes)
         ========================================================================= */}
      <section id="technology" className="py-24 bg-[#0a0a0b] border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <div className="text-[11px] font-mono tracking-widest text-[#f5a524] uppercase mb-3">
                COMPUTER VISION BACKBONE
              </div>
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-white mb-6">
                Survey-Grade Deep Learning with DeepLabV3-ResNet50
              </h2>
              <p className="text-white/70 leading-relaxed mb-6">
                RoadVision AI employs an Atrous Spatial Pyramid Pooling (ASPP) DeepLabV3 network paired with a ResNet-50 
                feature extractor to segment irregular pavement distress at multiple receptive fields.
              </p>

              <div className="space-y-4 font-mono text-xs">
                <div className="p-4 rounded-xl border border-white/10 bg-[#121214] flex items-start gap-3">
                  <CheckCircle2 className="w-4 h-4 text-[#f5a524] mt-0.5 shrink-0" />
                  <div>
                    <div className="text-white font-semibold mb-1">Three-Class Semantic Segmentation</div>
                    <div className="text-white/60">Class 0: Healthy Asphalt · Class 1: Pothole (#FF3B30) · Class 2: Surface Crack (#FFCC00).</div>
                  </div>
                </div>

                <div className="p-4 rounded-xl border border-white/10 bg-[#121214] flex items-start gap-3">
                  <CheckCircle2 className="w-4 h-4 text-[#f5a524] mt-0.5 shrink-0" />
                  <div>
                    <div className="text-white font-semibold mb-1">Native Resolution Alignment</div>
                    <div className="text-white/60">Bilinear inference upsampling paired with nearest-neighbour postprocessing to match original road coordinates.</div>
                  </div>
                </div>

                <div className="p-4 rounded-xl border border-white/10 bg-[#121214] flex items-start gap-3">
                  <CheckCircle2 className="w-4 h-4 text-[#f5a524] mt-0.5 shrink-0" />
                  <div>
                    <div className="text-white font-semibold mb-1">Deterministic Quantification</div>
                    <div className="text-white/60">Exact damage surface percentage: Pothole% + Crack% / Total Surface Pixels.</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-white/15 bg-[#121214] p-6 font-mono text-xs text-white/80 space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-white/10 text-white/50">
                <span className="flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-[#f5a524]" />
                  roadvision_inference_pipeline.py
                </span>
                <span>PyTorch 2.x</span>
              </div>

              <div className="text-[#f5a524]">// 1. Tensor Conversion & ASPP Forward Pass</div>
              <div className="bg-[#0a0a0b] p-3 rounded-lg border border-white/5 text-emerald-400">
                tensor = normalize(image_tensor).unsqueeze(0)<br />
                output = model(tensor)["out"] &nbsp;# (1, 3, 512, 512)<br />
                pred = torch.argmax(output, dim=1)
              </div>

              <div className="text-[#f5a524]">// 2. Deterministic Pixel Quantification</div>
              <div className="bg-[#0a0a0b] p-3 rounded-lg border border-white/5 text-sky-400">
                pothole_px = count(pred == 1)<br />
                crack_px = count(pred == 2)<br />
                damage_pct = ((pothole_px + crack_px) / total_px) * 100
              </div>

              <div className="pt-2 flex items-center justify-between text-white/40 text-[11px]">
                <span>Validated against held-out benchmark test sets</span>
                <span className="text-emerald-400">Status: Calibrated & Verified</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          WORKFLOW / ABOUT SECTION (5 Steps)
         ========================================================================= */}
      <section id="workflow" className="py-24 bg-[#121214] border-b border-white/10">
        <div id="about" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 scroll-mt-20">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <div className="text-[11px] font-mono tracking-widest text-[#f5a524] uppercase mb-3">
              SYSTEMATIC PROCESS
            </div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-white mb-4">
              From Camera Frame to Certified Report
            </h2>
            <p className="text-white/60 text-base">
              A streamlined five-step processing sequence ensuring auditable inspection outputs.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {[
              { step: "01", title: "Image Ingestion", desc: "Upload road camera frames or select highway benchmarks." },
              { step: "02", title: "ResNet50 Backbone", desc: "Extract multi-scale visual features of asphalt texture." },
              { step: "03", title: "DeepLabV3 ASPP", desc: "Segment pothole basins and hairline/alligator cracks." },
              { step: "04", title: "Area Quantification", desc: "Calculate exact pixel counts and damage percentage." },
              { step: "05", title: "Report Generation", desc: "Compile A4 certified engineering audit documents." }
            ].map((st) => (
              <div key={st.step} className="rounded-xl border border-white/10 bg-[#0a0a0b] p-5 relative">
                <div className="font-mono text-2xl font-bold text-[#f5a524] mb-3">
                  {st.step}
                </div>
                <h4 className="text-sm font-bold text-white mb-2">{st.title}</h4>
                <p className="text-xs text-white/60 leading-relaxed">{st.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* =========================================================================
          FINAL CTA (PIVAI Style Banner)
         ========================================================================= */}
      <section className="py-24 bg-[#0a0a0b] relative overflow-hidden">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-white/70 text-xs font-mono uppercase tracking-wider mb-6">
            Ready for Deployment
          </div>
          <h2 className="text-3xl sm:text-5xl font-bold tracking-tight text-white mb-6">
            Begin Your Autonomous Road Inspection
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-xl mx-auto mb-10">
            Sign in to access your pre-loaded highway inspection database, execute real-time model inference, 
            and generate certified engineering audit reports.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4">
            <button 
              type="button"
              onClick={(e) => handleNavAction(e, "signup")}
              className="btn-pivai-amber text-base font-semibold px-8 h-12"
            >
              Create Account
              <ArrowRight className="w-4 h-4 ml-1" />
            </button>
            <button 
              type="button"
              onClick={(e) => handleNavAction(e, "login")}
              className="btn-pivai-ghost text-base font-medium px-7 h-12"
            >
              Sign In to Workspace
            </button>
          </div>
        </div>
      </section>

      {/* =========================================================================
          FOOTER
         ========================================================================= */}
      <footer className="border-t border-white/10 bg-[#0a0a0b] py-12 text-white/50 text-xs font-mono">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-full bg-[#f5a524] flex items-center justify-center text-[#0a0a0b] font-bold">
              <Scan className="w-4 h-4" />
            </div>
            <span className="font-bold text-sm text-white">RoadVision AI</span>
            <span>· Semantic Segmentation for Civil Infrastructure</span>
          </div>

          <div className="flex flex-wrap items-center gap-6">
            <button type="button" onClick={(e) => scrollToSection(e, "home")} className="hover:text-white">Home</button>
            <button type="button" onClick={(e) => handleNavAction(e, "login")} className="hover:text-white">Sign In</button>
            <button type="button" onClick={(e) => handleNavAction(e, "signup")} className="hover:text-white">Register</button>
            <span className="text-white/30">|</span>
            <span>© 2026 RoadVision AI. All rights reserved.</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
