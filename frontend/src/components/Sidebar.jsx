import React from 'react';
import { 
  LayoutDashboard, 
  Scan, 
  Layers, 
  BarChart3, 
  History, 
  FileText, 
  LogOut,
  Cpu,
  ExternalLink,
  User,
  Shield
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Sidebar({ activePage, setActivePage, backendOnline, onNavigatePublic }) {
  const { currentUser, logout } = useAuth();

  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard, badge: null },
    { id: 'new-inspection', label: 'New Inspection', icon: Scan, badge: 'AI' },
    { id: 'results', label: 'Detection Results', icon: Layers, badge: null },
    { id: 'analytics', label: 'Analytics', icon: BarChart3, badge: null },
    { id: 'history', label: 'Inspection History', icon: History, badge: null },
    { id: 'reports', label: 'Reports', icon: FileText, badge: null },
  ];

  const handleLogout = async () => {
    try {
      await logout();
      if (onNavigatePublic) onNavigatePublic('login');
    } catch (e) {
      console.error(e);
    }
  };

  const displayName = currentUser?.displayName || currentUser?.email?.split('@')[0] || "Lead Inspector";
  const userEmail = currentUser?.email || "inspector@roadvision.ai";
  const initial = displayName.charAt(0).toUpperCase();

  return (
    <aside className="w-64 bg-[#0a0a0b] border-r border-white/10 text-white flex flex-col flex-shrink-0 select-none z-30 font-sans">
      {/* Brand Header */}
      <div className="h-16 px-5 flex items-center justify-between border-b border-white/10 bg-[#121214]/60">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-[#f5a524] flex items-center justify-center text-[#0a0a0b] font-bold shadow-sm">
            <Scan className="w-4 h-4" />
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-sm tracking-tight text-[#f3f1ec]">RoadVision<span className="text-[#f5a524]">.AI</span></span>
            <span className="font-mono text-[9.5px] text-white/50 tracking-wider uppercase">Pavement Intelligence</span>
          </div>
        </div>

        {onNavigatePublic && (
          <button 
            onClick={() => onNavigatePublic('landing')}
            className="text-white/40 hover:text-white p-1 rounded hover:bg-white/5 transition-colors"
            title="View Public Landing Page"
          >
            <ExternalLink className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* Navigation Section */}
      <div className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
        <div className="px-3 pb-2 text-[10px] font-mono tracking-widest text-[#f5a524] uppercase">
          WORKSPACE MODULES
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActivePage(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all group relative ${
                isActive
                  ? 'bg-[#f5a524] text-[#0a0a0b] shadow-sm font-bold'
                  : 'text-white/70 hover:text-white hover:bg-white/5'
              }`}
            >
              <Icon className={`w-4 h-4 flex-shrink-0 ${
                isActive ? 'text-[#0a0a0b]' : 'text-white/60 group-hover:text-white'
              }`} />
              <span className="flex-1 text-left truncate">{item.label}</span>
              {item.badge && (
                <span className={`text-[9.5px] font-mono px-1.5 py-0.5 rounded font-bold ${
                  isActive ? 'bg-[#0a0a0b] text-[#f5a524]' : 'bg-white/10 text-white/80'
                }`}>
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Model Engine Status Card */}
      <div className="p-3 mx-3 mb-3 rounded-xl bg-[#121214] border border-white/10 text-xs font-mono">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[10px] text-white/70 flex items-center gap-1.5 uppercase tracking-wider">
            <Cpu className="w-3 h-3 text-[#f5a524]" />
            DeepLabV3 Backbone
          </span>
          <span className={`inline-flex items-center gap-1 px-1.5 py-0.2 rounded text-[9.5px] font-bold ${
            backendOnline 
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
              : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${backendOnline ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
            {backendOnline ? 'Ready' : 'Offline'}
          </span>
        </div>
        
        <div className="text-[11px] text-white font-medium bg-[#0a0a0b] px-2 py-1 rounded border border-white/5 truncate">
          ResNet50 · 512×512 Argmax
        </div>
        
        <div className="mt-2 grid grid-cols-2 gap-1 text-[9.5px] text-white/50 pt-1.5 border-t border-white/5">
          <div><span className="text-white/30">Loss:</span> ASPP Aux</div>
          <div><span className="text-white/30">Classes:</span> 3 (BG,P,C)</div>
        </div>
      </div>

      {/* Authenticated User / Station Profile Footer */}
      <div className="p-3 border-t border-white/10 flex items-center justify-between bg-[#121214]">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-8 h-8 rounded-full bg-[#f5a524]/20 border border-[#f5a524]/40 flex items-center justify-center text-xs font-bold text-[#f5a524] flex-shrink-0">
            {initial}
          </div>
          <div className="leading-tight truncate">
            <p className="text-xs font-semibold text-white truncate">{displayName}</p>
            <p className="text-[10px] text-white/50 truncate font-mono">{userEmail}</p>
          </div>
        </div>
        
        <button 
          onClick={handleLogout}
          className="text-white/40 hover:text-red-400 p-1.5 rounded hover:bg-white/5 transition-colors"
          title="Sign Out"
        >
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </aside>
  );
}
