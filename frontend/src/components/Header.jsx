import React from 'react';
import { Scan, RefreshCw, ExternalLink, LogOut } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Header({ activePage, setActivePage, onRefresh, isRefreshing, backendOnline, onNavigatePublic }) {
  const { currentUser, logout } = useAuth();

  const titles = {
    'overview': { 
      title: 'Overview Dashboard', 
      desc: 'Continuous road network health surveillance and pavement distress indicators' 
    },
    'new-inspection': { 
      title: 'New Road Inspection', 
      desc: 'Upload high-resolution roadway imagery for DeepLabV3 semantic damage detection' 
    },
    'results': { 
      title: 'Detection Results', 
      desc: 'Deep learning pixel segmentation, distress quantification, and severity scoring' 
    },
    'analytics': { 
      title: 'Pavement Analytics', 
      desc: 'Network distress distribution, condition index breakdown, and historical trend analysis' 
    },
    'history': { 
      title: 'Inspection History', 
      desc: 'Centralized registry of surveyed roadways, severity ratings, and inspection records' 
    },
    'reports': { 
      title: 'Inspection Reports', 
      desc: 'Standardized infrastructure audit sheets and printable engineering documentation' 
    },
  };

  const current = titles[activePage] || { title: 'RoadVision AI', desc: '' };

  const handleLogout = async () => {
    try {
      await logout();
      if (onNavigatePublic) onNavigatePublic('login');
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <header className="h-16 bg-white border-b border-[#E3E8EF] px-6 flex items-center justify-between sticky top-0 z-20 flex-shrink-0">
      <div className="min-w-0 pr-4">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono font-bold text-[#f5a524] uppercase tracking-wider">MODULE</span>
          <span className="text-slate-300 font-mono">/</span>
          <h1 className="text-base font-bold text-[#121214] tracking-tight truncate">{current.title}</h1>
        </div>
        <p className="text-xs text-slate-500 hidden sm:block truncate max-w-xl">{current.desc}</p>
      </div>

      <div className="flex items-center gap-2.5 flex-shrink-0">
        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-50 rounded-lg border border-[#E3E8EF] transition-colors shadow-xs"
            title="Refresh database records"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-[#f5a524]' : ''}`} />
          </button>
        )}

        <div className="hidden md:flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-slate-50 border border-[#E3E8EF] text-xs font-semibold text-slate-700">
          <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-500 ring-2 ring-emerald-500/20' : 'bg-red-500'}`}></span>
          <span className="font-mono text-[11px]">API :8000</span>
        </div>

        {onNavigatePublic && (
          <button 
            onClick={() => onNavigatePublic('landing')}
            className="hidden lg:flex items-center gap-1.5 text-xs text-slate-600 hover:text-slate-900 px-3 py-1.5 rounded-lg border border-[#E3E8EF] hover:bg-slate-50 font-medium"
            title="View Public Website"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            <span>Public Site</span>
          </button>
        )}

        {activePage !== 'new-inspection' && (
          <button
            onClick={() => setActivePage('new-inspection')}
            className="btn-pivai-amber h-9 px-4 text-xs font-semibold shadow-xs"
          >
            <Scan className="w-3.5 h-3.5" />
            <span>New Inspection</span>
          </button>
        )}

        <button
          onClick={handleLogout}
          className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg border border-[#E3E8EF] transition-colors"
          title="Sign Out"
        >
          <LogOut className="w-3.5 h-3.5" />
        </button>
      </div>
    </header>
  );
}
