import React from 'react';
import { 
  Scan, 
  AlertTriangle, 
  Activity, 
  TrendingUp, 
  Layers, 
  ShieldAlert, 
  Calendar, 
  ChevronRight,
  CheckCircle2,
  AlertCircle,
  ArrowUpRight,
  Sparkles,
  BarChart2,
  FileText
} from 'lucide-react';
import MetricCard from '../components/MetricCard';
import SeverityBadge from '../components/SeverityBadge';
import EmptyState from '../components/EmptyState';

export default function OverviewPage({ analytics, inspections, setActivePage, onViewInspection }) {
  const total = analytics?.total_inspections || 0;
  const avgDamage = analytics?.average_damage_percentage || 0;
  const potholesDetected = analytics?.inspections_with_potholes || 0;
  const cracksDetected = analytics?.inspections_with_cracks || 0;
  const potholePx = analytics?.total_pothole_pixels || 0;
  const crackPx = analytics?.total_crack_pixels || 0;
  const totalPx = potholePx + crackPx || 1;
  const potholeShare = Math.round((potholePx / totalPx) * 100) || 0;
  const crackShare = 100 - potholeShare;

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Top Welcome & Quick Actions Enterprise Banner */}
      <div className="bg-[#101D30] rounded-xl p-6 text-white shadow-xs border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-5 relative overflow-hidden">
        {/* Subtle background glow */}
        <div className="absolute right-0 top-0 bottom-0 w-96 bg-gradient-to-l from-blue-600/10 to-transparent pointer-events-none"></div>

        <div className="relative z-10 max-w-2xl">
          <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full bg-blue-500/15 text-blue-400 text-[11px] font-bold mb-2 border border-blue-400/20">
            <Scan className="w-3.5 h-3.5" />
            <span>AI Automated Pavement Inspection System</span>
          </div>
          <h2 className="text-xl md:text-2xl font-bold tracking-tight text-white">
            Infrastructure Monitoring & Defect Intelligence
          </h2>
          <p className="text-slate-300 text-xs md:text-sm mt-1 leading-relaxed">
            Autonomous pixel-level pavement distress quantification powered by DeepLabV3-ResNet50. 
            Review network health, inspect high-priority roadway hazards, and export formal engineering audit sheets.
          </p>
        </div>

        <div className="flex items-center gap-3 relative z-10 flex-shrink-0">
          <button
            onClick={() => setActivePage('history')}
            className="bg-slate-800/80 hover:bg-slate-800 text-slate-200 border border-slate-700 font-semibold text-xs px-3.5 py-2.5 rounded-lg transition-colors flex items-center gap-1.5"
          >
            <span>View Registry</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
          
          <button
            onClick={() => setActivePage('new-inspection')}
            className="bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs px-4 py-2.5 rounded-lg flex items-center gap-2 shadow-sm shadow-blue-600/30 transition-all"
          >
            <Scan className="w-4 h-4" />
            <span>Start New Survey</span>
          </button>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Road Surveys"
          value={total}
          icon={Activity}
          iconColor="text-blue-600"
          iconBg="bg-blue-50"
          subtitle="Verified surveys in local registry"
          badge="100% Verified"
          badgeColor="bg-emerald-50 text-emerald-700 border border-emerald-200"
        />

        <MetricCard
          title="Potholes Identified"
          value={potholesDetected}
          icon={AlertTriangle}
          iconColor="text-red-600"
          iconBg="bg-red-50"
          valueColor="text-[#FF3B30]"
          subtitle="Sites with confirmed pothole voids"
          badge="Class 1"
          badgeColor="bg-red-50 text-red-700 border border-red-200"
        />

        <MetricCard
          title="Road Cracks Detected"
          value={cracksDetected}
          icon={Layers}
          iconColor="text-amber-600"
          iconBg="bg-amber-50"
          valueColor="text-amber-600"
          subtitle="Sites with longitudinal/alligator cracks"
          badge="Class 2"
          badgeColor="bg-amber-50 text-amber-700 border border-amber-200"
        />

        <MetricCard
          title="Network Mean Damage"
          value={avgDamage}
          unit="%"
          icon={TrendingUp}
          iconColor="text-indigo-600"
          iconBg="bg-indigo-50"
          valueColor="text-indigo-600"
          subtitle="Mean distress across surveyed lanes"
          badge="PCI Index"
          badgeColor="bg-indigo-50 text-indigo-700 border border-indigo-200"
        />
      </div>

      {/* Analytics Charts & Distributions Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Pothole vs Crack Distribution Card */}
        <div className="bg-white p-5 rounded-xl border border-[#E3E8EF] shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-bold text-[#172435]">Distress Volume Allocation</h3>
              <span className="text-[11px] text-slate-400 font-semibold uppercase">Pixel Footprint</span>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Relative proportion of detected pothole cavities vs pavement cracking fractures.
            </p>

            {/* Split Progress Bar */}
            <div className="h-6 w-full rounded-lg overflow-hidden bg-slate-100 flex p-1 gap-1 border border-slate-200 mb-4">
              <div 
                style={{ width: `${Math.max(potholeShare, 5)}%` }} 
                className="bg-[#FF3B30] rounded-md transition-all flex items-center justify-center text-[10px] font-bold text-white shadow-xs"
                title={`Potholes: ${potholeShare}%`}
              >
                {potholeShare > 10 ? `${potholeShare}%` : ''}
              </div>
              <div 
                style={{ width: `${Math.max(crackShare, 5)}%` }} 
                className="bg-[#FFCC00] rounded-md transition-all flex items-center justify-center text-[10px] font-bold text-slate-900 shadow-xs"
                title={`Cracks: ${crackShare}%`}
              >
                {crackShare > 10 ? `${crackShare}%` : ''}
              </div>
            </div>

            {/* Breakdown numbers */}
            <div className="space-y-2.5">
              <div className="flex items-center justify-between p-2.5 rounded-lg bg-red-50/60 border border-red-100">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded bg-[#FF3B30] flex-shrink-0"></span>
                  <div>
                    <span className="text-xs font-bold text-slate-800 block">Pothole Volume</span>
                    <span className="text-[10px] text-slate-500">Local patch maintenance required</span>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-xs font-bold text-[#FF3B30]">{potholeShare}%</span>
                  <span className="text-[10px] text-slate-500 block font-mono">{potholePx.toLocaleString()} px</span>
                </div>
              </div>

              <div className="flex items-center justify-between p-2.5 rounded-lg bg-amber-50/60 border border-amber-100">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded bg-[#FFCC00] flex-shrink-0"></span>
                  <div>
                    <span className="text-xs font-bold text-slate-800 block">Crack Fracture Volume</span>
                    <span className="text-[10px] text-slate-500">Bituminous sealing required</span>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-xs font-bold text-amber-600">{crackShare}%</span>
                  <span className="text-[10px] text-slate-500 block font-mono">{crackPx.toLocaleString()} px</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
            <span className="font-mono">Weights: P=1.5, C=1.5</span>
            <button 
              onClick={() => setActivePage('analytics')}
              className="text-blue-600 hover:text-blue-700 font-bold flex items-center gap-0.5"
            >
              Full Analytics <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Severity Classification Matrix */}
        <div className="bg-white p-5 rounded-xl border border-[#E3E8EF] shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-bold text-[#172435]">Pavement Condition Index (PCI)</h3>
              <span className="text-[11px] text-slate-400 font-semibold uppercase">FHWA Rating</span>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Structural health classification derived from total damaged area percentage.
            </p>

            <div className="grid grid-cols-2 gap-2.5">
              <div className="p-3 rounded-lg border border-emerald-200 bg-emerald-50/40">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-emerald-900">Good</span>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                </div>
                <div className="text-xl font-bold text-emerald-700">{analytics?.severity_counts?.['Good'] || 0}</div>
                <span className="text-[10px] text-emerald-600 font-medium">0% Damage</span>
              </div>

              <div className="p-3 rounded-lg border border-blue-200 bg-blue-50/40">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-blue-900">Low</span>
                  <AlertCircle className="w-3.5 h-3.5 text-blue-600" />
                </div>
                <div className="text-xl font-bold text-blue-700">{analytics?.severity_counts?.['Low'] || 0}</div>
                <span className="text-[10px] text-blue-600 font-medium">&lt; 1.5% Damage</span>
              </div>

              <div className="p-3 rounded-lg border border-amber-200 bg-amber-50/40">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-amber-900">Moderate</span>
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                </div>
                <div className="text-xl font-bold text-amber-700">{analytics?.severity_counts?.['Moderate'] || 0}</div>
                <span className="text-[10px] text-amber-600 font-medium">1.5% - 5.0% Damage</span>
              </div>

              <div className="p-3 rounded-lg border border-red-200 bg-red-50/40">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-red-900">Critical</span>
                  <ShieldAlert className="w-3.5 h-3.5 text-red-600" />
                </div>
                <div className="text-xl font-bold text-red-700">{analytics?.severity_counts?.['High / Critical'] || 0}</div>
                <span className="text-[10px] text-red-600 font-medium">&gt; 5.0% Damage</span>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500">
            Standard: ASTM D6433 / Federal Highway Administration Guidelines
          </div>
        </div>

        {/* Survey Activity Cadence Card */}
        <div className="bg-white p-5 rounded-xl border border-[#E3E8EF] shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-bold text-[#172435]">Survey Activity Cadence</h3>
              <span className="text-[11px] text-slate-400 font-semibold uppercase">Daily Stream</span>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Temporal distribution of recorded pavement surveys and observed mean distress.
            </p>

            <div className="space-y-2.5">
              {(analytics?.timeline || []).slice(-3).map((t, idx) => (
                <div key={idx} className="flex items-center justify-between p-2.5 rounded-lg bg-slate-50 border border-slate-200/80">
                  <div className="flex items-center gap-2.5">
                    <div className="w-7 h-7 rounded bg-white border border-slate-200 text-slate-700 flex items-center justify-center text-xs font-bold">
                      <Calendar className="w-3.5 h-3.5 text-slate-500" />
                    </div>
                    <div>
                      <p className="text-xs font-bold text-slate-800">{t.date}</p>
                      <p className="text-[10px] text-slate-500">{t.count} surveys registered</p>
                    </div>
                  </div>
                  <span className="text-xs font-bold text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200">
                    {t.avg_damage}% avg
                  </span>
                </div>
              ))}
              {(!analytics?.timeline || analytics?.timeline.length === 0) && (
                <div className="text-center py-8 text-xs text-slate-400">
                  No activity recorded yet. Run a new inspection to start.
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
            <span>Peak Damage: <strong>{analytics?.max_damage_percentage || 0}%</strong></span>
            <span className="font-semibold text-emerald-600 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              Live Database
            </span>
          </div>
        </div>
      </div>

      {/* Recent Inspection Records Table Card */}
      <div className="bg-white rounded-xl border border-[#E3E8EF] shadow-xs overflow-hidden">
        <div className="p-5 border-b border-[#E3E8EF] flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-[#172435]">Recent Inspection Records</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Verified pavement survey records stored in the SQLite database
            </p>
          </div>
          <button
            onClick={() => setActivePage('history')}
            className="text-xs font-bold text-blue-600 hover:text-blue-700 flex items-center gap-1"
          >
            <span>Full Registry ({inspections.length})</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {inspections.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-[#E3E8EF] text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                  <th className="py-3 px-4">Roadway / Survey Title</th>
                  <th className="py-3 px-4">Pothole %</th>
                  <th className="py-3 px-4">Crack %</th>
                  <th className="py-3 px-4">Total Damage</th>
                  <th className="py-3 px-4">Condition Severity</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {inspections.slice(0, 5).map((insp) => (
                  <tr key={insp.id} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-3">
                        {insp.overlay_image ? (
                          <img 
                            src={insp.overlay_image} 
                            alt="thumbnail" 
                            className="w-12 h-9 object-cover rounded border border-slate-200 bg-slate-900 flex-shrink-0"
                          />
                        ) : (
                          <div className="w-12 h-9 rounded bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-400">
                            <Scan className="w-4 h-4" />
                          </div>
                        )}
                        <div className="min-w-0">
                          <span className="font-bold text-slate-900 block truncate max-w-xs">{insp.title}</span>
                          <span className="text-[10px] text-slate-400 font-mono block">
                            {insp.id} • {new Date(insp.created_at).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className="inline-flex items-center gap-1 font-semibold text-[#FF3B30]">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#FF3B30]"></span>
                        {insp.pothole_area_percentage}%
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className="inline-flex items-center gap-1 font-semibold text-amber-600">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#FFCC00]"></span>
                        {insp.crack_area_percentage}%
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className="font-bold text-slate-900 bg-slate-100 px-2.5 py-0.5 rounded border border-slate-200 text-xs">
                        {insp.total_damaged_area_percentage}%
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <SeverityBadge severity={insp.severity} />
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => onViewInspection(insp.id)}
                        className="px-2.5 py-1 text-xs font-bold text-blue-700 bg-blue-50 hover:bg-blue-100 rounded border border-blue-200 transition-colors"
                      >
                        Review
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon={Scan}
            title="No inspection records found"
            description="The local registry has no saved road surveys. Upload an image to run automated semantic inspection."
            actionLabel="Start New Inspection"
            onAction={() => setActivePage('new-inspection')}
          />
        )}
      </div>
    </div>
  );
}
