import React, { useState } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  PieChart, 
  AlertTriangle, 
  ShieldCheck, 
  Layers, 
  Filter,
  Calendar,
  Sparkles,
  Info,
  CheckCircle2,
  ShieldAlert,
  ArrowUpRight
} from 'lucide-react';
import MetricCard from '../components/MetricCard';
import SeverityBadge from '../components/SeverityBadge';
import EmptyState from '../components/EmptyState';

export default function AnalyticsPage({ analytics, inspections }) {
  const [dateFilter, setDateFilter] = useState('all');
  const [severityFilter, setSeverityFilter] = useState('all');

  const total = analytics?.total_inspections || 0;
  const avgDamage = analytics?.average_damage_percentage || 0;
  const avgPothole = analytics?.average_pothole_percentage || 0;
  const avgCrack = analytics?.average_crack_percentage || 0;
  const maxDamage = analytics?.max_damage_percentage || 0;
  const potholePx = analytics?.total_pothole_pixels || 0;
  const crackPx = analytics?.total_crack_pixels || 0;
  const totalDamagePx = potholePx + crackPx || 1;
  const potholeRatio = Math.round((potholePx / totalDamagePx) * 100);
  const crackRatio = 100 - potholeRatio;

  // Filter inspections based on date and severity
  const filtered = (inspections || []).filter(i => {
    let matchesDate = true;
    if (dateFilter !== 'all') {
      const daysAgo = (new Date() - new Date(i.created_at)) / (1000 * 60 * 60 * 24);
      if (dateFilter === '7d') matchesDate = daysAgo <= 7;
      if (dateFilter === '30d') matchesDate = daysAgo <= 30;
    }

    let matchesSeverity = true;
    if (severityFilter !== 'all') {
      matchesSeverity = i.severity === severityFilter;
    }

    return matchesDate && matchesSeverity;
  });

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Top Header & Filter Controls Bar */}
      <div className="bg-white p-5 rounded-xl border border-[#E3E8EF] shadow-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold text-blue-600 mb-1">
            <BarChart3 className="w-4 h-4" />
            <span>Aggregate Infrastructure Intelligence</span>
          </div>
          <h2 className="text-xl font-bold text-[#172435] tracking-tight">Network Pavement Analytics</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Statistical distress metrics computed from {total} verified database inspection records.
          </p>
        </div>

        {/* Filter Pills Group */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {/* Timeframe selector */}
          <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200">
            <span className="text-[10px] font-bold text-slate-400 px-1.5 uppercase">Time:</span>
            {['all', '30d', '7d'].map((period) => (
              <button
                key={period}
                onClick={() => setDateFilter(period)}
                className={`px-2.5 py-1 rounded text-xs font-semibold transition-all ${
                  dateFilter === period
                    ? 'bg-white text-blue-700 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {period === 'all' ? 'All Time' : period === '30d' ? '30 Days' : '7 Days'}
              </button>
            ))}
          </div>

          {/* Severity selector */}
          <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200">
            <span className="text-[10px] font-bold text-slate-400 px-1.5 uppercase">Severity:</span>
            {['all', 'Good', 'Moderate', 'High / Critical'].map((sev) => (
              <button
                key={sev}
                onClick={() => setSeverityFilter(sev)}
                className={`px-2 py-1 rounded text-xs font-semibold transition-all ${
                  severityFilter === sev
                    ? 'bg-white text-blue-700 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {sev === 'all' ? 'All' : sev === 'High / Critical' ? 'Critical' : sev}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Primary Analytics Summary Metric Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Network Mean Damage"
          value={avgDamage}
          unit="%"
          icon={TrendingUp}
          iconColor="text-blue-600"
          iconBg="bg-blue-50"
          subtitle={`Pothole: ${avgPothole}% • Crack: ${avgCrack}%`}
        />

        <MetricCard
          title="Peak Distress Observed"
          value={maxDamage}
          unit="%"
          icon={AlertTriangle}
          iconColor="text-red-600"
          iconBg="bg-red-50"
          valueColor="text-[#FF3B30]"
          subtitle="Maximum single-lane distress"
          badge="Max Observed"
          badgeColor="bg-red-50 text-red-700 border border-red-200"
        />

        <MetricCard
          title="Pothole Surface Share"
          value={potholeRatio}
          unit="%"
          icon={AlertTriangle}
          iconColor="text-[#FF3B30]"
          iconBg="bg-red-50"
          valueColor="text-[#FF3B30]"
          subtitle={`${potholePx.toLocaleString()} damaged pixels`}
          badge="Class 1"
          badgeColor="bg-red-50 text-red-700 border border-red-200"
        />

        <MetricCard
          title="Crack Surface Share"
          value={crackRatio}
          unit="%"
          icon={Layers}
          iconColor="text-amber-600"
          iconBg="bg-amber-50"
          valueColor="text-amber-600"
          subtitle={`${crackPx.toLocaleString()} damaged pixels`}
          badge="Class 2"
          badgeColor="bg-amber-50 text-amber-700 border border-amber-200"
        />
      </div>

      {/* Comparative Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Survey-by-Survey Damage Comparison Chart */}
        <div className="bg-white p-5 rounded-xl border border-[#E3E8EF] shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-[#172435]">Survey Distress Breakdown</h3>
            <span className="text-[11px] text-slate-400 font-bold uppercase">Class-wise Allocation</span>
          </div>
          <p className="text-xs text-slate-500">
            Damaged area percentage per surveyed roadway in the registry.
          </p>

          <div className="space-y-3 pt-1">
            {filtered.map((item, idx) => (
              <div key={idx} className="p-3 rounded-lg border border-slate-200 bg-slate-50/50 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <div className="min-w-0 pr-2">
                    <span className="font-bold text-slate-900 block truncate max-w-xs">{item.title}</span>
                    <span className="text-[10px] text-slate-400 font-mono">{item.id}</span>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <span className="font-extrabold text-slate-900 text-sm">{item.total_damaged_area_percentage}%</span>
                    <span className="text-[10px] text-slate-400 block">Total Damage</span>
                  </div>
                </div>

                {/* Stacked Horizontal Bar */}
                <div className="h-3.5 w-full rounded bg-slate-200 overflow-hidden flex border border-slate-300/60 shadow-xs">
                  <div
                    style={{ width: `${Math.min(item.pothole_area_percentage * 3, 100)}%` }}
                    className="bg-[#FF3B30] transition-all"
                    title={`Pothole: ${item.pothole_area_percentage}%`}
                  ></div>
                  <div
                    style={{ width: `${Math.min(item.crack_area_percentage * 3, 100)}%` }}
                    className="bg-[#FFCC00] transition-all"
                    title={`Crack: ${item.crack_area_percentage}%`}
                  ></div>
                </div>

                <div className="flex items-center justify-between text-[11px] text-slate-600 pt-0.5">
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-xs bg-[#FF3B30]"></span>
                      <span>Pothole: <strong>{item.pothole_area_percentage}%</strong></span>
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-xs bg-[#FFCC00]"></span>
                      <span>Crack: <strong>{item.crack_area_percentage}%</strong></span>
                    </span>
                  </div>
                  <SeverityBadge severity={item.severity} />
                </div>
              </div>
            ))}

            {filtered.length === 0 && (
              <div className="text-center py-8 text-xs text-slate-400">
                No inspections matching the selected date and severity filters.
              </div>
            )}
          </div>
        </div>

        {/* Severity Classification Breakdown & Recommendations */}
        <div className="bg-white p-5 rounded-xl border border-[#E3E8EF] shadow-xs space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-bold text-[#172435]">Condition Severity Portfolio</h3>
              <span className="text-[11px] text-slate-400 font-bold uppercase">Maintenance Urgency</span>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Categorization of surveyed roadways across engineering priority tiers.
            </p>

            <div className="space-y-3">
              {[
                { 
                  label: 'Good (0% Damage)', 
                  desc: 'Routine monitoring, no immediate maintenance', 
                  count: analytics?.severity_counts?.['Good'] || 0, 
                  color: 'bg-emerald-500', 
                  textColor: 'text-emerald-700',
                  badgeColor: 'bg-emerald-50 border-emerald-200' 
                },
                { 
                  label: 'Low (< 1.5% Damage)', 
                  desc: 'Minor surface defect, schedule preventative check', 
                  count: analytics?.severity_counts?.['Low'] || 0, 
                  color: 'bg-blue-500', 
                  textColor: 'text-blue-700',
                  badgeColor: 'bg-blue-50 border-blue-200' 
                },
                { 
                  label: 'Moderate (1.5% - 5.0% Damage)', 
                  desc: 'Noticeable degradation, planned treatment required', 
                  count: analytics?.severity_counts?.['Moderate'] || 0, 
                  color: 'bg-amber-500', 
                  textColor: 'text-amber-700',
                  badgeColor: 'bg-amber-50 border-amber-200' 
                },
                { 
                  label: 'Critical (> 5.0% Damage)', 
                  desc: 'Prompt intervention required: patch and sealing', 
                  count: analytics?.severity_counts?.['High / Critical'] || 0, 
                  color: 'bg-red-500', 
                  textColor: 'text-red-700',
                  badgeColor: 'bg-red-50 border-red-200' 
                },
              ].map((tier, idx) => {
                const pct = total > 0 ? Math.round((tier.count / total) * 100) : 0;
                return (
                  <div key={idx} className="p-3.5 rounded-lg border border-slate-200/90 bg-slate-50/50 space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <div>
                        <span className="font-bold text-slate-800">{tier.label}</span>
                        <p className="text-[10px] text-slate-500">{tier.desc}</p>
                      </div>
                      <span className={`font-extrabold text-sm ${tier.textColor}`}>
                        {tier.count} ({pct}%)
                      </span>
                    </div>
                    <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden">
                      <div
                        style={{ width: `${pct}%` }}
                        className={`h-full ${tier.color} transition-all`}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="p-3.5 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-900 flex items-start gap-2.5 mt-4">
            <Info className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
            <p className="leading-relaxed">
              <strong>Engineering Protocol:</strong> Roadways categorized as <strong>Moderate</strong> or <strong>Critical</strong> should be dispatched for mechanized crack sealing and asphalt infill within 14 business days.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
