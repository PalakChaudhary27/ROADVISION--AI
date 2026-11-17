import React from 'react';

export default function MetricCard({ 
  title, 
  value, 
  unit = '', 
  subtitle, 
  icon: Icon, 
  iconColor = 'text-blue-600', 
  iconBg = 'bg-blue-50',
  badge,
  badgeColor = 'text-slate-500 bg-slate-100',
  valueColor = 'text-slate-900',
  className = ''
}) {
  return (
    <div className={`bg-white p-5 rounded-xl border border-[#E3E8EF] shadow-xs flex flex-col justify-between ${className}`}>
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
            {title}
          </span>
          {Icon && (
            <div className={`w-8 h-8 rounded-lg ${iconBg} ${iconColor} flex items-center justify-center flex-shrink-0`}>
              <Icon className="w-4 h-4" />
            </div>
          )}
        </div>
        
        <div className="flex items-baseline gap-1 mt-1">
          <span className={`text-2xl font-bold tracking-tight ${valueColor}`}>
            {value}
          </span>
          {unit && (
            <span className="text-sm font-semibold text-slate-500">
              {unit}
            </span>
          )}
        </div>
      </div>

      {(subtitle || badge) && (
        <div className="mt-3 pt-2.5 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
          {subtitle && (
            <span className="truncate pr-2">{subtitle}</span>
          )}
          {badge && (
            <span className={`px-2 py-0.5 rounded text-[10px] font-semibold flex-shrink-0 ${badgeColor}`}>
              {badge}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
