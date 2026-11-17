import React from 'react';

export default function SeverityBadge({ severity, size = 'sm' }) {
  const configs = {
    'Good': {
      bg: 'bg-emerald-50 text-emerald-800 border-emerald-200',
      dot: 'bg-emerald-500',
      label: 'Good (0% Damage)'
    },
    'Low': {
      bg: 'bg-blue-50 text-blue-800 border-blue-200',
      dot: 'bg-blue-500',
      label: 'Low (< 1.5%)'
    },
    'Moderate': {
      bg: 'bg-amber-50 text-amber-800 border-amber-200',
      dot: 'bg-amber-500',
      label: 'Moderate (1.5% - 5%)'
    },
    'High / Critical': {
      bg: 'bg-red-50 text-red-800 border-red-200',
      dot: 'bg-red-500',
      label: 'Critical (> 5%)'
    }
  };

  const current = configs[severity] || {
    bg: 'bg-slate-100 text-slate-700 border-slate-200',
    dot: 'bg-slate-400',
    label: severity || 'Unrated'
  };

  const sizeClasses = size === 'lg' 
    ? 'px-3 py-1 text-xs font-semibold' 
    : 'px-2.5 py-0.5 text-[11px] font-medium';

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border ${current.bg} ${sizeClasses} tracking-tight`}>
      <span className={`w-1.5 h-1.5 rounded-full ${current.dot} flex-shrink-0`}></span>
      <span>{severity || 'Unknown'}</span>
    </span>
  );
}
