import React from 'react';
import { Inbox } from 'lucide-react';

export default function EmptyState({ 
  icon: Icon = Inbox, 
  title = "No records found", 
  description = "No data is currently available for this section.", 
  actionLabel, 
  onAction 
}) {
  return (
    <div className="bg-white rounded-xl border border-dashed border-[#E3E8EF] p-12 text-center max-w-lg mx-auto my-6 shadow-xs">
      <div className="w-14 h-14 rounded-full bg-slate-50 border border-slate-200 text-slate-400 flex items-center justify-center mx-auto mb-3.5">
        <Icon className="w-6 h-6 text-slate-500" />
      </div>
      <h3 className="text-sm font-bold text-slate-800 tracking-tight">{title}</h3>
      <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto leading-relaxed">
        {description}
      </p>
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold shadow-xs transition-colors"
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
}
