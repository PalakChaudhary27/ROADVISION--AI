import React, { useState } from 'react';
import { 
  Search, 
  Filter, 
  Trash2, 
  Eye, 
  FileText, 
  Calendar, 
  AlertTriangle, 
  CheckCircle2, 
  Scan,
  AlertCircle,
  ArrowUpDown,
  Download
} from 'lucide-react';
import SeverityBadge from '../components/SeverityBadge';
import EmptyState from '../components/EmptyState';

export default function InspectionHistoryPage({ 
  inspections, 
  onViewInspection, 
  onGenerateReport, 
  onDeleteInspection 
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [severityFilter, setSeverityFilter] = useState('all');
  const [sortBy, setSortBy] = useState('date-desc'); // 'date-desc' | 'date-asc' | 'damage-desc' | 'damage-asc'
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);

  // Filter
  const filtered = (inspections || []).filter(item => {
    const matchesSearch = 
      item.title?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.id?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.filename?.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesSeverity = 
      severityFilter === 'all' || item.severity === severityFilter;

    return matchesSearch && matchesSeverity;
  });

  // Sort
  const sorted = [...filtered].sort((a, b) => {
    if (sortBy === 'date-desc') {
      return new Date(b.created_at) - new Date(a.created_at);
    }
    if (sortBy === 'date-asc') {
      return new Date(a.created_at) - new Date(b.created_at);
    }
    if (sortBy === 'damage-desc') {
      return (b.total_damaged_area_percentage || 0) - (a.total_damaged_area_percentage || 0);
    }
    if (sortBy === 'damage-asc') {
      return (a.total_damaged_area_percentage || 0) - (b.total_damaged_area_percentage || 0);
    }
    return 0;
  });

  const handleDelete = (id) => {
    onDeleteInspection(id);
    setDeleteConfirmId(null);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Top Search, Filter & Sort Controls Bar */}
      <div className="bg-white p-5 rounded-xl border border-[#E3E8EF] shadow-xs flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-4">
        {/* Search Input */}
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by roadway title, survey ID, or filename..."
            className="w-full text-xs pl-9 pr-3 py-2.5 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-slate-800"
          />
        </div>

        {/* Severity Filters & Sorting */}
        <div className="flex flex-wrap items-center gap-2.5 text-xs">
          {/* Severity Chips */}
          <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200">
            <span className="text-[10px] font-bold text-slate-400 px-1 uppercase">Severity:</span>
            {['all', 'Good', 'Low', 'Moderate', 'High / Critical'].map((sev) => (
              <button
                key={sev}
                onClick={() => setSeverityFilter(sev)}
                className={`px-2.5 py-1 rounded text-xs font-semibold transition-all ${
                  severityFilter === sev
                    ? 'bg-white text-blue-700 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {sev === 'all' ? 'All' : sev === 'High / Critical' ? 'Critical' : sev}
              </button>
            ))}
          </div>

          {/* Sort Dropdown */}
          <div className="flex items-center gap-1.5 bg-slate-100 px-2.5 py-1 rounded-lg border border-slate-200">
            <ArrowUpDown className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-transparent text-xs font-semibold text-slate-700 focus:outline-none cursor-pointer"
            >
              <option value="date-desc">Newest First</option>
              <option value="date-asc">Oldest First</option>
              <option value="damage-desc">Highest Damage %</option>
              <option value="damage-asc">Lowest Damage %</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Table Card */}
      <div className="bg-white rounded-xl border border-[#E3E8EF] shadow-xs overflow-hidden">
        <div className="p-5 border-b border-[#E3E8EF] flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-[#172435]">Inspection Survey Registry</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Showing {sorted.length} of {inspections.length} recorded pavement surveys
            </p>
          </div>
        </div>

        {sorted.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-[#E3E8EF] text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                  <th className="py-3 px-4">Roadway / Survey Title</th>
                  <th className="py-3 px-4">Survey Date</th>
                  <th className="py-3 px-4">Pothole %</th>
                  <th className="py-3 px-4">Crack %</th>
                  <th className="py-3 px-4">Total Damage</th>
                  <th className="py-3 px-4">Condition Severity</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {sorted.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-3">
                        {item.overlay_image ? (
                          <img 
                            src={item.overlay_image} 
                            alt="thumbnail" 
                            className="w-14 h-10 object-cover rounded border border-slate-200 bg-slate-900 flex-shrink-0"
                          />
                        ) : (
                          <div className="w-14 h-10 rounded bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-400">
                            <Scan className="w-4 h-4" />
                          </div>
                        )}
                        <div className="min-w-0">
                          <span className="font-bold text-slate-900 block truncate max-w-xs">{item.title}</span>
                          <span className="text-[10px] text-slate-400 font-mono block">
                            {item.id} • {item.image_dimensions?.width}×{item.image_dimensions?.height} px
                          </span>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-slate-600 whitespace-nowrap">
                      <div className="flex items-center gap-1.5">
                        <Calendar className="w-3.5 h-3.5 text-slate-400" />
                        <span>{new Date(item.created_at).toLocaleDateString()}</span>
                        <span className="text-[10px] text-slate-400">
                          {new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className="font-semibold text-[#FF3B30] flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#FF3B30]"></span>
                        {item.pothole_area_percentage}%
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className="font-semibold text-amber-600 flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#FFCC00]"></span>
                        {item.crack_area_percentage}%
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className="font-extrabold text-slate-900 bg-slate-100 px-2 py-0.5 rounded border border-slate-200 text-xs">
                        {item.total_damaged_area_percentage}%
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <SeverityBadge severity={item.severity} />
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <button
                          onClick={() => onViewInspection(item.id)}
                          className="px-2.5 py-1 text-xs font-bold text-blue-700 bg-blue-50 hover:bg-blue-100 rounded border border-blue-200 transition-colors flex items-center gap-1"
                          title="View Detailed Detection Results"
                        >
                          <Eye className="w-3 h-3" />
                          <span>Review</span>
                        </button>

                        <button
                          onClick={() => onGenerateReport(item)}
                          className="p-1.5 text-indigo-700 bg-indigo-50 hover:bg-indigo-100 rounded border border-indigo-200 transition-colors"
                          title="Generate PDF Audit Report"
                        >
                          <FileText className="w-3.5 h-3.5" />
                        </button>

                        <button
                          onClick={() => setDeleteConfirmId(item.id)}
                          className="p-1.5 text-red-500 hover:text-red-700 hover:bg-red-50 rounded border border-transparent hover:border-red-200 transition-colors"
                          title="Delete Inspection Record"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon={Search}
            title="No matching inspections found"
            description="No saved records match your search terms or filter selection. Try adjusting your query or filters."
            actionLabel={searchTerm || severityFilter !== 'all' ? "Clear Filters" : undefined}
            onAction={() => {
              setSearchTerm('');
              setSeverityFilter('all');
            }}
          />
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {deleteConfirmId && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-xl max-w-sm w-full p-5 shadow-2xl border border-slate-200 space-y-4 animate-in fade-in zoom-in-95 duration-100">
            <div className="w-10 h-10 rounded-full bg-red-100 text-red-600 flex items-center justify-center mx-auto">
              <AlertCircle className="w-5 h-5" />
            </div>
            <div className="text-center">
              <h4 className="text-base font-bold text-slate-900">Delete Inspection Record?</h4>
              <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                Are you sure you want to permanently delete record <strong className="font-mono text-slate-800">{deleteConfirmId}</strong>? 
                This action will remove all saved damage segmentation masks and cannot be undone.
              </p>
            </div>
            <div className="flex items-center gap-2 pt-2">
              <button
                onClick={() => setDeleteConfirmId(null)}
                className="flex-1 py-2 text-xs font-bold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDelete(deleteConfirmId)}
                className="flex-1 py-2 text-xs font-bold text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors shadow-xs"
              >
                Yes, Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
