import React from 'react';
import { 
  Printer, 
  FileText, 
  Calendar, 
  CheckCircle2,
  Scan,
  ShieldCheck,
  Building2,
  Award
} from 'lucide-react';
import SeverityBadge from '../components/SeverityBadge';
import EmptyState from '../components/EmptyState';

export default function ReportsPage({ inspections, selectedInspection, onSelectInspection }) {
  const current = selectedInspection || (inspections.length > 0 ? inspections[0] : null);

  const handlePrintOrPdf = () => {
    window.print();
  };

  if (!current) {
    return (
      <EmptyState
        icon={FileText}
        title="No Inspection Reports Available"
        description="Conduct a road survey in New Inspection and save it to generate formal engineering audit sheets."
      />
    );
  }

  const {
    id,
    title,
    filename,
    created_at,
    image_dimensions,
    pothole_area_percentage,
    crack_area_percentage,
    total_damaged_area_percentage,
    background_area_percentage,
    severity,
    severity_description,
    metrics,
    original_image,
    overlay_image,
    mask_image,
    notes
  } = current;

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Top Controls Bar (Hidden during print) */}
      <div className="bg-white p-4 rounded-xl border border-[#E3E8EF] shadow-xs flex flex-wrap items-center justify-between gap-4 no-print">
        {/* Inspection Selector Dropdown */}
        <div className="flex items-center gap-3">
          <span className="text-xs font-bold text-slate-500">Selected Audit Survey:</span>
          <select
            value={current.id}
            onChange={(e) => onSelectInspection(e.target.value)}
            className="text-xs font-bold text-slate-800 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500/20 max-w-md truncate"
          >
            {inspections.map((i) => (
              <option key={i.id} value={i.id}>
                {i.title} ({i.id}) — {i.total_damaged_area_percentage}% Damage [{i.severity}]
              </option>
            ))}
          </select>
        </div>

        {/* Print / PDF Action */}
        <button
          onClick={handlePrintOrPdf}
          className="bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-4 py-2.5 rounded-lg flex items-center gap-2 shadow-sm transition-all"
        >
          <Printer className="w-4 h-4" />
          <span>Print / Export PDF Audit Report</span>
        </button>
      </div>

      {/* Printable Report Document (Formal Infrastructure Engineering Audit Sheet) */}
      <div className="bg-white rounded-xl border border-[#E3E8EF] shadow-md p-8 print:p-0 print:border-none print:shadow-none space-y-6 text-[#172435] font-sans print-container">
        
        {/* Report Document Header */}
        <div className="border-b-2 border-slate-900 pb-5 flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2.5 mb-1.5">
              <div className="w-8 h-8 rounded-lg bg-slate-900 text-white flex items-center justify-center font-bold text-xs">
                RV
              </div>
              <div>
                <span className="font-extrabold text-lg tracking-tight text-slate-900">ROADVISION AI</span>
                <span className="text-[10px] font-bold ml-2 px-2 py-0.5 rounded bg-blue-100 text-blue-800 uppercase tracking-wider">
                  Official Audit
                </span>
              </div>
            </div>
            <p className="text-xs text-slate-600 font-semibold">
              Bureau of Pavement Asset Management & Autonomous Infrastructure Surveillance
            </p>
          </div>

          <div className="text-right">
            <span className="text-[10px] font-mono text-slate-400 block uppercase font-bold tracking-wider">Document Control No.</span>
            <span className="text-sm font-bold text-slate-900 font-mono">{id}</span>
            <span className="text-xs text-slate-500 block font-medium mt-0.5">
              Survey Date: {new Date(created_at).toLocaleDateString()}
            </span>
          </div>
        </div>

        {/* Survey Overview Metadata Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 rounded-lg bg-slate-50 border border-slate-200 text-xs">
          <div>
            <span className="text-slate-400 block text-[10px] uppercase font-bold">Survey Roadway</span>
            <span className="font-bold text-slate-800 truncate block mt-0.5">{title}</span>
          </div>
          <div>
            <span className="text-slate-400 block text-[10px] uppercase font-bold">Inference Engine</span>
            <span className="font-bold text-slate-800 block mt-0.5 font-mono">DeepLabV3-ResNet50</span>
          </div>
          <div>
            <span className="text-slate-400 block text-[10px] uppercase font-bold">Native Resolution</span>
            <span className="font-bold text-slate-800 font-mono block mt-0.5">
              {image_dimensions?.width} × {image_dimensions?.height} px
            </span>
          </div>
          <div>
            <span className="text-slate-400 block text-[10px] uppercase font-bold">PCI Condition Rating</span>
            <div className="mt-0.5">
              <SeverityBadge severity={severity} size="sm" />
            </div>
          </div>
        </div>

        {/* Executive Engineering Assessment */}
        <div className="border border-slate-200 rounded-lg p-4 bg-white space-y-2">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-900 flex items-center gap-1.5">
            <FileText className="w-3.5 h-3.5 text-blue-600" />
            <span>Executive Engineering Assessment</span>
          </h4>
          <p className="text-xs text-slate-700 leading-relaxed">
            The automated DeepLabV3-ResNet50 semantic segmentation model evaluated the pavement surface and quantified a total distressed area of 
            <strong> {total_damaged_area_percentage}%</strong> across the surveyed lane. Pothole cavities occupy 
            <strong> {pothole_area_percentage}%</strong> ({metrics?.pixels?.pothole?.toLocaleString()} px), while crack fractures comprise 
            <strong> {crack_area_percentage}%</strong> ({metrics?.pixels?.crack?.toLocaleString()} px). Pavement health condition is rated as 
            <strong> {severity}</strong>: <em>"{severity_description}"</em>.
          </p>
          {notes && (
            <p className="text-xs text-slate-600 italic bg-slate-50 p-2.5 rounded border border-slate-200/60">
              Observer Notes: {notes}
            </p>
          )}
        </div>

        {/* Visual Inspection Evidence: 3-Panel Side-by-Side */}
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-900 mb-2.5">
            Visual Inspection Evidence & Segmentation
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* Figure 1: Original */}
            <div className="rounded-lg border border-slate-300 overflow-hidden bg-slate-900 text-center">
              <div className="p-1.5 bg-slate-100 border-b border-slate-200 text-[10px] font-bold text-slate-700">
                Figure 1: Source Road Surface
              </div>
              <img
                src={original_image || overlay_image}
                alt="Source Image"
                className="w-full h-44 object-cover"
              />
            </div>

            {/* Figure 2: Mask */}
            <div className="rounded-lg border border-slate-300 overflow-hidden bg-slate-900 text-center">
              <div className="p-1.5 bg-slate-100 border-b border-slate-200 text-[10px] font-bold text-slate-700">
                Figure 2: AI Segmentation Mask
              </div>
              <img
                src={mask_image}
                alt="Segmentation Mask"
                className="w-full h-44 object-cover"
              />
            </div>

            {/* Figure 3: Overlay */}
            <div className="rounded-lg border border-slate-300 overflow-hidden bg-slate-900 text-center">
              <div className="p-1.5 bg-slate-100 border-b border-slate-200 text-[10px] font-bold text-slate-700">
                Figure 3: Pavement Distress Overlay
              </div>
              <img
                src={overlay_image}
                alt="Distress Overlay"
                className="w-full h-44 object-cover"
              />
            </div>
          </div>
          <p className="text-[10px] text-slate-500 mt-2 italic">
            Classification Color Key: Red (#FF3B30) designates pothole voids; Yellow (#FFCC00) designates longitudinal/alligator cracks; Dark Navy (#14243A) designates healthy road surface.
          </p>
        </div>

        {/* Quantitative Surface Distress Measurements Table */}
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-900 mb-2">
            Quantitative Surface Distress Measurements
          </h4>
          <table className="w-full text-left border-collapse text-xs border border-slate-200">
            <thead>
              <tr className="bg-slate-100 border-b border-slate-200 font-bold text-slate-800">
                <th className="py-2.5 px-3">Distress Classification</th>
                <th className="py-2.5 px-3">Pixel Allocation</th>
                <th className="py-2.5 px-3">Surface Share (%)</th>
                <th className="py-2.5 px-3">Recommended Engineering Intervention</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              <tr>
                <td className="py-2 px-3 font-semibold text-slate-800 flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded bg-[#FF3B30] flex-shrink-0"></span>
                  Pothole (Class 1)
                </td>
                <td className="py-2 px-3 font-mono">{metrics?.pixels?.pothole?.toLocaleString()} px</td>
                <td className="py-2 px-3 font-bold text-[#FF3B30]">{pothole_area_percentage}%</td>
                <td className="py-2 px-3 text-slate-700">Cold patching or hot-mix asphalt infill</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-semibold text-slate-800 flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded bg-[#FFCC00] flex-shrink-0"></span>
                  Road Crack (Class 2)
                </td>
                <td className="py-2 px-3 font-mono">{metrics?.pixels?.crack?.toLocaleString()} px</td>
                <td className="py-2 px-3 font-bold text-amber-600">{crack_area_percentage}%</td>
                <td className="py-2 px-3 text-slate-700">Bituminous crack sealing or micro-surfacing</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-semibold text-slate-800 flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded bg-[#14243A] flex-shrink-0"></span>
                  Intact Pavement (Class 0)
                </td>
                <td className="py-2 px-3 font-mono">{metrics?.pixels?.background?.toLocaleString()} px</td>
                <td className="py-2 px-3 font-bold text-slate-800">{background_area_percentage}%</td>
                <td className="py-2 px-3 text-slate-700">Satisfactory structural integrity</td>
              </tr>
              <tr className="bg-slate-50 font-bold border-t-2 border-slate-300">
                <td className="py-2.5 px-3">Total Evaluated Area</td>
                <td className="py-2.5 px-3 font-mono">{metrics?.pixels?.total?.toLocaleString()} px</td>
                <td className="py-2.5 px-3 text-slate-900">100.0%</td>
                <td className="py-2.5 px-3 text-blue-700 font-extrabold">Total Damaged: {total_damaged_area_percentage}%</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Engineering Certification Sign-off Block */}
        <div className="pt-6 border-t-2 border-slate-900 grid grid-cols-2 sm:grid-cols-3 gap-6 text-xs text-slate-600">
          <div>
            <span className="text-[10px] uppercase font-bold text-slate-400 block">Inspected By</span>
            <span className="font-bold text-slate-900 block mt-1">Autonomous Vision Station 01</span>
            <span className="text-[11px] text-slate-500 block">RoadVision AI Engine v1.0</span>
          </div>

          <div>
            <span className="text-[10px] uppercase font-bold text-slate-400 block">Certified Civil Engineer</span>
            <div className="mt-4 border-b border-slate-400 w-40"></div>
            <span className="text-[10px] text-slate-400 block mt-1">Professional Engineer Signature</span>
          </div>

          <div>
            <span className="text-[10px] uppercase font-bold text-slate-400 block">Next Scheduled Audit</span>
            <span className="font-bold text-slate-900 block mt-1">
              {new Date(new Date(created_at).getTime() + 90 * 24 * 60 * 60 * 1000).toLocaleDateString()}
            </span>
            <span className="text-[10px] text-slate-500 block">90-Day Pavement Review Cycle</span>
          </div>
        </div>
      </div>
    </div>
  );
}
