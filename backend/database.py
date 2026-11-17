"""
RoadVision AI - SQLite Database for Saved Inspection Records
Provides persistent storage, querying, and analytics aggregation for inspections.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = Path(__file__).resolve().parent / "roadvision.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                pothole_area_percentage REAL NOT NULL,
                crack_area_percentage REAL NOT NULL,
                total_damaged_area_percentage REAL NOT NULL,
                background_area_percentage REAL NOT NULL,
                severity TEXT NOT NULL,
                severity_description TEXT NOT NULL,
                pothole_pixels INTEGER NOT NULL,
                crack_pixels INTEGER NOT NULL,
                background_pixels INTEGER NOT NULL,
                total_pixels INTEGER NOT NULL,
                original_image TEXT NOT NULL,
                overlay_image TEXT NOT NULL,
                mask_image TEXT NOT NULL,
                notes TEXT DEFAULT '',
                bbox_image TEXT DEFAULT '',
                bounding_boxes TEXT DEFAULT '[]',
                damage_statistics TEXT DEFAULT '{}'
            )
        """)
        # Safe migration if table already exists from older version
        for col_def in [
            "ALTER TABLE inspections ADD COLUMN bbox_image TEXT DEFAULT ''",
            "ALTER TABLE inspections ADD COLUMN bounding_boxes TEXT DEFAULT '[]'",
            "ALTER TABLE inspections ADD COLUMN damage_statistics TEXT DEFAULT '{}'"
        ]:
            try:
                conn.execute(col_def)
            except sqlite3.OperationalError:
                pass
        conn.commit()


def create_inspection(data: Dict[str, Any]) -> Dict[str, Any]:
    insp_id = data.get("id") or f"INSP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    created_at = data.get("created_at") or datetime.now().isoformat()
    bbox_img = data.get("bbox_image", data.get("images", {}).get("bbox", ""))
    bboxes_json = json.dumps(data.get("bounding_boxes", data.get("boxes", [])))
    dmg_stats_json = json.dumps(data.get("damage_statistics", {}))
    
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO inspections (
                id, title, filename, created_at,
                image_width, image_height,
                pothole_area_percentage, crack_area_percentage,
                total_damaged_area_percentage, background_area_percentage,
                severity, severity_description,
                pothole_pixels, crack_pixels, background_pixels, total_pixels,
                original_image, overlay_image, mask_image, notes,
                bbox_image, bounding_boxes, damage_statistics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            insp_id,
            data.get("title", data.get("filename", "Road Inspection")),
            data.get("filename", "unnamed.jpg"),
            created_at,
            data["image_dimensions"]["width"],
            data["image_dimensions"]["height"],
            data["pothole_area_percentage"],
            data["crack_area_percentage"],
            data["total_damaged_area_percentage"],
            data["background_area_percentage"],
            data["severity"],
            data.get("severity_description", ""),
            data["metrics"]["pixels"]["pothole"],
            data["metrics"]["pixels"]["crack"],
            data["metrics"]["pixels"]["background"],
            data["metrics"]["pixels"]["total"],
            data.get("original_image", ""),
            data.get("overlay_image", data.get("images", {}).get("overlay", "")),
            data.get("mask_image", data.get("images", {}).get("mask", "")),
            data.get("notes", ""),
            bbox_img,
            bboxes_json,
            dmg_stats_json
        ))
        conn.commit()

    return get_inspection_by_id(insp_id)


def get_inspections(search: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_db() as conn:
        query = """
            SELECT id, title, filename, created_at,
                   image_width, image_height,
                   pothole_area_percentage, crack_area_percentage,
                   total_damaged_area_percentage, background_area_percentage,
                   severity, severity_description,
                   pothole_pixels, crack_pixels, background_pixels, total_pixels,
                   overlay_image, bbox_image, bounding_boxes, damage_statistics, notes
            FROM inspections
        """
        params = []
        if search:
            query += " WHERE title LIKE ? OR filename LIKE ? OR severity LIKE ?"
            term = f"%{search}%"
            params.extend([term, term, term])
            
        query += " ORDER BY created_at DESC"
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for r in rows:
            try:
                bboxes = json.loads(r["bounding_boxes"]) if r["bounding_boxes"] else []
            except Exception:
                bboxes = []
            try:
                dmg_stats = json.loads(r["damage_statistics"]) if r["damage_statistics"] else {}
            except Exception:
                dmg_stats = {}

            results.append({
                "id": r["id"],
                "title": r["title"],
                "filename": r["filename"],
                "created_at": r["created_at"],
                "image_dimensions": {
                    "width": r["image_width"],
                    "height": r["image_height"]
                },
                "pothole_area_percentage": r["pothole_area_percentage"],
                "crack_area_percentage": r["crack_area_percentage"],
                "total_damaged_area_percentage": r["total_damaged_area_percentage"],
                "background_area_percentage": r["background_area_percentage"],
                "severity": r["severity"],
                "severity_description": r["severity_description"],
                "metrics": {
                    "pixels": {
                        "pothole": r["pothole_pixels"],
                        "crack": r["crack_pixels"],
                        "background": r["background_pixels"],
                        "total": r["total_pixels"]
                    }
                },
                "overlay_image": r["overlay_image"],
                "bbox_image": r["bbox_image"] if "bbox_image" in r.keys() else "",
                "bounding_boxes": bboxes,
                "damage_statistics": dmg_stats,
                "notes": r["notes"]
            })
        return results


def get_inspection_by_id(insp_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM inspections WHERE id = ?", (insp_id,))
        r = cursor.fetchone()
        if not r:
            return None
        
        row_keys = r.keys()
        try:
            bboxes = json.loads(r["bounding_boxes"]) if "bounding_boxes" in row_keys and r["bounding_boxes"] else []
        except Exception:
            bboxes = []
        try:
            dmg_stats = json.loads(r["damage_statistics"]) if "damage_statistics" in row_keys and r["damage_statistics"] else {}
        except Exception:
            dmg_stats = {}

        bbox_img = r["bbox_image"] if "bbox_image" in row_keys and r["bbox_image"] else ""

        return {
            "id": r["id"],
            "title": r["title"],
            "filename": r["filename"],
            "created_at": r["created_at"],
            "image_dimensions": {
                "width": r["image_width"],
                "height": r["image_height"]
            },
            "pothole_area_percentage": r["pothole_area_percentage"],
            "crack_area_percentage": r["crack_area_percentage"],
            "total_damaged_area_percentage": r["total_damaged_area_percentage"],
            "background_area_percentage": r["background_area_percentage"],
            "severity": r["severity"],
            "severity_description": r["severity_description"],
            "metrics": {
                "pothole_area_percentage": r["pothole_area_percentage"],
                "crack_area_percentage": r["crack_area_percentage"],
                "total_damaged_area_percentage": r["total_damaged_area_percentage"],
                "background_area_percentage": r["background_area_percentage"],
                "pixels": {
                    "pothole": r["pothole_pixels"],
                    "crack": r["crack_pixels"],
                    "background": r["background_pixels"],
                    "total": r["total_pixels"]
                }
            },
            "bounding_boxes": bboxes,
            "boxes": bboxes,
            "damage_statistics": dmg_stats,
            "original_image": r["original_image"],
            "overlay_image": r["overlay_image"],
            "mask_image": r["mask_image"],
            "bbox_image": bbox_img,
            "images": {
                "original": r["original_image"],
                "overlay": r["overlay_image"],
                "mask": r["mask_image"],
                "bbox": bbox_img
            },
            "notes": r["notes"]
        }


def delete_inspection(insp_id: str) -> bool:
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM inspections WHERE id = ?", (insp_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_analytics_summary() -> Dict[str, Any]:
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_inspections,
                SUM(CASE WHEN pothole_pixels > 0 THEN 1 ELSE 0 END) as inspections_with_potholes,
                SUM(CASE WHEN crack_pixels > 0 THEN 1 ELSE 0 END) as inspections_with_cracks,
                AVG(total_damaged_area_percentage) as avg_damage_pct,
                AVG(pothole_area_percentage) as avg_pothole_pct,
                AVG(crack_area_percentage) as avg_crack_pct,
                MAX(total_damaged_area_percentage) as max_damage_pct,
                SUM(pothole_pixels) as total_pothole_pixels,
                SUM(crack_pixels) as total_crack_pixels,
                SUM(total_pixels) as grand_total_pixels
            FROM inspections
        """)
        summary_row = cursor.fetchone()

        total_insp = summary_row["total_inspections"] or 0
        if total_insp == 0:
            return {
                "total_inspections": 0,
                "inspections_with_potholes": 0,
                "inspections_with_cracks": 0,
                "average_damage_percentage": 0.0,
                "average_pothole_percentage": 0.0,
                "average_crack_percentage": 0.0,
                "max_damage_percentage": 0.0,
                "total_pothole_pixels": 0,
                "total_crack_pixels": 0,
                "pothole_vs_crack_ratio": {"pothole": 0, "crack": 0},
                "severity_counts": {"Good": 0, "Low": 0, "Moderate": 0, "High / Critical": 0},
                "recent_inspections": [],
                "timeline": []
            }

        # Severity breakdown
        sev_cursor = conn.execute("""
            SELECT severity, COUNT(*) as count 
            FROM inspections 
            GROUP BY severity
        """)
        sev_counts = {"Good": 0, "Low": 0, "Moderate": 0, "High / Critical": 0}
        for s in sev_cursor.fetchall():
            sev_counts[s["severity"]] = s["count"]

        # Timeline (grouped by day)
        time_cursor = conn.execute("""
            SELECT SUBSTR(created_at, 1, 10) as day, 
                   COUNT(*) as count, 
                   AVG(total_damaged_area_percentage) as avg_damage
            FROM inspections
            GROUP BY day
            ORDER BY day ASC
        """)
        timeline = [
            {"date": t["day"], "count": t["count"], "avg_damage": round(t["avg_damage"], 2)}
            for t in time_cursor.fetchall()
        ]

        # Recent records
        recent_cursor = conn.execute("""
            SELECT id, title, filename, created_at, total_damaged_area_percentage, pothole_area_percentage, crack_area_percentage, severity
            FROM inspections
            ORDER BY created_at DESC
            LIMIT 5
        """)
        recent = [dict(r) for r in recent_cursor.fetchall()]

        tot_pothole_px = summary_row["total_pothole_pixels"] or 0
        tot_crack_px = summary_row["total_crack_pixels"] or 0

        return {
            "total_inspections": total_insp,
            "inspections_with_potholes": summary_row["inspections_with_potholes"] or 0,
            "inspections_with_cracks": summary_row["inspections_with_cracks"] or 0,
            "average_damage_percentage": round(summary_row["avg_damage_pct"] or 0.0, 2),
            "average_pothole_percentage": round(summary_row["avg_pothole_pct"] or 0.0, 2),
            "average_crack_percentage": round(summary_row["avg_crack_pct"] or 0.0, 2),
            "max_damage_percentage": round(summary_row["max_damage_pct"] or 0.0, 2),
            "total_pothole_pixels": tot_pothole_px,
            "total_crack_pixels": tot_crack_px,
            "pothole_vs_crack_ratio": {
                "pothole": tot_pothole_px,
                "crack": tot_crack_px
            },
            "severity_counts": sev_counts,
            "recent_inspections": recent,
            "timeline": timeline
        }
