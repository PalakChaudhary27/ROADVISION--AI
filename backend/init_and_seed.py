from database import init_db, get_inspections, get_analytics_summary
import main

print("Initializing DB...")
main.init_db()
print("Loading model...")
main.model_manager.load()
print("Seeding initial real inspections...")
main.seed_initial_inspections()

insps = get_inspections()
print(f"Total seeded inspections: {len(insps)}")
for i in insps:
    print(f" - {i['id']}: {i['title']} (Damage: {i['total_damaged_area_percentage']}%)")

analytics = get_analytics_summary()
print("Analytics Summary:")
print(f"  Inspections: {analytics['total_inspections']}")
print(f"  Avg Damage: {analytics['average_damage_percentage']}%")
print(f"  Pothole px: {analytics['total_pothole_pixels']}, Crack px: {analytics['total_crack_pixels']}")
print("Initialization and real data seeding complete!")
