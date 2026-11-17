import httpx
import time

client = httpx.Client(timeout=10.0)

for attempt in range(1, 15):
    try:
        r_root = client.get('http://127.0.0.1:8000/')
        print(f"Attempt {attempt}: Root HTML status = {r_root.status_code}, Length = {len(r_root.text)} bytes")
        assert '<div id="root"></div>' in r_root.text, "Root HTML missing root div"
        
        r_health = client.get('http://127.0.0.1:8000/health')
        health = r_health.json()
        print(f"Health status = {r_health.status_code}, model_loaded = {health['model_loaded']}")
        
        r_ana = client.get('http://127.0.0.1:8000/api/analytics')
        ana = r_ana.json()
        print(f"Analytics status = {r_ana.status_code}, total_inspections = {ana['total_inspections']}, avg_damage = {ana['average_damage_percentage']}%")
        
        r_insp = client.get('http://127.0.0.1:8000/api/inspections')
        insps = r_insp.json()
        print(f"Inspections status = {r_insp.status_code}, count = {len(insps)}")
        
        print("\n========================================================")
        print(" [ALL TESTS PASSED] RoadVision AI Unified Server Live!")
        print(" Frontend: http://localhost:8000")
        print(" Backend API: http://localhost:8000/health, http://localhost:8000/predict")
        print("========================================================")
        break
    except Exception as e:
        print(f"Waiting for server... ({e})")
        time.sleep(1)
