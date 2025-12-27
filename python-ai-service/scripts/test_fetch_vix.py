import sys
import os
this_dir = os.path.dirname(__file__)
# ensure the local `python-ai-service` folder is on sys.path so `data` imports work
python_ai_service_dir = os.path.abspath(os.path.join(this_dir, '..'))
if python_ai_service_dir not in sys.path:
    sys.path.insert(0, python_ai_service_dir)

from data.volatility_features import fetch_vix_data
import datetime as dt

start = dt.date.today() - dt.timedelta(days=30)
end = dt.date.today()
try:
    df = fetch_vix_data(start, end)
    print("Fetched VIX rows:", len(df))
    print(df.head().to_string())
except Exception as e:
    print("ERROR:", e)
