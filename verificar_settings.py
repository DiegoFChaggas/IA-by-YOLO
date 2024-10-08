#Codigo de refência: https://docs.ultralytics.com/pt/quickstart/#use-ultralytics-with-python
from ultralytics import settings

# View all settings
print(settings)

# Return a specific setting
value = settings["runs_dir"]