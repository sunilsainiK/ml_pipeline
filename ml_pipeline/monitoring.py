import os
import time
import requests

def check_for_new_data(data_directory: str, last_checked_time: float) -> bool:
    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        if os.path.isfile(filepath):
            file_mod_time = os.path.getmtime(filepath)
            if file_mod_time > last_checked_time:
                return True
    return False

def trigger_training_pipeline(project_id: str, token: str):
    url = f"https://gitlab.com/api/v4/projects/{project_id}/trigger/pipeline"
    data = {
        "ref": "master",  # The branch you want to trigger
        "token": token    # Your GitLab trigger token
    }
    response = requests.post(url, data=data)
    return response.json()

if __name__ == "__main__":
    data_directory = "path/to/data"
    last_checked_time = time.time()  # Start time

    while True:
        time.sleep(240)  # Check every 60 seconds
        if check_for_new_data(data_directory, last_checked_time):
            print("New data detected. Triggering the training pipeline.")
            trigger_training_pipeline("your_project_id", "your_trigger_token")
            last_checked_time = time.time()  # Update the last checked time
