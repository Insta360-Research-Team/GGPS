
import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tfevents(event_file, output_csv):
    print(f"Extracting data from {event_file}...")
    
    # Initialize EventAccumulator with a high size_guidance to load all data
    event_acc = EventAccumulator(event_file, size_guidance={'scalars': 0})
    event_acc.Reload()
    
    tags = event_acc.Tags()['scalars']
    if not tags:
        print("No scalar tags found.")
        return

    all_data = []
    
    for tag in tags:
        events = event_acc.Scalars(tag)
        for event in events:
            all_data.append({
                'wall_time': event.wall_time,
                'step': event.step,
                'tag': tag,
                'value': event.value
            })
    
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved to {output_csv}")

if __name__ == "__main__":
    event_path = "output/<run>/lightning_logs/version_0/events.out.tfevents.<...>"
    output_path = "output/<run>/extracted_metrics.csv"
    extract_tfevents(event_path, output_path)
