import hopsworks
import pandas as pd

def upload_existing_csv_to_hopsworks(csv_path="testing_2.csv"):
    print("🚀 Connecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Load CSV
    print(f"📂 Loading CSV from {csv_path} ...")
    df_all = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df_all)} records.")

    # Convert datetime if needed
    if 'datetime' in df_all.columns:
        df_all['datetime'] = pd.to_datetime(df_all['datetime'])

    # Try creating new feature group
    print("🆕 Creating new feature group 'testing_2' ...")
    try:
        feature_group = fs.create_feature_group(
            name="testing_2",
            version=1,
            description="Hourly weather + AQI data for Karachi",
            primary_key=["datetime"],
            event_time="datetime",
            online_enabled=False
        )
        print("✅ Feature group 'testing_2' created successfully.")
    except Exception as e:
        print(f"⚠️ Feature group creation failed: {e}")
        print("Trying to get existing feature group instead...")
        feature_group = fs.get_feature_group("testing_2", version=1)

    # Double-check
    if feature_group is None:
        print("❌ Could not create or fetch feature group. Aborting.")
        return

    print("🚀 Inserting all records into Hopsworks...")
    feature_group.insert(df_all)
    print("✅ Full CSV uploaded successfully to feature group 'testing_2'!")

if __name__ == "__main__":
    upload_existing_csv_to_hopsworks("testing_2.csv")
