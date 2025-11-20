# upload_model_fixed.py
from huggingface_hub import HfApi, create_repo
import os

def upload_model_to_hf():
    # Use your actual username
    HF_USERNAME = "ImranAliNaeem"  # ✅ Your actual username
    MODEL_NAME = "bert-sentiment-analysis"
    REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"
    
    print("🚀 Uploading model to Hugging Face Hub...")
    print(f"Repository: {REPO_ID}")
    
    # Create repository
    try:
        create_repo(REPO_ID, repo_type="model", private=False)
        print(f"✅ Created repository: {REPO_ID}")
    except Exception as e:
        print(f"ℹ️ Repository might already exist: {e}")
        print("Continuing with upload...")
    
    # Initialize API
    api = HfApi()
    
    # Upload entire sentiment_model folder
    print("📤 Uploading model files...")
    
    # Check if sentiment_model folder exists
    if not os.path.exists("./sentiment_model"):
        print("❌ sentiment_model folder not found!")
        print("Current directory contents:")
        for item in os.listdir("."):
            print(f"  - {item}")
        return
    
    # List files to be uploaded
    print("Files to upload:")
    for root, dirs, files in os.walk("./sentiment_model"):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"  - {file} ({file_size} bytes)")
    
    # Upload the folder
    api.upload_folder(
        folder_path="./sentiment_model",
        repo_id=REPO_ID,
        repo_type="model"
    )
    
    print(f"✅ Model successfully uploaded to: https://huggingface.co/{REPO_ID}")
    print("🎉 Your model is now publicly available!")

if __name__ == "__main__":
    upload_model_to_hf()
