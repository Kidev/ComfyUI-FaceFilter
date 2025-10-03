from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
LOCAL_MODELS = THIS_DIR / "models" / "insightface"
HOME_MODELS = Path.home() / ".insightface" / "models"

def ensure_dirs():
    LOCAL_MODELS.mkdir(parents=True, exist_ok=True)
    HOME_MODELS.mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()
    print("InsightFace model locations ready")
    print(f"Local packs folder: {LOCAL_MODELS}")
    print(f"Home cache folder:  {HOME_MODELS}")
    print("Drop a pack folder (example antelopev2) into either path")
    print("Nothing else to install here. Use requirements.txt for Python deps.")

if __name__ == "__main__":
    main()
