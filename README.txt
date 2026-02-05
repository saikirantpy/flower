▶️ Steps to Execute the Flower Detection Project
Follow these steps in order to run the project successfully on any system.
1️⃣ Prerequisites
Make sure the system has:
Python 3.10+
pip (Python package manager)
Internet connection (for downloading dataset & model weights)
Check Python version:
python --version
2️⃣ Clone or Download the Project
If using Git:
git clone <repository-url>
cd Flower
Or:
Download ZIP
Extract it
Open terminal inside the Flower folder
3️⃣ Create a Virtual Environment (Recommended)
macOS / Linux
python3 -m venv flower_env
source flower_env/bin/activate
Windows
python -m venv flower_env
flower_env\Scripts\activate
After activation, you should see:
(flower_env)
4️⃣ Install Required Libraries
Install all dependencies:
pip install tensorflow streamlit pillow numpy matplotlib
(Optional but recommended)
pip install --upgrade pip
5️⃣ Download Flower Dataset
Run the following commands inside the project folder:
curl -L -o flowers.tgz https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
tar -xvzf flowers.tgz
mv flower_photos flowers
Rename folders:
mv flowers/roses flowers/rose
mv flowers/sunflowers flowers/sunflower
mv flowers/tulips flowers/tulip
Final dataset structure:
flowers/
 ├── daisy/
 ├── dandelion/
 ├── rose/
 ├── sunflower/
 └── tulip/
6️⃣ Train the Model
Run the training script:
python train.py
This will:
Train the EfficientNet model
Achieve ~90%+ validation accuracy
Save the trained model as:
flower_model/
⏳ Training time: ~5–10 minutes (depends on system)
7️⃣ Run the Streamlit Web App
Start the application:
streamlit run app.py
A browser window will open automatically.
8️⃣ Use the Application
Upload a flower image (.jpg, .png)
The app will display:
Uploaded image
Top 3 predicted flower classes
Confidence scores
Final predicted flower is shown clearly 🌸
9️⃣ Stop the Application
To stop Streamlit:
Ctrl + C
📂 Project Structure
Flower/
 ├── app.py            # Streamlit web app
 ├── train.py          # Model training script
 ├── flower_model/     # Saved trained model
 ├── flowers/          # Dataset
 ├── flower_env/       # Virtual environment
 └── README.md
⚠️ Common Issues & Fixes
❌ Wrong predictions
✔ Ensure:
train.py is run before app.py
EfficientNet preprocessing is used in app.py
❌ Module not found
✔ Activate virtual environment:
source flower_env/bin/activate
❌ Streamlit not opening
✔ Run:
streamlit run app.py
✅ Final Notes
This project uses transfer learning (EfficientNetB0)
Deployed as a real-time ML web app
Suitable for:
College mini / major project
GitHub portfolio
ML demonstrations