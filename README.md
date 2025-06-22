# 🧠 Handwritten Digit Generator Web App

This project allows users to generate **realistic-looking handwritten digits (0–9)** using a custom-trained PyTorch model on the **MNIST dataset**. The model is exposed via a simple and interactive **Streamlit web app**.

## 🚀 Features

- 🔢 Generate handwritten digits 0 through 9.
- 🎲 Randomized generation of 5 different images per digit.
- 🖥️ Clean web interface using Streamlit.
- 🧠 Generator model trained from scratch using PyTorch (no pre-trained weights).
- 🎓 MNIST dataset used for training (28x28 grayscale digits).
- ☁️ Easy deployment to Streamlit Cloud for public access.

---

## 🏗️ Project Structure

```

Handwritten-Computer-Vision/
│
├── app.py                 # Streamlit app
├── model.py               # PyTorch model architecture (DigitGenerator)
├── models/
│   └── digit\_generator.pth   # Trained model weights
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

````

---

## 📦 Installation & Setup (Locally)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Giwa-ibrahim/Handwritten_generator.git
   cd Handwritten_generator
   ````

2. **Install dependencies using pip or uv:**

   ```bash
   pip install -r requirements.txt
   # or
   uv pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deployed Web App

👉 [Access the live app here](https://handwritten-gen.streamlit.app/)
*(Note: Link will stay active for at least 2 weeks as per requirement)*

---

## 🧠 Model Training

* Trained from scratch on Google Colab using a **T4 GPU**, compliant with contest rules.
* Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/)
* Training code is available in `train_mnist_generator.py` (not shown here but included in submission).

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Streamlit
* torchvision
* matplotlib

---

## 📄 License

MIT License – feel free to use, modify, and distribute this project with attribution.

---

