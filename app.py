import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from models.digit_gen import DigitGenerator  # Custom model file

# Constants
MODEL_PATH = "models/digit_generator.pth"
LATENT_DIM = 100

# Load the model
@st.cache_resource
def load_model():
    model = DigitGenerator(latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

generator = load_model()

# Streamlit UI
st.title("🧠 Handwritten Digit Generator (0–9)")
digit = st.selectbox("Select a digit to generate:", list(range(10)))

if st.button("Generate 5 Images"):
    noise = torch.randn(5, LATENT_DIM)
    labels = torch.full((5,), digit, dtype=torch.long)
    with torch.no_grad():
        generated = generator(noise, labels)
    
    grid = make_grid(generated, nrow=5, normalize=True)

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    ax.axis('off')
    st.pyplot(fig)
