import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Page Configuration
st.set_page_config(
    page_title="English to Itsekiri Translator",
    page_icon="🗣️",
    layout="centered"
)

# 2. Load Model & Tokenizer (Cached)
# @st.cache_resource was used so the model loads only once when the app starts,
# not every time a user clicks the button.
@st.cache_resource
def load_model_pipeline():
    model_path = "Beecroft/trans-work"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

# Load the model (this shows a spinner while loading)
with st.spinner('Loading Model... Please wait...'):
    tokenizer, model = load_model_pipeline()

# 3. The Translation Function
def translate(sentence):
    
    prefix = "translate English to Itsekiri: " 
    
    input_text = prefix + sentence
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    
    # Generate translation
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. User Interface
st.title("🗣️ English to Itsekiri Translator")
st.markdown("### Neural Machine Translation System")
st.write("This tool uses an AI model fine-tuned to translate English text into the Itsekiri language.")

# Text Input Area
text_input = st.text_area("Enter English text:", height=100, placeholder="e.g., Hello, how are you today?")

# Translate Button
if st.button("Translate", type="primary"):
    if text_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Translating..."):
            try:
                translation = translate(text_input)
                st.success("Translation:")
                st.markdown(f"#### {translation}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer
st.markdown("---") 
st.caption("Powered by AfriTeVa & Hugging Face | Developed by Beecroft")