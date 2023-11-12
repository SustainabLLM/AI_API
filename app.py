from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from codecarbon import EmissionsTracker
import scrapper
import os
import nltk
#import summarization as sumlib
tracker = EmissionsTracker()

nltk.download("punkt")
nltk.download("stopwords")

# Initialize FastAPI app
app = FastAPI()

torch.set_default_device("cuda")

def load_distilbart_model():
    """
    Loads and returns the tokenizer and model from the 'sshleifer/distilbart-cnn-6-6' pretrained model.

    This function is specifically designed for loading the DistilBART model which is a distilled version
    of the BART model fine-tuned for summarization tasks.

    Returns:
        tokenizer (AutoTokenizer): The tokenizer for the 'sshleifer/distilbart-cnn-6-6' model.
        model (AutoModelForSeq2SeqLM): The sequence-to-sequence language model.
    """
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6")
    return tokenizer, model

def summarizer(model, tokenizer, prompt):
    """
    Generates a summary for a given text prompt using the specified model and tokenizer.

    This function takes a text prompt, tokenizes it using the provided tokenizer, and then
    generates a summary using the specified model. The function is designed to work with
    models that are suitable for summarization tasks.

    Parameters:
        model (Any): A pre-trained model from the Hugging Face library which is capable of sequence-to-sequence tasks.
        tokenizer (Any): The tokenizer corresponding to the model, used for converting the prompt text into a format suitable for the model.
        prompt (str): The text to be summarized.

    Returns:
        str: The generated summary of the input text.
    
    Example:
        model, tokenizer = load_distilbart_model()
        text = "Your long text to summarize goes here."
        summary = summarize(model, tokenizer, text)
        print(summary)

    Note:
        This function assumes that the 'model' and 'tokenizer' provided are compatible and 
        properly configured for summarization tasks. It does not perform error checking for 
        model and tokenizer compatibility.
    """
    inputs = tokenizer(prompt, max_length=2048, return_tensors="pt")
    
    summary_ids = model.generate(inputs["input_ids"], min_length=125, max_length=300)

    return tokenizer.batch_decode(summary_ids)[0]
  
# Define a request model
class PromptRequest(BaseModel):
    prompt: str

# Endpoint to process prompts
@app.post("/generate-text/")
async def generate_text(request: PromptRequest, max_length=500, temperature=1.0, top_k=50):
    try:
        # Tokenize the prompt
        tracker.start()
        inputs = tokenizer(request.prompt, return_tensors="pt", return_attention_mask=False)

               # Generate text
        outputs = model.generate(
            **inputs, 
            max_length=max_length, 
            temperature=temperature, 
            top_k=top_k, 
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=400
        )
        
        generated_text = tokenizer.batch_decode(outputs)[0]
        tracker.stop()
        return {"generated_text": generated_text}
    except Exception as e:
        tracker.stop()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/summarize/")
async def summarize(request: PromptRequest):
    try:
        tracker.start()
        generated_text = summarizer(request.prompt)
        tracker.stop()
        return {"generated_text": generated_text}
    except Exception as e:
        tracker.stop()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/carbon/", response_class=FileResponse)
async def get_carbon():
    file_path = "emissions.csv"
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename="emissions.csv", media_type='text/csv')
    else:
        return {"error": "File not found"}
    
@app.post("/scrape/")
async def scrape(request: PromptRequest):
    try:
        tracker.start()
        information = scrapper.inference(request.prompt)
        tracker.stop()
        return {"scraped_info": information}
    except Exception as e:
        tracker.stop()
        raise HTTPException(status_code=500, detail=str(e))
