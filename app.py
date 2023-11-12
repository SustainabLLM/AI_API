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
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

tokenizer_sum = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-3")
model_sum = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-xsum-12-3")

def summarizer(text):
  inputs_sum = tokenizer_sum(text, 
                    max_length=1024, 
                    truncation=True,
                    return_tensors="pt")
    
  summary_ids = model_sum.generate(inputs_sum["input_ids"])
  summary = tokenizer_sum.batch_decode(summary_ids, 
                                  skip_special_tokens=True, 
                                  clean_up_tokenization_spaces=False)
  plot = list(summary[0].split('.'))
  return plot

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
