# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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
    
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=2000, max_length=3000)

    return tokenizer.batch_decode(summary_ids)[0]