import torch
import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig, pipeline

# GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Vision model.
image_model_name = "unclecode/folrence-2-large-4bit"
image_processor_model_name = "microsoft/Florence-2-large"
config = AutoConfig.from_pretrained(image_model_name, trust_remote_code=True)
config.vision_config.model_type = 'davit'
image_model = AutoModelForCausalLM.from_pretrained(image_model_name, trust_remote_code=True,  config=config)
image_processor = AutoProcessor.from_pretrained(image_processor_model_name, trust_remote_code=True)

# Instruct model.
instruct_model_id = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
instruct_pipe = pipeline("text-generation", model=instruct_model_id, torch_dtype=torch.bfloat16)

# Visual question answerer - responding to a prompt on an image.
class VQA:
    
    # Initialize with the default instruct prompt.
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt   


    # Gets a response for the image URL and optional prompt (uses system_prompt if not overriden here).
    def analyze(self, image_url, prompt=None):
        # Use visual model to get caption..
        caption = self.__get_detailed_caption(image_url)
        
        # Analyze image using the instruct prompt (default to system_prompt).
        outcome = self.__instruct_on_caption(caption, prompt or self.system_prompt)

        # Return result.
        return outcome

    # PRIVATE: Gets a caption for the image URL.
    def __get_detailed_caption(self, image_url):
        # Open image from URL.
        image = Image.open(requests.get(image_url, stream=True).raw) 


        # Process through model to get caption.
        image_model_prompt = "<MORE_DETAILED_CAPTION>"
        inputs = image_processor(text=image_model_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = image_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=100,
            num_beams=3
        )

        generated_text = image_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = image_processor.post_process_generation(generated_text, task=image_model_prompt, image_size=(image.width, image.height))

        return parsed_answer

    # PRIVATE: Gets a short sub-label from the caption.
    def __instruct_on_caption(self, caption, prompt): 
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": caption}
        ]
        
        outputs = instruct_pipe(
            messages,
            max_new_tokens=256
        )

        content = outputs[0]['generated_text'][2]['content']

        return content
