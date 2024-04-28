import sys
sys.path.append("./Llava1.5/LLaVA")
# Detailed model can be viewed at https://github.com/haotian-liu/LLaVA
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "./pretrained_model/llava-v1.5-7b" # "liuhaotian/llava-v1.5-7b"
prompt = "Please describe the people in the image, including their gender, age, clothing, facial expressions, and any other distinguishing features."
image_file = "./demo.png"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    # load_4bit=True
) # device="cuda"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

outputs = eval_model(args, tokenizer, model, image_processor)
print(f"The caption is: {outputs}")










