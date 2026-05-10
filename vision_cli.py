import sys
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    image_path = sys.argv[1]
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    try:
        # Caricamento Modello e Processor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float32, 
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained(model_id)

        # Preparazione dell'immagine
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Descrivi brevemente cosa vedi in questa immagine."},
                ],
            }
        ]

        # Processamento input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cpu")

        # Generazione
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        # Pulizia e stampa dell'output (solo il testo generato)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print(output_text[0])

    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()
