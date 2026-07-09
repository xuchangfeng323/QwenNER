from model import Qwen4NER
from transformers import AutoTokenizer
from utils import Arguments
import argparse
from template import template_dict
class Predictor:
    def __init__(self, config: Arguments, weight: str):
        self.config = config
        self.tokenizer = config.tokenizer
        self.model_config = Qwen4NER(config)
        self.model = self.model_config.load_adapter(weight)
        self.model = self.model.to(config.device)
        self.model.eval()
        self.template = template_dict[config.template_name]
        
    def build_prompt(self, text: str):
        messages = [
        {"role": "system", "content": f"{self.config.prompt}"},
        {"role": "user", "content": f"{text}"}
    ]
        return messages
    def predict(self, text: str):
        messages = self.build_prompt(text)
        inputs=self.tokenizer.apply_chat_template(messages,add_generation_prompt=True, tokenize=False)
        inputs=self.tokenizer(inputs, return_tensors="pt")
        inputs=inputs.to(self.config.device)
        generated_ids = self.model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=self.config.max_new_tokens,
        use_cache=True,
    )
        output = generated_ids[:, inputs.input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return response
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg', type=str, default='./args/arg1.json')
    parser.add_argument('--weight', type=str, default='../autodl-tmp/checkpoint/exp1/')
    parser.add_argument('--text', type=str, default='Using the same approach we have shown that hFIRE binds the stimulatory proteins Sp1 and Sp3 in addition to CBF')
    # ground truth: hFIRE(GENE), Sp1(GENE), Sp3(GENE), CBF(GENE)
    args = parser.parse_args()
    config=Arguments(args.arg)
    weight=args.weight
    predictor=Predictor(config, weight)
    response=predictor.predict(args.text)
    print(response)
        
        
        
        

    

        
        