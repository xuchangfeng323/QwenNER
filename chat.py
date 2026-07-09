from model import Qwen4NER
from transformers import AutoTokenizer
from utils import Arguments
from template import template_dict
class Chat:
    def __init__(self, config: Arguments):
        self.config = config
        self.tokenizer = config.tokenizer
        self.model_config = Qwen4NER(config)
        self.model = self.model_config.get_model()
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
        generated_ids = self.model.generate(
        inputs.input_ids,
        max_new_tokens=config.max_new_tokens,
        use_cache=True,
    )
        output=generated_ids[len(inputs.input_ids):]
        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return response
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg', type=str, default='./args/arg1.json')
    parser.add_argument('--text', type=str, default='你好')
    args = parser.parse_args()
    args=Arguments(args.arg)
    chat=Chat(args)
    response=chat.predict(args.text)
    print(response)
        
        
        
        

    

        
        