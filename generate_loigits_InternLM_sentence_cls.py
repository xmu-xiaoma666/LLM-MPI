from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

class SentenceGenerator:
    def __init__(self, model_name="internlm/internlm2_5-7b-chat"):
        # 加载 tokenizer 和模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            exit()

        # 检查 CUDA 是否可用并选择设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 设置模型为评估模式
        self.model.eval()

    def generate_response(self, user_question):
        system_prompt = "You are InternLM2-Chat, a harmless AI assistant."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ]

        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        tokenized_chat = tokenized_chat.to(self.device)

        generated_ids = self.model.generate(
            tokenized_chat,
            max_new_tokens=1024,
            temperature=1,
            repetition_penalty=1.005,
            top_k=40,
            top_p=0.8,
            do_sample=True
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description='问题生成器')
    parser.add_argument('--user_question', type=str, default="9.11和9.9哪个大？", help='输入的问题')

    # 解析命令行参数
    args = parser.parse_args()


    # user_question = "9.11和9.9哪个大？"
    args.user_question = "如果一个人从房子里走出来，他是在室内还是在室外？"
    args.user_question = "一张纸的厚度是0.1毫米，折叠10次后，纸的厚度是多少？（假设每次折叠后厚度翻倍）"
    args.user_question = "一个长方形的周长为24厘米，宽是长的二分之一。这个长方形的长和宽分别是多少？"
    args.user_question = "在一次考试中，学生们的平均分是80分。如果有一名新学生的分数是100分，新的平均分是什么？（假设总共有n名学生）"

    sentence_generator = SentenceGenerator()
    
    response = sentence_generator.generate_response(args.user_question)
    print('-' * 100)
    print(response)
