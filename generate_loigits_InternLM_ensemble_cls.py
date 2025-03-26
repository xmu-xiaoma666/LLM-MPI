from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

class QuestionRefiner:
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
        self.model.eval()

    def choose_template(self, user_question, num_variants):
        # 判断问题的语言（中文或英文）
        if any('\u4e00' <= char <= '\u9fff' for char in user_question):  # 检测是否包含中文字符
            template = """
            您是一位优化和精炼问题的专家，擅长提升问题的清晰度和精准性。  
            根据用户提出的原始问题：{user_question}，生成 {num_variants} 个不同但清晰且结构良好的问题版本。  
            每个版本应准确传达用户的潜在意图，同时避免歧义。确保各个版本在措辞和结构上具有多样性。  
            将 {num_variants} 个问题以 {num_variants} 行形式返回，每个问题占一行，不添加额外的解释或编号。
            """
        else:  # 默认为英文
            template = """
            You are an expert at refining and optimizing questions for clarity and precision. 
            Given the user's original question: {user_question}, generate {num_variants} distinct, clear, and well-structured version/versions of the question. 
            Each version should effectively capture the user's underlying intent while avoiding ambiguity. Ensure diversity in phrasing and structure across the variants. 
            Return the {num_variants} question/questions in {num_variants} line/lines, one question per line, without additional explanations or numbering.
            """
        return template.format(user_question=user_question, num_variants=num_variants)

    def generate_variants(self, user_question, num_variants):
        variant_generation_prompt = self.choose_template(user_question, num_variants)

        messages = [
            {"role": "system", "content": "You are InternLM2-Chat, a harmless AI assistant."},
            {"role": "user", "content": variant_generation_prompt},
        ]

        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_ids = tokenized_chat.to(self.device)

        generated_ids = self.model.generate(input_ids, max_new_tokens=1024, temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.9, do_sample=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)]
        variant = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if('\n' in variant):
            variants = variant.strip().split('\n')  # 去除多余空格或换行符
        else:
            variants = [variant.strip()]
        
        return variants

    def sample_top_k(self, probs, k=50):
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        sampled_index = torch.multinomial(top_k_probs, num_samples=1)
        return top_k_indices.gather(-1, sampled_index)

    def generate_with_variants(self, variants, max_new_tokens):
        print("Original Questions:")
        print(variants)
        
        # 初始化每个变体的输入
        variant_inputs = [self.tokenizer.encode(v, return_tensors="pt").to(self.device) for v in variants]
        initial_length = variant_inputs[0].shape[1]

        for _ in range(max_new_tokens):
            variant_inputs_decode = [self.tokenizer.decode(ids[0], skip_special_tokens=True) for ids in variant_inputs]

            # print("Current Variants:")
            # for i, decoded_variant in enumerate(variant_inputs_decode):
            #     print(f"{i + 1}. {decoded_variant}")

            all_logits = []
            
            # 对每个变体进行前向传播，获取 logits
            for input_ids in variant_inputs:
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits[:, -1, :]
                    all_logits.append(logits)

            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            next_token_id = torch.argmax(avg_logits, dim=-1, keepdim=True)

            # probs = torch.softmax(avg_logits, dim=-1)
            # next_token_id = self.sample_top_k(probs, k=10)  # 使用 top-k 采样

            token = self.tokenizer.decode(next_token_id[0])
            print(token)
            variant_inputs = [torch.cat([ids, next_token_id], dim=-1) for ids in variant_inputs]

            if(token=='<|im_end|>'):
                break

        generated_ids = variant_inputs[0][:, initial_length:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_response(self, user_question, num_variants=5, max_new_tokens=1024):
        variants = self.generate_variants(user_question, num_variants)
        response = self.generate_with_variants(variants, max_new_tokens)
        return response


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description='问题细化器')
    parser.add_argument('--user_question', type=str, default='如果一个人从房子里走出来，他是在室内还是在室外？')
    parser.add_argument('--num_variants', type=int, default=5, help='生成的变体数量')

    # 解析命令行参数
    args = parser.parse_args()

    # args.user_question = "9.11和9.9，哪个大？"
    args.user_question = "如果一个人从房子里走出来，他是在室内还是在室外？"
    args.user_question = "一张纸的厚度是0.1毫米，折叠10次后，纸的厚度是多少？（假设每次折叠后厚度翻倍）"
    args.user_question = "一个长方形的周长为24厘米，宽是长的二分之一。这个长方形的长和宽分别是多少？"
    args.user_question = "在一次考试中，学生们的平均分是80分。如果有一名新学生的分数是100分，新的平均分是什么？（假设总共有n名学生）"

    question_refiner = QuestionRefiner()
    response = question_refiner.generate_response(args.user_question, num_variants=args.num_variants)
    
    print('-' * 100)
    print(response)


