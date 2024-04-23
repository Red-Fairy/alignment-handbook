from transformers import AutoTokenizer
import torch

# tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
print(len(tokenizer))

ids = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
tokens = tokenizer.convert_ids_to_tokens(ids)
print(' '.join(tokens))

print(tokenizer.unk_token, tokenizer.unk_token_id,
      tokenizer.pad_token, tokenizer.pad_token_id,
      tokenizer.eos_token, tokenizer.eos_token_id,
      tokenizer.bos_token, tokenizer.bos_token_id,)

visual_tokens_to_add = ['<v' + str(i) + '>' for i in range(0, 2048)]
num_added_visual_tokens = tokenizer.add_special_tokens({'additional_special_tokens': visual_tokens_to_add})
print(tokenizer.convert_tokens_to_ids(['<v1>']))

action_tokens_to_add = ['<a' + str(i) + '>' for i in range(0, 256)]
num_added_action_tokens = tokenizer.add_special_tokens({'additional_special_tokens': action_tokens_to_add})


special_tokens = ['<bot_i>', '<eot_i>', '<bov_i>', '<eov_i>', '<boa_i>', '<eoa_i>', 
                        '<bov_o>', '<eov_o>', '<boa_o>', '<eoa_o>']

num_added_special_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
# print('map', tokenizer.special_tokens_map['additional_special_tokens']['<boa_i>'])

print("We have added", num_added_special_tokens, "tokens")

# text = "test"
# tokens = tokenizer(text, padding="max_length", truncation=True, max_length=32, return_tensors="pt")

# print(tokens)

# ids = '34314 34314 34314 34314 34314 34314 34314 34314 34314 34314 34314 34314 34314 34314 34314 34314 1 34304 12018 264 3031 28722 1807 5498 302 26020 298 264 4498 271 693 15702 11971 304 22403 368 298 7900 28718 34305 34306 33256 33328 33549 33166 33438 33536 34307 34308 34164 34101 34165 34050 34191 34262 34309 34310 33976 32970 33035 33353 32735 32783 34311 34312 34247 34149 34251 34199 34203 34164 34313 523 28706 385 28767'
# ids = [int(i) for i in ids.split()]
# tokens = tokenizer.convert_ids_to_tokens(ids)
# print(tokens)

text_with_new_tokens = "How are you?"
# text_with_new_tokens = "<bov_i>This is a sample text.<bov_i><v1><v2><eoa_i><v3>"
tokens_with_new_tokens = tokenizer(text_with_new_tokens)

print(tokens_with_new_tokens)