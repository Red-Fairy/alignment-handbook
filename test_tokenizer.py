from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')

print(len(tokenizer))

print(tokenizer.unk_token, tokenizer.unk_token_id,
      tokenizer.pad_token, tokenizer.pad_token_id,
      tokenizer.eos_token, tokenizer.eos_token_id,
      tokenizer.bos_token, tokenizer.bos_token_id,)

num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
print("We have added", num_added_toks, "tokens")

text = "test"
tokens = tokenizer(text, padding="max_length", truncation=True, max_length=32, return_tensors="pt")

print(tokens)

text_with_new_tokens = "This is a sample text. new_tok1 my_new-tok2 <unk>"
tokens_with_new_tokens = tokenizer(text_with_new_tokens)

print(tokens_with_new_tokens)