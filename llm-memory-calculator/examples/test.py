from llm_memory_calculator import calculate_memory
result = calculate_memory("meta-llama/Llama-4-Scout-17B-16E-Instruct", seq_length=4096, batch_size=1, precision='fp16')
print(result)