"""Story Bot: llama_cpp

https://llama-cpp-python.readthedocs.io/en/latest/
https://huggingface.co/TheBloke/OpenOrca-Platypus2-13B-GGUF/resolve/main/openorca-platypus2-13b.Q4_K_M.gguf
"""
from llama_cpp import Llama


def generate_story(user_prompt: str):
    system_prompt = "You are a masterful storyteller. You will be given a " \
                    "prompt, please respond with a short story based on the " \
                    "prompt. Do not discuss the story or explain anything. " \
                    "Your response should be the story text by itself in " \
                    "simple markdown."
    prompt = f"### Instruction: {system_prompt}\n\n" \
             f"{user_prompt}\n\n" \
             f"### Response:\n"
    llm = Llama(
        model_path="./models/openorca-platypus2-13b.Q4_K_M.gguf",
        n_ctx=4096,
    )
    raw_output = llm(
        prompt,
        stop=["###"],
        max_tokens=-1,
        # temperature=1.0,
        # top_k=100,
        # top_p=1.0,
        # presence_penalty=0.0,
        # frequency_penalty=0.0,
        # echo=False,
    )
    reply = raw_output.get("choices")[0].get("text").strip()
    return reply


if __name__ == '__main__':
    from time import perf_counter
    from datetime import timedelta

    start = perf_counter()
    print(generate_story("Tell me a fantasy story about the adventures of a hamster"))
    stop = perf_counter()
    print(f"\nTotal Time: {timedelta(seconds=stop - start)}")
