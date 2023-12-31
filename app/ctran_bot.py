"""Story Bot: ctransformers

https://github.com/marella/ctransformers
https://huggingface.co/TheBloke/OpenOrca-Platypus2-13B-GGUF/resolve/main/openorca-platypus2-13b.Q4_K_M.gguf
"""
from ctransformers import AutoModelForCausalLM


def generate_story(user_prompt: str):
    system_prompt = "You are a masterful storyteller. Please respond with a " \
                    "short story based on the following prompt. Please respond " \
                    "with just the story."
    prompt = f"### Instruction: {system_prompt}\n\n" \
             f"{user_prompt}\n\n" \
             f"### Response:\n"
    llm = AutoModelForCausalLM.from_pretrained(
        model_path_or_repo_id="./models/openorca-platypus2-13b.Q4_K_M.gguf",
        model_type="llama",
        context_length=4096,
        max_new_tokens=2048,
        temperature=0.8,
        stop=["###"],
    )
    reply = llm(prompt)
    return reply


if __name__ == '__main__':
    from time import perf_counter
    from datetime import timedelta

    start = perf_counter()
    print(generate_story("Tell me a fantasy story about the adventures of a hamster"))
    stop = perf_counter()
    print(f"\nTotal Time: {timedelta(seconds=stop - start)}")
