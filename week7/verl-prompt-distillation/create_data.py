"""
Data generation script for prompt distillation using vLLM.

This script generates training data for prompt distillation by using a teacher model
to generate language classification labels with a detailed prompt, which will then be
used to train a student model that internalizes the prompt.

Based on the tinker cookbook prompt distillation recipe.
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm_asyncio
from vllm import LLM, SamplingParams

LANGUAGE_CLASSIFICATION_PROMPT = """You are a precise language classifier.

Goal: Classify the language of the provided text into exactly one of these labels:
ar (Arabic), de (German), el (Greek), en (English), es (Spanish), fr (French),
hi (Hindi), ru (Russian), tr (Turkish), ur (Urdu), vi (Vietnamese),
zh (Chinese - Simplified), ot (Other/Unknown).

Instructions:
1) Preprocess carefully (without changing the intended meaning):
   - Trim whitespace.
   - Ignore URLs, emails, file paths, hashtags, user handles, and emojis.
   - Ignore numbers, math expressions, and standalone punctuation.
   - If there is code, IGNORE code syntax (keywords, operators, braces) and focus ONLY on human language in comments and string literals.
   - Preserve letters and diacritics; do NOT strip accents.
   - If after ignoring the above there are no alphabetic letters left, output 'ot'.

2) Script-based rules (highest priority):
   - Devanagari script → hi.
   - Greek script → el.
   - Cyrillic script → ru.
   - Han characters (中文) → zh. (Treat Traditional as zh too.)
   - Arabic script → ar vs ur:
       • If Urdu-only letters appear (e.g., ے, ڑ, ں, ھ, ٹ, ڈ, کھ, گ, چ with Urdu forms), or clear Urdu words, choose ur.
       • Otherwise choose ar.
   (If multiple scripts appear, pick the script that contributes the majority of alphabetic characters. If tied, go to step 5.)

3) Latin-script heuristics (use when text is mainly Latin letters):
   - vi: presence of Vietnamese-specific letters/diacritics (ă â ê ô ơ ư đ, plus dense diacritics across many words).
   - tr: presence of Turkish-specific letters (ı İ ğ Ğ ş Ş ç Ç ö Ö ü Ü) and common function words (ve, bir, için, değil, ama, çok).
   - de: presence of umlauts (ä ö ü) or ß and common function words (und, der, die, das, nicht, ist).
   - es: presence of ñ, ¿, ¡ and common words (y, de, la, el, es, no, por, para, con, gracias, hola).
   - fr: frequent French diacritics (é è ê à ç ô â î û ù) and common words (et, le, la, les, des, une, est, avec, pour, merci, bonjour).
   - en: default among Latin languages if strong evidence for others is absent, but ONLY if English function words are present (the, and, is, are, to, of, in, for, on, with). If evidence is insufficient for any Latin language, prefer 'ot' over guessing.

4) Named entities & loanwords:
   - Do NOT decide based on a single proper noun, brand, or place name.
   - Require at least two function words or repeated language-specific signals (diacritics/letters) before assigning a Latin-language label.

5) Mixed-language text:
   - Determine the dominant language by counting indicative tokens (language-specific letters/diacritics/function words) AFTER preprocessing.
   - If two or more languages are equally dominant or the text is a deliberate multi-language mix, return 'ot'.

6) Very short or noisy inputs:
   - If the text is ≤2 meaningful words or too short to be confident, return 'ot' unless there is a very strong language-specific signal (e.g., "bonjour" → fr, "hola" → es).

7) Transliteration/romanization:
   - If Hindi/Urdu/Arabic/Chinese/Russian/Greek is written purely in Latin letters (romanized) without clear, repeated language-specific cue words, return 'ot'. (Only classify as hi/ur/ar/zh/ru/el when native scripts or highly distinctive romanized patterns are clearly present.)

8) Code-heavy inputs:
   - If the text is mostly code with minimal or no natural-language comments/strings, return 'ot'.
   - If comments/strings clearly indicate a language per rules above, use that label.

9) Ambiguity & confidence:
   - When in doubt, choose 'ot' rather than guessing.

Output format:
- Respond with EXACTLY one line: "Final Answer: xx"
- Where xx ∈ {{ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot}} and nothing else.

Text to classify:
{text}
"""


def parse_final_answer(response: str) -> Optional[str]:
    """Parse the final answer from the model response."""
    search_response = re.search(r"Final Answer: (\w+)", response)
    return search_response.group(1) if search_response else None


async def generate_distillation_data(
    input_file: str,
    output_file: str,
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    temperature: float = 0.15,
    max_tokens: int = 1000,
    tensor_parallel_size: int = 1,
):
    """
    Generate prompt distillation training data.
    
    Args:
        input_file: Path to file containing sentences to classify (one per line)
        output_file: Path to save the generated training data (JSONL format)
        model_name: Teacher model to use for generating labels
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
    """
    print(f"Loading input sentences from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(sentences)} sentences")
    
    # Initialize vLLM model
    print(f"Initializing teacher model: {model_name}")
    print(f"Using tensor parallelism across {tensor_parallel_size} GPU(s)")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,  # Use 90% of GPU memory for better throughput
        max_model_len=32768,  # Match training max length
    )
    
    # Prepare prompts
    prompts = [LANGUAGE_CLASSIFICATION_PROMPT.format(text=sentence) for sentence in sentences]
    
    # Set sampling parameters - matching tinker's settings
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["\n\n"],  # Stop at double newline
    )
    
    print("Generating labels with teacher model...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Process outputs and save
    valid_count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence, output in zip(sentences, outputs):
            response = output.outputs[0].text
            final_answer = parse_final_answer(response)
            
            if final_answer:
                # Format as conversational data for verl SFT
                data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": sentence,
                        },
                        {
                            "role": "assistant",
                            "content": final_answer,
                        },
                    ]
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                valid_count += 1
    
    print(f"\nData generation complete!")
    print(f"Total sentences: {len(sentences)}")
    print(f"Valid labels generated: {valid_count}")
    print(f"Success rate: {valid_count/len(sentences)*100:.2f}%")
    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate prompt distillation training data using a teacher model"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./example-data/multilingual.txt",
        help="Path to input file with sentences (one per line)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./data/prompt_distillation_lang.jsonl",
        help="Path to save generated training data",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Teacher model name (matches tinker's Qwen3-30B-A3B, now open source)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.15,
        help="Sampling temperature (matching tinker's 0.15)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (recommend 2-4 for 30B model on H100)",
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Generate data
    asyncio.run(
        generate_distillation_data(
            input_file=args.input_file,
            output_file=args.output_file,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    )


if __name__ == "__main__":
    main()

