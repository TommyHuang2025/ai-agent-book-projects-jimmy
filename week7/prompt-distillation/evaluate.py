"""
Evaluation script for the distilled prompt model.

This script evaluates the student model's performance on language classification
without providing the detailed prompt.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


def load_model(model_path: str, base_model: str = "Qwen/Qwen3-4B-Instruct-2507"):
    """Load the fine-tuned model with LoRA adapters."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapters from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    return model, tokenizer


def parse_language_label(response: str) -> Optional[str]:
    """Extract the language label from model response."""
    # Try to match common patterns
    patterns = [
        r"^([a-z]{2})$",  # Just "en", "fr", etc.
        r"^([a-z]{2})\s*$",  # With trailing whitespace
        r"Final Answer:\s*([a-z]{2})",  # With "Final Answer:" prefix
        r"Language:\s*([a-z]{2})",  # With "Language:" prefix
    ]
    
    response = response.strip().lower()
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    # If response is short and looks like a language code
    if len(response) == 2 and response.isalpha():
        return response
    
    return None


def evaluate_model(
    model,
    tokenizer,
    test_sentences: List[str],
    ground_truth_labels: Optional[List[str]] = None,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> Dict:
    """Evaluate the distilled model on test sentences."""
    
    predictions = []
    correct = 0
    total = 0
    
    print("\nEvaluating model...")
    for idx, sentence in enumerate(tqdm(test_sentences)):
        # Format as user message (no system prompt!)
        messages = [
            {"role": "user", "content": sentence}
        ]
        
        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        # Decode
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        # Parse prediction
        pred_label = parse_language_label(response)
        predictions.append(pred_label)
        
        # Check correctness if ground truth available
        if ground_truth_labels and idx < len(ground_truth_labels):
            gt_label = ground_truth_labels[idx]
            if pred_label == gt_label:
                correct += 1
            total += 1
            
            # Show some examples
            if idx < 10 or (pred_label != gt_label and idx < 50):
                status = "✓" if pred_label == gt_label else "✗"
                print(f"\n{status} Example {idx + 1}:")
                print(f"  Input: {sentence[:80]}...")
                print(f"  Predicted: {pred_label}")
                print(f"  Ground Truth: {gt_label}")
                print(f"  Raw Response: {response[:100]}")
    
    # Calculate metrics
    results = {
        "predictions": predictions,
        "total": len(test_sentences),
        "predicted": sum(1 for p in predictions if p is not None),
        "unparseable": sum(1 for p in predictions if p is None),
    }
    
    if ground_truth_labels:
        results["accuracy"] = correct / total if total > 0 else 0.0
        results["correct"] = correct
        results["evaluated"] = total
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the distilled prompt model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapters",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base student model name",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="../tinker-cookbook/example-data/multilingual.txt",
        help="Test sentences file (one per line)",
    )
    parser.add_argument(
        "--ground_truth_file",
        type=str,
        default=None,
        help="Ground truth labels file (optional, one per line)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./evaluation_results.json",
        help="Where to save evaluation results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)",
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    # Load test data
    print(f"\nLoading test sentences from: {args.test_file}")
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_sentences = [line.strip() for line in f if line.strip()]
    
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth_file:
        print(f"Loading ground truth from: {args.ground_truth_file}")
        with open(args.ground_truth_file, "r", encoding="utf-8") as f:
            ground_truth = [line.strip() for line in f if line.strip()]
    
    # Limit samples if requested
    if args.max_samples:
        test_sentences = test_sentences[:args.max_samples]
        if ground_truth:
            ground_truth = ground_truth[:args.max_samples]
        print(f"Limiting evaluation to {args.max_samples} samples")
    
    print(f"Total test sentences: {len(test_sentences)}")
    
    # Evaluate
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_sentences=test_sentences,
        ground_truth_labels=ground_truth,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {results['total']}")
    print(f"Successfully predicted: {results['predicted']}")
    print(f"Unparseable responses: {results['unparseable']}")
    print(f"Parse rate: {results['predicted']/results['total']*100:.2f}%")
    
    if "accuracy" in results:
        print(f"\nAccuracy: {results['accuracy']*100:.2f}%")
        print(f"Correct: {results['correct']}/{results['evaluated']}")
    
    # Save results
    output = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "test_file": args.test_file,
        "metrics": {
            k: v for k, v in results.items() if k != "predictions"
        },
    }
    
    # Optionally include predictions
    if args.output_file:
        output["predictions"] = results["predictions"]
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()

