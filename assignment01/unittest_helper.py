import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

def test_part1_tokenization(en_tokens, vi_tokens):
    if isinstance(en_tokens, list) and len(en_tokens) > 0 and isinstance(vi_tokens, list) and len(vi_tokens) > 0:
        logging.info('\033[92m[SUCCESS] Part 1: Tokenization logic verified!\033[0m')
        return True
    logging.warning('\033[91m[FAIL] Part 1: Tokens are still empty.\033[0m')
    return False

def test_part2_chatml(prompt, response):
    try:
        assert '<|im_start|>system' in prompt, 'Missing system tag'
        assert '<|im_start|>user' in prompt, 'Missing user tag'
        assert '<|im_start|>assistant' in prompt, 'Missing assistant tag'
        assert '<|im_end|>' in prompt, 'Missing end tags'
        logging.info('\033[92m[SUCCESS] Part 2: ChatML formatting logic passed!\033[0m')
        return True
    except AssertionError as e:
        logging.warning(f'\033[91m[FAIL] Part 2: {str(e)}\033[0m')
        return False

def test_part2_collator(inputs, targets):
    if isinstance(inputs, torch.Tensor) and (targets == -100).any():
        logging.info('\033[92m[SUCCESS] Part 2: Collator masking logic verified!\033[0m')
        return True
    logging.warning('\033[91m[FAIL] Part 2: Collator masking incorrect.\033[0m')
    return False

def test_part3_dataset(train_ds):
    try:
        assert len(train_ds) > 0, 'Dataset length cannot be 0'
        item = train_ds[0]
        assert isinstance(item, tuple) and len(item) == 2, 'Should return (en, vi) tuple'
        logging.info('\033[92m[SUCCESS] Part 3: Dataset implementation passed!\033[0m')
        return True
    except Exception as e:
        logging.warning(f'\033[91m[FAIL] Part 3: {str(e)}\033[0m')
        return False


def test_part4_translation(translation_fn, model, tokenizer):
    try:
        test_input = 'Hello'
        output = translation_fn(test_input, strategy='greedy')

        # 1. Check type
        assert isinstance(output, str), 'Translation must return a string'

        # 2. Check that it's not empty
        assert len(output.strip()) > 0, 'Translation output is empty'

        # 3. Check that it doesn't contain ChatML tags (Isolation check)
        forbidden_tokens = ['<|im_start|>', '<|im_end|>', 'assistant', 'system', 'user']
        for token in forbidden_tokens:
            assert token not in output, f'Output contains internal ChatML tag or role identifier: {token}'

        # 4. Check that it's not just returning the prompt
        assert test_input not in output or len(output) < len(test_input) * 2, 'Output looks like it might still contain the full prompt history'

        logging.info('\033[92m[SUCCESS] Part 4: Rigorous translation logic verified!\033[0m')
        return True
    except Exception as e:
        logging.warning(f'\033[91m[FAIL] Part 4: {str(e)}\033[0m')
        return False
def test_part5_icl(few_shot_fn):
    try:
        examples = [('Apple', 'Quả táo'), ('Banana', 'Quả chuối')]
        prompt = few_shot_fn('Orange', examples)
        assert 'Quả táo' in prompt and 'Quả chuối' in prompt, 'Few-shot examples missing from prompt'
        assert '<|im_start|>assistant' in prompt, 'Prompt missing final assistant turn'
        logging.info('\033[92m[SUCCESS] Part 5: ICL Prompting logic verified!\033[0m')
        return True
    except Exception as e:
        logging.warning(f'\033[91m[FAIL] Part 5: {str(e)}\033[0m')
        return False

def test_part6_bleu(bleu_score):
    if isinstance(bleu_score, (int, float)) and bleu_score >= 0:
        logging.info(f'\033[92m[SUCCESS] Part 6: BLEU score calculation verified! Score: {bleu_score:.2f}\033[0m')
        return True
    logging.warning('\033[91m[FAIL] Part 6: Invalid BLEU score.\033[0m')
    return False