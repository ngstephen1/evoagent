import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

# Constants for Validation
MAX_EPOCHS = 5
ALPHA_REQUIRED_ACC = 0.98
BETA_TARGET_ACC = 0.80
PARAM_LIMIT = 2000
GAMMA_TOLERANCE = 1e-4

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
        output = translation_fn('Hello', strategy='greedy')
        assert isinstance(output, str), 'Translation must return a string'
        logging.info('\033[92m[SUCCESS] Part 4: Translation logic verified!\033[0m')
        return True
    except Exception as e:
        logging.warning(f'\033[91m[FAIL] Part 4: {str(e)}\033[0m')
        return False

class VirusValidator:
    """Collection of static validation helpers for assignment 'strains'."""
    @staticmethod
    def verify_translation_quality(bleu_score: float) -> bool:
        if bleu_score > 20.0:
            logging.info("[SUCCESS] Translation quality verified! (BLEU: %.2f)", bleu_score)
            return True
        logging.warning("[FAIL] Translation too weak. (BLEU: %.2f)", bleu_score)
        return False