import numpy as np


class SoftmaxCCE:
    @staticmethod
    def compute_loss(logits, reference_answers):
        logits_for_answers = logits[np.arange(len(logits)), reference_answers]
        xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))
        return xentropy

    @staticmethod
    def compute_grad(logits, reference_answers):
        ones_for_answers = np.zeros_like(logits)
        ones_for_answers[np.arange(len(logits)), reference_answers] = 1
        softmax = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        return (- ones_for_answers + softmax) / logits.shape[0]
