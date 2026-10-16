import re
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def post_process_report(raw_text, predicted_class):
    text = raw_text.strip()
    text = re.sub(r'\[CLS=[^\]]*\]', '', text).strip()
    text = re.sub(r'\[AI-DenseNet:[^\]]*\]', '', text).strip()
    text = re.sub(r'Describe this chest X-ray\.?', '', text, flags=re.IGNORECASE).strip()
    text = text.lstrip('"\'-:; ')
    
    has_f = 'FINDINGS:' in text.upper()
    has_i = 'IMPRESSION:' in text.upper()
    if has_f and has_i:
        return text
    if has_f and not has_i:
        imp = "Findings consistent with pneumonia. Clinical follow-up recommended." if predicted_class == 1 else "No acute cardiopulmonary abnormality. Normal study."
        return f"{text}\nIMPRESSION: {imp}"
    if not has_f and has_i:
        return f"FINDINGS: {text}"
    imp = "Findings consistent with pneumonia. Clinical correlation advised." if predicted_class == 1 else "No acute cardiopulmonary abnormality. Normal study."
    return f"FINDINGS: {text}\nIMPRESSION: {imp}"

def compute_nlg_metrics(hypotheses, references):
    smooth = SmoothingFunction().method1
    b1, b2, b3, b4 = [], [], [], []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rl = []
    for hyp, ref in zip(hypotheses, references):
        ht = hyp.lower().split()
        rt = [ref.lower().split()]
        b1.append(sentence_bleu(rt, ht, weights=(1, 0, 0, 0), smoothing_function=smooth))
        b2.append(sentence_bleu(rt, ht, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
        b3.append(sentence_bleu(rt, ht, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth))
        b4.append(sentence_bleu(rt, ht, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
        rl.append(scorer.score(ref, hyp)['rougeL'].fmeasure)
    return {
        'BLEU-1': float(np.mean(b1)),
        'BLEU-2': float(np.mean(b2)),
        'BLEU-3': float(np.mean(b3)),
        'BLEU-4': float(np.mean(b4)),
        'ROUGE-L': float(np.mean(rl)),
        'bleu_4_per_case': b4
    }

def compute_paired_statistics(scores_a, scores_b):
    diff = np.array(scores_b) - np.array(scores_a)
    t_stat, t_pval = stats.ttest_rel(scores_b, scores_a)
    w_stat, w_pval = stats.wilcoxon(scores_b, scores_a)
    cohen_d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
    return {
        't_stat': float(t_stat), 't_pvalue': float(t_pval),
        'wilcoxon_stat': float(w_stat), 'wilcoxon_pvalue': float(w_pval),
        'cohens_d': cohen_d
    }
