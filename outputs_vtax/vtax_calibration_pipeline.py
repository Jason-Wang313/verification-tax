"""
VTax NeurIPS 2026 Main Track Review Calibration Pipeline
Adapted from MIRROR scoring methodology v2.0
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime

DATA_PATH = 'C:/Users/wangz/Downloads/stereology/stereology/neurips_scraper/data/neurips_main_all_papers.jsonl'
OUTPUT_DIR = 'C:/Users/wangz/verification tax/outputs_vtax'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PHASE 1: LOAD + FILTER
# ============================================================

def load_papers(path):
    papers = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            papers.append(json.loads(line))
    print(f"Loaded {len(papers)} papers")
    return papers


THEORY_KEYWORDS = [
    'minimax', 'lower bound', 'upper bound', 'sample complexity',
    'calibration', 'estimation', 'nonparametric', 'hypothesis test',
    'statistical', 'information-theoretic', 'le cam', 'assouad',
    'functional estimation', 'convergence rate', 'optimal rate',
    'verification', 'auditing', 'fairness', 'uncertainty',
    'ece', 'expected calibration error', 'scoring rule',
    'phase transition', 'scaling law',
    'lipschitz', 'regression', 'density estimation',
    'confidence interval', 'testing', 'detection',
]


def is_theory_relevant(paper):
    text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
    return sum(1 for kw in THEORY_KEYWORDS if kw in text) >= 2


def filter_theory_papers(papers):
    theory = [p for p in papers if is_theory_relevant(p)]
    print(f"Filtered {len(papers)} -> {len(theory)} theory-relevant")
    return theory


# ============================================================
# PHASE 2: VTAX SIMILARITY MATCHING
# ============================================================

def compute_vtax_similarity(paper):
    """
    4-signal similarity scoring against VTax's profile.
    Returns (score, breakdown) where score in [0, 1].
    """
    text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()

    # === TOPIC (0.35) ===
    topic_signals = {
        'calibration_core': [
            'calibration', 'calibrated', 'miscalibration', 'recalibration',
        ],
        'ece_specific': [
            'expected calibration error', 'ece', 'binned ece',
            'calibration error', 'reliability diagram',
        ],
        'verification_auditing': [
            'verification', 'auditing', 'audit', 'verif',
            'monitoring', 'evaluation cost',
        ],
        'statistical_theory': [
            'minimax', 'lower bound', 'upper bound', 'sample complexity',
            'le cam', 'assouad', 'nonparametric',
            'hypothesis test', 'detection',
        ],
        'ai_safety_regulation': [
            'ai safety', 'regulation', 'regulatory', 'deployment',
            'trustworth', 'responsible ai',
        ],
    }

    topic_score = 0
    topic_breakdown = {}
    for signal, keywords in topic_signals.items():
        hits = sum(1 for kw in keywords if kw in text)
        val = min(hits / 2, 1.0)
        topic_breakdown[signal] = val
        topic_score += val
    topic_score = topic_score / len(topic_signals)

    # === STRUCTURE (0.25) ===
    has_theory = any(kw in text for kw in ['theorem', 'proof', 'lemma'])
    has_empirical = any(kw in text for kw in ['experiment', 'empiric', 'synthetic'])
    has_matched = bool(re.search(
        r'lower.*bound.*upper.*bound|upper.*bound.*lower.*bound|minimax.*rate|tight.*bound|match.*bound',
        text
    ))

    structure_breakdown = {
        'theorem_proof': 1.0 if has_theory else 0.0,
        'empirical_validation': 1.0 if has_empirical else 0.0,
        'matched_bounds': 1.0 if has_matched else 0.0,
        'hybrid_theory_empirical': 1.0 if (has_theory and has_empirical) else 0.0,
    }
    structure_score = sum(structure_breakdown.values()) / len(structure_breakdown)

    # === SCALE (0.15) ===
    formal_count = len(re.findall(r'theorem|lemma|corollary|proposition', text))
    scale_score = min(formal_count / 5, 1.0)

    # === CLAIM (0.25) ===
    claim_signals = {
        'impossibility': [
            'impossible', 'impossibility', 'cannot', 'no estimator',
            'fundamental limit',
        ],
        'phase_transition': [
            'phase transition', 'threshold', 'critical',
        ],
        'practical_tools': [
            'practition', 'practical', 'guideline', 'formula',
            'holdout', 'sample size',
        ],
        'scaling_consequence': [
            'scaling law', 'horizon', 'as models improve',
            'capability', 'grows with',
        ],
    }

    claim_score = 0
    claim_breakdown = {}
    for claim, keywords in claim_signals.items():
        hits = sum(1 for kw in keywords if kw in text)
        val = min(hits / 2, 1.0)
        claim_breakdown[claim] = val
        claim_score += val
    claim_score = claim_score / len(claim_signals)

    # === WEIGHTED TOTAL ===
    total = (topic_score * 0.35 + structure_score * 0.25 +
             scale_score * 0.15 + claim_score * 0.25)

    return total, {
        'total': total,
        'topic': topic_score,
        'structure': structure_score,
        'scale': scale_score,
        'claim': claim_score,
        'breakdown': {
            'topic': topic_breakdown,
            'structure': structure_breakdown,
            'claim': claim_breakdown,
        },
    }


def find_comparable_papers(theory_papers, min_similarity=0.3):
    """Score every theory paper against VTax. Return all above threshold."""
    scored = []

    for paper in theory_papers:
        sim, breakdown = compute_vtax_similarity(paper)
        if sim >= min_similarity:
            scored.append({
                'paper': paper,
                'similarity': sim,
                'similarity_breakdown': breakdown,
                'tier': (
                    'A_highly_similar' if sim >= 0.7 else
                    'B_moderately_similar' if sim >= 0.5 else
                    'C_weakly_similar'
                ),
            })

    scored.sort(key=lambda x: -x['similarity'])
    return scored


MANDATORY_COMPARABLES = [
    'On the Estimation of Expected Calibration Error',
    'An Information-Theoretic Analysis of Expected Calibration Error',
    'Testing Calibration in Nearly-Linear Time',
    'Auditing Fairness by Betting',
    'A Unifying Theory of Distance from Calibration',
    'Finite-Sample Guarantees for Calibration Error',
    'Active Testing',
]


def ensure_mandatory_comparables(comparables, all_papers):
    found_titles = {c['paper']['title'].lower() for c in comparables}

    for target_title in MANDATORY_COMPARABLES:
        if target_title.lower() not in found_titles:
            for paper in all_papers:
                if target_title.lower() in paper.get('title', '').lower():
                    sim, breakdown = compute_vtax_similarity(paper)
                    comparables.append({
                        'paper': paper,
                        'similarity': max(sim, 0.8),
                        'similarity_breakdown': breakdown,
                        'tier': 'A_highly_similar',
                        'manually_added': True,
                    })
                    print(f"  MANUALLY ADDED: {paper['title']}")
                    break
            else:
                print(f"  WARNING: '{target_title}' not found in scraped data")

    comparables.sort(key=lambda x: -x['similarity'])
    return comparables


# ============================================================
# PHASE 3: EXTRACT REVIEW PATTERNS FROM ACCEPTED PAPERS
# ============================================================

THEORY_WEAKNESS_CATEGORIES = {
    'novelty_technique': [
        'incremental', 'standard technique', 'well-known',
        'straightforward application', 'not novel',
        'direct consequence', 'follows from',
    ],
    'novelty_problem': [
        'artificial', 'contrived', 'not motivated',
        'unclear why', 'narrow setting',
    ],
    'assumptions_strong': [
        'strong assumption', 'restrictive', 'unrealistic',
        'lipschitz', 'iid', 'independence',
        'bounded density', 'regularity',
    ],
    'gap_bounds': [
        'gap', 'log factor', 'logarithmic', 'not tight',
        'loose', 'suboptimal',
    ],
    'experiments_weak': [
        'synthetic only', 'toy', 'limited experiment',
        'real data', 'no real', 'practical relevance',
    ],
    'presentation_dense': [
        'dense', 'hard to follow', 'notation',
        'too many results', 'theorem dump', 'clarity',
        'difficult to read',
    ],
    'scope_too_broad': [
        'too many', 'unfocused', 'tries to do too much',
        'would be better as', 'split into',
    ],
    'missing_related': [
        'missing reference', 'prior work', 'not cited',
        'already known', 'overlap with',
    ],
}


def classify_weaknesses(review_text):
    text = (review_text or '').lower()
    hits = {}
    for cat, keywords in THEORY_WEAKNESS_CATEGORIES.items():
        count = sum(1 for kw in keywords if kw in text)
        if count > 0:
            hits[cat] = count
    return hits


def extract_accepted_review_patterns(comparables):
    accepted = [c for c in comparables if c['paper'].get('is_accepted')]

    if not accepted:
        print("  WARNING: No accepted papers in comparables")
        return {}

    all_scores = []
    for c in accepted:
        for r in c['paper']['reviews']:
            if r.get('score'):
                all_scores.append(r['score'])

    mean_scores_per_paper = []
    for c in accepted:
        scores = [r['score'] for r in c['paper']['reviews'] if r.get('score')]
        if scores:
            mean_scores_per_paper.append(sum(scores) / len(scores))

    weakness_freq = defaultdict(int)
    total_reviews = 0
    for c in accepted:
        for r in c['paper']['reviews']:
            cats = classify_weaknesses(r.get('weaknesses', ''))
            for cat in cats:
                weakness_freq[cat] += 1
            total_reviews += 1

    survivable_weaknesses = {
        cat: {
            'count': count,
            'rate': round(count / total_reviews, 3) if total_reviews > 0 else 0,
            'interpretation': f'Raised in {count}/{total_reviews} reviews on accepted papers -- survivable',
        }
        for cat, count in weakness_freq.items()
    }

    return {
        'num_accepted_comparables': len(accepted),
        'score_stats': {
            'all_scores': all_scores,
            'mean': round(sum(all_scores) / len(all_scores), 2) if all_scores else None,
            'min': min(all_scores) if all_scores else None,
            'max': max(all_scores) if all_scores else None,
            'per_paper_means': [round(s, 2) for s in mean_scores_per_paper],
            'score_floor': round(min(mean_scores_per_paper), 2) if mean_scores_per_paper else None,
        },
        'survivable_weaknesses': survivable_weaknesses,
        'total_reviews_analyzed': total_reviews,
    }


# ============================================================
# PHASE 4: BUILD CALIBRATION JSON
# ============================================================

def build_calibration_output(all_papers, theory_papers, comparables, patterns):
    accepted_papers = [p for p in all_papers if p.get('is_accepted')]
    all_accepted_scores = []
    for p in accepted_papers:
        for r in p['reviews']:
            if r.get('score'):
                all_accepted_scores.append(r['score'])

    scores_by_year = defaultdict(list)
    for p in all_papers:
        year = p.get('year')
        for r in p['reviews']:
            if r.get('score'):
                scores_by_year[year].append(r['score'])

    output = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'target_paper': 'The Verification Tax: Fundamental Limits of AI Auditing in the Rare-Error Regime',
            'target_venue': 'NeurIPS 2026 Main Track',
            'data_source': f'Existing NeurIPS scrape ({len(all_papers)} papers)',
            'theory_papers_filtered': len(theory_papers),
            'comparable_papers_found': len(comparables),
            'DATA_LIMITATION': (
                'NeurIPS does NOT publish rejected papers by default. '
                'Only opt-in rejected papers are visible. This data is '
                'overwhelmingly accepted papers. We CANNOT compute '
                'fatality rates, decision boundaries, or score-to-acceptance '
                'mappings. All analysis is relative to accepted paper baselines.'
            ),
        },
        'global_stats': {
            'total_accepted': len(accepted_papers),
            'mean_accepted_score': round(sum(all_accepted_scores) / len(all_accepted_scores), 2) if all_accepted_scores else None,
            'scores_by_year': {
                str(year): {
                    'count': len(scores),
                    'mean': round(sum(scores) / len(scores), 2) if scores else None,
                }
                for year, scores in scores_by_year.items()
            },
        },
        'accepted_review_patterns': patterns,
        'comparable_papers': [],
    }

    for entry in comparables:
        paper = entry['paper']
        scores = [r['score'] for r in paper['reviews'] if r.get('score')]

        paper_entry = {
            'title': paper['title'],
            'year': paper.get('year'),
            'similarity': round(entry['similarity'], 3),
            'tier': entry['tier'],
            'scores': scores,
            'mean_score': round(sum(scores) / len(scores), 2) if scores else None,
            'decision': paper.get('decision'),
            'is_accepted': paper.get('is_accepted'),
            'manually_added': entry.get('manually_added', False),
        }

        if entry['tier'] == 'A_highly_similar':
            paper_entry['reviews_full'] = [{
                'score': r.get('score'),
                'confidence': r.get('confidence'),
                'strengths': (r.get('strengths', '') or '')[:2000],
                'weaknesses': (r.get('weaknesses', '') or '')[:2000],
                'questions': (r.get('questions', '') or '')[:1000],
            } for r in paper['reviews']]
            paper_entry['meta_review'] = paper.get('meta_review')
            paper_entry['abstract'] = paper.get('abstract', '')[:1500]

        elif entry['tier'] == 'B_moderately_similar':
            weakness_cats = defaultdict(int)
            for r in paper['reviews']:
                cats = classify_weaknesses(r.get('weaknesses', ''))
                for cat, count in cats.items():
                    weakness_cats[cat] += count
            paper_entry['weakness_categories'] = dict(weakness_cats)
            paper_entry['abstract'] = paper.get('abstract', '')[:500]

        output['comparable_papers'].append(paper_entry)

    return output


def save_calibration(output):
    path = os.path.join(OUTPUT_DIR, 'vtax_calibration_data.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nSaved calibration data to {path}")
    size_bytes = os.path.getsize(path)
    size_tokens_approx = size_bytes // 4
    print(f"  File size: {size_bytes / 1024:.1f} KB (~{size_tokens_approx} tokens)")
    if size_tokens_approx > 30000:
        print("  WARNING: File may be too large for context window.")
        print("  Consider trimming Tier C papers or reducing review text length.")
    return path


# ============================================================
# MAIN RUNNER
# ============================================================

if __name__ == '__main__':
    all_papers = load_papers(DATA_PATH)
    theory_papers = filter_theory_papers(all_papers)

    comparables = find_comparable_papers(theory_papers)
    comparables = ensure_mandatory_comparables(comparables, all_papers)

    print(f"\nComparable papers: {len(comparables)}")
    for tier in ['A_highly_similar', 'B_moderately_similar', 'C_weakly_similar']:
        tier_papers = [c for c in comparables if c['tier'] == tier]
        print(f"  {tier}: {len(tier_papers)}")
        for c in tier_papers[:5]:
            p = c['paper']
            scores = [r['score'] for r in p['reviews'] if r.get('score')]
            mean_s = f"{sum(scores)/len(scores):.1f}" if scores else "N/A"
            dec = p.get('decision', '?')
            manually = " [MANUAL]" if c.get('manually_added') else ""
            print(f"    [{mean_s}] [{dec}] {p.get('title', '')[:80]}{manually}")

    patterns = extract_accepted_review_patterns(comparables)
    print(f"\nScore floor among accepted comparables: {patterns.get('score_stats', {}).get('score_floor')}")
    print(f"Survivable weaknesses:")
    for cat, info in patterns.get('survivable_weaknesses', {}).items():
        print(f"  {cat}: {info['count']}/{patterns['total_reviews_analyzed']} reviews")

    output = build_calibration_output(all_papers, theory_papers, comparables, patterns)
    cal_path = save_calibration(output)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Comparable papers: {len(comparables)}")
    print(f"  Tier A: {len([c for c in comparables if c['tier'] == 'A_highly_similar'])}")
    print(f"  Tier B: {len([c for c in comparables if c['tier'] == 'B_moderately_similar'])}")
    print(f"  Tier C: {len([c for c in comparables if c['tier'] == 'C_weakly_similar'])}")
    print(f"  Output: {cal_path}")
    print(f"{'='*60}")
    print(f"\nNEXT STEP: Give your LLM reviewer:")
    print(f"  1. vtax_review_prompt.md")
    print(f"  2. vtax_calibration_data.json")
    print(f"  3. The VTax PDF")
