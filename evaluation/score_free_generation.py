"""Critic-score free-generation outputs written by experiments.residual.free_generation.

Scores every free_*.jsonl in --dir with the rule's existing critic and writes
free_*_critic.jsonl, free_critic_summary.json and critic_metadata.json. Resumable.

    python evaluation/score_free_generation.py --rule s1 --dir data/experiments/baselines/3b/s1
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.score_clause_order import BITS, SCHEMA, SYSTEM, aggregate_judgment, clean  # noqa: E402
from evaluation.score_cot_alignment import CoTStanceCritic, annotate_record as annotate_s1, clean_chain_of_thought  # noqa: E402
from evaluation.score_voice_alignment import CoTVoiceCritic, annotate_record as annotate_voice  # noqa: E402

FOLLOW = {'s1': 'critic_follows_first_sentence', 'voice': 'critic_follows_voice',
          'clause': 'critic_follows_clause_order', 'lexical': 'rule_follow'}


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, rows):
    tmp = Path(path).with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    tmp.replace(path)


class ClauseCritic:
    def __init__(self, folder):
        from stencils.openai_client import OpenAIClient
        self.client = OpenAIClient()
        self.cache_path = Path(folder) / 'clause_critic_sentence_cache.json'
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    def annotate(self, row):
        text = clean(row['raw_generation'])
        key = hashlib.sha256((self.client.deployment + SYSTEM + text).encode()).hexdigest()
        if key not in self.cache:
            self.cache[key] = aggregate_judgment(self.client.chat_json_with_retries(
                SYSTEM, 'Reasoning text:\n' + text, SCHEMA, 'clause_order_sentence_critic')) if text else aggregate_judgment({'sentences': []})
            tmp = self.cache_path.with_suffix('.tmp'); tmp.write_text(json.dumps(self.cache, indent=2)); tmp.replace(self.cache_path)
        judged = self.cache[key]
        row.update(critic_order=judged['order'], critic_sentences=judged['sentences'],
                   critic_follows_clause_order=judged['order'] in BITS and row.get('prediction') == BITS[judged['order']])


def summarize(rows, rule):
    key = FOLLOW[rule]
    n = len(rows)
    parsed = [r for r in rows if r.get('prediction') in (0, 1)]
    follows = sum(bool(r.get(key)) for r in rows)
    return {'n': n, 'parsed': len(parsed), 'follow_field': key,
            'follow_rate_all': follows / n if n else None,
            'follow_rate_parsed': follows / len(parsed) if parsed else None,
            'accuracy_all': sum(r.get('prediction') == r['gold'] for r in rows) / n if n else None,
            'critic_errors': sum('critic_error' in r for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', required=True, type=Path)
    parser.add_argument('--rule', required=True, choices=sorted(FOLLOW))
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    sources = sorted(p for p in args.dir.glob('free_*.jsonl') if not p.stem.endswith('_critic'))
    assert sources, f'No free_*.jsonl in {args.dir}'
    if args.rule == 's1':
        critic = CoTStanceCritic('ethics')
        annotate = lambda row: annotate_s1(row, critic, scope='first_sentence')
        deployment = critic.client.deployment
    elif args.rule == 'voice':
        critic = CoTVoiceCritic()
        annotate = lambda row: annotate_voice(row, critic)
        deployment = critic.client.deployment
    elif args.rule == 'clause':
        critic = ClauseCritic(args.dir)
        annotate = critic.annotate
        deployment = critic.client.deployment
    else:
        annotate, deployment = None, 'deterministic lexical_score (no LLM)'
    summary = {}
    for source in sources:
        out = source.with_name(source.stem + '_critic.jsonl')
        rows = read(out) if out.exists() and not args.no_resume else read(source)
        for i, row in enumerate(rows):
            if args.rule in ('s1', 'voice'):
                row['chain_of_thought'] = clean_chain_of_thought(row['raw_generation'])
            if annotate and FOLLOW[args.rule] not in row and row.get('critic_scored') is not True:
                try:
                    annotate(row)
                except RuntimeError as error:
                    row[FOLLOW[args.rule]] = None
                    row['critic_error'] = str(error)
                row['critic_scored'] = True
                if i % 10 == 0:
                    write(out, rows)
        write(out, rows)
        arm = source.stem.removeprefix('free_')
        summary[arm] = summarize(rows, args.rule)
        print(args.rule, arm, json.dumps(summary[arm]), flush=True)
    (args.dir / 'free_critic_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.dir / 'critic_metadata.json').write_text(json.dumps({
        'rule': args.rule, 'backend': os.getenv('CRITIC_BACKEND', 'azure'), 'deployment': deployment,
        'chain_of_thought': 'raw_generation truncated before the first "Final answer:" marker',
        'follow_rate_all': 'rows judged following / all rows (unparsed or unresolved count as not following)',
        's1_scope': 'first_sentence', 'clause_prompt_sha256': hashlib.sha256(SYSTEM.encode()).hexdigest()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
