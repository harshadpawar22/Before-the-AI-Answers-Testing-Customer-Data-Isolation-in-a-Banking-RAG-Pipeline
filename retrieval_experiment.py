"""Deterministic retrieval-boundary experiment, not an LLM benchmark.
Run: python3 retrieval_experiment.py
No dependencies, network calls, credentials or real customer data.
"""
import collections, hashlib, json, math, pathlib, platform, re
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent
TOPICS = {
 'balance': ('The outstanding loan principal balance is INR {n}. This statement records the remaining debt.', 'What is my outstanding loan principal balance?'),
 'instalment': ('The next monthly instalment repayment amount is INR {n}. The scheduled due date is 15 October.', 'What is my next monthly instalment repayment amount and scheduled due date?'),
 'closure': ('The account closure settlement quote is INR {n}. This figure describes the full payoff amount.', 'What is my account closure settlement quote for full payoff?'),
}
DOCS = []
for customer in range(1, 13):
 for topic, (template, query) in TOPICS.items():
  DOCS.append({'id': f'C{customer:02d}-{topic}', 'owner': f'C{customer:02d}',
               'topic': topic, 'text': template.format(n=10000 + customer*137)})
def terms(text):
 # Deliberate simplification: numbers are excluded; ownership is metadata.
 return re.findall(r'[a-z]+', text.lower())
df = collections.Counter(t for d in DOCS for t in set(terms(d['text'])))
idf = {t: math.log((1+len(DOCS))/(1+n))+1 for t,n in df.items()}
def vector(text):
 c = collections.Counter(terms(text))
 v = {t:n*idf[t] for t,n in c.items() if t in idf}
 norm = math.sqrt(sum(x*x for x in v.values()))
 return {t:x/norm for t,x in v.items()} if norm else {}
DV = {d['id']:vector(d['text']) for d in DOCS}
def rank(query, docs, k):
 qv=vector(query)
 scored=[(sum(v*DV[d['id']].get(t,0) for t,v in qv.items()),d) for d in docs]
 # Stable ID tie-break is intentional, explicit and reproducible.
 return [d for score,d in sorted(scored,key=lambda x:(-x[0],x[1]['id']))[:k] if score>0]
rows=[]
for k in (1,3,5,12):
 for mode in ('unfiltered','filter_after_top_k','filter_before_top_k'):
  for customer in range(1,13):
   principal=f'C{customer:02d}'  # trusted fixture identity; not parsed from a prompt
   for topic,(_,query) in TOPICS.items():
    pool=[d for d in DOCS if d.get('owner')==principal] if mode=='filter_before_top_k' else DOCS
    candidates=rank(query,pool,k)
    context=[d for d in candidates if d.get('owner')==principal] if mode=='filter_after_top_k' else candidates
    target=f'{principal}-{topic}'
    rows.append({'k':k,'mode':mode,'principal':principal,'query':query,'target':target,
                 'candidate_ids':[d['id'] for d in candidates],
                 'context_ids':[d['id'] for d in context],
                 'unauthorized_chunks':sum(d.get('owner')!=principal for d in context),
                 'target_present':any(d['id']==target for d in context),
                 'empty_context':not context})
summary=[]
for k in (1,3,5,12):
 for mode in ('unfiltered','filter_after_top_k','filter_before_top_k'):
  cells=[r for r in rows if r['k']==k and r['mode']==mode]
  summary.append({'k':k,'mode':mode,'queries':len(cells),
                  'contexts_with_unauthorized_data':sum(r['unauthorized_chunks']>0 for r in cells),
                  'unauthorized_chunk_occurrences':sum(r['unauthorized_chunks'] for r in cells),
                  'target_present':sum(r['target_present'] for r in cells),
                  'empty_context':sum(r['empty_context'] for r in cells)})
output={'method':{'executed_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
                 'documents':36,'synthetic_customers':12,'distinct_queries_by_principal':36,
                 'configurations':len(rows),'ranking':'word TF-IDF cosine, stable document-ID tie-break',
                 'llm_calls':0,'embedding_model_calls':0,
                 'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()},
        'summary':summary,'corpus':DOCS,'observations':rows}
(ROOT/'retrieval_results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps({'method':output['method'],'summary':summary},indent=2))
