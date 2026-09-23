"""Rebuild a content-deduplicated historical documentation index without moving files."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
worktrees=[Path(line[9:]) for line in subprocess.check_output(['git','worktree','list','--porcelain'],cwd=ROOT,text=True).splitlines() if line.startswith('worktree ')]
paths=[]
for root in worktrees:paths.extend((root/'docs').rglob('*.md'))
paths.extend((ROOT.parent/'G1_Recovery_Below_Block/evidence/terrain_history').glob('*.md'))
by_hash={}
for p in sorted(paths):
 data=p.read_bytes();sha=hashlib.sha256(data).hexdigest();text=data.decode('utf8')
 item=by_hash.setdefault(sha,{'sha256':sha,'title':next((x.lstrip('# ').strip() for x in text.splitlines() if x.startswith('#')),''),'filenames':[],'paths':[]})
 item['paths'].append(str(p));item['filenames']=sorted(set(item['filenames']+[p.name]))
out=ROOT/'configs/recovery_study/documents.json';out.write_text(json.dumps({'status':'navigation snapshot; duplicated prose is not independent evidence','documents':list(by_hash.values())},ensure_ascii=False,indent=2)+'\n')
print(len(paths),'files;',len(by_hash),'distinct document contents')
