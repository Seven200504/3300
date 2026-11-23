"""Single entry point: recompute R5–R7, validate and regenerate the same report."""
from pathlib import Path
import argparse,hashlib,importlib.util,importlib.metadata,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--pdf-python',help='Python executable with ReportLab installed')
args=parser.parse_args()
pdf_python=args.pdf_python or os.environ.get('PHASE3_PDF_PYTHON')
if not pdf_python:
    if importlib.util.find_spec('reportlab'):pdf_python=sys.executable
    else:
        bundled=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
        if bundled.exists():pdf_python=str(bundled)
if not pdf_python:raise SystemExit('Install reportlab in this Python, or pass --pdf-python. Set PHASE3_CJK_FONT if needed.')
(ROOT/'output').mkdir(exist_ok=True)
with (ROOT/'output/run.log').open('w') as log:
    for script in ['r5.py','r6.py','r6_predict.py','r7.py','validate.py','build_report.py','render_report.py']:
        print('Running',script,flush=True)
        log.write('\n### '+script+'\n');log.flush()
        executable=pdf_python if script=='render_report.py' else sys.executable
        subprocess.run([executable,str(ROOT/'code'/script)],cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
from csv_store import consolidate
consolidate(ROOT)
files=sorted(p for folder in ['code','output','report'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json')
files += [ROOT/'README.md', ROOT/'requirements.txt']
manifest={'version':'2026-11-23','python':sys.version,'analysis_packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib']},'files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
(ROOT/'output/manifest.json').write_text(json.dumps(manifest,indent=2))
print('Completed. Open phase3_xyt/report/Phase3_Report.html or Phase3_Report.pdf.',flush=True)
