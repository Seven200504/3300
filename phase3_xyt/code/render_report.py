"""Render the shared report content using ReportLab and an embedded Chinese font."""
from pathlib import Path
import os,json,html
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.utils import ImageReader
ROOT=Path(__file__).resolve().parents[1]
font_paths=[os.environ.get('PHASE3_CJK_FONT',''),'/System/Library/Fonts/Supplemental/Arial Unicode.ttf','/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']
font=next((Path(p) for p in font_paths if p and Path(p).is_file()),None)
if font is None:raise RuntimeError('Set PHASE3_CJK_FONT to a Chinese TrueType font path.')
pdfmetrics.registerFont(TTFont('CJK',str(font)))
pages=json.loads((ROOT/'report/report_content.json').read_text())
width,height=A4; body_width=width-88; body_height=height-98
styles={
'kicker':ParagraphStyle('kicker',fontName='CJK',fontSize=8,leading=12,textColor=colors.HexColor('#60808a')),
 'title':ParagraphStyle('title',fontName='CJK',fontSize=17,leading=24,textColor=colors.HexColor('#164d5b'),spaceAfter=8),
 'body':ParagraphStyle('body',fontName='CJK',fontSize=9.5,leading=15,wordWrap='CJK',spaceAfter=7),
 'caption':ParagraphStyle('caption',fontName='CJK',fontSize=8,leading=12,textColor=colors.HexColor('#546c77'),wordWrap='CJK',spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='CJK',fontSize=7.7,leading=11,wordWrap='CJK'),
}
def paragraph(text,kind='body'):return Paragraph(html.escape(text),styles[kind])
def make_table(t):
    data=[[paragraph(v,'cell') for v in t['headers']]]+[[paragraph(v,'cell') for v in row] for row in t['rows']]
    weights=t.get('widths') or [1/len(t['headers'])]*len(t['headers'])
    tb=Table(data,colWidths=[w*body_width for w in weights],repeatRows=1,hAlign='LEFT')
    tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5f0f0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8aafb9')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d9e5e8')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7fafb')]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    return tb
story=[];layout=[]
for ix,p in enumerate(pages,1):
    before=[paragraph(f'MAT3300  /  PHASE 3  /  {ix:02d}','kicker'),Spacer(1,6),paragraph(p['title'],'title')]+[paragraph(t) for t in p['paragraphs']]
    after=[]
    if p['caption']:after.extend([paragraph(p['caption'],'caption')])
    for t in p['tables']:after.extend([Spacer(1,6),make_table(t)])
    def height_of(items):
        return sum(x.wrap(body_width,body_height)[1]+getattr(x,'spaceAfter',0)+getattr(x,'spaceBefore',0) for x in items)
    base_height=height_of(before+after)
    story.extend(before)
    ih=0
    if p['figure']:
        path=ROOT/'output'/p['figure'];iw,orig_h=ImageReader(str(path)).getSize()
        max_h=min(p['figure_max_height'],body_height-base_height-24)
        if max_h<140:raise ValueError(f'Report section {ix} leaves only {max_h:.1f} pt for its figure')
        scale=min(body_width/iw,max_h/orig_h)
        ih=orig_h*scale
        story.extend([Spacer(1,5),Image(str(path),width=iw*scale,height=ih,hAlign='CENTER'),Spacer(1,6)])
    elif base_height>body_height-10:raise ValueError(f'Report section {ix} is too long')
    story.extend(after)
    layout.append({'section':ix,'text_table_height':round(base_height,1),'figure_height':round(ih,1),'available_height':round(body_height,1)})
    if ix<len(pages):story.append(PageBreak())
def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#d7e3e6'));c.line(44,37,width-44,37)
    c.setFont('CJK',8);c.setFillColor(colors.HexColor('#60808a'))
    c.drawString(44,24,'MAT3300 · R5–R7  |  2026-11-23');c.drawRightString(width-44,24,str(doc.page))
doc=SimpleDocTemplate(str(ROOT/'report/Phase3_Report.pdf'),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=44,bottomMargin=54,title='音乐演化、代际传承与文化联系',author='MAT3300 Phase 3',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'output/report_layout.json').write_text(json.dumps(layout,indent=2))
print(f'Rendered PDF from {len(pages)} shared sections using {font.name}.')
