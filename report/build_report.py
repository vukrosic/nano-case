"""Build the nano-case technical report PDF (single column, US-letter, Times).

Own-work variant: a technical report by Vuk Rosic. Figures are original plots
of the model's own measured results. Every number traces to the shipped
benchmark (scripts/eval_cases_full.py), reproduced from the published weights.

    python build_report.py   ->  nano-case-report.pdf
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, KeepTogether)

BASE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE, "..", "assets")
OUT = os.path.join(BASE, "nano-case-report.pdf")

# --- fonts: embed real Times New Roman so 'c' with acute renders ---
SUP = "/System/Library/Fonts/Supplemental"
pdfmetrics.registerFont(TTFont("TNR", f"{SUP}/Times New Roman.ttf"))
pdfmetrics.registerFont(TTFont("TNR-B", f"{SUP}/Times New Roman Bold.ttf"))
pdfmetrics.registerFont(TTFont("TNR-I", f"{SUP}/Times New Roman Italic.ttf"))
pdfmetrics.registerFont(TTFont("TNR-BI", f"{SUP}/Times New Roman Bold Italic.ttf"))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B",
                              italic="TNR-I", boldItalic="TNR-BI")
MONO = "Courier"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="TNR-B", fontSize=12.5,
                    spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#111111"))
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontName="TNR", fontSize=10.3,
                      leading=14.2, alignment=TA_JUSTIFY, spaceAfter=6)
TITLE = ParagraphStyle("TITLE", fontName="TNR-B", fontSize=18, leading=21,
                       alignment=TA_CENTER, spaceAfter=4)
BYLINE = ParagraphStyle("BYLINE", fontName="TNR", fontSize=10.5, alignment=TA_CENTER,
                        textColor=colors.HexColor("#222222"), spaceAfter=2)
ATTRIB = ParagraphStyle("ATTRIB", fontName="TNR", fontSize=9.5, alignment=TA_CENTER,
                        textColor=colors.HexColor("#222222"), spaceAfter=2)
CAP = ParagraphStyle("CAP", fontName="TNR-I", fontSize=8.8, leading=11.5,
                     alignment=TA_CENTER, textColor=colors.HexColor("#333333"),
                     spaceBefore=3, spaceAfter=10)
CODE = ParagraphStyle("CODE", fontName=MONO, fontSize=8.6, leading=11,
                      textColor=colors.HexColor("#111111"), spaceAfter=6)

USABLE = letter[0] - 2 * 0.95 * inch


def fig(name, width_frac=1.0):
    p = os.path.join(ASSETS, name)
    from PIL import Image as PILImage
    w, h = PILImage.open(p).size
    W = USABLE * width_frac
    return Image(p, width=W, height=W * h / w)


def tbl(data, col_widths, header=True):
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "TNR"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.3),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#111111")),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.HexColor("#111111")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.HexColor("#111111")),
    ]
    if header:
        style.append(("FONTNAME", (0, 0), (-1, 0), "TNR-B"))
    t.setStyle(TableStyle(style))
    return t


def deco(canvas, doc):
    canvas.saveState()
    canvas.setFont("TNR", 8.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(letter[0] / 2, 0.55 * inch, str(canvas.getPageNumber()))
    canvas.restoreState()


def P(t, s=BODY):
    return Paragraph(t, s)


story = []
story += [
    P("nano-case: A 1M-Parameter Transformer that Reads Identifiers a Regex Cannot Split", TITLE),
    P("A Technical Report by Vuk Rosić", BYLINE),
    P("Subject: nano-case, an open-source (MIT) ~1M-parameter byte-level transformer "
      "for case conversion of boundary-destroyed identifiers, trained 100% on code-generated data.", ATTRIB),
    Spacer(1, 8),
]

story += [P("Abstract", H1)]
story += [P(
    "Converting a <i>clean</i> identifier between case styles &mdash; <font name='Courier'>snake_case</font>, "
    "<font name='Courier'>kebab-case</font>, <font name='Courier'>camelCase</font>, <font name='Courier'>PascalCase</font>, "
    "<font name='Courier'>CONST_CASE</font> &mdash; is a solved, free problem: a regular expression splits on the "
    "separators and case humps and re-renders. nano-case targets the slice a regex provably cannot handle: "
    "<b>boundary-destroyed</b> inputs where the separators and casing are gone (<font name='Courier'>sdkmodel</font>, "
    "<font name='Courier'>httprequesthandler</font>), leaving nothing to split on. The only route back to the intended "
    "words is a learned vocabulary. On a held-out test set, nano-case reaches <b>99.8% &plusmn; 0.0%</b> exact-match overall "
    "(3 training seeds) versus <b>61.8%</b> for a standard regex converter, and on the regex-killer &lsquo;smushed&rsquo; slice "
    "(N=1410) <b>99.7% &plusmn; 0.0%</b> versus the regex&rsquo;s <b>8.2%</b>. The model has 1,016,960 parameters, runs on a CPU in "
    "milliseconds, and was trained entirely on data generated by code &mdash; no scraping, no labelling, no distillation. "
    "We also report where it breaks: its segmentation prior <i>is</i> its ~120-token training vocabulary, so it collapses "
    "on out-of-vocabulary words (2%) and on chains longer than seen in training (33%).")]

story += [KeepTogether([fig("fig1_model_vs_regex.png", 0.92), P(
    "Figure 1. Exact-match accuracy, nano-case vs a standard regex case-converter. "
    "Left: overall held-out set (N=4000). Right: the &lsquo;smushed&rsquo; regex-killer slice (N=1410) &mdash; "
    "boundary-destroyed, single-case, multi-word inputs. The 8.2% vs 99.7% gap is the &lsquo;you genuinely "
    "need a model here&rsquo; result. Model bars are the mean over 3 training seeds.", CAP)])]

story += [P("1. The problem: when a script runs out of signal", H1)]
story += [P(
    "A deterministic converter works by reading structure that is still present in the input. Given "
    "<font name='Courier'>userTableHandler</font> it splits on the case humps; given <font name='Courier'>user_table_handler</font> "
    "it splits on underscores. That is genuinely free, so building a model for it would be dishonest. The interesting "
    "case is when a developer (or an upstream tool) has <i>destroyed</i> those boundaries: an all-lowercase, "
    "separator-free token such as <font name='Courier'>userprofilecache</font>, or an all-caps "
    "<font name='Courier'>HTTPREQUESTHANDLER</font>. A regex has no signal left to split on &mdash; the word boundaries "
    "exist only in a reader&rsquo;s knowledge of which character sequences are words. That knowledge is exactly what a "
    "small language model can hold, and what a script cannot.")]

story += [P("2. Data: correct by construction, generated entirely by code", H1)]
story += [P(
    "There is no dataset to download. Each training example is synthesised: sample 1&ndash;4 words from a fixed "
    "~120-token vocabulary (common code words, acronyms, digit tokens), render the <b>gold target</b> canonically in "
    "the chosen case style, then corrupt a copy of the input &mdash; about 45% of the time by destroying all boundaries "
    "(lower/upper smush). Because the target is rendered from the known word list, every label is correct <i>by "
    "construction</i>; there is nothing to mislabel. A held-out set uses a disjoint RNG seed, and a test checks its "
    "overlap with the 100k-example training set stays below 25%.")]
story += [P(
    "Training is supervised fine-tuning with the prompt masked: the loss is computed only on the target identifier "
    "and a one-byte newline end-of-sequence marker, never on the <font name='Courier'>&lt;case&gt; | &lt;messy input&gt; =&gt;</font> "
    "prompt. The model is a ~1M-parameter byte-level decoder-only transformer (RMSNorm, RoPE, grouped-query attention, "
    "SwiGLU; dim 128, 4 layers), trained for 12k steps with AdamW and a cosine schedule.")]

story += [P("3. Results", H1)]
story += [P(
    "We evaluate exact-match accuracy on a held-out set (seed 987654321, N=4000) against "
    "<font name='Courier'>regex_convert</font>, a standard separator-and-hump case converter. Headline numbers are the "
    "mean &plusmn; standard deviation over three independent training seeds (0/1/2); the variance is below 0.05%, so it "
    "rounds to 0.0%.", BODY)]
story += [tbl([
    ["", "nano-case (model)", "regex script"],
    ["Overall (N=4000)", "99.8% ± 0.0%", "61.8%"],
    ["Smushed slice (N=1410)", "99.7% ± 0.0%", "8.2%"],
], [2.2 * inch, 1.9 * inch, 1.5 * inch])]
story += [Spacer(1, 5), P(
    "Table 1. The regex scores 61.8% overall because the held-out set mixes easy (still-structured) inputs it can "
    "handle with hard (boundary-destroyed) ones it cannot. Restricting to the smushed slice isolates the hard case: "
    "the script collapses to 8.2% while the model holds at 99.7%.", CAP)]

story += [P("4. Where it breaks (out-of-distribution)", H1)]
story += [P(
    "An honest nano model reports its ceiling. nano-case&rsquo;s segmentation ability <i>is</i> its training vocabulary, "
    "so we probe three out-of-distribution regimes against the in-vocabulary control.", BODY)]
story += [KeepTogether([fig("fig2_ood.png", 0.86), P(
    "Figure 2. OOD breaking-point probe (seed-0 model, N=400 each). The model nails smushing of <i>known</i> words "
    "(100%), but has no segmentation prior for out-of-vocabulary tokens (2%) and degrades on chains longer than the "
    "1&ndash;4 words seen in training (33%). This is the expected ceiling of a 1M-parameter model on a vocabulary "
    "task &mdash; reported, not hidden.", CAP)])]

story += [P("5. Scientific rigor and reproducibility", H1)]
story += [P(
    "Every claim in this report is regenerated from the published weights. The release ships: a self-contained "
    "inference file (<font name='Courier'>modeling_nano_case.py</font>, torch + safetensors only); the data generator "
    "(<font name='Courier'>data_cases.py</font>); the benchmark (<font name='Courier'>eval_nano_case.py</font>); and a "
    "test suite (<font name='Courier'>test_nano_case.py</font>) that checks labels are correct by construction, "
    "train/test overlap stays low, decoding is deterministic, and &mdash; as a regression gate &mdash; that the shipped "
    "<font name='Courier'>model.safetensors</font> still performs the hard, regex-defeating conversions. Reproduce:", BODY)]
story += [P("pip install -r requirements.txt<br/>"
            "python eval_nano_case.py --n 4000&nbsp;&nbsp;&nbsp;# model vs regex, overall + smushed slice<br/>"
            "pytest test_nano_case.py -q&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# labels / overlap / determinism / weights regression", CODE)]

story += [P("Author's note", H1)]
story += [P(
    "Independent technical report by Vuk Rosić, 30 June 2026. nano-case and the voidlab training harness are the "
    "author&rsquo;s own work, released under the MIT license. All figures are original plots of the model&rsquo;s own "
    "measured results; all numbers are reproduced from the published weights with the shipped benchmark. Any "
    "interpretation errors are the author&rsquo;s own.", ParagraphStyle(
        "note", parent=BODY, fontSize=9.2, textColor=colors.HexColor("#333333")))]

doc = SimpleDocTemplate(OUT, pagesize=letter, topMargin=0.85 * inch,
                        bottomMargin=0.8 * inch, leftMargin=0.95 * inch,
                        rightMargin=0.95 * inch, title="nano-case Technical Report",
                        author="Vuk Rosic")
doc.build(story, onFirstPage=deco, onLaterPages=deco)
print("wrote", OUT)
