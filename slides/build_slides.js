// Builds slides/presentation.pptx from the numbers in results/.
//
//   npm install pptxgenjs        (once, anywhere on the Node path)
//   node slides/build_slides.js
//
// You don't need to run this to change the slides: open presentation.pptx in PowerPoint,
// Keynote or Google Slides and edit it there. This script is kept as the record of how the
// deck was made. Its colours come from a theme that is applied by a helper script named in
// the PPTX_APPLY_THEME environment variable; without it the deck gets PowerPoint's default
// colours.
//
// The charts are real PowerPoint charts filled from the result files, so the deck cannot
// drift away from the measured numbers. Text and speaker notes are written below: edit them
// here and rebuild, or edit the .pptx directly once you are happy with the structure.
// The first slide uses slides/qr_demo.png (the repository); the last slide uses slides/qr_colab.png, which opens
// the demo notebook in Google Colab. Both were made with the Python package `segno`.
// The demo slide has a clickable link to the Colab notebook instead of a QR code.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const read = (name) => JSON.parse(fs.readFileSync(path.join(ROOT, "results", name), "utf8"));

const ours = read("distilbert-squad2.json").metrics;
const roberta = read("roberta-base-squad2.json").metrics;
const training = read("distilbert-squad2_train_summary.json");
const sweep = read("threshold_sweep.json").models;
const errors = read("error_breakdown.json").models;
const efficiency = read("efficiency.json").models;
const longdoc = read("long_documents.json").articles;

const OURS = "distilbert-squad2", BASE = "roberta-base-squad2";
const effOurs = efficiency["Ingsyling/distilbert-squad2"], effBase = efficiency["deepset/roberta-base-squad2"];
const f1 = (x) => x.toFixed(1);

// ---------------------------------------------------------------------------------------
// Theme, layouts
// ---------------------------------------------------------------------------------------
const THEME = {
  name: "SQuAD QA",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0F1B2D", lt1: "FFFFFF", dk2: "46566B", lt2: "EEF3F9",
    accent1: "2A78D6",   // our model, everywhere
    accent2: "EB6834",   // the RoBERTa baseline, everywhere
    accent3: "1BAF7A", accent4: "EDA100", accent5: "6250D6", accent6: "E34948",
    hlink: "1C5CAB", folHlink: "6250D6",
  },
};
const HEX = THEME.colors;
const GRID = "DDE3EA";   // chart gridlines take hex only

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";          // 13.33 x 7.5 inches
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Extractive Question Answering on SQuAD 2.0 with a Fine-Tuned DistilBERT";
pres.author = "David Faleye and Kosisochukwu Ogbuanya";
const C = pres.SchemeColor;
const W = 13.33, MARGIN = 0.65;
const FOOTER = "CSCI 390 · Question answering on SQuAD 2.0";

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MARGIN, y: 2.0, w: 9.4, h: 2.2, fontSize: 40, bold: true,
                                color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: MARGIN, y: 4.45, w: 9.4, h: 1.6, fontSize: 20,
                                color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MARGIN, y: 2.5, w: 11.5, h: 1.5, fontSize: 40, bold: true,
                                color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: MARGIN, y: 4.1, w: 11.5, h: 1.2, fontSize: 20,
                                color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - 1.2, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MARGIN, y: 0.4, w: W - 2 * MARGIN, h: 1.0, fontSize: 32,
                                bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: FOOTER, options: { x: MARGIN, y: 6.95, w: 6, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: W - 1.25, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
});

// ---------------------------------------------------------------------------------------
// Small helpers. Each call builds fresh option objects (pptxgenjs changes them in place).
// ---------------------------------------------------------------------------------------
let section = "";
function startSection(title) { section = title; pres.addSection({ title }); }
function content(title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  s.addNotes(notes);
  return s;
}
function card(s, x, y, w, h, fill = C.background2, name = "card") {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { type: "none" }, objectName: name });
}
function text(s, value, x, y, w, h, opts = {}) {
  s.addText(value, { x, y, w, h, isTextBox: true, margin: 0, fontSize: 16, color: C.text1, valign: "top", ...opts });
}
function stat(s, x, y, w, number, label, color = C.accent1) {
  card(s, x, y, w, 2.0, C.background2, "stat card");
  text(s, number, x + 0.25, y + 0.2, w - 0.5, 1.0, { fontSize: 38, bold: true, color, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, label, x + 0.25, y + 1.2, w - 0.5, 0.7, { fontSize: 14, color: C.text2 });
}
function bullets(s, items, x, y, w, h, fontSize = 16) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } })),
            { x, y, w, h, isTextBox: true, margin: 0, fontSize, color: C.text1, valign: "top" });
}
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
                    titleFontFace: "+mn-lt" };
function chartBase(extra = {}) {
  return {
    ...chartText, catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 12, valAxisLabelFontSize: 12,
    legendFontSize: 12, legendColor: HEX.dk1, dataLabelFontSize: 11, dataLabelColor: HEX.dk2,
    valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" }, showTitle: false,
    chartColors: [HEX.accent1, HEX.accent2], showLegend: true, legendPos: "b", ...extra,
  };
}

// =======================================================================================
// 1. Title
// =======================================================================================
startSection("Introduction");
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: section });
  s.addText("Teaching a small model to answer questions, and to know when it can't", { placeholder: "title" });
  s.addText([
    { text: "Extractive question answering on SQuAD 2.0 with a fine-tuned DistilBERT", options: { breakLine: true } },
    { text: "David Faleye · Kosisochukwu Ogbuanya", options: { breakLine: true, fontSize: 18 } },
    { text: "CSCI 390: Natural Language Processing and Applications · Dr. Qi Li", options: { fontSize: 16 } },
  ], { placeholder: "body" });
  s.addImage({ path: path.join(__dirname, "qr_demo.png"), x: 10.55, y: 2.05, w: 2.1, h: 2.1, altText: "QR code to the project repository and demo" });
  s.addText("Code and demo", { x: 10.55, y: 4.25, w: 2.1, h: 0.35, isTextBox: true, margin: 0, fontSize: 14, color: C.background2, align: "center" });
  s.addNotes("SPEAKER: David. (0.5 min)\nIntroduce both of you and the one-sentence summary: we fine-tuned a small language model, DistilBERT, to find answers inside a passage, and to say 'no answer' when the passage doesn't contain one. Everything is public; the QR code goes to the repository, where the demo can be started. Outline: the task, how we built it, how well it works, what it gets wrong, a live demo.");
}

// Outline
{
  const s = content("What we'll cover",
    "SPEAKER: David. (0.5 min)\nFive parts. First the task and the data. Then how we built the system, step by step. Then how well it works against a published model. Kosi takes the fourth part: what it gets wrong and why, including our extension to long documents. We finish with a live demo and what we learned. Questions at the end.");
  const rows = [
    ["1", "The task and the data", "What extractive question answering is, and what makes SQuAD 2.0 hard"],
    ["2", "How we built it", "The model, preprocessing, training, and turning scores into answers"],
    ["3", "How well it works", "Our scores next to a larger published model (RoBERTa), and what being smaller costs"],
    ["4", "What it gets wrong, and why", "Tuning when to say \"no answer\", sorting all 4,258 errors, and testing on whole articles"],
    ["5", "Live demo and what we learned", "The system running, its limits, and our conclusions"],
  ];
  rows.forEach(([n, head, what], i) => {
    const y = 1.65 + i * 0.98;
    card(s, MARGIN, y, W - 2 * MARGIN, 0.84, C.background2, `outline row ${n}`);
    s.addShape(pres.shapes.OVAL, { x: MARGIN + 0.25, y: y + 0.14, w: 0.56, h: 0.56, fill: { color: C.accent1 }, line: { type: "none" }, objectName: `outline badge ${n}` });
    s.addText(n, { x: MARGIN + 0.25, y: y + 0.14, w: 0.56, h: 0.56, isTextBox: true, margin: 0, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
    text(s, head, MARGIN + 1.1, y + 0.1, 4.6, 0.64, { fontSize: 20, bold: true, fontFace: THEME.headFontFace, valign: "middle" });
    text(s, what, MARGIN + 5.8, y + 0.1, W - 2 * MARGIN - 6.1, 0.64, { fontSize: 15, color: C.text2, valign: "middle" });
  });
}

// 2. The task
{
  const s = content("The model points at the answer, or says there is none",
    "SPEAKER: David. (1.5 min)\nExplain 'extractive': the answer is always a piece of the passage, copied, never written freely. Walk through the passage. First question: the answer 'France' is in the text, highlighted. Second question looks just as reasonable, but the passage never says France is a region of anything, so the right output is 'no answer'. That second kind is what SQuAD 2.0 added, and it is the hard part: a system that always answers would confidently say 'Normandy'. Keep this example in mind, it comes back in the demo.");
  card(s, MARGIN, 1.65, 6.3, 4.7, C.background2, "passage card");
  text(s, "PASSAGE", MARGIN + 0.3, 1.85, 5.7, 0.3, { fontSize: 12, bold: true, color: C.text2, charSpacing: 2 });
  text(s, [
    { text: "The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in " },
    { text: "France", options: { bold: true, color: C.accent3 } },
    { text: ". They were descended from Norse raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia." },
  ], MARGIN + 0.3, 2.25, 5.7, 3.9, { fontSize: 20, lineSpacingMultiple: 1.15 });

  const x2 = 7.35, w2 = W - MARGIN - x2;
  card(s, x2, 1.65, w2, 2.2, C.background2, "answerable card");
  text(s, "ANSWERABLE", x2 + 0.3, 1.85, w2 - 0.6, 0.3, { fontSize: 12, bold: true, color: C.accent3, charSpacing: 2 });
  text(s, "In what country is Normandy located?", x2 + 0.3, 2.2, w2 - 0.6, 0.7, { fontSize: 18 });
  text(s, "France", x2 + 0.3, 2.95, w2 - 0.6, 0.6, { fontSize: 26, bold: true, color: C.accent3, fontFace: THEME.headFontFace });
  card(s, x2, 4.15, w2, 2.2, C.background2, "unanswerable card");
  text(s, "UNANSWERABLE", x2 + 0.3, 4.35, w2 - 0.6, 0.3, { fontSize: 12, bold: true, color: C.accent6, charSpacing: 2 });
  text(s, "What is France a region of?", x2 + 0.3, 4.7, w2 - 0.6, 0.7, { fontSize: 18 });
  text(s, "No answer in the passage", x2 + 0.3, 5.45, w2 - 0.6, 0.6, { fontSize: 26, bold: true, color: C.accent6, fontFace: THEME.headFontFace });
}

// Why it matters
{
  const s = content("Why this kind of question answering is useful",
    "SPEAKER: David. (0.75 min)\nWhy this kind of system? Three reasons. One: the answer is a quote from the source, so the reader can check it, which a chatbot's free-form answer doesn't give you. Two: it is the reading step inside search engines and document assistants, where a retriever finds passages and a reader like ours picks the answer out. Three, the reason SQuAD 2.0 exists: a system that always produces an answer will be confidently wrong whenever the text doesn't contain one. Saying 'I don't know' is part of being useful.");
  const w = (W - 2 * MARGIN - 2 * 0.3) / 3;
  [["You can check it", "The answer is copied word for word from the passage and shown there, so you can see exactly where it came from.", C.accent1],
   ["Used in search", "Search engines and document assistants work in two steps: first find passages that might help, then read them to pick out the answer. Our model does the second step.", C.accent1],
   ["Knowing when to stop", "A system that always gives an answer will be confidently wrong whenever the passage doesn't contain one. SQuAD 2.0 tests exactly that.", C.accent6],
  ].forEach(([head, body, color], i) => {
    const x = MARGIN + i * (w + 0.3);
    card(s, x, 1.7, w, 3.6, C.background2, `${head} card`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.95, w: 0.6, h: 0.6, fill: { color }, line: { type: "none" }, objectName: `${head} badge` });
    s.addText(String(i + 1), { x: x + 0.3, y: 1.95, w: 0.6, h: 0.6, isTextBox: true, margin: 0, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
    text(s, head, x + 0.3, 2.75, w - 0.6, 0.5, { fontSize: 22, bold: true, color, fontFace: THEME.headFontFace });
    text(s, body, x + 0.3, 3.35, w - 0.6, 1.8, { fontSize: 16 });
  });
  text(s, "Our question: how well can a small model do this, if we train it ourselves on a free GPU?", MARGIN, 5.7, W - 2 * MARGIN, 0.6, { fontSize: 20, bold: true, fontFace: THEME.headFontFace });
}

// 3. Data
{
  const s = content("SQuAD 2.0: half of the test questions have no answer",
    "SPEAKER: David. (1 min)\nSQuAD 2.0 is the standard benchmark for this task. Crowd workers wrote questions about Wikipedia paragraphs, and then wrote extra questions that look answerable but are not. We trained on 130 thousand questions. The official test set is hidden, so like most papers we report on the development set: about 12 thousand questions that the model never saw in training, half of them unanswerable. Answers are short, typically two words, and paragraphs are short too, which matters later for the long-document experiment.");
  const w = (W - 2 * MARGIN - 3 * 0.3) / 4;
  stat(s, MARGIN, 1.75, w, "130,319", "training questions, a third of them unanswerable");
  stat(s, MARGIN + (w + 0.3), 1.75, w, "11,873", "test questions (the \"development set\"), never seen in training");
  stat(s, MARGIN + 2 * (w + 0.3), 1.75, w, "50%", "of the test questions have no answer", C.accent6);
  stat(s, MARGIN + 3 * (w + 0.3), 1.75, w, "2 words", "median answer length");
  bullets(s, [
    "Paragraphs are short (about 142 word pieces), so 99% fit into the model in one go",
    "63% of questions start with \"what\"; only 1.6% ask \"why\"",
    "Two scores, explained on the next slide: exact match (EM) and F1",
  ], MARGIN, 4.3, W - 2 * MARGIN, 2.3, 20);
}

// How answers are scored
{
  const s = content("How answers are scored: exact match and F1",
    "SPEAKER: David. (1.25 min)\nTwo numbers appear on every slide from here, so here is what they mean. Exact match, EM, is all or nothing: one if the prediction equals a reference answer, zero otherwise. F1 gives partial credit by counting shared words. Take the reference 'the Nobel Prize in Physics'. Predicting exactly that scores one and one. Predicting 'Nobel Prize' is not an exact match, but both of its words are right, so precision is 100%, and it covers two of the four reference words, so recall is 50%; F1 combines them to 0.67. A wrong answer scores zero on both. Before comparing, text is lower-cased and punctuation and the words a, an, the are removed. For an unanswerable question the only correct prediction is 'no answer'. We wrote these metrics ourselves and checked them against the official SQuAD script.");
  const head = (v, align = "left") => ({ text: v, options: { bold: true, color: C.text2, fontSize: 14, align } });
  const cell = (v, opts = {}) => ({ text: v, options: { fontSize: 16, color: C.text1, ...opts } });
  s.addTable([
    [head("Correct answer"), head("Prediction"), head("EM", "right"), head("F1", "right"), head("Why")],
    [cell("the Nobel Prize in Physics"), cell("the Nobel Prize in Physics", { bold: true }), cell("1", { align: "right" }), cell("1.00", { align: "right" }), cell("exactly the same", { color: C.text2, fontSize: 14 })],
    [cell("the Nobel Prize in Physics"), cell("Nobel Prize", { bold: true }), cell("0", { align: "right" }), cell("0.67", { align: "right" }), cell("both words are right, but it has only 2 of the 4 correct words", { color: C.text2, fontSize: 14 })],
    [cell("the Nobel Prize in Physics"), cell("Chemistry", { bold: true }), cell("0", { align: "right" }), cell("0.00", { align: "right" }), cell("no shared words", { color: C.text2, fontSize: 14 })],
    [cell("(no answer)"), cell("(no answer)", { bold: true }), cell("1", { align: "right" }), cell("1.00", { align: "right" }), cell("correctly said \"no answer\"", { color: C.text2, fontSize: 14 })],
    [cell("(no answer)"), cell("Marie Curie", { bold: true }), cell("0", { align: "right" }), cell("0.00", { align: "right" }), cell("answered an unanswerable question", { color: C.text2, fontSize: 14 })],
  ], { x: MARGIN, y: 1.65, w: W - 2 * MARGIN, colW: [3.2, 3.2, 0.8, 0.9, 3.93], rowH: 0.52, border: { type: "none" }, valign: "middle",
       margin: [3, 8, 3, 0], objectName: "scoring examples" });
  const w = (W - 2 * MARGIN - 0.3) / 2;
  card(s, MARGIN, 5.0, w, 1.6, C.background2, "f1 formula card");
  text(s, "EM is all or nothing. F1 gives partial credit.", MARGIN + 0.3, 5.15, w - 0.6, 0.5, { fontSize: 17, bold: true });
  text(s, "F1 combines precision (how much of the prediction is right) and recall (how much of the correct answer was found).", MARGIN + 0.3, 5.7, w - 0.6, 0.8, { fontSize: 14, color: C.text2 });
  card(s, MARGIN + w + 0.3, 5.0, w, 1.6, C.background2, "normalisation card");
  text(s, "Small differences are ignored", MARGIN + w + 0.6, 5.15, w - 0.6, 0.5, { fontSize: 17, bold: true });
  text(s, "Capital letters, punctuation and the words a / an / the don't count. If there are several correct answers, the closest one is used.", MARGIN + w + 0.6, 5.7, w - 0.6, 0.8, { fontSize: 14, color: C.text2 });
}

// =======================================================================================
startSection("How we built it");
// 4. Pipeline
{
  const s = content("We built the whole pipeline, from raw data to a demo",
    "SPEAKER: David. (0.75 min)\nFive steps, each one a tested module in the repository. Load the data. Preprocess: turn text into numbered tokens with labels. Train: fine-tune the model on a free Colab GPU. Post-process: turn the model's scores back into an answer or 'no answer'. Evaluate: score it and compare with a published model. Our target, set in the proposal, was 65 F1. The next three slides go through the middle steps.");
  const steps = [
    ["1", "Load", "Download the SQuAD 2.0 questions and passages"],
    ["2", "Preprocess", "Cut the text into word pieces and mark where each answer starts and ends"],
    ["3", "Train", "Train DistilBERT on the questions, twice through the data"],
    ["4", "Post-process", "Turn the model's scores into an answer, or \"no answer\""],
    ["5", "Evaluate", "Score it, and compare with a larger published model"],
  ];
  const gap = 0.3, w = (W - 2 * MARGIN - 4 * gap) / 5;
  steps.forEach(([n, name, what], i) => {
    const x = MARGIN + i * (w + gap);
    card(s, x, 1.8, w, 2.9, C.background2, `step ${n}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 2.05, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { type: "none" }, objectName: `step ${n} badge` });
    s.addText(n, { x: x + 0.3, y: 2.05, w: 0.6, h: 0.6, isTextBox: true, margin: 0, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
    text(s, name, x + 0.3, 2.85, w - 0.5, 0.5, { fontSize: 20, bold: true, fontFace: THEME.headFontFace });
    text(s, what, x + 0.3, 3.4, w - 0.5, 1.2, { fontSize: 14, color: C.text2 });
  });
  card(s, MARGIN, 5.05, W - 2 * MARGIN, 1.4, C.text1, "goal card");
  text(s, [
    { text: "Our target: an F1 score of at least 65 out of 100 on the test questions", options: { bold: true, fontSize: 22, breakLine: true } },
    { text: "with a model small enough to train on a free GPU, and a demo anyone can run", options: { fontSize: 16 } },
  ], MARGIN + 0.4, 5.25, W - 2 * MARGIN - 0.8, 1.0, { color: C.background1, valign: "middle" });
}

// The model
{
  const s = content("The model: a pretrained DistilBERT plus one new layer",
    "SPEAKER: David. (1.25 min)\nWe did not train a language model from scratch. DistilBERT was already trained by its authors on a huge amount of English text to predict missing words, so it already represents meaning. It is a compressed version of BERT: six layers instead of twelve, 66 million parameters, about 40% smaller. We added one small layer on top. For every token it outputs two numbers: how likely the answer starts here, and how likely it ends here. That layer starts from random numbers, which is why the untrained model scores almost zero. Fine-tuning means training the whole thing, the old layers and the new one, on the SQuAD questions. In the picture at the bottom, the model has learned to put its highest start score and end score on 'France'.");
  const w = (W - 2 * MARGIN - 2 * 0.6) / 3;
  [["Pretrained DistilBERT", "A language model with 66 million parameters. Its authors already taught it English by having it fill in missing words in Wikipedia and books.", C.text2],
   ["+ one new layer", "We add one small layer. For every word piece it gives two scores: \"the answer starts here\" and \"the answer ends here\". At first they are random.", C.accent4],
   ["Fine-tuning", "We train the whole model on 130,319 SQuAD questions, until the two highest scores land on the answer.", C.accent1],
  ].forEach(([head, body, color], i) => {
    const x = MARGIN + i * (w + 0.6);
    card(s, x, 1.65, w, 2.3, C.background2, `${head} card`);
    text(s, head, x + 0.3, 1.8, w - 0.6, 0.5, { fontSize: 20, bold: true, color, fontFace: THEME.headFontFace });
    text(s, body, x + 0.3, 2.35, w - 0.6, 1.5, { fontSize: 14 });
    if (i < 2) text(s, "→", x + w + 0.05, 2.45, 0.5, 0.6, { fontSize: 28, bold: true, color: C.text2, align: "center", valign: "middle" });
  });
  text(s, "What the new layer learns to do", MARGIN, 4.3, 8, 0.35, { fontSize: 14, bold: true, color: C.text2 });
  const tokens = ["[CLS]", "in", "what", "country", "…", "[SEP]", "…", "a", "region", "in", "France", ".", "[SEP]"];
  const tw = (W - 2 * MARGIN) / tokens.length;
  tokens.forEach((tok, i) => {
    const x = MARGIN + i * tw, hit = tok === "France";
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.04, y: 5.35, w: tw - 0.08, h: 0.55, rectRadius: 0.05, fill: { color: hit ? C.accent1 : C.background2 }, line: { type: "none" }, objectName: `token ${i}` });
    s.addText(tok, { x: x + 0.04, y: 5.35, w: tw - 0.08, h: 0.55, isTextBox: true, margin: 0, fontSize: 14, bold: hit, color: hit ? C.background1 : C.text1, align: "center", valign: "middle" });
    if (hit) text(s, "the highest \"starts here\" and \"ends here\" scores", x - 2.6, 4.8, tw + 5.2, 0.4, { fontSize: 14, bold: true, color: C.accent1, align: "center" });
  });
  text(s, "question", MARGIN + tw, 6.0, 3 * tw, 0.3, { fontSize: 12, color: C.text2, align: "center" });
  text(s, "passage", MARGIN + 7 * tw, 6.0, 5 * tw, 0.3, { fontSize: 12, color: C.text2, align: "center" });
}

// 5. Preprocessing
{
  const s = content("Preparing the text: word pieces, windows and labels",
    "SPEAKER: David. (1.5 min)\nThe model reads the question and the passage as one sequence of at most 384 tokens. Top row: a special CLS token, the question, a separator, the passage. If the passage is too long it is cut into windows that share 128 tokens, so an answer near a cut is always complete in at least one window. The dataset gives the answer as a character position, the model predicts token positions, so we convert between them. The key trick for 'no answer': unanswerable questions are labelled with position zero, the CLS token. So 'pointing at CLS' means 'there is no answer'. We checked this step on 2,000 training questions by turning the labels back into text: 1,310 of the 1,314 answers matched exactly, that is 99.7%, the other 4 differed by a character or so, and none were lost.");
  // The input sequence
  text(s, "What the model reads: one sequence of at most 384 word pieces (\"tokens\")", MARGIN, 1.6, 10, 0.35, { fontSize: 14, bold: true, color: C.text2 });
  const blocks = [["[CLS]", 1.0, C.accent6], ["question", 2.6, C.accent4], ["[SEP]", 0.9, C.text2], ["passage", 6.63, C.accent1], ["[SEP]", 0.9, C.text2]];
  let x = MARGIN;
  blocks.forEach(([label, w, fill]) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.0, w: w - 0.06, h: 0.7, rectRadius: 0.05, fill: { color: fill }, line: { type: "none" }, objectName: `token block ${label}` });
    s.addText(label, { x, y: 2.0, w: w - 0.06, h: 0.7, isTextBox: true, margin: 0, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
    x += w;
  });
  text(s, "Pointing at [CLS], the first position, means \"no answer\"", MARGIN, 2.78, 7, 0.3, { fontSize: 12, color: C.accent6, bold: true });
  // Overlapping windows
  text(s, "A passage that is too long is cut into overlapping windows", MARGIN, 3.45, 7, 0.35, { fontSize: 14, bold: true, color: C.text2 });
  [[MARGIN, "window 1"], [MARGIN + 2.5, "window 2"], [MARGIN + 5.0, "window 3"]].forEach(([wx, label], i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: wx, y: 3.9 + i * 0.62, w: 3.9, h: 0.5, rectRadius: 0.05, fill: { color: C.accent1, transparency: 25 }, line: { type: "none" }, objectName: label });
    s.addText(label, { x: wx, y: 3.9 + i * 0.62, w: 3.9, h: 0.5, isTextBox: true, margin: 0, fontSize: 14, color: C.background1, align: "center", valign: "middle", bold: true });
  });
  text(s, "Each window holds 384 tokens and shares 128 with the next, so no answer is cut in half", MARGIN, 5.85, 8.8, 0.35, { fontSize: 14, color: C.text2 });
  // Right column
  const rx = 9.75, rw = W - MARGIN - rx;
  card(s, rx, 3.45, rw, 3.05, C.background2, "check card");
  text(s, "99.7%", rx + 0.25, 3.6, rw - 0.5, 0.9, { fontSize: 44, bold: true, color: C.accent1, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, "of the answers we checked still matched the correct answer exactly after this step: 1,310 of 1,314, from 2,000 training questions. None were lost.", rx + 0.25, 4.6, rw - 0.5, 1.7, { fontSize: 14, color: C.text2 });
}

// 6. Training (native line chart of the loss)
{
  const s = content("Training: under an hour on a free GPU",
    "SPEAKER: David. (1.25 min)\nDistilBERT is a smaller, faster version of BERT: 66 million parameters. We added one small layer that gives every token a start score and an end score, and trained all the weights for two passes over the data. Settings on the left are standard for this task. The chart is the training loss, how wrong the model is, recorded every 100 steps. It starts at 5.8, which is what a blind guess among 384 positions gives, and falls to about 0.9. The small step down in the middle is the start of the second epoch, when the model sees the same questions again. It ran in Google Colab on a T4 GPU in 44 minutes, and we ran a 50-step test first to make sure everything worked.");
  const rows = [["Model", "distilbert-base-uncased, 66 M parameters"], ["Data", "130,319 questions, 131,754 windows"], ["Passes over the data", "2 (\"epochs\"), 16,470 steps"],
                ["Windows per step", "16"], ["Learning rate", "0.00003: raised gently at first, then lowered"], ["Computer", "a free Tesla T4 GPU in Google Colab"]];
  s.addTable(rows.map(([k, v]) => [
    { text: k, options: { bold: true, color: C.text2, fontSize: 14 } }, { text: v, options: { color: C.text1, fontSize: 14 } }]),
    { x: MARGIN, y: 1.75, w: 5.2, colW: [2.05, 3.15], rowH: 0.62, border: { type: "none" }, margin: [4, 6, 4, 0], valign: "middle", objectName: "training settings" });
  const steps = training.losses.map((p) => String(p[0])), losses = training.losses.map((p) => p[1]);
  s.addChart(pres.charts.LINE, [{ name: "Training loss", labels: steps, values: losses }], chartBase({
    x: 6.0, y: 1.6, w: W - MARGIN - 6.0, h: 4.6, chartColors: [HEX.accent1], showLegend: false, lineSize: 2.25, lineDataSymbol: "none",
    catAxisLabelFrequency: 40, catAxisTitle: "training step", showCatAxisTitle: true, catAxisTitleFontSize: 12, catAxisTitleColor: HEX.dk2,
    valAxisTitle: "loss", showValAxisTitle: true, valAxisTitleFontSize: 12, valAxisTitleColor: HEX.dk2, valAxisMinVal: 0, valAxisMaxVal: 6,
    altText: "Line chart: training loss falls from 5.8 to about 0.9 over 16,470 steps",
  }));
  text(s, `The loss measures how wrong the model is. It falls from ${losses[0].toFixed(2)} to about 0.9. The small drop near step 8,300 is where the second pass begins.`, 6.0, 6.2, W - MARGIN - 6.0, 0.65, { fontSize: 14, color: C.text2 });
}

// 7. Post-processing
{
  const s = content("How the model's scores become an answer",
    "SPEAKER: David. (1.5 min)\nThe model does not output text, it outputs numbers, and this slide is how we turn them into an answer. Step one: for every word piece of the passage the model gives two numbers, a start score and an end score. High start score means 'I think the answer begins here'. Step two: any stretch of the passage could be the answer. We score a stretch by adding the start score of its first word piece and the end score of its last one, and the stretch with the highest total is the model's best answer. Step three: 'no answer' needs a score too. CLS is a special marker placed in front of every question. During training, unanswerable questions were labelled as 'the answer is at CLS', so the model learned to point there when it finds nothing. Step four: compare. Higher score wins. The example is real output from our model: 'France' has start score 6.9 and end score 7.4, total 14.3; 'no answer' totals 0.2; so it answers France. Last line: the threshold tau is a number we choose that says how much 'no answer' has to win by. At zero the higher score simply wins. At minus 1.4 the model only answers if its best answer leads by more than 1.4, so it says 'no answer' more often. Kosi will show what happens when we change it.");
  const gap = 0.25, w = (W - 2 * MARGIN - 3 * gap) / 4;
  [["1", "Score every word piece", "The model gives each word piece of the passage two numbers: a start score (how likely the answer begins here) and an end score (how likely it ends here). It learned to do this in training.", C.accent1],
   ["2", "Find the best answer", "Any stretch of the passage could be the answer. Its score is the start score of its first word piece plus the end score of its last one. The stretch with the highest score is the model's best answer.", C.accent1],
   ["3", "Score \"no answer\"", "[CLS] is a marker we put in front of every question. In training, the model learned to point at it when the passage has no answer. Its start score plus its end score is the \"no answer\" score.", C.accent6],
   ["4", "Compare the two", "If the best answer's score is higher, the model gives that answer. If the \"no answer\" score is higher, it says \"no answer\".", C.text1],
  ].forEach(([n, head, body, color], i) => {
    const x = MARGIN + i * (w + gap);
    card(s, x, 1.6, w, 2.95, C.background2, `score step ${n}`);
    text(s, `${n}.  ${head}`, x + 0.22, 1.75, w - 0.44, 0.65, { fontSize: 16, bold: true, color, fontFace: THEME.headFontFace });
    text(s, body, x + 0.22, 2.5, w - 0.44, 2.0, { fontSize: 13 });
  });
  card(s, MARGIN, 4.75, W - 2 * MARGIN, 1.85, C.background2, "example card");
  text(s, "EXAMPLE: \"In what country is Normandy located?\"   (start score + end score)", MARGIN + 0.3, 4.88, 11.4, 0.3, { fontSize: 12, bold: true, color: C.text2, charSpacing: 1 });
  text(s, [
    { text: "\"France\": 6.9 + 7.4 = 14.3", options: { bold: true, color: C.accent1 } },
    { text: "      \"no answer\": −0.1 + 0.3 = 0.2", options: { bold: true, color: C.accent6 } },
    { text: "      →  the model answers \"France\"", options: { bold: true } },
  ], MARGIN + 0.3, 5.22, W - 2 * MARGIN - 0.6, 0.45, { fontSize: 17 });
  text(s, "The threshold τ (tau) is a number we choose: how much \"no answer\" has to win by. At τ = 0 the higher score simply wins. At τ = −1.4 the model only answers if its best answer leads by more than 1.4, so it says \"no answer\" more often.",
       MARGIN + 0.3, 5.78, W - 2 * MARGIN - 0.6, 0.75, { fontSize: 14, color: C.text2 });
}

// How we know the code is right
{
  const s = content("How we know the code is right",
    "SPEAKER: David. (0.75 min)\nA wrong label or a wrong metric gives you a number that looks fine and means nothing, so we built checks in. Every module has unit tests, 133 in total, that run in seconds without internet or a GPU, using a tiny model. Every setting lives in one config file, never in a notebook. Before the 44-minute run we ran a 50-step test to confirm the loss falls and checkpoints save. Our evaluation script reproduces RoBERTa's published score exactly and refuses to run if it disagrees with the official metric. And the README lists eight commands with the number each should print, so anyone can reproduce our results.");
  const w = (W - 2 * MARGIN - 2 * 0.3) / 3;
  stat(s, MARGIN, 1.65, w, "133", "automatic tests of our code; they run in about 10 seconds");
  stat(s, MARGIN + w + 0.3, 1.65, w, "79.9 / 82.9", "RoBERTa's published EM / F1. Our scoring code gets exactly the same.", C.accent2);
  stat(s, MARGIN + 2 * (w + 0.3), 1.65, w, "8", "numbered steps in our README rebuild every number in this talk", C.accent3);
  bullets(s, [
    "Every setting lives in one file, and a misspelt setting stops the program instead of being ignored",
    "Before the real training, a one-minute trial run checked that the model learns and that its progress is saved",
    "Our scores must match the official SQuAD 2.0 scoring program, or our script refuses to continue",
    "Anyone can repeat our work: the code, the trained model and all results are public",
  ], MARGIN, 4.1, W - 2 * MARGIN, 2.5, 17);
}

// =======================================================================================
startSection("Results");
// 8. Main results
{
  const s = content(`We beat our target: ${f1(ours.f1)} F1 on the test questions`,
    "SPEAKER: David. (1.5 min)\nThree numbers. Left: the same DistilBERT before our training scores 4.6 F1, essentially nothing, so everything the model can do comes from fine-tuning. Middle: our model, 67.4 F1 and 64.1 exact match, above the 65 target. Right: a RoBERTa model published by deepset, about twice the size, scores 82.9. We measured it with our own code and got exactly its published number, which is how we know our evaluation is right. The table splits the score: our model is better on answerable questions than on unanswerable ones, and the gap to RoBERTa is biggest on unanswerable questions. Remember that; it is the theme of the second half.");
  const w = (W - 2 * MARGIN - 2 * 0.3) / 3;
  stat(s, MARGIN, 1.65, w, "4.6", "Before: F1 of the same DistilBERT before we trained it (first 1,000 questions)", C.text2);
  stat(s, MARGIN + w + 0.3, 1.65, w, f1(ours.f1), `After: F1 of our trained model (exact match: ${f1(ours.exact)}). Our target was 65.`, C.accent1);
  stat(s, MARGIN + 2 * (w + 0.3), 1.65, w, f1(roberta.f1), `For comparison: F1 of RoBERTa, a larger published model (exact match: ${f1(roberta.exact)})`, C.accent2);
  const head = (t) => ({ text: t, options: { bold: true, color: C.text2, fontSize: 14, align: t === "F1 by kind of question" ? "left" : "right" } });
  const cell = (t, color = C.text1, bold = false) => ({ text: t, options: { color, fontSize: 16, bold, align: "right" } });
  s.addTable([
    [head("F1 by kind of question"), head("All questions"), head("Questions with an answer"), head("Questions with no answer")],
    [{ text: "DistilBERT (ours)", options: { fontSize: 16, bold: true, color: C.accent1 } }, cell(f1(ours.f1)), cell(f1(ours.HasAns_f1)), cell(f1(ours.NoAns_f1))],
    [{ text: "RoBERTa (deepset)", options: { fontSize: 16, bold: true, color: C.accent2 } }, cell(f1(roberta.f1)), cell(f1(roberta.HasAns_f1)), cell(f1(roberta.NoAns_f1))],
    [{ text: "How far we are behind", options: { fontSize: 16, bold: true, color: C.text2 } }, cell(f1(roberta.f1 - ours.f1), C.text2), cell(f1(roberta.HasAns_f1 - ours.HasAns_f1), C.text2),
     cell(f1(roberta.NoAns_f1 - ours.NoAns_f1), C.accent6, true)],
  ], { x: MARGIN, y: 3.95, w: W - 2 * MARGIN, colW: [3.6, 2.4, 3.0, 3.03], rowH: 0.5, border: { type: "none" }, valign: "middle",
       margin: [4, 8, 4, 0], objectName: "answerable vs unanswerable" });
  text(s, "The gap is largest on questions with no answer: knowing when not to answer is where the smaller model loses most.",
       MARGIN, 6.1, W - 2 * MARGIN, 0.6, { fontSize: 16, color: C.text2 });
}

// 9. Trade-off (native bar chart, indexed to RoBERTa)
{
  const pct = (a, b) => Math.round((100 * a) / b);
  const s = content("The trade-off: 81% of the score at about half the size",
    "SPEAKER: David. (1 min)\nWhy use the smaller model at all? This chart sets RoBERTa to 100% on three measures. Our model reaches 81% of its F1, with 54% of the size, 266 megabytes instead of 496, and it answers a question in 31 milliseconds on a laptop CPU instead of 57, so it is 1.8 times as fast. For a demo that has to run on a free CPU, that matters. That is the trade-off DistilBERT was designed for, and our numbers match it. Hand over to Kosi: now, what exactly does it get wrong?");
  s.addChart(pres.charts.BAR, [
    { name: "DistilBERT (ours)", labels: ["F1 score", "Model size", "Time per question"],
      values: [pct(ours.f1, roberta.f1), pct(effOurs.size_mb, effBase.size_mb), pct(effOurs.median_ms, effBase.median_ms)] },
    { name: "RoBERTa (deepset)", labels: ["F1 score", "Model size", "Time per question"], values: [100, 100, 100] },
  ], chartBase({ x: MARGIN, y: 2.0, w: 7.6, h: 4.5, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60, showValue: true,
                 dataLabelPosition: "outEnd", dataLabelFormatCode: "0\"%\"", valAxisMinVal: 0, valAxisMaxVal: 120, valAxisLabelFormatCode: "0\"%\"",
                 altText: "Bar chart: our model as a percentage of RoBERTa on F1, size and CPU time" }));
  text(s, "Each pair of bars sets RoBERTa to 100%. For size and time, lower is better.", MARGIN, 1.5, 7.6, 0.4, { fontSize: 14, color: C.text2 });
  const rx = 8.7, rw = W - MARGIN - rx;
  [[`${f1(ours.f1)} vs ${f1(roberta.f1)}`, "F1 on the test questions"],
   [`${Math.round(effOurs.size_mb)} vs ${Math.round(effBase.size_mb)} MB`, `${effOurs.parameters_millions.toFixed(0)} million vs ${effBase.parameters_millions.toFixed(0)} million parameters`],
   [`${Math.round(effOurs.median_ms)} vs ${Math.round(effBase.median_ms)} ms`, "to answer one question on a laptop"]].forEach(([big, small], i) => {
    card(s, rx, 1.7 + i * 1.62, rw, 1.42, C.background2, "trade-off card");
    text(s, big, rx + 0.25, 1.8 + i * 1.62, rw - 0.5, 0.6, { fontSize: 24, bold: true, fontFace: THEME.headFontFace, valign: "middle" });
    text(s, small, rx + 0.25, 2.45 + i * 1.62, rw - 0.5, 0.5, { fontSize: 14, color: C.text2 });
  });
}

// =======================================================================================
startSection("What it gets wrong");
// 10. Divider
{
  const s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: section });
  s.addText("What does it get wrong, and why?", { placeholder: "title" });
  s.addText("One number says how good a model is. It doesn't say what is wrong with it.", { placeholder: "body" });
  s.addNotes("SPEAKER: Kosi takes over. (0.25 min)\nOne F1 number tells you how good a model is, not what is wrong with it. In this part: tuning the threshold, sorting all 4,258 errors by type, reading 60 of them by hand, and testing on whole documents.");
}

// 11. Threshold sweep (native line chart)
{
  const o = sweep[OURS], b = sweep[BASE];
  const s = content("Being more cautious helps, a little",
    "SPEAKER: Kosi. (1.5 min)\nThis is the dial from David's slide. The horizontal axis is the threshold tau; to the right the model answers more often, to the left it abstains more. Because we saved every question's scores, we could try every threshold without running the model again. Both curves peak slightly left of zero: both models are a bit too eager to answer. For ours the best value is minus 1.4. Important detail: choosing the threshold on the same questions you score on flatters the result, that gives 68.3. So we split the 35 articles in half, chose the threshold on one half and scored the other. The honest number is 68.1, a gain of 0.7. Small, because the curve is flat at the top. And it is a trade: we gain on unanswerable questions and lose on answerable ones.");
  // A scatter chart with lines, because it has a real numeric x axis (a line chart treats
  // the thresholds as category names, and negative labels wrap).
  s.addChart(pres.charts.SCATTER, [
    { name: "tau", values: o.curve.tau },
    { name: "DistilBERT (ours)", values: o.curve.f1 },
    { name: "RoBERTa (deepset)", values: b.curve.f1 },
  ], chartBase({ x: MARGIN, y: 2.15, w: 7.7, h: 4.4, lineSize: 2.25, lineDataSymbol: "none",
                 catAxisTitle: "no-answer threshold τ  (higher = answers more often)", showCatAxisTitle: true, catAxisTitleFontSize: 12,
                 catAxisTitleColor: HEX.dk2, valAxisTitle: "F1", showValAxisTitle: true, valAxisTitleFontSize: 12, valAxisTitleColor: HEX.dk2,
                 valAxisMinVal: 40, valAxisMaxVal: 90, catAxisMinVal: -8, catAxisMaxVal: 8, catAxisMajorUnit: 2,
                 altText: "Line chart: F1 against the no-answer threshold for both models, peaking just below zero" }));
  text(s, "τ (tau) is the no-answer threshold: how much \"no answer\" has to win by before the model says it. A lower τ makes the model say \"no answer\" more often.", MARGIN, 1.45, 7.7, 0.65, { fontSize: 14, color: C.text2 });
  const rx = 8.8, rw = W - MARGIN - rx;
  card(s, rx, 1.65, rw, 2.05, C.background2, "tuned f1 card");
  text(s, `${o.at_tau_0.f1.toFixed(1)} → ${o.cross_validated.scores.f1.toFixed(1)}`, rx + 0.25, 1.8, rw - 0.5, 0.9, { fontSize: 36, bold: true, color: C.accent1, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, `our F1 when τ is lowered to ${o.best_tau.toFixed(1)}. We chose τ on half of the articles and measured on the other half.`, rx + 0.25, 2.75, rw - 0.5, 0.85, { fontSize: 14, color: C.text2 });
  bullets(s, [
    `Choosing τ on the same questions we measure on would give ${o.at_best_tau.f1.toFixed(1)}, which flatters the model`,
    `It is a trade: F1 on answerable questions falls from ${o.at_tau_0.HasAns_f1.toFixed(0)} to ${o.at_best_tau.HasAns_f1.toFixed(0)}; on unanswerable ones it rises from ${o.at_tau_0.NoAns_f1.toFixed(0)} to ${o.at_best_tau.NoAns_f1.toFixed(0)}`,
    `RoBERTa gains about as little: ${b.at_tau_0.f1.toFixed(1)} to ${b.cross_validated.scores.f1.toFixed(1)}`,
  ], rx, 4.0, rw, 2.5, 15);
}

// Threshold: the trade-off
{
  const o = sweep[OURS];
  const s = content("The threshold trades one kind of error for the other",
    "SPEAKER: Kosi. (1 min)\nWhy is the gain so small? This is our model only, split by kind of question. The solid line is F1 on answerable questions: it rises as the threshold goes up, because the model answers more. The dashed line is unanswerable questions: it falls, for the same reason. They cross near minus one. Moving the threshold never improves both; it only chooses where on these two lines you sit. At zero we get 72 and 63. At the best overall value we get 66 and 70. To really improve, the model's scores themselves have to separate the two kinds of question better, and that takes a better model, not a better threshold.");
  s.addChart(pres.charts.SCATTER, [
    { name: "tau", values: o.curve.tau },
    { name: "Answerable questions", values: o.curve.HasAns_f1 },
    { name: "Unanswerable questions", values: o.curve.NoAns_f1 },
  ], chartBase({ x: MARGIN, y: 1.55, w: 7.7, h: 5.0, lineSize: 2.25, lineDataSymbol: "none", chartColors: [HEX.accent1, HEX.accent6],
                 catAxisTitle: "no-answer threshold τ  (higher = answers more often)", showCatAxisTitle: true, catAxisTitleFontSize: 12,
                 catAxisTitleColor: HEX.dk2, valAxisTitle: "F1 (our model)", showValAxisTitle: true, valAxisTitleFontSize: 12, valAxisTitleColor: HEX.dk2,
                 valAxisMinVal: 0, valAxisMaxVal: 100, catAxisMinVal: -8, catAxisMaxVal: 8, catAxisMajorUnit: 2,
                 altText: "Line chart: as the threshold rises, F1 on answerable questions rises and F1 on unanswerable questions falls" }));
  const rx = 8.8, rw = W - MARGIN - rx;
  const row = (label, a, b) => [{ text: label, options: { fontSize: 15, bold: true, color: C.text1 } },
                                { text: a, options: { fontSize: 15, align: "right", color: C.accent1, bold: true } },
                                { text: b, options: { fontSize: 15, align: "right", color: C.accent6, bold: true } }];
  s.addTable([
    [{ text: "Our F1 at", options: { fontSize: 13, bold: true, color: C.text2 } }, { text: "Answerable", options: { fontSize: 13, bold: true, color: C.text2, align: "right" } },
     { text: "Unanswerable", options: { fontSize: 13, bold: true, color: C.text2, align: "right" } }],
    row("τ = 0 (default)", o.at_tau_0.HasAns_f1.toFixed(1), o.at_tau_0.NoAns_f1.toFixed(1)),
    row(`τ = ${o.best_tau.toFixed(1)} (best)`, o.at_best_tau.HasAns_f1.toFixed(1), o.at_best_tau.NoAns_f1.toFixed(1)),
  ], { x: rx, y: 1.7, w: rw, colW: [1.6, 1.05, rw - 2.65], rowH: 0.55, border: { type: "none" }, valign: "middle", margin: [3, 4, 3, 0], objectName: "trade-off table" });
  bullets(s, [
    `At τ = 0 the model answers ${o.at_tau_0.answered_pct.toFixed(0)}% of all questions; at τ = ${o.best_tau.toFixed(1)} it answers ${o.at_best_tau.answered_pct.toFixed(0)}%`,
    "No value of τ improves both lines at the same time",
    "To gain more, the model itself must get better at telling the two kinds of question apart",
  ], rx, 3.75, rw, 2.7, 15);
}

// 12. Error types (native bar chart)
{
  const eo = errors[OURS], eb = errors[BASE];
  const types = ["false answer", "missed answer", "too long", "too short", "partial overlap", "wrong span"];
  const decisionShare = Math.round((100 * (eo.outcomes["false answer"] + eo.outcomes["missed answer"])) / eo.errors);
  const s = content(`${decisionShare}% of our errors are about whether to answer, not where`,
    "SPEAKER: Kosi. (1.25 min)\nWe sorted every one of our model's 4,258 wrong predictions into one type, automatically. A false answer means the question was unanswerable and the model answered anyway: that is more than half of all errors. A missed answer is the opposite: there was an answer and the model said 'no answer'. Together those two are three quarters of the errors. Only the bottom four bars are about finding the wrong span. Look at the orange bars: RoBERTa makes half as many false answers. Most of the gap between the two models is right there, in the top bar.");
  const label = (t) => t.charAt(0).toUpperCase() + t.slice(1);
  s.addChart(pres.charts.BAR, [
    { name: "DistilBERT (ours)", labels: types.map(label), values: types.map((t) => eo.outcomes[t]) },
    { name: "RoBERTa (deepset)", labels: types.map(label), values: types.map((t) => eb.outcomes[t]) },
  ], chartBase({ x: MARGIN, y: 1.55, w: 7.9, h: 5.0, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 45, showValue: true,
                 dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0", valAxisLabelFormatCode: "#,##0", valAxisMinVal: 0, valAxisMaxVal: 2500,
                 catAxisOrientation: "maxMin", valAxisHidden: true, valGridLine: { style: "none" },
                 altText: "Bar chart: number of errors of each type for both models; false answers are the largest group" }));
  const rx = 9.0, rw = W - MARGIN - rx;
  card(s, rx, 1.65, rw, 2.1, C.background2, "decision errors card");
  const decisions = eo.outcomes["false answer"] + eo.outcomes["missed answer"];
  text(s, `${decisionShare}%`, rx + 0.25, 1.8, rw - 0.5, 0.9, { fontSize: 40, bold: true, color: C.accent6, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, `${eo.outcomes["false answer"].toLocaleString("en-US")} false answers + ${eo.outcomes["missed answer"].toLocaleString("en-US")} missed answers = ${decisions.toLocaleString("en-US")} of our ${eo.errors.toLocaleString("en-US")} errors`, rx + 0.25, 2.75, rw - 0.5, 0.85, { fontSize: 14, color: C.text2 });
  bullets(s, [
    `False answer: the question had no answer, but the model gave one (${eo.error_shares_pct["false answer"].toFixed(0)}% of errors)`,
    `Missed answer: there was an answer, but the model said \"no answer\" (${eo.error_shares_pct["missed answer"].toFixed(0)}%)`,
    `RoBERTa gives half as many false answers (${eb.outcomes["false answer"].toLocaleString("en-US")})`,
    "Picking the wrong words is the smaller problem (the four short bars)",
  ], rx, 4.0, rw, 2.6, 14);
}

// 13. Why: hand-labelled examples
{
  const s = content("60 questions analysed: it matches wording, not meaning",
    "SPEAKER: Kosi. (1.75 min)\nThe automatic types say what happened, not why. So we took a random sample of 60 errors, covering every type, and read each one. Three patterns. First, swapped detail: the question changes one thing. The passage says Jacksonville's Filipino community has ties to the Navy; the question asks about the Arab population. The model sees the same sentence shape and answers 'United States Navy'. Second, changed meaning: the question keeps the names but changes what happened, in our sample always to the opposite. The passage says Genghis Khan united these tribes, the question asks which tribes he fought against, and the model gives the same tribes. Together those two explain 24 of the 27 false answers we read. Third, a reassuring one: all nine 'too long or too short' errors were answers a person would accept, like 'Karakorum in Mongolia' for 'Karakorum'. So exact match is a bit too strict. Only one error out of 60 needed real reasoning.");
  const w = (W - 2 * MARGIN - 2 * 0.3) / 3;
  const cards = [
    ["Swapped detail", "The question changes one detail. 14 of the 27 false answers we read.", "Passage: \"Much of Jacksonville's Filipino community has ties to the United States Navy.\"",
     "Q: What military branch does much of Jacksonville's Arab population have ties to?", "Our model: United States Navy (should be: no answer)", C.accent6],
    ["Changed meaning", "The question changes what happened. 10 of the 27 false answers we read.", "Passage: \"Genghis Khan united the Mongol and Turkic tribes of the steppes.\"",
     "Q: Which tribes did Genghis Khan fight against?", "Our model: Mongol and Turkic (should be: no answer)", C.accent6],
    ["Right, but marked wrong", "A person would accept it. All 9 \"too long / too short\" errors we read.", "Passage: \"...the Mongol capital from Karakorum in Mongolia to Khanbaliq...\"",
     "Q: Where had the Mongol capital been before Kublai moved it?", "Our model: Karakorum in Mongolia (correct answer: Karakorum)", C.accent3],
  ];
  cards.forEach(([head, count, passage, q, model, color], i) => {
    const x = MARGIN + i * (w + 0.3);
    card(s, x, 1.65, w, 4.85, C.background2, `${head} card`);
    text(s, head, x + 0.3, 1.85, w - 0.6, 0.45, { fontSize: 20, bold: true, color, fontFace: THEME.headFontFace });
    text(s, count, x + 0.3, 2.35, w - 0.6, 0.6, { fontSize: 14, bold: true, color: C.text2 });
    text(s, passage, x + 0.3, 3.1, w - 0.6, 1.1, { fontSize: 14, italic: true });
    text(s, q, x + 0.3, 4.3, w - 0.6, 1.0, { fontSize: 14 });
    text(s, model, x + 0.3, 5.4, w - 0.6, 0.9, { fontSize: 15, bold: true, color });
  });
}

// 14. Question words (native bar chart)
{
  const eo = errors[OURS], eb = errors[BASE];
  const order = ["how many", "when", "how much", "which", "who", "where", "what", "how", "why"];
  const get = (model, t) => model.by_question_type_answerable.find((r) => r.question_type === t).f1;
  const len = (model, t) => model.by_answer_length.find((r) => r.answer_length === t).f1;
  const s = content("Numbers and dates are easy; explanations are hard",
    "SPEAKER: Kosi. (0.75 min)\nF1 on answerable questions, by the question's first word. At the top, 'how many' and 'when': the answer is a number or a date, short, with an obvious shape, and the model scores about 85. At the bottom, 'how' and 'why': the answer is an explanation, long, with fuzzy edges, and the score drops to around 60. RoBERTa, in orange, has the same pattern at a higher level, so this is partly the task, not just our model. The same thing shows up by answer length: steady up to four words, then it falls to 51 for answers of ten words or more.");
  s.addChart(pres.charts.BAR, [
    { name: "DistilBERT (ours)", labels: order, values: order.map((t) => get(eo, t)) },
    { name: "RoBERTa (deepset)", labels: order, values: order.map((t) => get(eb, t)) },
  ], chartBase({ x: MARGIN, y: 1.55, w: 7.9, h: 5.0, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 40, showValue: true,
                 dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMinVal: 0, valAxisMaxVal: 100, catAxisOrientation: "maxMin",
                 valAxisHidden: true, valGridLine: { style: "none" },
                 altText: "Bar chart: F1 on answerable questions by question word for both models; why and how are lowest" }));
  const rx = 9.0, rw = W - MARGIN - rx;
  card(s, rx, 1.65, rw, 2.1, C.background2, "why card");
  text(s, `${get(eo, "how many").toFixed(0)} vs ${get(eo, "why").toFixed(0)}`, rx + 0.25, 1.8, rw - 0.5, 0.9, { fontSize: 40, bold: true, color: C.accent1, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, "our F1 on \"how many\" questions against \"why\" questions", rx + 0.25, 2.75, rw - 0.5, 0.85, { fontSize: 14, color: C.text2 });
  bullets(s, [
    "Questions whose answer is a number or a date are the easiest",
    "\"Why\" and \"how\" ask for explanations, which are longer and have less clear edges",
    "RoBERTa shows the same pattern, just with higher scores",
  ], rx, 4.05, rw, 2.4, 15);
}

// Answer length
{
  const eo = errors[OURS], eb = errors[BASE];
  const buckets = ["1 word", "2 words", "3-4 words", "5-9 words", "10+ words"];
  const len = (model, b) => model.by_answer_length.find((r) => r.answer_length === b).f1;
  const n = (b) => eo.by_answer_length.find((r) => r.answer_length === b).questions;
  const s = content("The longer the correct answer, the lower the score",
    "SPEAKER: Kosi. (0.75 min)\nThe same story by answer length. For answers of one to four words, F1 is steady, around 72 to 74. From five words it drops, and for ten words or more it is 51, while RoBERTa holds 73. Long answers are rare in training, most are one or two words, and every extra word is another chance to cut the span too early or too late. That is also where the 'too long' and 'too short' errors come from. When we read nine of those, all nine were answers a person would accept, so part of this drop is the metric being strict.");
  s.addChart(pres.charts.BAR, [
    { name: "DistilBERT (ours)", labels: buckets, values: buckets.map((b) => len(eo, b)) },
    { name: "RoBERTa (deepset)", labels: buckets, values: buckets.map((b) => len(eb, b)) },
  ], chartBase({ x: MARGIN, y: 1.55, w: 7.9, h: 5.0, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55, showValue: true,
                 dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMinVal: 0, valAxisMaxVal: 100, valAxisTitle: "F1, answerable questions",
                 showValAxisTitle: true, valAxisTitleFontSize: 12, valAxisTitleColor: HEX.dk2,
                 altText: "Bar chart: F1 by length of the reference answer for both models; it falls for answers of five words or more" }));
  const rx = 9.0, rw = W - MARGIN - rx;
  card(s, rx, 1.65, rw, 2.1, C.background2, "length card");
  text(s, `${len(eo, "3-4 words").toFixed(0)} → ${len(eo, "10+ words").toFixed(0)}`, rx + 0.25, 1.8, rw - 0.5, 0.9, { fontSize: 40, bold: true, color: C.accent1, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, "our F1 drops from 74 for answers of 3-4 words to 51 for answers of 10 words or more", rx + 0.25, 2.75, rw - 0.5, 0.85, { fontSize: 14, color: C.text2 });
  bullets(s, [
    `Long answers are rare: ${n("10+ words")} of ${eo.by_answer_length.reduce((a, r) => a + r.questions, 0).toLocaleString("en-US")} answerable questions have 10+ words`,
    `${(eo.outcomes["too long"] + eo.outcomes["too short"]).toLocaleString("en-US")} of our errors are answers in the right place that are slightly too long or too short`,
    "We read 9 of those and a person would accept all 9. Example: \"Karakorum in Mongolia\" when the correct answer is \"Karakorum\"",
  ], rx, 4.0, rw, 2.6, 14);
}

// 15. Long documents (native bar chart)
{
  const titles = ["Amazon_rainforest", "Normans", "Oxygen"];
  const names = titles.map((t) => `${t.replace("_", " ")} (${longdoc[t].windows_per_question.toFixed(0)} windows)`);
  const s = content("Extension: on whole articles, it stops saying \"no answer\"",
    "SPEAKER: Kosi. (1.25 min)\nSQuAD hands the model the one paragraph that contains the answer. A real user has a whole document. So for our extension we rebuilt three complete Wikipedia articles from their paragraphs and asked all 907 of their questions against the whole article, which the model reads in 12 to 28 overlapping windows. Grey is the normal setting with the right paragraph given. Blue: the whole article with our normal rule. F1 drops by 16 to 23 points. Interesting part: the model still finds the right place, 73 to 94% of the time. What breaks is abstaining. Our rule compared the best span from any window with the lowest no-answer score from any window, and with that many windows, extremes turn up by chance, so it almost always answers. Green is our fix: let each window judge itself and let the most confident window decide. That wins back 6 to 7 points with no retuning, and changes nothing on single paragraphs.");
  s.addChart(pres.charts.BAR, [
    { name: "Given the right paragraph (the normal setting)", labels: names, values: titles.map((t) => longdoc[t].paragraph_mode.f1) },
    { name: "Whole article, original rule", labels: names, values: titles.map((t) => longdoc[t].rules.global.at_tau_0.f1) },
    { name: "Whole article, our new rule", labels: names, values: titles.map((t) => longdoc[t].rules.per_window.at_tau_0.f1) },
  ], chartBase({ x: MARGIN, y: 2.1, w: 7.9, h: 4.45, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55, showValue: true,
                 dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 80,
                 chartColors: ["9AA7B5", HEX.accent1, HEX.accent3], valAxisTitle: "F1", showValAxisTitle: true, valAxisTitleFontSize: 12,
                 valAxisTitleColor: HEX.dk2, altText: "Bar chart: F1 on three whole articles under three settings" }));
  const rx = 9.0, rw = W - MARGIN - rx;
  text(s, "We joined all the paragraphs of three Wikipedia articles and asked each article's questions against the whole text.", MARGIN, 1.45, 7.9, 0.6, { fontSize: 14, color: C.text2 });
  const ng = titles.map((t) => longdoc[t].rules.global.at_tau_0.NoAns_f1);
  const np = titles.map((t) => longdoc[t].paragraph_mode.NoAns_f1);
  card(s, rx, 1.65, rw, 2.1, C.background2, "unanswerable collapse card");
  text(s, `${Math.min(...ng).toFixed(0)} to ${Math.max(...ng).toFixed(0)}`, rx + 0.25, 1.8, rw - 0.5, 0.9, { fontSize: 40, bold: true, color: C.accent6, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, `F1 on unanswerable questions, down from ${Math.min(...np).toFixed(0)} to ${Math.max(...np).toFixed(0)}: the model almost never says "no answer"`, rx + 0.25, 2.75, rw - 0.5, 0.85, { fontSize: 14, color: C.text2 });
  bullets(s, [
    "It still finds the right place: its answer is in the correct paragraph 73% to 94% of the time",
    "Our new rule wins back 6 to 7 F1 points (next slide)",
    "A threshold τ chosen for single paragraphs is wrong for long articles",
  ], rx, 4.05, rw, 2.4, 15);
}

// Long documents: why, and the fix
{
  const titles = ["Amazon_rainforest", "Normans", "Oxygen"];
  const gain = titles.map((t) => longdoc[t].rules.per_window.at_tau_0.f1 - longdoc[t].rules.global.at_tau_0.f1);
  const s = content("Why it breaks, and our fix: let each window judge itself",
    "SPEAKER: Kosi. (1.25 min)\nHere is the mechanism, with made-up numbers to show it. Window A covers the wrong part of the article. It finds a span that scores 10, but its own no-answer score is 14, so window A itself says 'the answer is not here'. Window B covers the right part: its span scores only 6, but its no-answer score is 0, so it is confident. Our original rule, built for single paragraphs, takes the best span from anywhere, that is 10 from window A, and compares it with the lowest no-answer score from anywhere, that is 0 from window B. Ten beats zero, so it confidently returns the wrong window's span. The per-window rule asks each window for its own margin and trusts the most confident one: window B, the right answer. On real articles this gained 6 to 7 F1 points, and on single paragraphs the two rules are identical, so nothing else changed.");
  const head = (v, align = "left") => ({ text: v, options: { bold: true, color: C.text2, fontSize: 14, align } });
  const cell = (v, opts = {}) => ({ text: v, options: { fontSize: 16, color: C.text1, align: "right", ...opts } });
  s.addTable([
    [head("A made-up example"), head("Best answer found", "right"), head("No-answer score", "right"), head("What this window thinks", "right")],
    [cell("Window A (wrong part of the article)", { align: "left", bold: true }), cell("10"), cell("14"), cell("the answer is not here", { color: C.accent6 })],
    [cell("Window B (right part of the article)", { align: "left", bold: true }), cell("6"), cell("0"), cell("the answer is here", { color: C.accent3 })],
  ], { x: MARGIN, y: 1.65, w: W - 2 * MARGIN, colW: [4.6, 1.7, 2.3, 3.43], rowH: 0.55, border: { type: "none" }, valign: "middle", margin: [3, 8, 3, 0],
       objectName: "window illustration" });
  const w = (W - 2 * MARGIN - 0.3) / 2;
  card(s, MARGIN, 3.65, w, 2.9, C.background2, "global rule card");
  text(s, "Original rule (\"global\")", MARGIN + 0.3, 3.8, w - 0.6, 0.45, { fontSize: 19, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
  text(s, "It takes the highest answer score from any window (10, from A) and the lowest no-answer score from any window (0, from B).", MARGIN + 0.3, 4.35, w - 0.6, 0.9, { fontSize: 15 });
  text(s, "10 beats 0, so it returns window A's answer: wrong, and confident.", MARGIN + 0.3, 5.3, w - 0.6, 0.9, { fontSize: 16, bold: true, color: C.accent6 });
  card(s, MARGIN + w + 0.3, 3.65, w, 2.9, C.background2, "per-window rule card");
  text(s, "Our new rule (\"per-window\")", MARGIN + w + 0.6, 3.8, w - 0.6, 0.45, { fontSize: 19, bold: true, color: C.accent3, fontFace: THEME.headFontFace });
  text(s, "Each window compares its own answer with its own no-answer score. The window that is most sure decides.", MARGIN + w + 0.6, 4.35, w - 0.6, 0.9, { fontSize: 15 });
  text(s, `Window B decides, so we get the right answer. On real articles: +${Math.min(...gain).toFixed(1)} to +${Math.max(...gain).toFixed(1)} F1.`, MARGIN + w + 0.6, 5.3, w - 0.6, 0.9, { fontSize: 16, bold: true, color: C.accent3 });
}

// =======================================================================================
startSection("Demo and conclusion");
// 16. Demo
{
  const COLAB = "https://colab.research.google.com/github/Davidfaleye360/AI-Question-Answering-System/blob/main/notebooks/04_demo_colab.ipynb";
  const s = content("Live demo",
    "SPEAKER: David runs the demo, Kosi narrates. (3 min)\nBEFORE THE TALK: the demo takes about two minutes to start, so start it in advance. Click the blue button on this slide (or open notebook 04 in Colab), run both cells, wait for the line 'Running on public URL', open that gradio.live link in a tab, and leave the tab open. Have the backup video ready in case the network fails.\n1. Normans passage, 'In what country is Normandy located?': France, highlighted, with the candidate table.\n2. Same passage, 'What is France a region of?': our model says 'Normandy' (a false answer, the swapped-detail error from the analysis). Switch to RoBERTa: it abstains. This is the gap from the results slide, live.\n3. Oxygen passage, 'Which element is the most abundant in the universe by mass?': our model abstains at the default threshold. Drag the slider up past minus 1.1: it answers 'oxygen', which is wrong. This is what the threshold protects against.\n4. Paste something long, or take a question from the audience. Point out the line that says how many windows it read.");
  const lw = 4.55;
  card(s, MARGIN, 1.75, lw, 4.65, C.background2, "open the demo card");
  text(s, "Open the demo", MARGIN + 0.3, 1.95, lw - 0.6, 0.45, { fontSize: 22, bold: true, fontFace: THEME.headFontFace });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MARGIN + 0.3, y: 2.6, w: lw - 0.6, h: 0.75, rectRadius: 0.08, fill: { color: C.accent1 }, line: { type: "none" },
                                              hyperlink: { url: COLAB, tooltip: "Open notebook 04 in Google Colab" }, objectName: "colab button" });
  s.addText([{ text: "Open in Google Colab  ↗", options: { hyperlink: { url: COLAB, tooltip: "Open notebook 04 in Google Colab" }, color: "FFFFFF", bold: true } }],
            { x: MARGIN + 0.3, y: 2.6, w: lw - 0.6, h: 0.75, isTextBox: true, margin: 0, fontSize: 18, align: "center", valign: "middle" });
  // Numbers are typed, not auto-numbered: some viewers restart automatic numbering on every line.
  ["1.  Run both cells", "2.  Wait for \"Running on public URL\"", "3.  Open the gradio.live link it prints"].forEach((line, k) => {
    text(s, line, MARGIN + 0.3, 3.6 + k * 0.45, lw - 0.6, 0.4, { fontSize: 15 });
  });
  text(s, "It takes about two minutes to start, so we started it before the talk. The model runs on Colab's free CPU.", MARGIN + 0.3, 5.2, lw - 0.6, 1.0, { fontSize: 14, color: C.text2 });
  const rx = 5.5, rw = W - MARGIN - rx;
  const steps = [
    ["1", "An answerable question", "The answer is highlighted in the passage, with the top candidates and their scores"],
    ["2", "A trick question", "Our model is fooled; RoBERTa correctly says \"no answer\""],
    ["3", "The threshold slider", "Move τ and watch the answer flip, without re-running the model"],
    ["4", "A long passage, or your question", "Too long to read at once, so the model reads it in windows"],
  ];
  steps.forEach(([n, head, what], i) => {
    const y = 1.75 + i * 1.2;
    card(s, rx, y, rw, 1.05, C.background2, `demo step ${n}`);
    s.addShape(pres.shapes.OVAL, { x: rx + 0.25, y: y + 0.22, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { type: "none" }, objectName: `demo step ${n} badge` });
    s.addText(n, { x: rx + 0.25, y: y + 0.22, w: 0.6, h: 0.6, isTextBox: true, margin: 0, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
    text(s, head, rx + 1.1, y + 0.12, rw - 1.3, 0.4, { fontSize: 18, bold: true });
    text(s, what, rx + 1.1, y + 0.55, rw - 1.3, 0.4, { fontSize: 14, color: C.text2 });
  });
}

// 17. Limitations
{
  const s = content("What it can't do: put two sentences together",
    "SPEAKER: Kosi. (1.25 min)\nAn example David found while testing the demo, not from the dataset. A passage about restoring a fresco: Professor Aris argues the bright colours are a side effect of the chemical solvent, and two sentences later the board votes to stop the chemical treatments. A person connects the two. The right answer, in green on the slide, is that Aris argued the bright colours came from the chemical solvent and not from the original painting. But no single piece of text says that. Our model returns the clause next to the question's words, 'they allowed the physical scraping to proceed', and RoBERTa, the stronger model, makes the same mistake. So this is a limit of the method: an extractive model copies one span and matches wording. Fixing it needs a different kind of model, one that generates answers. Other limits on the right: it is too willing to answer, we tested one model size on Wikipedia text only, and long documents are slower and less accurate.");
  card(s, MARGIN, 1.65, 6.6, 4.95, C.background2, "fresco example card");
  text(s, "AN EXAMPLE FROM OUTSIDE THE DATASET", MARGIN + 0.3, 1.85, 6.0, 0.3, { fontSize: 12, bold: true, color: C.text2, charSpacing: 2 });
  text(s, [
    { text: "\"...Professor Aris maintained that the sudden brilliance was merely an artifact of the chemical solvent... While the museum board ultimately voted to halt the chemical treatments, " },
    { text: "they allowed the physical scraping to proceed under strict supervision", options: { bold: true, color: C.accent6 } },
    { text: "...\"" },
  ], MARGIN + 0.3, 2.25, 6.0, 1.85, { fontSize: 16, italic: true, lineSpacingMultiple: 1.1 });
  text(s, "Q: Why did the museum board decide to stop using the chemical treatments?", MARGIN + 0.3, 4.2, 6.0, 0.6, { fontSize: 15, bold: true });
  text(s, "Both models answer with the words in red.", MARGIN + 0.3, 4.85, 6.0, 0.3, { fontSize: 14, bold: true, color: C.accent6 });
  text(s, "Right answer: because Professor Aris argued that the bright colours came from the chemical solvent, not from the original painting.", MARGIN + 0.3, 5.2, 6.0, 0.75, { fontSize: 14, bold: true, color: C.accent3 });
  text(s, "A reader has to connect two sentences to see that. No single stretch of text says it.", MARGIN + 0.3, 5.98, 6.0, 0.5, { fontSize: 13, color: C.text2 });
  const rx = 7.65, rw = W - MARGIN - rx;
  bullets(s, [
    "It can only copy one stretch of text. It cannot combine sentences or explain.",
    "It is too willing to answer: half of its errors are answers to unanswerable questions",
    "We tested one model size, on Wikipedia text only",
    "On long documents it is slower (up to 0.6 seconds per question) and less accurate",
    "Next steps: a second model that only decides \"answerable or not\", and a search step that finds the right paragraphs first",
  ], rx, 1.8, rw, 4.6, 16);
}

// 18. Conclusion
{
  const o = sweep[OURS];
  const s = content("What we learned",
    "SPEAKER: David. (1 min)\nFour takeaways. One: a small model trained for 44 minutes on a free GPU gets to 67.4 F1, 68.1 with a tuned threshold, above our target. Two: it costs 15 points against a model twice the size, and buys half the size and almost twice the speed. Three: the hard part of SQuAD 2.0 is not finding answers, it is knowing when there is none. That showed up three times: in the error counts, in the threshold experiment, and most strongly on long documents. Four: everything is reproducible. Every number on these slides comes from a command in the README, and the model and demo are public.");
  const items = [
    [f1(ours.f1), "F1 from a small model that we trained in under an hour on a free GPU (" + o.cross_validated.scores.f1.toFixed(1) + " after tuning τ)", C.accent1],
    ["54%", "of RoBERTa's size and 1.8 times as fast, at a cost of 15.5 F1 points", C.accent1],
    [`${Math.round((100 * (errors[OURS].outcomes["false answer"] + errors[OURS].outcomes["missed answer"])) / errors[OURS].errors)}%`, "of errors are about whether to answer at all. The same weakness showed up in the error counts, the threshold and long documents.", C.accent6],
    ["100%", "repeatable: every number comes from a command in our README, and the model and demo are public", C.accent3],
  ];
  const w = (W - 2 * MARGIN - 0.3) / 2;
  items.forEach(([big, small, color], i) => {
    const x = MARGIN + (i % 2) * (w + 0.3), y = 1.65 + Math.floor(i / 2) * 2.5;
    card(s, x, y, w, 2.2, C.background2, "takeaway card");
    text(s, big, x + 0.3, y + 0.2, 2.2, 1.8, { fontSize: 44, bold: true, color, fontFace: THEME.headFontFace, valign: "middle" });
    text(s, small, x + 2.6, y + 0.25, w - 2.9, 1.7, { fontSize: 16, valign: "middle" });
  });
}

// 19. Questions
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: section });
  s.addText("Questions?", { placeholder: "title" });
  s.addText([
    { text: "Code, results and report: github.com/Davidfaleye360/AI-Question-Answering-System", options: { breakLine: true, fontSize: 18 } },
    { text: "Model: huggingface.co/Ingsyling/distilbert-squad2", options: { breakLine: true, fontSize: 18 } },
    { text: "David Faleye · Kosisochukwu Ogbuanya", options: { fontSize: 16 } },
  ], { placeholder: "body" });
  s.addImage({ path: path.join(__dirname, "qr_colab.png"), x: 10.55, y: 2.05, w: 2.1, h: 2.1, altText: "QR code that opens the demo notebook in Google Colab" });
  s.addText("Run the demo in Colab", { x: 10.55, y: 4.25, w: 2.1, h: 0.35, isTextBox: true, margin: 0, fontSize: 14, color: C.background2, align: "center" });
  s.addNotes("SPEAKER: both. (remaining time)\nLikely questions and short answers:\n- Why not a bigger model? It was the trade-off we wanted to measure; RoBERTa shows what size buys.\n- Why is the threshold gain so small? The curve is flat near its peak; the model's scores for 'answer' and 'no answer' overlap a lot.\n- How do you know the evaluation is right? Our code reproduces RoBERTa's published 79.9 / 82.9 exactly and matches the official metric.\n- Did you tune on the test set? Only the threshold, and we report the cross-validated number for that reason.\n- What would you do next? A separate answerability classifier, and retrieval for long documents.");
}

// ---------------------------------------------------------------------------------------
(async () => {
  const out = path.join(__dirname, "presentation.pptx");
  await pres.writeFile({ fileName: out });
  const themeScript = process.env.PPTX_APPLY_THEME;
  if (themeScript) {
    const { applyTheme } = require(themeScript);
    await applyTheme(out, THEME);
  } else {
    console.log("Note: PPTX_APPLY_THEME is not set, so the deck keeps PowerPoint's default theme colours.");
  }
  console.log("Wrote", path.relative(ROOT, out));
})();
