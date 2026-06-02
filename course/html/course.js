/* Shared course helpers: glossary tooltips + check-yourself quizzes.
   No dependencies. Safe to include on every page. */
(() => {
  // --- Glossary: term -> short kid-friendly definition (also lives in glossary.html) ---
  const GLOSSARY = {
    "model": "A bag of numbers plus a rule for turning input into output.",
    "weights": "The numbers a model multiplies its inputs by. Together with the biases, these are the values it learns.",
    "weight": "One number a model multiplies an input by — a single adjustable value it learns.",
    "bias": "An extra number a model adds on after multiplying by weights — it shifts the result up or down.",
    "biases": "The extra numbers a model adds on after multiplying by weights — they shift each result up or down.",
    "parameter": "One adjustable number inside a model. A tiny model has 2; PRAGMA has a billion.",
    "parameters": "The adjustable numbers inside a model. A tiny model has 2; PRAGMA has a billion.",
    "training": "Showing a model examples and nudging its numbers until its guesses get good.",
    "loss": "One number that measures how wrong the model's guesses are. Lower is better.",
    "gradient": "The slope of the loss — which way to nudge a weight to make the loss smaller.",
    "gradient descent": "Repeatedly nudging every weight downhill on the loss, a tiny step at a time.",
    "learning rate": "How big each nudge is. Too big overshoots; too small crawls.",
    "optimiser": "The helper that does the nudging step for you (e.g. SGD or Adam).",
    "SGD": "Stochastic Gradient Descent — the simplest optimiser: step opposite the gradient.",
    "Adam": "A smarter optimiser that adapts the step size per weight.",
    "MSE": "Mean Squared Error — average of (guess − truth)². The loss for predicting numbers.",
    "cross-entropy": "The loss for predicting a choice (like which word). Low when confident and right.",
    "token": "One chunk of input after splitting it up — often a word.",
    "tokenise": "Split input into chunks and give each chunk an ID number.",
    "embedding": "A list of numbers that represents a word (or token). Similar words get similar lists.",
    "embeddings": "Lists of numbers that represent words. Similar words get similar lists.",
    "vector": "Just a list of numbers, e.g. [0.3, -1.2, 0.4].",
    "dot product": "Multiply matching slots of two lists and add them up. Big = the lists are similar.",
    "softmax": "Turns a row of scores into percentages that add up to 100%.",
    "attention": "Letting every token look at every other token and blend in the ones that matter.",
    "self-attention": "Attention where the tokens look at each other (not at some other sequence).",
    "query": "In attention, 'what am I looking for?' — one of the Q/K/V projections.",
    "key": "In attention, 'what do I offer to others?' — one of the Q/K/V projections.",
    "value": "In attention, 'what do I actually pass along if matched?' — the V in Q/K/V.",
    "multi-head": "Running attention several times in parallel so each 'head' can specialise.",
    "feed-forward": "A small MLP applied to each token: Linear → activation → Linear.",
    "FFN": "Feed-forward network — a small MLP inside each Transformer block.",
    "MLP": "Multi-Layer Perceptron — Linear → ReLU → Linear. The simplest neural network.",
    "ReLU": "An activation that clips negatives to 0. Adds the 'bend' that lets nets fit curves.",
    "GELU": "A smoother cousin of ReLU used inside Transformers.",
    "activation": "The non-linear step (ReLU/GELU) that lets stacked layers learn curves.",
    "encoder": "The stack of Transformer blocks that turns tokens into context-aware vectors.",
    "Transformer": "The model made of stacked (attention + feed-forward) blocks. Powers BERT, GPT, PRAGMA.",
    "layer": "One (attention + feed-forward) block. BERT-style PRAGMA stacks 18 of them.",
    "residual": "A shortcut that adds a block's input to its output so deep stacks train well.",
    "LayerNorm": "A step that keeps each token's numbers in a tidy range so training is stable.",
    "logits": "The raw scores the model outputs before softmax turns them into percentages.",
    "MLM": "Masked Language Modelling — hide a token and train the model to fill it back in.",
    "masked language modelling": "Hide a token and train the model to guess it. Free training data!",
    "pre-training": "Training on lots of unlabelled data with the fill-in-the-blank game first.",
    "fine-tuning": "Lightly adjusting a pre-trained model for a specific new task.",
    "freeze": "Lock a part of the model so training leaves its weights unchanged.",
    "foundation model": "One big pre-trained model reused for many tasks. PRAGMA is one.",
    "recall": "Of all the real positives, what fraction did we catch? Key when one class is rare.",
    "RNN": "Recurrent Neural Network — the older, sequential model attention replaced.",
    "epoch": "One full pass through all the training data.",
    "batch": "A small group of examples processed together in one training step."
  };

  function slug(term) {
    return term.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  }

  function linkGlossary() {
    const wrap = document.querySelector(".wrap");
    if (!wrap) return;
    // Skip these containers entirely.
    const SKIP = new Set(["PRE", "CODE", "A", "BUTTON", "H1", "H2", "H3", "H4",
                          "SCRIPT", "STYLE", "SELECT", "OPTION", "FIGCAPTION"]);
    const skipClass = ["lesson-toc", "lesson-links", "quiz", "mission", "roadmap"];
    // Longest terms first so "gradient descent" beats "gradient".
    const terms = Object.keys(GLOSSARY).sort((a, b) => b.length - a.length);
    const used = new Set();              // one wrap per term per page

    const walker = document.createTreeWalker(wrap, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        for (let el = node.parentElement; el && el !== wrap; el = el.parentElement) {
          if (SKIP.has(el.tagName)) return NodeFilter.FILTER_REJECT;
          if (el.classList && skipClass.some(c => el.classList.contains(c)))
            return NodeFilter.FILTER_REJECT;
          if (el.classList && el.classList.contains("term"))
            return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });

    const targets = [];
    let n;
    while ((n = walker.nextNode())) targets.push(n);

    for (const node of targets) {
      for (const term of terms) {
        if (used.has(term)) continue;
        const re = new RegExp("\\b" + term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\b",
                              term[0] === term[0].toUpperCase() ? "" : "i");
        const m = node.nodeValue.match(re);
        if (!m) continue;
        const idx = m.index;
        const before = node.nodeValue.slice(0, idx);
        const hit = node.nodeValue.slice(idx, idx + m[0].length);
        const after = node.nodeValue.slice(idx + m[0].length);
        const a = document.createElement("a");
        a.className = "term";
        a.href = "glossary.html#" + slug(term);
        a.dataset.def = GLOSSARY[term];
        a.textContent = hit;
        const frag = document.createDocumentFragment();
        if (before) frag.appendChild(document.createTextNode(before));
        frag.appendChild(a);
        if (after) frag.appendChild(document.createTextNode(after));
        node.parentNode.replaceChild(frag, node);
        used.add(term);
        break;  // node is now detached; move on
      }
    }
  }

  // --- Quiz renderer ---
  // renderQuiz(containerId, [{ q, options:[...], answer: <index>, explain }])
  window.renderQuiz = function (id, questions) {
    const box = document.getElementById(id);
    if (!box) return;
    box.classList.add("quiz");
    let html = '<h3>✅ Check yourself</h3>';
    questions.forEach((item, qi) => {
      html += `<div class="q"><div class="q-text">${qi + 1}. ${item.q}</div>`;
      item.options.forEach((opt, oi) => {
        html += `<button class="opt" data-q="${qi}" data-o="${oi}">${opt}</button>`;
      });
      html += `<div class="explain" id="${id}-ex-${qi}"></div></div>`;
    });
    box.innerHTML = html;

    box.querySelectorAll(".opt").forEach(btn => {
      btn.addEventListener("click", () => {
        const qi = +btn.dataset.q, oi = +btn.dataset.o;
        const item = questions[qi];
        const sibs = box.querySelectorAll(`.opt[data-q="${qi}"]`);
        sibs.forEach(b => { b.disabled = true; });
        const ex = document.getElementById(`${id}-ex-${qi}`);
        if (oi === item.answer) {
          btn.classList.add("correct");
          btn.innerHTML += '<span class="mark">✓</span>';
          ex.className = "explain show ok";
          ex.textContent = "Correct! " + (item.explain || "");
        } else {
          btn.classList.add("wrong");
          btn.innerHTML += '<span class="mark">✗</span>';
          sibs[item.answer].classList.add("correct");
          ex.className = "explain show no";
          ex.textContent = "Not quite. " + (item.explain || "");
        }
      });
    });
  };

  // --- Predict-before-reveal ---
  // predict(id, { q, options:[...], correct:<idx>, reveal:"..." })
  window.predict = function (id, cfg) {
    const box = document.getElementById(id);
    if (!box) return;
    box.classList.add("predict");
    box.innerHTML =
      `<div class="p-head"></div><div class="p-q">${cfg.q}</div>` +
      cfg.options.map((o, i) => `<button class="p-opt" data-i="${i}">${o}</button>`).join("") +
      `<div class="p-reveal" id="${id}-rev"></div>`;
    box.querySelectorAll(".p-opt").forEach(btn => {
      btn.addEventListener("click", () => {
        const i = +btn.dataset.i;
        box.querySelectorAll(".p-opt").forEach((b, j) => {
          b.disabled = true;
          if (j === cfg.correct) b.classList.add("correct");
        });
        btn.classList.add("chosen");
        const hit = i === cfg.correct;
        const rev = document.getElementById(`${id}-rev`);
        rev.className = "p-reveal show";
        rev.innerHTML =
          `<span class="verdict ${hit ? "hit" : "miss"}">${hit ? "🎯 You called it!" : "🤔 Good guess — here's what actually happens:"}</span> ` +
          cfg.reveal;
      });
    });
  };

  // --- Inline checkpoint (slim single MCQ) ---
  // check(id, { q, options:[...], answer:<idx>, explain:"..." })
  window.check = function (id, cfg) {
    const box = document.getElementById(id);
    if (!box) return;
    box.classList.add("check");
    box.innerHTML =
      `<div class="c-q">${cfg.q}</div>` +
      cfg.options.map((o, i) => `<button class="c-opt" data-i="${i}">${o}</button>`).join("") +
      `<div class="c-ex" id="${id}-ex"></div>`;
    box.querySelectorAll(".c-opt").forEach(btn => {
      btn.addEventListener("click", () => {
        const i = +btn.dataset.i;
        const opts = box.querySelectorAll(".c-opt");
        opts.forEach(b => { b.disabled = true; });
        const ex = document.getElementById(`${id}-ex`);
        if (i === cfg.answer) { btn.classList.add("correct"); }
        else { btn.classList.add("wrong"); opts[cfg.answer].classList.add("correct"); }
        ex.className = "c-ex show";
        ex.textContent = (i === cfg.answer ? "✓ " : "✗ ") + (cfg.explain || "");
      });
    });
  };

  // gradeTryit — turn each ".tryit" capstone into a self-graded check.
  // Reveal each answer, mark ✓/✗; PASS shows only when more than 80% are correct.
  function gradeTryit() {
    document.querySelectorAll(".tryit").forEach(function (box) {
      var items = Array.prototype.slice.call(box.querySelectorAll("ol > li"));
      if (!items.length) return;
      var state = items.map(function () { return null; });  // null | true | false
      var NEED = Math.floor(items.length * 0.8) + 1;        // smallest count that is > 80%
      var res = document.createElement("div");
      res.className = "ti-result";
      res.setAttribute("aria-live", "polite");
      box.appendChild(res);
      function update() {
        var graded = state.filter(function (s) { return s !== null; }).length;
        var correct = state.filter(function (s) { return s === true; }).length;
        if (graded < items.length) {
          res.className = "ti-result";
          res.innerHTML = '<span class="ti-prog">Reveal each answer and mark it — grade all ' + items.length +
            ' to see your result (' + graded + '/' + items.length + ' done). Aim for more than 80% to pass.</span>';
          return;
        }
        var pct = Math.round(correct / items.length * 100);
        var pass = pct > 80;
        res.className = "ti-result " + (pass ? "pass" : "fail");
        res.innerHTML = '<span class="ti-badge">' + (pass ? "✅ PASS" : "🔁 Keep trying") + '</span>' +
          '<span class="ti-score">' + correct + ' / ' + items.length + ' correct (' + pct + '%)</span>' +
          (pass ? '' : '<span class="ti-hint">You need more than 80% (' + NEED + ' of ' + items.length +
            '). Re-read the ones you missed, then mark them again.</span>');
      }
      function mkBtn(kind, label) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "ti-gbtn ti-" + kind;
        b.textContent = label;
        b.setAttribute("aria-pressed", "false");
        return b;
      }
      items.forEach(function (li, i) {
        var g = document.createElement("div");
        g.className = "ti-grade";
        var q = document.createElement("span");
        q.className = "ti-q";
        q.textContent = "Did you get it?";
        var yes = mkBtn("yes", "✓ Yes");
        var no = mkBtn("no", "✗ Not quite");
        function set(v) {
          state[i] = v;
          yes.classList.toggle("on", v === true);
          no.classList.toggle("on", v === false);
          yes.setAttribute("aria-pressed", v === true ? "true" : "false");
          no.setAttribute("aria-pressed", v === false ? "true" : "false");
          update();
        }
        yes.addEventListener("click", function () { set(true); });
        no.addEventListener("click", function () { set(false); });
        g.appendChild(q); g.appendChild(yes); g.appendChild(no);
        li.appendChild(g);
      });
      update();
    });
  }

  document.addEventListener("DOMContentLoaded", linkGlossary);
  document.addEventListener("DOMContentLoaded", gradeTryit);
})();
