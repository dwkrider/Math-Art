// The UI layer: controls and readouts, and nothing else.
//
// This file knows about sliders, selects and text boxes. It does NOT
// know any mathematics and never touches three.js. A page declares the
// controls it wants as data -- the same way math_art's operators
// declare their properties -- and gets back an object of current values
// plus a change callback.
//
// Declaring them as data rather than writing the DOM by hand is what
// removes the per-page wiring loop that every module had its own copy
// of, formatters and all.

const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

/** Build a control panel from a declaration.
 *
 *  Each control is an object:
 *
 *    { id, type: 'range', label, min, max, step, value, format }
 *    { id, type: 'select', label, options: [[value, text], ...], value }
 *    { id, type: 'text', label, value, placeholder }
 *    { id, type: 'toggle', label, value }
 *    { id, type: 'button', label, onClick }       // not a value
 *    { type: 'row', controls: [...] }             // side by side
 *
 *  `format` turns the value into what the readout beside the label
 *  shows; without one the value is shown as it stands. Returns
 *  { values, set, get, element, enable }.
 */
export function mountControls(root, declaration, onChange) {
  const values = {};
  const nodes = new Map();
  const formats = new Map();

  const build = (spec, parent) => {
    if (spec.type === 'row' || spec.type === 'group') {
      const box = el(spec.type === 'row' ? 'div' : 'span',
                     spec.type === 'row' ? 'belt-row' : 'belt-snaps');
      for (const child of spec.controls) build(child, box);
      parent.append(box);
      return;
    }
    if (spec.type === 'button') {
      const button = el('button', 'stage-action', spec.label);
      button.type = 'button';
      button.id = spec.id;
      button.addEventListener('click', () => spec.onClick(values, button));
      parent.append(button);
      nodes.set(spec.id, button);
      return;
    }
    if (spec.type === 'toggle') {
      const label = el('label', 'belt-check');
      const input = el('input');
      input.type = 'checkbox';
      input.id = spec.id;
      input.checked = !!spec.value;
      values[spec.id] = !!spec.value;
      input.addEventListener('change', () => {
        values[spec.id] = input.checked;
        onChange(spec.id, values);
      });
      label.append(input, document.createTextNode(' ' + spec.label));
      parent.append(label);
      nodes.set(spec.id, input);
      return;
    }

    const label = el('label', 'belt-field'
      + (spec.type === 'range' ? ' belt-slider' : ''));
    const head = el('span');
    head.append(document.createTextNode(spec.label + ' '));
    const out = el('output');
    out.id = `${spec.id}-out`;
    out.setAttribute('for', spec.id);
    head.append(out);
    label.append(head);

    let input;
    if (spec.type === 'select') {
      input = el('select');
      for (const [value, text] of spec.options) {
        const option = el('option', null, text);
        option.value = value;
        if (value === spec.value) option.selected = true;
        input.append(option);
      }
    } else {
      input = el('input');
      input.type = spec.type === 'range' ? 'range' : 'text';
      if (spec.type === 'range') {
        input.min = spec.min;
        input.max = spec.max;
        input.step = spec.step ?? 1;
      }
      if (spec.placeholder) input.placeholder = spec.placeholder;
      input.value = String(spec.value);
    }
    input.id = spec.id;
    if (spec.title) input.setAttribute('aria-label', spec.title);

    const read = () => (spec.type === 'range' ? Number(input.value) : input.value);
    values[spec.id] = read();
    formats.set(spec.id, spec.format || ((v) => (spec.type === 'range' ? String(v) : '')));
    out.textContent = formats.get(spec.id)(values[spec.id]);

    const event = spec.type === 'select' || spec.type === 'text' ? 'change' : 'input';
    input.addEventListener(event, () => {
      values[spec.id] = read();
      out.textContent = formats.get(spec.id)(values[spec.id]);
      onChange(spec.id, values);
    });
    // a text box should also commit on Enter, not only on blur
    if (spec.type === 'text') {
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
      });
    }
    label.append(input);
    parent.append(label);
    nodes.set(spec.id, input);
  };

  for (const spec of declaration) build(spec, root);

  return {
    values,
    element: root,
    node: (id) => nodes.get(id),
    get: (id) => values[id],
    /** Set a value from code -- used when a preset drives the other
     *  controls -- without firing a change for each one. */
    set(id, value, { silent = true } = {}) {
      const node = nodes.get(id);
      if (!node) return;
      if (node.type === 'checkbox') { node.checked = !!value; values[id] = !!value; }
      else { node.value = String(value); values[id] = node.type === 'range' ? Number(value) : node.value; }
      const out = document.getElementById(`${id}-out`);
      if (out && formats.has(id)) out.textContent = formats.get(id)(values[id]);
      if (!silent) onChange(id, values);
    },
    enable(id, on) {
      const node = nodes.get(id);
      if (node) node.disabled = !on;
    },
  };
}

/** Call `fn` no more often than `ms`, on the trailing edge.
 *
 *  A slider fires an event per step of a drag, and a build that takes a
 *  second would then queue a second per step. UI concern, so it lives
 *  here rather than in anyone's maths. */
export function debounce(fn, ms = 120) {
  let timer = null;
  return (...args) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => { timer = null; fn(...args); }, ms);
  };
}

/** A line of text under the controls, for whatever the build reports. */
export function readout(node) {
  return {
    set(text) { node.textContent = text; },
    clear() { node.textContent = ''; },
  };
}
