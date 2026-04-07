# Awesome Algorithms

> ✨ A growing, execution-driven algorithm repository that aims to turn abstract names on a blueprint into runnable Python MVPs, readable notes, and source-level algorithm breakdowns.

If you have ever opened an algorithm repository and thought:

- "This list is impressive, but where is the runnable code?"
- "I know the algorithm name, but what does it actually do step by step?"
- "Can I learn the idea, the history, the complexity, and the implementation path in one place?"

This repo is built for exactly that.

## Why This Repo Exists 🚀

Most algorithm repositories do one of these well, but not all of them together:

- They collect famous algorithm names.
- They show a few code snippets.
- They explain theory in isolation.
- They focus only on interview-style implementations.

This project is trying to go much further.

The goal is to build a large-scale algorithm library where each completed algorithm becomes its own small, useful learning unit:

- 📚 clear enough for readers who want intuition
- 🧠 deep enough for readers who want theory
- 🛠 runnable enough for readers who want code
- 🔬 structured enough for readers who want to inspect the real algorithm path instead of a black-box API call

In short:

This is not meant to be just a list of algorithms.  
It is meant to become an algorithm workspace you can actually learn from, run, and reuse.

## What Is Inside Right Now 🗺️

The repository is organized around a single authoritative blueprint:

- [`Docs/Stage0_Blueprint.md`](Docs/Stage0_Blueprint.md)

That blueprint tracks a deduplicated cross-discipline algorithm universe covering:

- Mathematics
- Physics
- Computer Science

Each algorithm is intended to become its own folder under:

- `Algorithms/领域-分类-编号-名字/`

When an item is completed, it is no longer just a checkbox in the blueprint.  
It becomes a concrete artifact with code, notes, and implementation detail.

## How To Use This Repo 🧭

### 1. Start from the blueprint

Open the blueprint first:

- [`Docs/Stage0_Blueprint.md`](Docs/Stage0_Blueprint.md)

It tells you:

- what algorithms are covered
- how the repo is categorized
- what research fields each algorithm must complete
- what the expected MVP and source-level breakdown standards are

### 2. Jump into a finished algorithm folder

Completed algorithms live in:

- `Algorithms/领域-分类-编号-名字/`

Each finished folder is meant to include:

- `README.md`
- `demo.py`
- `meta.json`

That gives you a compact workflow:

1. Read the algorithm note
2. Run the MVP
3. Inspect the core steps
4. Compare idea vs implementation

### 3. Run the Python MVP

This repo uses `uv` to manage Python dependencies.

Setup:

```bash
uv sync
```

Then run a finished algorithm demo:

```bash
cd "Algorithms/<领域-分类-编号-名字>"
uv run python demo.py
```

This matters because the repo is not trying to be "just theory."  
If an algorithm is marked complete, you should be able to actually run something.

### 4. Learn at two levels

Each finished algorithm is designed to support two reading modes:

- ⚡ Quick mode: read the summary, complexity, example, and run the demo
- 🧩 Deep mode: inspect the source-level breakdown and understand the algorithm in 3-10 concrete steps

That second mode is especially important here.

The project does **not** want to stop at:

- "import package"
- "call one function"
- "done"

If a third-party library can solve something in one line, this repo still tries to trace the real algorithm structure underneath it.

## What You Can Get From This Repo When The Blueprint Is Fully Finished 🎁

If the blueprint is completed end-to-end, this repo becomes much more than a reference list.

You get:

- 🧠 A broad algorithm map across math, physics, and CS in one place
- 🗂 A consistent folder-per-algorithm structure
- 📖 For each algorithm: concept, history, context, meaning, dependencies, and applications
- ⏱ Complexity analysis with both theory and practical cost awareness
- 🧪 Runnable Python MVPs, not just pseudocode
- 🔍 Source-level algorithm decomposition, so readers can see how an implementation is really built
- 🧱 A reusable base for teaching, self-study, demos, and experimentation
- 🧰 A practical "algorithm lab" where you can open a folder and immediately start learning or testing

At that point, this repo becomes useful for several very different people:

- students who want intuition + examples
- engineers who want runnable baselines
- researchers who want structured algorithm notes
- builders who want to prototype from classical methods upward

## Why It Is Worth Reading Even Before It Is Finished 👀

Because this repo is being built in a highly explicit way.

You can already see:

- how the blueprint defines quality
- how completed algorithms are shaped
- how MVPs are expected to be runnable
- how black-box usage is pushed back into readable algorithm steps

That means even the unfinished version is not random accumulation.  
It is a visible construction process.

And that makes it interesting.

You are not just reading a repository.  
You are watching a large algorithm system being assembled.

## Project Philosophy ⚙️

The intended standard for each completed algorithm is:

- readable
- runnable
- structured
- comparable
- reusable

The ideal finished folder should answer all of these:

- What is this algorithm for?
- Why was it invented?
- What does it depend on?
- How expensive is it?
- How do I run a minimum working demo?
- What are the actual implementation steps?
- Where does it sit in the larger algorithm landscape?

If a folder can answer those well, it is no longer just a code sample.  
It becomes a real learning object.

## Repo Structure At A Glance 📦

```text
Docs/
  Stage0_Blueprint.md
  researches/
Algorithms/
  <领域-分类-编号-名字>/
    README.md
    demo.py
    meta.json
scripts/
  generate_stage0_blueprint.py
```

## Final Hook 🌟

There are many places to find algorithm names.  
There are fewer places to find algorithms that are:

- organized at scale
- explained in plain language
- backed by runnable MVPs
- traced down to real implementation steps

That is the bet this repo is making.

If it reaches the end of its blueprint, it will not just be a catalog.  
It will be a serious, hands-on algorithm knowledge base.
