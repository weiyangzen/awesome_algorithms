# Awesome Algorithms

> 🌍 **Language / 言語 / 语言**: [中文](#中文) | [日本語](#日本語) | [English](#english)

> ✨ A large-scale algorithm repository that turns algorithm names into runnable Python MVPs, readable notes, and source-level breakdowns.

---

## 中文

> 想找的不只是“算法名字大全”，而是一套**能读、能跑、能拆、能复用**的算法资料库？  
> 这个仓库就是朝这个方向做的。

### 为什么值得看 🚀

很多算法仓库只解决了其中一部分问题：

- 有列表，但没有可运行代码
- 有代码，但没有解释为什么这样做
- 有理论，但没有最小可验证 Demo
- 有 API 调用，但看不到算法到底是怎么一步步工作的

这个仓库想把这些东西合起来。

它的目标不是“再做一份算法清单”，而是做成一个真正可用的算法工作区：

- 📚 你能快速理解算法在干什么
- 🧠 你能看到它的背景、意义、复杂度与依赖链
- 🛠 你能直接运行 Python MVP
- 🔍 你能看到源码级 3-10 步拆解，而不是停留在黑盒包调用

### 怎么用 🧭

#### 1. 先看蓝图

从这里开始：

- [`Docs/Stage0_Blueprint.md`](Docs/Stage0_Blueprint.md)

它定义了：

- 仓库覆盖了哪些算法
- 数学 / 物理 / 计算机三大领域如何分类
- 每个算法需要补哪些研究字段
- 每个算法文件夹应交付什么

如果你想看最终交付总览，再看：

- [`Docs/Stage0_Final_Summary.md`](Docs/Stage0_Final_Summary.md)

#### 2. 再看具体算法目录

每个已完成算法都对应一个目录：

- `Algorithms/领域-分类-编号-名字/`

目录里通常包含：

- `README.md`
- `demo.py`
- `meta.json`

这意味着你可以直接：

1. 读说明
2. 跑 Demo
3. 看实现
4. 理解它的算法路径

#### 3. 跑 Python MVP

这个仓库用 `uv` 管理 Python 环境。

先同步依赖：

```bash
uv sync
```

再进入某个算法目录运行：

```bash
cd "Algorithms/<领域-分类-编号-名字>"
uv run python demo.py
```

### 蓝图完成后你能得到什么 🎁

现在 Stage0 已经完成，所以这些收益不是“未来式”，而是你已经可以开始使用的东西：

- 🗺 一张跨数学、物理、计算机的算法地图
- 🗂 一套统一的“每个算法一个目录”的结构
- 📖 每个算法都有概念、背景、意义、复杂度、依赖、应用说明
- 🧪 每个算法都有可运行的 Python MVP
- 🔬 每个算法都尽量给出源码级拆解，而不是只写“调用某个库函数”
- 🧱 一个适合学习、教学、演示、原型验证的算法资料库

### 这个仓库适合谁 👀

- 想系统补算法的人
- 想把“听过名字”变成“真正理解”的人
- 想直接运行最小 demo 的工程师
- 想做讲解、课程、分享、演示的人
- 想从经典算法继续往上做原型的人

### 一句话总结 🌟

你在这里拿到的，不是一份“算法名单”。  
你拿到的是一套**可以阅读、可以运行、可以拆解、可以继续扩展**的算法基础设施。

---

## 日本語

> 「アルゴリズム名の一覧」だけではなく、**読める・動く・分解できる・再利用できる**リポジトリが欲しいなら、この repo はそのために作られています。

### なぜこのリポジトリが面白いのか 🚀

多くのアルゴリズム系リポジトリは、次のどれか一つには強いです。

- 名前の網羅
- コード例
- 理論解説
- 実装スニペット

でも、この repo はそこを一つにまとめようとしています。

目標は、各アルゴリズムを「名前」ではなく、**小さくても完成した学習単位**として提供することです。

- 📚 何をするアルゴリズムかが分かる
- 🧠 背景・意味・計算量・依存関係が分かる
- 🛠 Python MVP を実際に動かせる
- 🔍 ブラックボックス API ではなく、3〜10 ステップの実装分解まで追える

### 使い方 🧭

#### 1. まず Blueprint を見る

入口はここです：

- [`Docs/Stage0_Blueprint.md`](Docs/Stage0_Blueprint.md)

ここに以下がまとまっています：

- 何のアルゴリズムをカバーしているか
- 数学 / 物理 / 計算機科学の分類
- 各アルゴリズムに必要な研究項目
- 完成フォルダに求められる成果物

最終的な集計を見たい場合は：

- [`Docs/Stage0_Final_Summary.md`](Docs/Stage0_Final_Summary.md)

#### 2. 完成済みアルゴリズムのフォルダへ行く

完成済みアルゴリズムは次にあります：

- `Algorithms/分野-分類-番号-名前/`

通常、各フォルダには以下があります：

- `README.md`
- `demo.py`
- `meta.json`

つまり、次の流れで使えます：

1. 説明を読む
2. Demo を動かす
3. 実装を見る
4. アルゴリズムの流れを理解する

#### 3. Python MVP を実行する

この repo は `uv` で Python 環境を管理しています。

最初に：

```bash
uv sync
```

その後、任意のアルゴリズムフォルダで：

```bash
cd "Algorithms/<分野-分類-番号-名前>"
uv run python demo.py
```

### Blueprint 完了後に何が得られるか 🎁

Stage0 はすでに完了しているので、今この repo から得られるものは次の通りです。

- 🗺 数学・物理・計算機を横断するアルゴリズム地図
- 🗂 アルゴリズムごとに統一されたフォルダ構造
- 📖 背景・意味・計算量・依存・応用まで含む説明
- 🧪 実際に動く Python MVP
- 🔬 ソースレベルの手順分解
- 🧱 学習・教育・デモ・試作に使えるアルゴリズム基盤

### どんな人に向いているか 👀

- アルゴリズムを体系的に学びたい人
- 名前だけでなく中身まで理解したい人
- 最小実装をすぐ動かしたい人
- 教材・講義・発表の素材を探している人
- 古典アルゴリズムから実験的に広げたい人

### ひとことで言うと 🌟

これは単なる一覧ではありません。  
**読めて、動かせて、分解できて、再利用できるアルゴリズム基盤**です。

---

## English

> Looking for more than an algorithm list?  
> This repository is designed to be **readable, runnable, explainable, and reusable**.

### Why This Repo Matters 🚀

Most algorithm repositories are strong in one dimension:

- name collection
- code snippets
- theory notes
- implementation demos

This repo tries to combine all of them.

The goal is not to build another long list of algorithm names.  
The goal is to turn each completed algorithm into a compact but useful learning unit:

- 📚 easy to read
- 🧠 rich in theory and context
- 🛠 runnable as a Python MVP
- 🔍 traceable at the source-level instead of ending at a black-box package call

### How To Use It 🧭

#### 1. Start with the blueprint

Begin here:

- [`Docs/Stage0_Blueprint.md`](Docs/Stage0_Blueprint.md)

It defines:

- what the repository covers
- how math / physics / computer science are structured
- what research fields each algorithm is expected to include
- what a completed algorithm folder should contain

For the final delivered summary, open:

- [`Docs/Stage0_Final_Summary.md`](Docs/Stage0_Final_Summary.md)

#### 2. Jump into a completed algorithm folder

Completed algorithms live under:

- `Algorithms/<discipline-category-id-name>/`

Each finished folder is expected to contain:

- `README.md`
- `demo.py`
- `meta.json`

That gives you a simple workflow:

1. Read the note
2. Run the MVP
3. Inspect the implementation
4. Understand the algorithm path

#### 3. Run the Python MVP

This repo uses `uv` for Python dependency management.

Setup:

```bash
uv sync
```

Then run a finished demo:

```bash
cd "Algorithms/<discipline-category-id-name>"
uv run python demo.py
```

### What You Get After The Blueprint Is Finished 🎁

Stage0 is already complete, so this is not hypothetical anymore.

You get:

- 🗺 a cross-discipline algorithm map across math, physics, and CS
- 🗂 a consistent one-folder-per-algorithm structure
- 📖 concept, history, significance, complexity, dependencies, and applications
- 🧪 runnable Python MVPs
- 🔬 source-level breakdowns instead of black-box references
- 🧱 a reusable algorithm knowledge base for learning, teaching, demos, and prototyping

### Who This Repo Is For 👀

- learners who want structure
- engineers who want runnable baselines
- researchers who want organized notes
- educators who want reusable teaching material
- builders who want to prototype from classical algorithms upward

### Final Hook 🌟

This is not just a catalog of names.  
It is an algorithm infrastructure you can actually **read, run, inspect, and build on top of**.
