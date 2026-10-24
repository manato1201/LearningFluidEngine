# メモ: ナビエ–ストークス問題の話題の扱い方(2026-10)

学習用リポジトリとして、**誤情報を避ける**ための覚え書き。ここは数学の結論を述べる場所ではなく、
「何が報じられていて、何が未確定か」と「実装にどう影響させるか(させないか)」だけを残す。

## 1. 報じられていること(2026-10 時点・**争点あり**)

- 2026-09-08、OpenAI が「内部の未公開モデルが、**強制項つき**の 3 次元ナビエ–ストークス方程式の有限時間ブローアップを証明した」と発表した。Lean での形式化にも言及している(報道による)。
- **Clay 数学研究所の懸賞問題は強制項のない場合**で、別の問題。懸賞の条件を満たすかは未確定で、Clay は受理していない。独立した検証も済んでいない。
- 同時期に、別の研究者(Buckmaster 氏ら)が関連する方程式のブローアップを示す Lean 検証つきのプレプリントを公開したと報じられ、**功績・先後関係をめぐる論争**がある。両者の主張は食い違っており、報道ごとに細部(日付の差・証明の長さ・関与した人物名)も一致しない。
- 証明そのものは、確認した範囲では未公開。

**したがって「ナビエ–ストークス問題が解かれた」と書くのは不正確**。「ある発表があり、強制項つきの話で、検証と論争の途中」と書く。

## 2. このリポジトリでの扱い(決めごと)

1. 数値シミュレーションの話題では、**強制項(重力・外力)がある場合とない場合は別の問題**として扱う。
2. 理論的な結論(滑らかな解が必ず存在する/しない)を**前提にした実装・説明は書かない**。
3. 本リポジトリの SPH / FLIP / 格子法 / GNN / カールノイズは、方程式を厳密に解くものではなく、
   **近似**である(ビューアにも「近似であり、物理的に厳密ではありません」を固定表示している)。
4. 新しい話題を足すときは、出典の URL と「争点あり」の注記を付ける。

## 3. 出典(検索で確認した報道。一次資料ではない)

- [OpenAI Claims a Navier–Stokes Proof, Amid a Dispute Over Credit(NYU Shanghai)](https://rits.shanghai.nyu.edu/ai/openai-navier-stokes-proof-credit-dispute/)
- [OpenAI Navier–Stokes: Proof Dispute(lilting.ch)](https://lilting.ch/en/articles/navier-stokes-openai-blowup-dispute)
- [The OpenAI Navier–Stokes controversy explained(Analytics India Magazine)](https://analyticsindiamag.com/ai-features/the-openai-navierstokes-controversy-explained)
- [OpenAI Navier–Stokes claim: verification and credit(The Next Web)](https://thenextweb.com/news/openai-navier-stokes-claim-verification-credit)
- [Did AI solve the Navier-Stokes problem? OpenAI faces scrutiny(YourStory)](https://yourstory.com/ai-story/navier-stokes-math-dispute-openai-proof-claim)

## 4. 注意・未確認

- 数学的な真偽は、このメモの筆者には判断できない。上は**報道・検索結果の範囲**の整理で、一次資料(論文・プレプリント)は確認していない。
- 報道の間で細部が食い違うため、日付・人名・ページ数などは**書かない**ことにした。
- 状況は変わりうる。参照するときは最新の一次資料と Clay 研究所の発表を確認すること。
