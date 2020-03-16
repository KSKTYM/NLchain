NLchain
=================================================

NLG(Natural Language Generation)とNLU(Natural Language Understanding)のチェーン学習を行う


1. 構成
 * common/
   * 共通コード

 * corpus/
   * 学習用コーパスデータ

 * evaluation/
   * 評価プログラム及びデータ → 未実装

 * model/
   * ネットワークモデルコード

 * parameter/
   * ネットワークパラメータ

 * inference/
   * 推論コード

 * training/
   * 学習コード

2. 必要なもの
 * Python3
   * spacy
   * Matplotlib
   * japanize_matplotlib
   * cloudpickle
   * dill
   * torch
   * torchtext
 
3. 実行方法
 * 学習
   * training/EXE-##.csh を参照のこと
 * 推論
   * inference/README.md を参照のこと

4. Reference
 * Speech Chain
   * Andros Tjandra at el., "Listening while Speeking: Speech Chain by Deep Learning"
     * https://arxiv.org/abs/1707.04879
 * Transformer
   * https://github.com/bentrevett/pytorch-seq2seq/

Copyright 2020 Keisuke Toyama / AHC Lab / NAIST
