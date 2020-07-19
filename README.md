NLchain
=================================================

NLG(Natural Language Generation)とNLU(Natural Language Understanding)のチェーン学習を行う


1. 構成
 * common/
   * 共通コード

 * corpus/
   * 学習用コーパスデータ

 * evaluation/
   * 評価プログラム及びデータ

 * model/
   * ネットワークモデルコード

 * parameter/
   * 学習済ネットワークパラメータ

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
 * コーパス生成
   * corpus/script/EXE-E2E.csh を実行する
 * 学習
   * training/EXE-E2E.csh を実行する
 * 推論
   * inference/README.md を参照のこと
 * 評価
   * evaluation/EXE.csh を実行する

4. Reference
 * Speech Chain
   * Andros Tjandra at el., "Listening while Speeking: Speech Chain by Deep Learning"
     * https://arxiv.org/abs/1707.04879
 * Transformer
   * https://github.com/bentrevett/pytorch-seq2seq/

Copyright 2020 Keisuke Toyama / AHC Lab / NAIST
